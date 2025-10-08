import requests

import pandas as pd
import numpy as np
from sqlmodel import Session, create_engine, select

from policyengine_us_data.storage import STORAGE_FOLDER

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    SourceType,
)
from policyengine_us_data.utils.census import STATE_ABBREV_TO_FIPS
from policyengine_us_data.utils.db import parse_ucgid, get_geographic_strata
from policyengine_us_data.utils.db_metadata import (
    get_or_create_source,
    get_or_create_variable_group,
    get_or_create_variable_metadata,
)


def extract_medicaid_data(year):
    base_url = (
        f"https://api.census.gov/data/{year}/acs/acs1/subject?get=group(S2704)"
    )
    url = f"{base_url}&for=congressional+district:*"
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    headers = data[0]
    data_rows = data[1:]
    cd_survey_df = pd.DataFrame(data_rows, columns=headers)

    item = "6165f45b-ca93-5bb5-9d06-db29c692a360"
    response = requests.get(
        f"https://data.medicaid.gov/api/1/metastore/schemas/dataset/items/{item}?show-reference-ids=false"
    )
    metadata = response.json()

    data_url = metadata["distribution"][0]["data"]["downloadURL"]
    state_admin_df = pd.read_csv(data_url)

    return cd_survey_df, state_admin_df


def transform_medicaid_data(state_admin_df, cd_survey_df, year):

    reporting_period = year * 100 + 12
    print(f"Reporting period is {reporting_period}")
    state_df = state_admin_df.loc[
        (state_admin_df["Reporting Period"] == reporting_period)
        & (state_admin_df["Final Report"] == "Y"),
        [
            "State Abbreviation",
            "Reporting Period",
            "Total Medicaid Enrollment",
        ],
    ]

    state_df["FIPS"] = state_df["State Abbreviation"].map(STATE_ABBREV_TO_FIPS)

    cd_df = cd_survey_df[
        ["GEO_ID", "state", "congressional district", "S2704_C02_006E"]
    ]

    nc_cd_sum = cd_df.loc[cd_df.state == "37"].S2704_C02_006E.astype(int).sum()
    nc_state_sum = state_df.loc[state_df.FIPS == "37"][
        "Total Medicaid Enrollment"
    ].values[0]
    assert nc_cd_sum > 0.5 * nc_state_sum
    assert nc_cd_sum <= nc_state_sum

    state_df = state_df.rename(
        columns={"Total Medicaid Enrollment": "medicaid_enrollment"}
    )
    state_df["ucgid_str"] = "0400000US" + state_df["FIPS"].astype(str)

    cd_df = cd_df.rename(
        columns={
            "S2704_C02_006E": "medicaid_enrollment",
            "GEO_ID": "ucgid_str",
        }
    )
    cd_df = cd_df.loc[cd_df.state != "72"]

    out_cols = ["ucgid_str", "medicaid_enrollment"]
    return state_df[out_cols], cd_df[out_cols]


def load_medicaid_data(long_state, long_cd, year):

    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        # Get or create sources
        admin_source = get_or_create_source(
            session,
            name="Medicaid T-MSIS",
            source_type=SourceType.ADMINISTRATIVE,
            vintage=f"{year} Final Report",
            description="Medicaid Transformed MSIS administrative enrollment data",
            url="https://data.medicaid.gov/",
            notes="State-level Medicaid enrollment from administrative records"
        )
        
        survey_source = get_or_create_source(
            session,
            name="Census ACS Table S2704",
            source_type=SourceType.SURVEY,
            vintage=f"{year} ACS 1-year estimates",
            description="American Community Survey health insurance coverage data",
            url="https://data.census.gov/",
            notes="Congressional district level Medicaid coverage from ACS"
        )
        
        # Get or create Medicaid variable group
        medicaid_group = get_or_create_variable_group(
            session,
            name="medicaid_recipients",
            category="benefit",
            is_histogram=False,
            is_exclusive=False,
            aggregation_method="sum",
            display_order=3,
            description="Medicaid enrollment and spending"
        )
        
        # Create variable metadata
        # Note: The actual target variable used is "person_count" with medicaid_enrolled==True constraint
        # This metadata entry is kept for consistency with the actual variable being used
        get_or_create_variable_metadata(
            session,
            variable="person_count",
            group=medicaid_group,
            display_name="Medicaid Enrollment",
            display_order=1,
            units="count",
            notes="Number of people enrolled in Medicaid (person_count with medicaid_enrolled==True)"
        )
        
        # Fetch existing geographic strata
        geo_strata = get_geographic_strata(session)
        
        # National ----------------
        # Create a Medicaid stratum as child of the national geographic stratum
        nat_stratum = Stratum(
            parent_stratum_id=geo_strata["national"],
            stratum_group_id=5,  # Medicaid strata group
            notes="National Medicaid Enrolled",
        )
        nat_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="medicaid_enrolled",
                operation="==",
                value="True",
            ),
        ]
        # No target at the national level is provided at this time.

        session.add(nat_stratum)
        session.flush()
        medicaid_stratum_lookup = {"national": nat_stratum.stratum_id, "state": {}}

        # State -------------------
        for _, row in long_state.iterrows():
            # Parse the UCGID to get state_fips
            geo_info = parse_ucgid(row['ucgid_str'])
            state_fips = geo_info["state_fips"]
            
            # Get the parent geographic stratum
            parent_stratum_id = geo_strata["state"][state_fips]
            
            note = f"State FIPS {state_fips} Medicaid Enrolled"

            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=5,  # Medicaid strata group
                notes=note,
            )
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(state_fips),
                ),
                StratumConstraint(
                    constraint_variable="medicaid_enrolled",
                    operation="==",
                    value="True",
                ),
            ]
            new_stratum.targets_rel.append(
                Target(
                    variable="person_count",
                    period=year,
                    value=row["medicaid_enrollment"],
                    source_id=admin_source.source_id,
                    active=True,
                )
            )
            session.add(new_stratum)
            session.flush()
            medicaid_stratum_lookup["state"][state_fips] = new_stratum.stratum_id

        # District -------------------
        for _, row in long_cd.iterrows():
            # Parse the UCGID to get district info
            geo_info = parse_ucgid(row['ucgid_str'])
            cd_geoid = geo_info["congressional_district_geoid"]
            
            # Get the parent geographic stratum
            parent_stratum_id = geo_strata["district"][cd_geoid]
            
            note = f"Congressional District {cd_geoid} Medicaid Enrolled"

            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=5,  # Medicaid strata group
                notes=note,
            )
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="congressional_district_geoid",
                    operation="==",
                    value=str(cd_geoid),
                ),
                StratumConstraint(
                    constraint_variable="medicaid_enrolled",
                    operation="==",
                    value="True",
                ),
            ]
            new_stratum.targets_rel.append(
                Target(
                    variable="person_count",
                    period=year,
                    value=row["medicaid_enrollment"],
                    source_id=survey_source.source_id,
                    active=True,
                )
            )
            session.add(new_stratum)
            session.flush()

        session.commit()


if __name__ == "__main__":

    year = 2023

    # Extract ------------------------------
    cd_survey_df, state_admin_df = extract_medicaid_data(year)

    # Transform -------------------
    long_state, long_cd = transform_medicaid_data(
        state_admin_df, cd_survey_df, year
    )

    # Load -----------------------
    load_medicaid_data(long_state, long_cd, year)
