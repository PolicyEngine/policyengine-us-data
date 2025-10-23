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
from policyengine_us_data.utils.census import (
    STATE_ABBREV_TO_FIPS,
    pull_acs_table,
)
from policyengine_us_data.utils.db import parse_ucgid, get_geographic_strata
from policyengine_us_data.utils.db_metadata import (
    get_or_create_source,
    get_or_create_variable_group,
    get_or_create_variable_metadata,
)


def extract_administrative_medicaid_data(year):
    item = "6165f45b-ca93-5bb5-9d06-db29c692a360"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        session = requests.Session()
        session.headers.update(headers)

        metadata_url = f"https://data.medicaid.gov/api/1/metastore/schemas/dataset/items/{item}?show-reference-ids=false"
        print(f"Attempting to fetch Medicaid metadata from: {metadata_url}")

        response = session.get(metadata_url, timeout=30)
        response.raise_for_status()

        metadata = response.json()

        if (
            "distribution" not in metadata
            or len(metadata["distribution"]) == 0
        ):
            raise ValueError(
                f"No distribution found in metadata for item {item}"
            )

        data_url = metadata["distribution"][0]["data"]["downloadURL"]
        print(f"Downloading Medicaid data from: {data_url}")

        try:
            state_admin_df = pd.read_csv(data_url)
            print(
                f"Successfully downloaded {len(state_admin_df)} rows of Medicaid administrative data"
            )
            return state_admin_df
        except Exception as csv_error:
            print(f"\nError downloading CSV from: {data_url}")
            print(f"Error: {csv_error}")
            print(
                f"\nThe metadata API returned successfully, but the data file doesn't exist."
            )
            print(f"This suggests the dataset has been updated/moved.")
            print(f"Please visit https://data.medicaid.gov/ and search for:")
            print(
                f"  - 'Medicaid Enrollment' or 'T-MSIS' or 'Performance Indicators'"
            )
            print(f"Then update the item ID in the code (currently: {item})\n")
            raise

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"\n404 Error: Medicaid metadata item not found.")
            print(f"The item ID '{item}' may have changed.")
            print(
                f"Please check https://data.medicaid.gov/ for updated dataset IDs."
            )
            print(f"Search for 'Medicaid Enrollment' or 'T-MSIS' datasets.\n")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error downloading Medicaid data: {e}")
        raise


def extract_survey_medicaid_data(year):
    return pull_acs_table("S2704", "District", year)


def transform_administrative_medicaid_data(state_admin_df, year):
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

    state_df = state_df.rename(
        columns={"Total Medicaid Enrollment": "medicaid_enrollment"}
    )
    state_df["ucgid_str"] = "0400000US" + state_df["FIPS"].astype(str)

    return state_df[["ucgid_str", "medicaid_enrollment"]]


def transform_survey_medicaid_data(cd_survey_df):
    cd_df = cd_survey_df[
        ["GEO_ID", "state", "congressional district", "S2704_C02_006E"]
    ]

    cd_df = cd_df.rename(
        columns={
            "S2704_C02_006E": "medicaid_enrollment",
            "GEO_ID": "ucgid_str",
        }
    )
    cd_df = cd_df.loc[cd_df.state != "72"]

    return cd_df[["ucgid_str", "medicaid_enrollment"]]


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
            notes="State-level Medicaid enrollment from administrative records",
        )

        survey_source = get_or_create_source(
            session,
            name="Census ACS Table S2704",
            source_type=SourceType.SURVEY,
            vintage=f"{year} ACS 1-year estimates",
            description="American Community Survey health insurance coverage data",
            url="https://data.census.gov/",
            notes="Congressional district level Medicaid coverage from ACS",
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
            description="Medicaid enrollment and spending",
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
            notes="Number of people enrolled in Medicaid (person_count with medicaid_enrolled==True)",
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
        medicaid_stratum_lookup = {
            "national": nat_stratum.stratum_id,
            "state": {},
        }

        # State -------------------
        for _, row in long_state.iterrows():
            # Parse the UCGID to get state_fips
            geo_info = parse_ucgid(row["ucgid_str"])
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
            medicaid_stratum_lookup["state"][
                state_fips
            ] = new_stratum.stratum_id

        # District -------------------
        for _, row in long_cd.iterrows():
            # Parse the UCGID to get district info
            geo_info = parse_ucgid(row["ucgid_str"])
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


def main():
    year = 2023

    # Extract ------------------------------
    state_admin_df = extract_administrative_medicaid_data(year)
    cd_survey_df = extract_survey_medicaid_data(year)

    # Transform -------------------
    long_state = transform_administrative_medicaid_data(state_admin_df, year)
    long_cd = transform_survey_medicaid_data(cd_survey_df)

    # Validate consistency between sources
    nc_cd_sum = (
        long_cd.loc[long_cd.ucgid_str.str.contains("5001800US37")]
        .medicaid_enrollment.astype(int)
        .sum()
    )
    nc_state_sum = long_state.loc[long_state.ucgid_str == "0400000US37"][
        "medicaid_enrollment"
    ].values[0]
    assert (
        nc_cd_sum > 0.5 * nc_state_sum
    ), f"NC CD sum ({nc_cd_sum}) is too low compared to state sum ({nc_state_sum})"
    assert (
        nc_cd_sum <= nc_state_sum
    ), f"NC CD sum ({nc_cd_sum}) exceeds state sum ({nc_state_sum})"

    # Load -----------------------
    load_medicaid_data(long_state, long_cd, year)


if __name__ == "__main__":
    main()
