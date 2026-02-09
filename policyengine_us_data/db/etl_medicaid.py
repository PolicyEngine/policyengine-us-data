import logging

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
from policyengine_us_data.utils.db import (
    parse_ucgid,
    get_geographic_strata,
    etl_argparser,
)
from policyengine_us_data.utils.db_metadata import (
    get_or_create_source,
    get_or_create_variable_group,
    get_or_create_variable_metadata,
)
from policyengine_us_data.utils.raw_cache import (
    is_cached,
    cache_path,
    save_json,
    load_json,
    save_bytes,
)

logger = logging.getLogger(__name__)


def extract_administrative_medicaid_data(year):
    cms_cache = f"medicaid_enrollment_{year}.csv"
    if is_cached(cms_cache):
        logger.info(f"Using cached {cms_cache}")
        return pd.read_csv(cache_path(cms_cache))

    item = "6165f45b-ca93-5bb5-9d06-db29c692a360"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
    }

    session = requests.Session()
    session.headers.update(headers)

    metadata_url = f"https://data.medicaid.gov/api/1/metastore/schemas/dataset/items/{item}?show-reference-ids=false"
    print(f"Attempting to fetch Medicaid metadata from: {metadata_url}")

    response = session.get(metadata_url, timeout=30)
    response.raise_for_status()

    metadata = response.json()

    if "distribution" not in metadata or len(metadata["distribution"]) == 0:
        raise ValueError(f"No distribution found in metadata for item {item}")

    data_url = metadata["distribution"][0]["data"]["downloadURL"]
    print(f"Downloading Medicaid data from: {data_url}")

    state_admin_df = pd.read_csv(data_url)
    state_admin_df.to_csv(cache_path(cms_cache), index=False)
    print(
        f"Successfully downloaded {len(state_admin_df)} rows of Medicaid administrative data"
    )
    return state_admin_df


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
    ].copy()

    state_df["FIPS"] = state_df["State Abbreviation"].map(STATE_ABBREV_TO_FIPS)

    state_df = state_df.rename(
        columns={"Total Medicaid Enrollment": "medicaid_enrollment"}
    )

    # Handle states with 0 or NaN enrollment by using most recent non-zero value
    # This addresses data quality issues where some states have missing Dec data
    problem_states = state_df[
        (state_df["medicaid_enrollment"] == 0)
        | (state_df["medicaid_enrollment"].isna())
    ]["State Abbreviation"].tolist()

    if problem_states:
        print(
            f"Warning: States with 0/NaN enrollment in {reporting_period}: {problem_states}"
        )
        print("Attempting to use most recent non-zero values...")

        for state_abbrev in problem_states:
            # Find most recent non-zero final report for this state
            state_history = state_admin_df[
                (state_admin_df["State Abbreviation"] == state_abbrev)
                & (state_admin_df["Final Report"] == "Y")
                & (state_admin_df["Total Medicaid Enrollment"] > 0)
                & (state_admin_df["Reporting Period"] < reporting_period)
            ].sort_values("Reporting Period", ascending=False)

            if not state_history.empty:
                fallback_value = state_history.iloc[0][
                    "Total Medicaid Enrollment"
                ]
                fallback_period = state_history.iloc[0]["Reporting Period"]
                print(
                    f"  {state_abbrev}: Using {fallback_value:,.0f} from period {fallback_period}"
                )
                state_df.loc[
                    state_df["State Abbreviation"] == state_abbrev,
                    "medicaid_enrollment",
                ] = fallback_value
            else:
                print(f"  {state_abbrev}: No historical data found, keeping 0")

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

    DATABASE_URL = (
        f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    )
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
        if long_cd is None:
            session.commit()
            return

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
    _, year = etl_argparser("ETL for Medicaid calibration targets")

    # Extract ------------------------------
    state_admin_df = extract_administrative_medicaid_data(year)

    # TODO: Re-enable CD survey Medicaid targets once we handle the 119th
    # Congress district codes (5001900US) vs 118th Congress (5001800US)
    # mismatch. The 2024 ACS uses 119th Congress GEO_IDs but the DB
    # geographic strata use 118th Congress codes. Need a remapping step.
    # When re-enabling, also restore the NC validation assert below.
    #
    # cd_survey_df = extract_survey_medicaid_data(year)
    # long_cd = transform_survey_medicaid_data(cd_survey_df)
    # nc_cd_sum = (
    #     long_cd.loc[long_cd.ucgid_str.str.contains("5001800US37")]
    #     .medicaid_enrollment.astype(int)
    #     .sum()
    # )
    # nc_state_sum = long_state.loc[long_state.ucgid_str == "0400000US37"][
    #     "medicaid_enrollment"
    # ].values[0]
    # assert nc_cd_sum > 0.5 * nc_state_sum
    # assert nc_cd_sum <= nc_state_sum

    # Transform -------------------
    long_state = transform_administrative_medicaid_data(state_admin_df, year)

    # Load (state admin only, no CD survey) ---
    load_medicaid_data(long_state, long_cd=None, year=year)


if __name__ == "__main__":
    main()
