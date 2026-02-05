import logging
import requests
import zipfile
import io

import pandas as pd
import numpy as np
import us
from sqlmodel import Session, create_engine, select

from policyengine_us_data.storage import STORAGE_FOLDER

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    Source,
    VariableGroup,
    VariableMetadata,
)
from policyengine_us_data.utils.census import (
    pull_acs_table,
    STATE_NAME_TO_FIPS,
)
from policyengine_us_data.utils.db import parse_ucgid, get_geographic_strata
from policyengine_us_data.utils.db_metadata import (
    get_or_create_source,
    get_or_create_variable_group,
    get_or_create_variable_metadata,
)
from policyengine_us_data.utils.raw_cache import (
    is_cached,
    cache_path,
    save_bytes,
    load_bytes,
)

logger = logging.getLogger(__name__)


def extract_administrative_snap_data(year=2023):
    """
    Downloads and extracts annual state-level SNAP data from the USDA FNS zip file.
    """
    cache_file = "snap_fy69tocurrent.zip"
    if is_cached(cache_file):
        logger.info(f"Using cached {cache_file}")
        return zipfile.ZipFile(io.BytesIO(load_bytes(cache_file)))

    url = "https://www.fns.usda.gov/sites/default/files/resource-files/snap-zip-fy69tocurrent-6.zip"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        session.get(
            "https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap",
            timeout=30,
        )
    except Exception:
        pass

    response = session.get(url, timeout=30, allow_redirects=True)
    response.raise_for_status()
    save_bytes(cache_file, response.content)
    return zipfile.ZipFile(io.BytesIO(response.content))


def extract_survey_snap_data(year):
    return pull_acs_table("S2201", "District", year)


def transform_administrative_snap_data(zip_file, year):
    filename = f"FY{str(year)[-2:]}.xlsx"
    with zip_file.open(filename) as f:
        xls = pd.ExcelFile(f)
        tab_results = []
        for sheet_name in [
            "NERO",
            "MARO",
            "SERO",
            "MWRO",
            "SWRO",
            "MPRO",
            "WRO",
        ]:
            df_raw = pd.read_excel(
                xls, sheet_name=sheet_name, header=None, dtype={0: str}
            )

            state_row_mask = (
                df_raw[0].notna()
                & df_raw[1].isna()
                & ~df_raw[0].str.contains("Total", na=False)
                & ~df_raw[0].str.contains("Footnote", na=False)
            )

            df_raw["State"] = df_raw.loc[state_row_mask, 0]
            df_raw["State"] = df_raw["State"].ffill()
            total_rows = df_raw[df_raw[0].eq("Total")].copy()
            total_rows = total_rows.rename(
                columns={
                    1: "Households",
                    2: "Persons",
                    3: "Cost",
                }
            )

            state_totals = total_rows[
                [
                    "State",
                    "Households",
                    "Persons",
                    "Cost",  # Annual (Note: the CostPer* vars are monthly)
                ]
            ]

            tab_results.append(state_totals)

    results_df = pd.concat(tab_results)

    df_states = results_df.loc[
        results_df["State"].isin(STATE_NAME_TO_FIPS.keys())
    ].copy()
    df_states["STATE_FIPS"] = df_states["State"].map(STATE_NAME_TO_FIPS)
    df_states = (
        df_states.loc[~df_states["STATE_FIPS"].isna()]
        .sort_values("STATE_FIPS")
        .reset_index(drop=True)
    )
    df_states["ucgid_str"] = "0400000US" + df_states["STATE_FIPS"]

    return df_states


def transform_survey_snap_data(raw_df):
    df = raw_df.copy()
    return df[["GEO_ID", "S2201_C03_001E"]].rename(
        {"GEO_ID": "ucgid_str", "S2201_C03_001E": "snap_household_ct"}, axis=1
    )[
        ~df["GEO_ID"].isin(
            [  # Puerto Rico's state and district
                "0400000US72",
                "5001800US7298",
            ]
        )
    ]


def load_administrative_snap_data(df_states, year):

    DATABASE_URL = (
        f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    )
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        # Get or create the administrative source
        admin_source = get_or_create_source(
            session,
            name="USDA FNS SNAP Data",
            source_type="administrative",
            vintage=f"FY {year}",
            description="SNAP administrative data from USDA Food and Nutrition Service",
            url="https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap",
            notes="State-level administrative totals for households and costs",
        )

        # Get or create the SNAP variable group
        snap_group = get_or_create_variable_group(
            session,
            name="snap_recipients",
            category="benefit",
            is_histogram=False,
            is_exclusive=False,
            aggregation_method="sum",
            display_order=2,
            description="SNAP (food stamps) recipient counts and benefits",
        )

        # Get or create variable metadata
        get_or_create_variable_metadata(
            session,
            variable="snap",
            group=snap_group,
            display_name="SNAP Benefits",
            display_order=1,
            units="dollars",
            notes="Annual SNAP benefit costs",
        )

        get_or_create_variable_metadata(
            session,
            variable="household_count",
            group=snap_group,
            display_name="SNAP Household Count",
            display_order=2,
            units="count",
            notes="Number of households receiving SNAP",
        )

        # Fetch existing geographic strata
        geo_strata = get_geographic_strata(session)

        # National ----------------
        # Create a SNAP stratum as child of the national geographic stratum
        nat_stratum = Stratum(
            parent_stratum_id=geo_strata["national"],
            stratum_group_id=4,  # SNAP strata group
            notes="National Received SNAP Benefits",
        )
        nat_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="snap",
                operation=">",
                value="0",
            ),
        ]
        # No target at the national level is provided at this time. Keeping it
        # so that the state strata can have a parent stratum

        session.add(nat_stratum)
        session.flush()
        snap_stratum_lookup = {"national": nat_stratum.stratum_id, "state": {}}

        # State -------------------
        for _, row in df_states.iterrows():
            # Parse the UCGID to get state_fips
            geo_info = parse_ucgid(row["ucgid_str"])
            state_fips = geo_info["state_fips"]

            # Get the parent geographic stratum
            parent_stratum_id = geo_strata["state"][state_fips]

            note = f"State FIPS {state_fips} Received SNAP Benefits"

            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=4,  # SNAP strata group
                notes=note,
            )
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(state_fips),
                ),
                StratumConstraint(
                    constraint_variable="snap",
                    operation=">",
                    value="0",
                ),
            ]
            # Two targets now. Same data source. Same stratum
            new_stratum.targets_rel.append(
                Target(
                    variable="household_count",
                    period=year,
                    value=row["Households"],
                    source_id=admin_source.source_id,
                    active=True,
                )
            )
            new_stratum.targets_rel.append(
                Target(
                    variable="snap",
                    period=year,
                    value=row["Cost"],
                    source_id=admin_source.source_id,
                    active=True,
                )
            )
            session.add(new_stratum)
            session.flush()
            snap_stratum_lookup["state"][state_fips] = new_stratum.stratum_id

        session.commit()
    return snap_stratum_lookup


def load_survey_snap_data(survey_df, year, snap_stratum_lookup):
    """Use an already defined snap_stratum_lookup to load the survey SNAP data

    Note: snap_stratum_lookup should contain the SNAP strata created by
    load_administrative_snap_data, so we don't recreate them.
    """

    DATABASE_URL = (
        f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    )
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        # Get or create the survey source
        survey_source = get_or_create_source(
            session,
            name="Census ACS Table S2201",
            source_type="survey",
            vintage=f"{year} ACS 5-year estimates",
            description="American Community Survey SNAP/Food Stamps data",
            url="https://data.census.gov/",
            notes="Congressional district level SNAP household counts from ACS",
        )

        # Fetch existing geographic strata
        geo_strata = get_geographic_strata(session)

        # Create new strata for districts whose households recieve SNAP benefits
        district_df = survey_df.copy()
        for _, row in district_df.iterrows():
            # Parse the UCGID to get district info
            geo_info = parse_ucgid(row["ucgid_str"])
            cd_geoid = geo_info["congressional_district_geoid"]

            # Get the parent geographic stratum
            parent_stratum_id = geo_strata["district"][cd_geoid]

            note = f"Congressional District {cd_geoid} Received SNAP Benefits"

            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=4,  # SNAP strata group
                notes=note,
            )

            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="congressional_district_geoid",
                    operation="==",
                    value=str(cd_geoid),
                ),
                StratumConstraint(
                    constraint_variable="snap",
                    operation=">",
                    value="0",
                ),
            ]
            new_stratum.targets_rel.append(
                Target(
                    variable="household_count",
                    period=year,
                    value=row["snap_household_ct"],
                    source_id=survey_source.source_id,
                    active=True,
                )
            )
            session.add(new_stratum)
            session.flush()

        session.commit()

    return snap_stratum_lookup


def main():
    year = 2023

    # Extract ---------
    zip_file_admin = extract_administrative_snap_data()
    raw_survey_df = extract_survey_snap_data(year)

    # Transform -------
    state_admin_df = transform_administrative_snap_data(zip_file_admin, year)
    district_survey_df = transform_survey_snap_data(raw_survey_df)

    # Load -----------
    snap_stratum_lookup = load_administrative_snap_data(state_admin_df, year)
    load_survey_snap_data(district_survey_df, year, snap_stratum_lookup)


if __name__ == "__main__":
    main()
