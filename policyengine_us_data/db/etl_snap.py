import requests
import zipfile
import io

import pandas as pd
import numpy as np
import us
from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.census import (
    pull_acs_table,
    STATE_NAME_TO_FIPS,
)


def extract_administrative_snap_data(year=2024):
    """
    Downloads and extracts annual state-level SNAP data from the USDA FNS zip file.
    """
    url = "https://www.fns.usda.gov/sites/default/files/resource-files/snap-zip-fy69tocurrent-6.zip"

    # Note: extra complexity in request due to regional restrictions on downloads (e.g., Spain)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        session = requests.Session()
        session.headers.update(headers)

        # Try to visit the main page first to get any necessary cookies
        main_page = "https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap"
        try:
            session.get(main_page, timeout=30)
        except:
            pass  # Ignore errors on the main page

        response = session.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        # Try alternative URL or method
        try:
            alt_url = "https://www.fns.usda.gov/sites/default/files/resource-files/snap-zip-fy69tocurrent-6.zip"
            response = session.get(alt_url, timeout=30, allow_redirects=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e2:
            print(f"Alternative URL also failed: {e2}")
            return None

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

    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    stratum_lookup = {}

    with Session(engine) as session:
        # National ----------------
        nat_stratum = Stratum(
            parent_stratum_id=None,
            stratum_group_id=0,
            notes="Geo: 0100000US Received SNAP Benefits",
        )
        nat_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="in",
                value="0100000US",
            ),
            StratumConstraint(
                constraint_variable="snap",
                operation="greater_than",
                value="0",
            ),
        ]
        # No target at the national level is provided at this time. Keeping it
        # so that the state strata can have a parent stratum

        session.add(nat_stratum)
        session.flush()
        stratum_lookup["National"] = nat_stratum.stratum_id

        # State -------------------
        stratum_lookup["State"] = {}
        for _, row in df_states.iterrows():

            note = f"Geo: {row['ucgid_str']} Received SNAP Benefits"
            parent_stratum_id = nat_stratum.stratum_id

            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=0,
                notes=note,
            )
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="ucgid_str",
                    operation="in",
                    value=row["ucgid_str"],
                ),
                StratumConstraint(
                    constraint_variable="snap",
                    operation="greater_than",
                    value="0",
                ),
            ]
            # Two targets now. Same data source. Same stratum
            new_stratum.targets_rel.append(
                Target(
                    variable="household_count",
                    period=year,
                    value=row["Households"],
                    source_id=3,
                    active=True,
                )
            )
            new_stratum.targets_rel.append(
                Target(
                    variable="snap",
                    period=year,
                    value=row["Cost"],
                    source_id=3,
                    active=True,
                )
            )
            session.add(new_stratum)
            session.flush()
            stratum_lookup["State"][row["ucgid_str"]] = new_stratum.stratum_id

        session.commit()
    return stratum_lookup


def load_survey_snap_data(survey_df, year, stratum_lookup=None):
    """Use an already defined stratum_lookup to load the survey SNAP data"""

    if stratum_lookup is None:
        raise ValueError("stratum_lookup must be provided")

    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        # Create new strata for districts whose households recieve SNAP benefits
        district_df = survey_df.copy()
        for _, row in district_df.iterrows():
            note = f"Geo: {row['ucgid_str']} Received SNAP Benefits"
            state_ucgid_str = "0400000US" + row["ucgid_str"][9:11]
            state_stratum_id = stratum_lookup["State"][state_ucgid_str]
            new_stratum = Stratum(
                parent_stratum_id=state_stratum_id,
                stratum_group_id=0,
                notes=note,
            )

            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="ucgid_str",
                    operation="in",
                    value=row["ucgid_str"],
                ),
                StratumConstraint(
                    constraint_variable="snap",
                    operation="greater_than",
                    value="0",
                ),
            ]
            new_stratum.targets_rel.append(
                Target(
                    variable="household_count",
                    period=year,
                    value=row["snap_household_ct"],
                    source_id=4,
                    active=True,
                )
            )
            session.add(new_stratum)
            session.flush()

        session.commit()

    return stratum_lookup


def main():
    year = 2024

    # Extract ---------
    zip_file_admin = extract_administrative_snap_data()
    raw_survey_df = extract_survey_snap_data(year)

    # Transform -------
    state_admin_df = transform_administrative_snap_data(zip_file_admin, year)
    district_survey_df = transform_survey_snap_data(raw_survey_df)

    # Load -----------
    stratum_lookup = load_administrative_snap_data(state_admin_df, year)
    load_survey_snap_data(district_survey_df, year, stratum_lookup)


if __name__ == "__main__":
    main()
