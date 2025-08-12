import requests
import zipfile
import io
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import us
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.census import (
    get_census_docs,
    pull_acs_table,
    STATE_NAME_TO_FIPS,
)


STATE_NAME_TO_FIPS = {
    "Alabama": "01",
    "Alaska": "02",
    "Arizona": "04",
    "Arkansas": "05",
    "California": "06",
    "Colorado": "08",
    "Connecticut": "09",
    "Delaware": "10",
    "District of Columbia": "11",
    "Florida": "12",
    "Georgia": "13",
    "Hawaii": "15",
    "Idaho": "16",
    "Illinois": "17",
    "Indiana": "18",
    "Iowa": "19",
    "Kansas": "20",
    "Kentucky": "21",
    "Louisiana": "22",
    "Maine": "23",
    "Maryland": "24",
    "Massachusetts": "25",
    "Michigan": "26",
    "Minnesota": "27",
    "Mississippi": "28",
    "Missouri": "29",
    "Montana": "30",
    "Nebraska": "31",
    "Nevada": "32",
    "New Hampshire": "33",
    "New Jersey": "34",
    "New Mexico": "35",
    "New York": "36",
    "North Carolina": "37",
    "North Dakota": "38",
    "Ohio": "39",
    "Oklahoma": "40",
    "Oregon": "41",
    "Pennsylvania": "42",
    "Rhode Island": "44",
    "South Carolina": "45",
    "South Dakota": "46",
    "Tennessee": "47",
    "Texas": "48",
    "Utah": "49",
    "Vermont": "50",
    "Virginia": "51",
    "Washington": "53",
    "West Virginia": "54",
    "Wisconsin": "55",
    "Wyoming": "56",
}

# Administrative data ------------------------------------------------

def extract_administrative_snap_data(year=2023):
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


def load_administrative_snap_data(df_states, year):

    DATABASE_URL = "sqlite:///policy_data.db"
    engine = create_engine(DATABASE_URL)

    Session = sessionmaker(bind=engine)
    session = Session()

    stratum_lookup = {}

    # National ----------------
    nat_stratum = Stratum(
        parent_stratum_id=None, stratum_group_id=0, notes="Geo: 0100000US Received SNAP Benefits"
    )
    nat_stratum.constraints_rel = [
        StratumConstraint(
            constraint_variable="ucgid_str",
            operation="in",
            value="0100000US",
        ),
        StratumConstraint(
            constraint_variable="snap",
            operation="is_greater_than",
            value="0",
        ),
    ]
    # No target at the national level is provided at this time.

    session.add(nat_stratum)
    session.flush()
    stratum_lookup["National"] = nat_stratum.stratum_id

    # State -------------------
    stratum_lookup["State"] = {} 
    for _, row in df_states.iterrows():

        note = f"Geo: {row['ucgid_str']} Received SNAP Benefits"
        parent_stratum_id = nat_stratum.stratum_id

        new_stratum = Stratum(
            parent_stratum_id=parent_stratum_id, stratum_group_id=0, notes=note
        )
        new_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="ucgid_str",
                operation="in",
                value=row["ucgid_str"],
            ),
            StratumConstraint(
                constraint_variable="snap",
                operation="is_greater_than",
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
        stratum_lookup["State"][row['ucgid_str']] = new_stratum.stratum_id

    session.commit()
    return stratum_lookup


# Survey data ------------------------------------------------------

def extract_survey_snap_data(year):

    raw_dfs = {}
    for geo in ["District", "State", "National"]:
        df = pull_acs_table("S2201", geo, year)
        raw_dfs[geo] = df

    return raw_dfs


def transform_survey_snap_data(raw_dfs):

    dfs = {}
    for geo in raw_dfs.keys():
        df = raw_dfs[geo] 
        dfs[geo] = df_data = df[["GEO_ID", "S2201_C03_001E"]].rename({
            "GEO_ID": "ucgid_str",
            "S2201_C03_001E": "snap_household_ct"
            }, axis=1
        )[
            ~df["GEO_ID"].isin(
                [  # Puerto Rico's state and district
                    "0400000US72",
                    "5001800US7298",
                ]
            )
        ].copy()

    return dfs


def load_survey_snap_data(survey_dfs, year, stratum_lookup ={}):
    """Use an already defined stratum_lookup to load the survey SNAP data"""

    DATABASE_URL = "sqlite:///policy_data.db"
    engine = create_engine(DATABASE_URL)

    Session = sessionmaker(bind=engine)
    session = Session()

    # National. Use the stratum from the administrative data function
    nat_df = survey_dfs["National"]
    nat_stratum = session.get(Stratum, stratum_lookup["National"])

    nat_stratum.targets_rel.append(
        Target(
            variable="household_count",
            period=year,
            value=nat_df["snap_household_ct"],
            source_id=4,
            active=True,
        )
    )
    session.add(nat_stratum)
    session.flush()

    # Skipping state for now, but 
    # # State. Also use the stratum from the administrative data function
    # state_df = survey_dfs["State"]
    # for _, row in state_df.iterrows():
    #     print(row)
    #     state_stratum = session.get(Stratum, stratum_lookup["State"][row["ucgid_str"]])

    #     state_stratum.targets_rel.append(
    #         Target(
    #             variable="household_count",
    #             period=year,
    #             value=row["snap_household_ct"],
    #             source_id=4,
    #             active=True,
    #         )
    #     )
    #     session.add(state_stratum)
    #     session.flush()

    # You will need to create new strata for districts
    district_df = survey_dfs["District"]
    for _, row in district_df.iterrows():
        note = f"Geo: {row['ucgid_str']} Received SNAP Benefits"
        state_ucgid_str = '0400000US' + row['ucgid_str'][9:11]
        state_stratum_id = stratum_lookup['State'][state_ucgid_str]
        new_stratum = Stratum(
            parent_stratum_id=state_stratum_id, stratum_group_id=0, notes=note
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
                value='0',
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
    year = 2023

    # Extract ---------
    zip_file_admin = extract_administrative_snap_data()
    raw_survey_dfs = extract_survey_snap_data(year)

    # Transform -------
    state_admin_df = transform_administrative_snap_data(zip_file_admin, year)
    survey_dfs = transform_survey_snap_data(raw_survey_dfs)

    # Load -----------
    stratum_lookup = load_administrative_snap_data(state_admin_df, year)
    load_survey_snap_data(survey_dfs, year, stratum_lookup)


if __name__ == "__main__":
    main()
