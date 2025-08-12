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
    pull_subject_table,
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


def transform_snap_administrative_data(zip_file, year):
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

    # I don't think I need to make this long, because it's going to be 3 different variables
    #df_states[['ucgid_str', 'Households']]
    #df_states[['ucgid_str', 'Persons']]
    #df_states[['ucgid_str', 'Cost']]

    return df_states


def load_snap_administrative_data(?, year):

    year = 2023

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
            operation="equals",
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
                operation="equals",
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



# Moving  away from administrative data to get the survey data ------



def extract_survey_snap_data(year):

    # Household count data -----
    data = pull_acs_table("S2201", "National", 2023)
    data["S2201_C03_001E"]


    # Ha, this is off my a factor of 1000, and ACS does not report dollars in 1000s
    # TODO: try to figure it out.
    data = pull_acs_table("B19058", "State", 2023)
    np.sum(data["B19058_001E"].values.astype(int)) / 1E9



    raw_dfs = {}
    for geo in ["District", "State", "National"]:
        df = pull_subject_table(group, geo, year)
        df_data = df.rename(columns=rename_mapping)[
            ["GEO_ID", "NAME"] + list(label_to_short_name_mapping.values())
        ]
        if geo == "State":
            raw_dfs["DC"] = df_data[df_data["GEO_ID"].isin(["0400000US11"])]

        # Filter out Puerto Rico
        df_geos = df_data[
            ~df_data["GEO_ID"].isin(
                [
                    "5001800US7298",
                    "0400000US72",
                ]
            )
        ].copy()
        raw_dfs[geo] = df_geos
        SAVE_DIR = Path(get_data_directory() / "input" / "demographics")
        df_geos.to_csv(SAVE_DIR / f"raw_snap_{geo}.csv", index=False)

    folder_path = (
        f"{get_data_directory()}/targets/edition=raw/"
        f"base_period={year}/reference_period={year}/"
        f"variable=snap_households/"
    )
    raw_out = pd.concat([
        raw_dfs['National'][['GEO_ID', 'overall']],
        raw_dfs['State'][['GEO_ID', 'overall']],
        raw_dfs['DC'][['GEO_ID', 'overall']],
        raw_dfs['District'][['GEO_ID', 'overall']]
    ]).rename({"GEO_ID": "geography_id", "overall": "value"}, axis=1)
        
    raw_out.to_csv(os.path.join(folder_path, "part-001.csv"), index=False)     

    additive_dfs = enforce_geographic_self_consistency(raw_dfs, 'overall')    
    usda_snap_df = extract_usda_snap_data()
    adjusted_dfs = adjust_to_administrative_data(additive_dfs, 'overall', usda_snap_df)
    assert check_geographic_consistency(adjusted_dfs, 'overall') 

    folder_path = (
        f"{get_data_directory()}/targets/edition=cleaned/"
        f"base_period={year}/reference_period={year}/"
        f"variable=snap_households/"
    )
 
    clean_out = pd.concat([
        adjusted_dfs['National'][['GEO_ID', 'overall']],
        adjusted_dfs['State'][['GEO_ID', 'overall']],
        adjusted_dfs['DC'][['GEO_ID', 'overall']],
        adjusted_dfs['District'][['GEO_ID', 'overall']]
    ]).rename({"GEO_ID": "geography_id", "overall": "value"}, axis=1)
 
    clean_out.to_csv(os.path.join(folder_path, "part-001.csv"), index=False)


def reformat_cleaned_data():
    """Temporary conversion function"""
    benefits_dir = Path(get_data_directory() / 'input' / 'benefits')
   
    snap_filepath = Path(
        get_data_directory(),
        "targets",
        "edition=cleaned",
        "base_period=2023",
        "reference_period=2023",
        "variable=snap_households",
        "part-001.csv"
    )
    snap_data = pd.read_csv(snap_filepath)
    geo_hierarchies = pd.read_csv(Path(get_data_directory(), 'meta', 'geo_hierarchies.csv'))
    
    # Use Type II SCD to Filter geo_hierarchies for the year 2023
    geo_hierarchies['start_date'] = pd.to_datetime(geo_hierarchies['start_date'])
    geo_hierarchies['end_date'] = pd.to_datetime(geo_hierarchies['end_date'])
    geo_hierarchies_2023 = geo_hierarchies[
        (geo_hierarchies['start_date'] <= '2023-01-01') &
        (geo_hierarchies['end_date'] >= '2023-01-01')
    ]
    
    merged_data = pd.merge(snap_data, geo_hierarchies_2023, left_on='geography_id', right_on='geography_id')
    
    def create_cleaned_df(data, geo_name_map=None, geo_name_prefix=''):
        df = pd.DataFrame()
        df['GEO_ID'] = data['geography_id']
        if geo_name_map:
            df['GEO_NAME'] = data['geography_id'].map(geo_name_map)
        elif 'geography_name' in data.columns:
            df['GEO_NAME'] = data['geography_name']
        else:
            df['GEO_NAME'] = ''
    
        df['AGI_LOWER_BOUND'] = ''
        df['AGI_UPPER_BOUND'] = ''
        df['VALUE'] = data['value']
        df['IS_COUNT'] = 1
        df['VARIABLE'] = 'snap_households'
        return df
    
    # National data
    national_data = merged_data[merged_data['geography_type'] == 'nation'].copy()
    national_data['geography_name'] = 'US'
    cleaned_national = create_cleaned_df(national_data)
    cleaned_national.to_csv(Path(get_data_directory(), 'input', 'benefits', 'cleaned_snap_national.csv'), index=False)
    
    # State data
    state_data = merged_data[merged_data['geography_type'] == 'state-equivalent'].copy()
    # TODO: fix this redundancy if this becomes permanenent
    state_fips_map = {
        '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC',
        '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY',
        '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT',
        '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
        '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
        '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY'
    }
    state_data['state_fips'] = state_data['geography_id'].str[-2:]
    state_data['geography_name'] = state_data['state_fips'].map(state_fips_map)
    cleaned_state = create_cleaned_df(state_data)
    cleaned_state.to_csv(Path('us-congressional-districts/data/input/benefits/cleaned_snap_state.csv', index=False)
    cleaned_state.to_csv(Path(get_data_directory(), 'input', 'benefits', 'cleaned_snap_state.csv'), index=False)
    
    # District data
    district_data = merged_data[merged_data['geography_type'] == 'district'].copy()
    district_data['state_fips'] = district_data['geography_id'].str[9:11]
    district_data['district_num'] = district_data['geography_id'].str[11:]
    district_data['geography_name'] = district_data['state_fips'].map(state_fips_map) + ' - District ' + district_data['district_num']
    cleaned_district = create_cleaned_df(district_data)
    cleaned_district["VALUE"] = cleaned_district["VALUE"].round().astype(int)
    cleaned_district.to_csv(Path(get_data_directory(), 'input', 'benefits', 'cleaned_snap_district.csv'), index=False)



if __name__ == "__main__":
    process_snap_data(2023)



def main() -> None:
    year = 2023

    zip_file = extract_snap_data(2023)


if __name__ == "__main__":
    main()
