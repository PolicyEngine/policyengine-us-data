import requests
import zipfile
import io
import pandas as pd
import numpy as np

from policyengine_us_data.storage import CALIBRATION_FOLDER
from policyengine_us_data.storage.calibration_targets.pull_age_targets import (
    STATE_NAME_TO_ABBREV,
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


def extract_usda_snap_data(year=2023):
    """
    Downloads and extracts annual state-level SNAP data from the USDA FNS zip file.
    """
    url = "https://www.fns.usda.gov/sites/default/files/resource-files/snap-zip-fy69tocurrent-6.zip"

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

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

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
                    4: "CostPerHousehold",
                    5: "CostPerPerson",
                }
            )

            state_totals = total_rows[
                [
                    "State",
                    "Households",
                    "Persons",
                    "Cost",
                    "CostPerHousehold",
                    "CostPerPerson",
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
    df_states["GEO_ID"] = "0400000US" + df_states["STATE_FIPS"]
<<<<<<< HEAD
    df_states["GEO_NAME"] = "state_" + df_states["State"].map(
        STATE_NAME_TO_ABBREV
    )
=======
    df_states["GEO_NAME"] = df_states["State"].map(STATE_NAME_TO_ABBREV)
>>>>>>> 2b1e40a (start cleaning calibration targets)

    count_df = df_states[["GEO_ID", "GEO_NAME"]].copy()
    count_df["VALUE"] = df_states["Households"]
    count_df["IS_COUNT"] = 1.0
    count_df["DATA_SOURCE"] = "usda_fns"
    count_df["BREAKDOWN_VARIABLE"] = np.nan
    count_df["LOWER_BOUND"] = np.nan
    count_df["UPPER_BOUND"] = np.nan

    amount_df = df_states[["GEO_ID", "GEO_NAME"]].copy()
    amount_df["VALUE"] = df_states["Cost"]
    amount_df["IS_COUNT"] = 0.0
    amount_df["DATA_SOURCE"] = "usda_fns"
    amount_df["BREAKDOWN_VARIABLE"] = np.nan
    amount_df["LOWER_BOUND"] = np.nan
    amount_df["UPPER_BOUND"] = np.nan

    final_df = pd.concat([count_df, amount_df], ignore_index=True)
    final_df["VARIABLE"] = "snap"

    return final_df[
        [
            "DATA_SOURCE",
            "GEO_ID",
            "GEO_NAME",
            "VARIABLE",
            "VALUE",
            "IS_COUNT",
            "BREAKDOWN_VARIABLE",
            "LOWER_BOUND",
            "UPPER_BOUND",
        ]
    ]


def main() -> None:
    out_dir = CALIBRATION_FOLDER
    state_df = extract_usda_snap_data(2024)
    state_df.to_csv(out_dir / "snap.csv", index=False)


if __name__ == "__main__":
    main()
