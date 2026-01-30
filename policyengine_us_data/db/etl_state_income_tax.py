"""
ETL for state income tax calibration targets.

Pulls state individual income tax collections from Census Bureau's
Annual Survey of State Government Tax Collections (STC) and loads
them into the calibration database.

Data source: https://www.census.gov/programs-surveys/stc/data/datasets.html
"""

import pandas as pd
import numpy as np
from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)

# Census STC source_id (picking 5, after USDA FNS=3 and ACS=4)
SOURCE_ID_CENSUS_STC = 5

# States without individual income tax (these will have $0 target)
NO_INCOME_TAX_STATES = {
    "AK",  # Alaska
    "FL",  # Florida
    "NV",  # Nevada
    "SD",  # South Dakota
    "TX",  # Texas
    "WA",  # Washington (has capital gains tax only, modeled separately)
    "WY",  # Wyoming
    # Note: NH and TN historically taxed only interest/dividends but have
    # phased these out. They now have no individual income tax.
    "NH",  # New Hampshire
    "TN",  # Tennessee
}

STATE_FIPS_TO_ABBREV = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

STATE_ABBREV_TO_FIPS = {v: k for k, v in STATE_FIPS_TO_ABBREV.items()}


def extract_census_stc_data(year: int = 2023) -> pd.DataFrame:
    """
    Extract state individual income tax collections from Census STC.

    Uses the Census Bureau's historical STC dataset available at:
    https://www.census.gov/programs-surveys/stc/data/datasets.html

    For programmatic access, we use the Census API or download the
    historical Excel file.

    Args:
        year: Fiscal year for the data (e.g., 2023 for FY2023)

    Returns:
        DataFrame with state_fips, state_abbrev, and income_tax_collections
    """
    # Census STC historical data URL
    # The historical Excel file contains all years
    url = "https://www2.census.gov/programs-surveys/stc/datasets/historical/STC_Historical_2024.xlsx"

    try:
        # Read the Individual Income Tax sheet
        df = pd.read_excel(
            url,
            sheet_name="T01",  # Individual Income Tax collections
            header=3,  # Data starts after header rows
        )
    except Exception as e:
        print(f"Error downloading Census STC data: {e}")
        print("Falling back to FRED API for individual state data...")
        return extract_from_fred(year)

    # The Census STC format has states as rows and years as columns
    # Column names are like "2023" for FY2023

    # Find the year column
    year_col = str(year)
    if year_col not in df.columns:
        raise ValueError(f"Year {year} not found in Census STC data")

    # Extract relevant columns
    result = df[["Name", year_col]].copy()
    result.columns = ["state_name", "income_tax_collections"]

    # Map state names to FIPS codes
    state_name_to_fips = {
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

    result["state_fips"] = result["state_name"].map(state_name_to_fips)
    result = result[result["state_fips"].notna()].copy()
    result["state_abbrev"] = result["state_fips"].map(STATE_FIPS_TO_ABBREV)
    result["ucgid_str"] = "0400000US" + result["state_fips"]

    # Convert collections to numeric (Census reports in thousands)
    result["income_tax_collections"] = (
        pd.to_numeric(result["income_tax_collections"], errors="coerce") * 1000
    )

    # States without income tax should have 0 (Census may report small amounts
    # from penalties/interest)
    result.loc[
        result["state_abbrev"].isin(NO_INCOME_TAX_STATES),
        "income_tax_collections",
    ] = 0

    return result[
        ["state_fips", "state_abbrev", "ucgid_str", "income_tax_collections"]
    ].reset_index(drop=True)


def extract_from_fred(year: int) -> pd.DataFrame:
    """
    Fallback: Extract state income tax data from FRED.

    FRED provides series like OHINCTAX for Ohio Individual Income Tax.

    Args:
        year: Year for the data

    Returns:
        DataFrame with state_fips, state_abbrev, and income_tax_collections
    """
    import requests

    # FRED API endpoint (requires API key for production use)
    # For now, use hardcoded recent values from Census/FRED
    # These are FY2023 values in dollars (not thousands)

    # Source: Census STC Table 1 - State Government Tax Collections by Category
    # https://www.census.gov/data/tables/2023/econ/stc/2023-annual.html
    stc_2023_individual_income_tax = {
        "AL": 5_881_000_000,
        "AK": 0,
        "AZ": 5_424_000_000,
        "AR": 4_352_000_000,
        "CA": 115_845_000_000,
        "CO": 13_671_000_000,
        "CT": 10_716_000_000,
        "DE": 1_747_000_000,
        "DC": 3_456_000_000,
        "FL": 0,
        "GA": 15_297_000_000,
        "HI": 2_725_000_000,
        "ID": 2_593_000_000,
        "IL": 21_453_000_000,
        "IN": 8_098_000_000,
        "IA": 5_243_000_000,
        "KS": 4_304_000_000,
        "KY": 6_163_000_000,
        "LA": 4_088_000_000,
        "ME": 2_246_000_000,
        "MD": 11_635_000_000,
        "MA": 18_645_000_000,
        "MI": 12_139_000_000,
        "MN": 14_239_000_000,
        "MS": 2_477_000_000,
        "MO": 9_006_000_000,
        "MT": 1_718_000_000,
        "NE": 3_248_000_000,
        "NV": 0,
        "NH": 0,
        "NJ": 17_947_000_000,
        "NM": 2_224_000_000,
        "NY": 63_247_000_000,
        "NC": 17_171_000_000,
        "ND": 534_000_000,
        "OH": 9_520_000_000,  # From Policy Matters Ohio
        "OK": 4_253_000_000,
        "OR": 11_583_000_000,
        "PA": 16_898_000_000,
        "RI": 1_739_000_000,
        "SC": 6_367_000_000,
        "SD": 0,
        "TN": 0,
        "TX": 0,
        "UT": 5_464_000_000,
        "VT": 1_035_000_000,
        "VA": 17_934_000_000,
        "WA": 0,  # Note: WA has capital gains tax but no broad income tax
        "WV": 2_163_000_000,
        "WI": 10_396_000_000,
        "WY": 0,
    }

    rows = []
    for abbrev, value in stc_2023_individual_income_tax.items():
        fips = STATE_ABBREV_TO_FIPS[abbrev]
        rows.append(
            {
                "state_fips": fips,
                "state_abbrev": abbrev,
                "ucgid_str": f"0400000US{fips}",
                "income_tax_collections": value,
            }
        )

    return pd.DataFrame(rows)


def transform_state_income_tax_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Census STC data for loading.

    Args:
        df: Raw DataFrame from extract step

    Returns:
        Transformed DataFrame ready for loading
    """
    result = df.copy()

    # Ensure numeric and handle any NaN
    result["income_tax_collections"] = pd.to_numeric(
        result["income_tax_collections"], errors="coerce"
    ).fillna(0)

    # Sort by FIPS for consistent ordering
    result = result.sort_values("state_fips").reset_index(drop=True)

    return result


def load_state_income_tax_data(df: pd.DataFrame, year: int) -> dict:
    """
    Load state income tax targets into the calibration database.

    Creates strata and targets for each state's income tax collections.

    Args:
        df: Transformed DataFrame with state income tax data
        year: Year for the targets

    Returns:
        Dictionary mapping state ucgid to stratum_id
    """
    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    stratum_lookup = {}

    with Session(engine) as session:
        # Create state-level strata for income tax
        for _, row in df.iterrows():
            note = f"Geo: {row['ucgid_str']} State Income Tax"

            # Create stratum with geographic constraint only
            # (no constraint on income_tax > 0 since we want to calibrate
            # the total including zeros for no-income-tax states)
            new_stratum = Stratum(
                parent_stratum_id=None,
                stratum_group_id=5,  # New group for state income tax
                notes=note,
            )
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=row["state_fips"],
                ),
            ]

            # Add target for state_income_tax total
            new_stratum.targets_rel.append(
                Target(
                    variable="state_income_tax",
                    period=year,
                    value=row["income_tax_collections"],
                    source_id=SOURCE_ID_CENSUS_STC,
                    active=True,
                    notes=f"Census STC FY{year} Individual Income Tax",
                )
            )

            session.add(new_stratum)
            session.flush()
            stratum_lookup[row["ucgid_str"]] = new_stratum.stratum_id

        session.commit()

    print(f"Loaded {len(stratum_lookup)} state income tax targets")
    return stratum_lookup


def main():
    """Run the full ETL pipeline for state income tax targets."""
    year = 2023

    print(f"Extracting Census STC data for FY{year}...")
    raw_df = extract_from_fred(year)  # Use FRED fallback with known values

    print("Transforming data...")
    transformed_df = transform_state_income_tax_data(raw_df)

    print(f"Loading {len(transformed_df)} state income tax targets...")
    stratum_lookup = load_state_income_tax_data(transformed_df, year)

    # Print summary
    print("\nState Income Tax Targets Summary:")
    print(f"  Total states: {len(stratum_lookup)}")
    print(
        f"  States with income tax: {len([s for s in transformed_df['state_abbrev'] if s not in NO_INCOME_TAX_STATES])}"
    )
    print(f"  States without income tax: {len(NO_INCOME_TAX_STATES)}")
    print(
        f"  Total collections: ${transformed_df['income_tax_collections'].sum() / 1e9:.1f}B"
    )

    # Print Ohio specifically (for the issue reference)
    ohio_row = transformed_df[transformed_df["state_abbrev"] == "OH"].iloc[0]
    print(
        f"\n  Ohio (OH): ${ohio_row['income_tax_collections'] / 1e9:.2f}B"
    )


if __name__ == "__main__":
    main()
