"""
ETL for state income tax calibration targets.

Pulls state individual income tax collections from Census Bureau's
Annual Survey of State Government Tax Collections (STC) and loads
them into the calibration database.

Data source: https://www.census.gov/programs-surveys/stc/data/datasets.html

"""

import logging

import pandas as pd
from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.db import get_geographic_strata, etl_argparser
from policyengine_us_data.utils.raw_cache import (
    is_cached,
    save_json,
    load_json,
)

logger = logging.getLogger(__name__)

CENSUS_STC_FLAT_FILE_URLS = {
    2023: "https://www2.census.gov/programs-surveys/stc/datasets/2023/FY2023-Flat-File.txt",
}
LATEST_STC_YEAR = max(CENSUS_STC_FLAT_FILE_URLS)
CENSUS_STC_INDIVIDUAL_INCOME_TAX_ITEM = "T40"
CENSUS_STC_NOT_AVAILABLE = "X"

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


def extract_state_income_tax_data(year: int = 2023) -> pd.DataFrame:
    """
    Extract state individual income tax collections from Census STC.

    Parses the official FY2023 Census STC flat file and extracts item
    ``T40`` (Individual Income Taxes). Census reports amounts in
    thousands of dollars, so the returned values are converted to
    dollars. Cells marked ``X`` in the source are treated as 0.

    Args:
        year: Fiscal year for the data (currently only 2023 supported)

    Returns:
        DataFrame with state_fips, state_abbrev, and income_tax_collections
    """
    if year not in CENSUS_STC_FLAT_FILE_URLS:
        raise ValueError(
            f"Only years {sorted(CENSUS_STC_FLAT_FILE_URLS)} are supported, got {year}"
        )

    # Use a distinct cache key so existing bad hardcoded JSON cannot survive
    # the switch to the official Census T40 download.
    cache_file = f"census_stc_t40_individual_income_tax_{year}.json"

    if is_cached(cache_file):
        logger.info(f"Using cached {cache_file}")
        data = load_json(cache_file)
        return pd.DataFrame(data)

    logger.info(f"Building Census STC individual income tax data for FY{year}")
    stc_df = pd.read_csv(CENSUS_STC_FLAT_FILE_URLS[year], dtype=str)
    item_rows = stc_df.loc[stc_df["ITEM"] == CENSUS_STC_INDIVIDUAL_INCOME_TAX_ITEM]
    if len(item_rows) != 1:
        raise ValueError(
            f"Expected exactly one Census STC row for item "
            f"{CENSUS_STC_INDIVIDUAL_INCOME_TAX_ITEM}, found {len(item_rows)}"
        )
    item_row = item_rows.iloc[0]

    rows = []
    for abbrev in STATE_ABBREV_TO_FIPS:
        fips = STATE_ABBREV_TO_FIPS[abbrev]
        raw_value = item_row[abbrev]
        value = (
            0
            if pd.isna(raw_value) or raw_value == CENSUS_STC_NOT_AVAILABLE
            else int(raw_value) * 1000
        )
        rows.append(
            {
                "state_fips": fips,
                "state_abbrev": abbrev,
                "income_tax_collections": value,
            }
        )

    df = pd.DataFrame(rows)

    # Cache for future use
    save_json(cache_file, df.to_dict(orient="records"))

    return df


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


def load_state_income_tax_data(
    df: pd.DataFrame, year: int, source_year: int | None = None
) -> dict:
    """
    Load state income tax targets into the calibration database.

    Creates strata and targets for each state's income tax collections.
    Uses the geographic hierarchy strata as parents.

    Args:
        df: Transformed DataFrame with state income tax data
        year: Year for the targets

    Returns:
        Dictionary mapping state_fips to stratum_id
    """
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    DATABASE_URL = f"sqlite:///{db_path}"
    engine = create_engine(DATABASE_URL)

    stratum_lookup = {}

    with Session(engine) as session:
        # Get geographic strata to use as parents
        geo_strata = get_geographic_strata(session)
        state_strata = geo_strata.get("state", {})

        # Create state-level strata for income tax
        for _, row in df.iterrows():
            state_fips = row["state_fips"]
            state_abbrev = row["state_abbrev"]

            # Find the geographic stratum for this state
            parent_stratum_id = state_strata.get(int(state_fips))
            if parent_stratum_id is None:
                logger.warning(
                    f"No geographic stratum found for state {state_abbrev} "
                    f"(FIPS {state_fips}), skipping"
                )
                continue

            note = f"State Income Tax: {state_abbrev}"

            # Create stratum with state_fips constraint
            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                notes=note,
            )
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=state_fips,
                ),
            ]

            # Add target for state_income_tax total
            new_stratum.targets_rel.append(
                Target(
                    variable="state_income_tax",
                    period=year,
                    value=row["income_tax_collections"],
                    active=True,
                    source="Census STC",
                    notes=f"Census STC FY{source_year or year}",
                )
            )

            session.add(new_stratum)
            session.flush()
            stratum_lookup[state_fips] = new_stratum.stratum_id

        session.commit()

    logger.info(f"Loaded {len(stratum_lookup)} state income tax targets")
    return stratum_lookup


def main():
    """Run the full ETL pipeline for state income tax targets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _, year = etl_argparser(
        "ETL for state income tax calibration targets",
        allow_year=True,
    )

    data_year = min(year, LATEST_STC_YEAR)
    if data_year != year:
        logger.warning(
            f"Census STC data not available for {year}; "
            f"using latest available year ({LATEST_STC_YEAR})"
        )
    logger.info(f"Extracting Census STC data for FY{data_year}...")
    raw_df = extract_state_income_tax_data(data_year)

    logger.info("Transforming data...")
    transformed_df = transform_state_income_tax_data(raw_df)

    logger.info(f"Loading {len(transformed_df)} state income tax targets...")
    stratum_lookup = load_state_income_tax_data(
        transformed_df, year, source_year=data_year
    )

    # Print summary
    total_collections = transformed_df["income_tax_collections"].sum()
    states_with_tax = int((transformed_df["income_tax_collections"] > 0).sum())
    states_without_tax = len(transformed_df) - states_with_tax

    logger.info(
        f"State Income Tax Targets Summary:\n"
        f"  Total states loaded: {len(stratum_lookup)}\n"
        f"  States with income tax: {states_with_tax}\n"
        f"  States without income tax: {states_without_tax}\n"
        f"  Total collections: ${total_collections / 1e9:.1f}B"
    )

    # Print Ohio specifically (for the issue reference)
    ohio_row = transformed_df[transformed_df["state_abbrev"] == "OH"].iloc[0]
    logger.info(f"  Ohio (OH): ${ohio_row['income_tax_collections'] / 1e9:.2f}B")


if __name__ == "__main__":
    main()
