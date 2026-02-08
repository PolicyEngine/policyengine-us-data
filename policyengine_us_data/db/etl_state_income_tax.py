"""
ETL for state income tax calibration targets.

Pulls state individual income tax collections from Census Bureau's
Annual Survey of State Government Tax Collections (STC) and loads
them into the calibration database.

Data source: https://www.census.gov/programs-surveys/stc/data/datasets.html

Stratum Group ID: 7 (State Income Tax)
"""

import argparse
import logging
import pandas as pd
import numpy as np
from sqlmodel import Session, create_engine, select

from policyengine_us_data.storage import STORAGE_FOLDER

DEFAULT_DATASET = "hf://policyengine/policyengine-us-data/calibration/stratified_extended_cps.h5"
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
    Source,
    VariableGroup,
    VariableMetadata,
)
from policyengine_us_data.utils.db import get_geographic_strata
from policyengine_us_data.utils.db_metadata import (
    get_or_create_source,
    get_or_create_variable_group,
    get_or_create_variable_metadata,
)
from policyengine_us_data.utils.raw_cache import (
    is_cached,
    save_json,
    load_json,
)
from policyengine_us_data.utils.constraint_validation import (
    Constraint,
    ensure_consistent_constraint_set,
)

logger = logging.getLogger(__name__)

# Stratum group ID for state income tax targets
STRATUM_GROUP_ID_STATE_INCOME_TAX = 7

# States without individual income tax (these will have $0 target)
NO_INCOME_TAX_STATES = {
    "AK",  # Alaska
    "FL",  # Florida
    "NV",  # Nevada
    "SD",  # South Dakota
    "TX",  # Texas
    "WA",  # Washington (has capital gains tax only, modeled separately)
    "WY",  # Wyoming
    "NH",  # New Hampshire (phased out interest/dividends tax)
    "TN",  # Tennessee (phased out Hall income tax)
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


def extract_state_income_tax_data(year: int = 2023) -> pd.DataFrame:
    """
    Extract state individual income tax collections from Census STC.

    Uses hardcoded FY2023 values from Census Bureau's Annual Survey of
    State Government Tax Collections. These values are derived from
    Census STC Table 1: State Government Tax Collections by Category.

    Source: https://www.census.gov/data/tables/2023/econ/stc/2023-annual.html

    Args:
        year: Fiscal year for the data (currently only 2023 supported)

    Returns:
        DataFrame with state_fips, state_abbrev, and income_tax_collections
    """
    cache_file = f"census_stc_individual_income_tax_{year}.json"

    if is_cached(cache_file):
        logger.info(f"Using cached {cache_file}")
        data = load_json(cache_file)
        return pd.DataFrame(data)

    logger.info(f"Building Census STC individual income tax data for FY{year}")

    # FY2023 values in dollars from Census STC
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
        "OH": 9_520_000_000,  # Confirmed with Policy Matters Ohio
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
        "WA": 0,  # WA has capital gains tax but no broad income tax
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


def load_state_income_tax_data(df: pd.DataFrame, year: int) -> dict:
    """
    Load state income tax targets into the calibration database.

    Creates strata and targets for each state's income tax collections.
    Uses the geographic hierarchy strata (stratum_group_id=1) as parents.

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
        # Get or create the Census STC source
        source = get_or_create_source(
            session,
            name="Census Bureau Annual Survey of State Tax Collections",
            source_type="administrative",
            url="https://www.census.gov/programs-surveys/stc.html",
            notes="Individual income tax collections by state",
        )

        # Get or create variable group for state income tax
        var_group = get_or_create_variable_group(
            session,
            name="state_income_tax",
            category="taxes",
            description="State-level individual income tax collections",
        )

        # Get or create variable metadata
        get_or_create_variable_metadata(
            session,
            variable="state_income_tax",
            group=var_group,
            display_name="State Income Tax",
            units="USD",
            notes="Total state individual income tax collections",
        )

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
                stratum_group_id=STRATUM_GROUP_ID_STATE_INCOME_TAX,
                notes=note,
            )
            # Validate constraints before adding
            state_tax_constraints = [
                Constraint(
                    variable="state_fips",
                    operation="==",
                    value=state_fips,
                ),
            ]
            ensure_consistent_constraint_set(state_tax_constraints)
            new_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable=c.variable,
                    operation=c.operation,
                    value=c.value,
                )
                for c in state_tax_constraints
            ]

            # Add target for state_income_tax total
            new_stratum.targets_rel.append(
                Target(
                    variable="state_income_tax",
                    period=year,
                    value=row["income_tax_collections"],
                    source_id=source.source_id,
                    active=True,
                    notes=f"Census STC FY{year}",
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
    parser = argparse.ArgumentParser(
        description="ETL for state income tax calibration targets"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "Source dataset (local path or HuggingFace URL). "
            "The year for targets is derived from the dataset's "
            "default_calculation_period. Default: %(default)s"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Derive year from dataset
    from policyengine_us import Microsimulation

    logger.info(f"Loading dataset: {args.dataset}")
    sim = Microsimulation(dataset=args.dataset)
    year = int(sim.default_calculation_period)
    logger.info(f"Derived year from dataset: {year}")

    logger.info(f"Extracting Census STC data for FY{year}...")
    raw_df = extract_state_income_tax_data(year)

    logger.info("Transforming data...")
    transformed_df = transform_state_income_tax_data(raw_df)

    logger.info(f"Loading {len(transformed_df)} state income tax targets...")
    stratum_lookup = load_state_income_tax_data(transformed_df, year)

    # Print summary
    total_collections = transformed_df["income_tax_collections"].sum()
    states_with_tax = len(
        [
            s
            for s in transformed_df["state_abbrev"]
            if s not in NO_INCOME_TAX_STATES
        ]
    )

    logger.info(
        f"State Income Tax Targets Summary:\n"
        f"  Total states loaded: {len(stratum_lookup)}\n"
        f"  States with income tax: {states_with_tax}\n"
        f"  States without income tax: {len(NO_INCOME_TAX_STATES)}\n"
        f"  Total collections: ${total_collections / 1e9:.1f}B"
    )

    # Print Ohio specifically (for the issue reference)
    ohio_row = transformed_df[transformed_df["state_abbrev"] == "OH"].iloc[0]
    logger.info(
        f"  Ohio (OH): ${ohio_row['income_tax_collections'] / 1e9:.2f}B"
    )


if __name__ == "__main__":
    main()
