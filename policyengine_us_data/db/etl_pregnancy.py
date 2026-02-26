"""ETL for pregnancy calibration targets.

Fetches state-level birth counts from the CDC VSRR (Vital Statistics
Rapid Release) provisional natality dataset on data.cdc.gov, and female
population aged 15-44 from Census ACS (table B01001).  Creates
calibration strata targeting the number of pregnant people per state.

The calibration target is a point-in-time pregnancy count derived from
annual births: target = births * (39/52), where 39/52 accounts for
the ~9-month pregnancy duration within a 52-week year.

Data sources:
  - CDC VSRR: data.cdc.gov/resource/hmz2-vwda (Socrata API)
    "State and National Provisional Counts for Live Births,
     Deaths, and Infant Deaths"
  - Census ACS B01001: female population aged 15-44 by state
"""

import logging

import pandas as pd
import requests
from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.census import STATE_ABBREV_TO_FIPS
from policyengine_us_data.utils.db import (
    get_geographic_strata,
    etl_argparser,
)
from policyengine_us_data.utils.raw_cache import (
    is_cached,
    save_json,
    load_json,
)

logger = logging.getLogger(__name__)

# Weeks of pregnancy / weeks in a year.
PREGNANCY_DURATION_FRACTION = 39 / 52

# CDC VSRR Socrata dataset ID for provisional natality.
CDC_VSRR_DATASET = "hmz2-vwda"
CDC_VSRR_BASE = f"https://data.cdc.gov/resource/{CDC_VSRR_DATASET}.json"

# State name -> abbreviation for mapping CDC's uppercase names.
STATE_NAME_TO_ABBREV = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}


# ── Extract ──────────────────────────────────────────────────────────


def extract_cdc_births(year: int) -> pd.DataFrame:
    """Fetch state-level birth counts from CDC VSRR via Socrata API.

    Sums monthly provisional birth counts for the requested year.
    If fewer than 12 months are available (common for the current
    year), annualises by scaling up proportionally.

    Args:
        year: Calendar year to query.

    Returns:
        DataFrame with columns [state_abbrev, births].
    """
    cache_file = f"cdc_vsrr_births_{year}.json"
    if is_cached(cache_file):
        logger.info(f"Using cached {cache_file}")
        rows = load_json(cache_file)
    else:
        params = (
            f"$where=year='{year}'"
            f" AND indicator='Number of Live Births'"
            f" AND state!='UNITED STATES'"
            f" AND period='Monthly'"
            f"&$limit=5000"
        )
        url = f"{CDC_VSRR_BASE}?{params}"
        logger.info(f"Fetching CDC VSRR births for {year}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            raise ValueError(f"No CDC VSRR birth data returned for {year}")
        save_json(cache_file, rows)

    df = pd.DataFrame(rows)
    df["data_value"] = pd.to_numeric(df["data_value"])

    # Count months available per state and annualise if partial.
    months_per_state = df.groupby("state")["month"].nunique()
    annual = df.groupby("state")["data_value"].sum()
    n_months = months_per_state.reindex(annual.index).fillna(1)
    annual = (annual * 12 / n_months).round().astype(int)

    result = annual.reset_index()
    result.columns = ["state_name", "births"]
    result["state_abbrev"] = result["state_name"].map(STATE_NAME_TO_ABBREV)
    result = result.dropna(subset=["state_abbrev"])

    n_mo = int(n_months.mode().iloc[0])
    logger.info(
        f"CDC VSRR {year}: {len(result)} states, "
        f"{n_mo} months of data, "
        f"{result.births.sum():,} births (annualised)"
    )
    return result[["state_abbrev", "births"]]


def extract_female_population(year: int) -> pd.DataFrame:
    """Fetch state-level female population aged 15-44 from ACS B01001.

    Variables B01001_030E..B01001_038E cover female age groups
    15-17 through 40-44.

    Args:
        year: ACS vintage year to query.

    Returns:
        DataFrame with columns [state_abbrev, female_15_44].
    """
    cache_file = f"census_b01001_female_15_44_{year}.json"
    if is_cached(cache_file):
        logger.info(f"Using cached {cache_file}")
        data = load_json(cache_file)
    else:
        var_ids = ",".join([f"B01001_{i:03d}E" for i in range(30, 39)])
        url = (
            f"https://api.census.gov/data/{year}/acs/acs1"
            f"?get={var_ids}&for=state:*"
        )
        logger.info(f"Fetching ACS B01001 female 15-44 for {year}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        save_json(cache_file, data)

    headers, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=headers)
    age_cols = [c for c in df.columns if c.startswith("B01001_")]
    df[age_cols] = df[age_cols].astype(int)
    df["female_15_44"] = df[age_cols].sum(axis=1)
    fips_to_abbrev = {v: k for k, v in STATE_ABBREV_TO_FIPS.items()}
    df["state_abbrev"] = df["state"].map(fips_to_abbrev)
    return df[["state_abbrev", "female_15_44"]].dropna(subset=["state_abbrev"])


# ── Transform ────────────────────────────────────────────────────────


def transform_pregnancy_data(
    births_df: pd.DataFrame,
    pop_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute state-level pregnancy targets and rates.

    Args:
        births_df: From extract_cdc_births.
        pop_df: From extract_female_population.

    Returns:
        DataFrame with columns [state_abbrev, state_fips,
        ucgid_str, births, pregnancy_target, pregnancy_rate].
    """
    df = births_df.merge(pop_df, on="state_abbrev")
    df["state_fips"] = df["state_abbrev"].map(STATE_ABBREV_TO_FIPS)
    # Point-in-time pregnancy count.
    df["pregnancy_target"] = (
        df["births"] * PREGNANCY_DURATION_FRACTION
    ).round()
    # Rate for stochastic assignment in the CPS build.
    df["pregnancy_rate"] = (
        df["births"] / df["female_15_44"]
    ) * PREGNANCY_DURATION_FRACTION
    df["ucgid_str"] = "0400000US" + df["state_fips"]
    return df


# ── Load ─────────────────────────────────────────────────────────────


def load_pregnancy_data(
    df: pd.DataFrame,
    year: int,
) -> None:
    """Create pregnancy calibration strata and targets in the DB.

    Args:
        df: From transform_pregnancy_data.
        year: Target year for the calibration targets.
    """
    db_url = (
        f"sqlite:///" f"{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    )
    engine = create_engine(db_url)

    with Session(engine) as session:
        geo_strata = get_geographic_strata(session)

        # National parent stratum for pregnancy.
        nat_stratum = Stratum(
            parent_stratum_id=geo_strata["national"],
            notes="National Pregnant",
        )
        nat_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="is_pregnant",
                operation="==",
                value="True",
            ),
        ]
        session.add(nat_stratum)
        session.flush()

        # State-level strata with targets.
        for _, row in df.iterrows():
            state_fips = int(row["state_fips"])
            if state_fips not in geo_strata["state"]:
                logger.warning(
                    f"No geographic stratum for FIPS "
                    f"{state_fips}, skipping"
                )
                continue

            parent_id = geo_strata["state"][state_fips]
            stratum = Stratum(
                parent_stratum_id=parent_id,
                notes=(f"State FIPS {state_fips} Pregnant"),
            )
            stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(state_fips),
                ),
                StratumConstraint(
                    constraint_variable="is_pregnant",
                    operation="==",
                    value="True",
                ),
            ]
            stratum.targets_rel.append(
                Target(
                    variable="person_count",
                    period=year,
                    value=row["pregnancy_target"],
                    active=True,
                    source="CDC VSRR Natality",
                )
            )
            session.add(stratum)
            session.flush()

        session.commit()


# ── Public API for cps.py ────────────────────────────────────────────


def get_state_pregnancy_rates(
    cdc_year: int = 2023,
    acs_year: int = 2023,
) -> dict:
    """Return {state_abbrev: pregnancy_rate} for use by cps.py.

    This is the public entry point consumed by the CPS build
    pipeline to get state-level pregnancy rates for the stochastic
    draw.

    Args:
        cdc_year: Year to pull CDC birth counts for.
        acs_year: ACS vintage for female population denominators.

    Returns:
        dict mapping two-letter state abbreviation to pregnancy
        rate (probability that a woman aged 15-44 is currently
        pregnant).
    """
    births_df = extract_cdc_births(cdc_year)
    pop_df = extract_female_population(acs_year)
    df = transform_pregnancy_data(births_df, pop_df)
    return dict(zip(df["state_abbrev"], df["pregnancy_rate"]))


# ── CLI entry point ──────────────────────────────────────────────────


def main():
    _, year = etl_argparser("ETL for pregnancy calibration targets")

    # CDC VSRR has provisional data for the most recent 1-2 years.
    # ACS releases lag by ~1 year (e.g. ACS 2023 released Sep 2024).
    # Try the target year first for births, then fall back.
    births_df = None
    for cdc_year in [year, year - 1]:
        try:
            births_df = extract_cdc_births(cdc_year)
            print(f"Using CDC VSRR {cdc_year} birth data")
            break
        except Exception as e:
            logger.warning(f"CDC VSRR {cdc_year} not available: {e}")
    if births_df is None:
        raise RuntimeError(f"No CDC VSRR birth data for {year} or {year - 1}")

    pop_df = None
    for acs_year in [year - 1, year - 2]:
        try:
            pop_df = extract_female_population(acs_year)
            print(f"Using ACS {acs_year} female population data")
            break
        except Exception as e:
            logger.warning(f"ACS {acs_year} not available: {e}")
    if pop_df is None:
        raise RuntimeError(
            f"No ACS population data for " f"{year - 1} or {year - 2}"
        )

    df = transform_pregnancy_data(births_df, pop_df)

    total_births = df["births"].sum()
    total_target = df["pregnancy_target"].sum()
    print(f"Total births: {total_births:,.0f}")
    print(f"Pregnancy target (point-in-time): " f"{total_target:,.0f}")

    load_pregnancy_data(df, year)
    print("Pregnancy calibration targets loaded.")


if __name__ == "__main__":
    main()
