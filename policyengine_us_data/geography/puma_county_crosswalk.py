"""
Build a probabilistic crosswalk between PUMAs and counties.

PUMAs (Public Use Microdata Areas) are geographic units used in Census
microdata. They contain at least 100,000 people and are built from census
tracts. A single PUMA may span multiple counties (especially in rural areas),
and large counties may contain multiple PUMAs.

This module creates population-weighted allocation factors that allow
probabilistic assignment of counties to households that only have PUMA
information.

Methodology:
1. Download Census tract-to-PUMA relationship file
2. Download tract-level population from the 2020 Census
3. Derive county FIPS from tract GEOID (first 5 digits)
4. Calculate allocation factor = pop_in_county / total_puma_pop
"""

import io
import requests
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import us

from policyengine_us_data.storage import STORAGE_FOLDER


CROSSWALK_FILE = STORAGE_FOLDER / "puma_county_crosswalk.csv"


def fetch_tract_to_puma() -> pd.DataFrame:
    """
    Fetches the 2020 Census Tract to 2020 PUMA relationship file.

    Returns:
        DataFrame with columns: state_fips, county_fips, tract, puma
    """
    url = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt"
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(
        io.BytesIO(response.content),
        dtype=str,
        encoding="utf-8-sig",  # Handle BOM
    )

    # The file has STATEFP, COUNTYFP, TRACTCE, PUMA5CE columns
    df = df.rename(
        columns={
            "STATEFP": "state_fips",
            "COUNTYFP": "county_fips",
            "TRACTCE": "tract",
            "PUMA5CE": "puma",
        }
    )

    return df[["state_fips", "county_fips", "tract", "puma"]]


def fetch_tract_population(state: str) -> pd.DataFrame:
    """
    Download & parse the 2020 PL-94-171 redistricting data for one state.

    This provides population at the census tract level.

    Args:
        state: Two-letter state postal code or full state name.

    Returns:
        DataFrame with columns: tract_geoid, population
    """
    BASE = (
        "https://www2.census.gov/programs-surveys/decennial/2020/data/"
        "01-Redistricting_File--PL_94-171/{dir}/{abbr}2020.pl.zip"
    )

    # Handle DC specially since it's not in us.states
    if state.upper() in ["DC", "DISTRICT OF COLUMBIA"]:
        dir_name = "District_of_Columbia"
        abbr = "dc"
    else:
        st = us.states.lookup(state)
        if st is None:
            raise ValueError(f"Unrecognised state name/abbr: {state}")
        dir_name = st.name.replace(" ", "_")
        abbr = st.abbr.lower()

    url = BASE.format(dir=dir_name, abbr=abbr)

    zbytes = requests.get(url, timeout=120).content
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        raw = z.read(f"{abbr}geo2020.pl")
        try:
            geo_lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            geo_lines = raw.decode("latin-1").splitlines()

        p1_lines = z.read(f"{abbr}000012020.pl").decode("utf-8").splitlines()

    # GEO file: keep tracts (SUMLEV 140)
    geo_records = []
    for ln in geo_lines:
        parts = ln.split("|")
        if parts[2] == "140":  # summary level 140 = tracts
            logrecno = parts[7]
            geoid = parts[8][-11:]  # 11-digit tract GEOID
            geo_records.append((logrecno, geoid))

    geo_df = pd.DataFrame(geo_records, columns=["LOGRECNO", "tract_geoid"])

    # P-file: pull total-population
    p1_records = [
        (p[4], int(p[5])) for p in map(lambda x: x.split("|"), p1_lines)
    ]
    p1_df = pd.DataFrame(p1_records, columns=["LOGRECNO", "population"])

    result = geo_df.merge(p1_df, on="LOGRECNO", how="left")
    result["population"] = result["population"].fillna(0).astype(int)

    return result[["tract_geoid", "population"]]


def build_puma_county_crosswalk() -> pd.DataFrame:
    """
    Builds the PUMA-to-county crosswalk with population-weighted allocation
    factors.

    Returns:
        DataFrame with columns:
            - state_fips: 2-digit state FIPS
            - puma: 5-digit PUMA code
            - county_fips: 3-digit county FIPS within state
            - county_fips_full: 5-digit full county FIPS (state + county)
            - population: population in this PUMA-county intersection
            - allocation_factor: proportion of PUMA population in this county
    """
    print("Fetching tract-to-PUMA relationship file...")
    tract_puma = fetch_tract_to_puma()

    # Build tract GEOID for joining
    tract_puma["tract_geoid"] = (
        tract_puma["state_fips"]
        + tract_puma["county_fips"]
        + tract_puma["tract"]
    )

    # Fetch tract populations for all states
    print("Fetching tract-level population data...")
    state_pops = []

    # Include DC explicitly (not in us.states.STATES_AND_TERRITORIES)
    states_to_fetch = list(us.states.STATES_AND_TERRITORIES) + ["DC"]

    for s in states_to_fetch:
        # Handle both State objects and string abbreviations
        if hasattr(s, "abbr"):
            if s.is_territory or s.abbr in ["ZZ"]:
                continue
            abbr = s.abbr
            name = s.name
        else:
            abbr = s
            name = s

        print(f"  {name}")
        try:
            state_pops.append(fetch_tract_population(abbr))
        except Exception as e:
            print(f"    Warning: Could not fetch {name}: {e}")

    tract_pop = pd.concat(state_pops, ignore_index=True)

    # Join tract population to tract-PUMA mapping
    merged = tract_puma.merge(tract_pop, on="tract_geoid", how="left")
    merged["population"] = merged["population"].fillna(0).astype(int)

    # Aggregate to PUMA-county level
    print("Aggregating to PUMA-county level...")
    puma_county = (
        merged.groupby(["state_fips", "puma", "county_fips"])["population"]
        .sum()
        .reset_index()
    )

    # Calculate allocation factors (proportion of PUMA pop in each county)
    puma_county["puma_total_pop"] = puma_county.groupby(
        ["state_fips", "puma"]
    )["population"].transform("sum")

    puma_county["allocation_factor"] = (
        puma_county["population"] / puma_county["puma_total_pop"]
    )

    # Handle edge case where PUMA has zero population
    puma_county["allocation_factor"] = puma_county["allocation_factor"].fillna(
        0
    )

    # Create full 5-digit county FIPS
    puma_county["county_fips_full"] = (
        puma_county["state_fips"] + puma_county["county_fips"]
    )

    # Sort by state, puma, allocation factor (descending)
    puma_county = puma_county.sort_values(
        ["state_fips", "puma", "allocation_factor"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    # Select final columns
    result = puma_county[
        [
            "state_fips",
            "puma",
            "county_fips",
            "county_fips_full",
            "population",
            "allocation_factor",
        ]
    ]

    return result


def get_puma_county_crosswalk() -> pd.DataFrame:
    """
    Returns the PUMA-to-county crosswalk, loading from cache if available.

    Returns:
        DataFrame with PUMA-county allocation factors.
    """
    if CROSSWALK_FILE.exists():
        return pd.read_csv(CROSSWALK_FILE, dtype=str).assign(
            population=lambda df: df["population"].astype(int),
            allocation_factor=lambda df: df["allocation_factor"].astype(float),
        )

    crosswalk = build_puma_county_crosswalk()
    crosswalk.to_csv(CROSSWALK_FILE, index=False)
    return crosswalk


def assign_county_from_puma(
    state_fips: np.ndarray,
    puma: np.ndarray,
    random_state: np.random.RandomState = None,
) -> np.ndarray:
    """
    Probabilistically assigns county FIPS codes based on PUMA and state.

    For each household, uses the PUMA-to-county allocation factors to
    randomly assign a county, weighted by population.

    Args:
        state_fips: Array of 2-digit state FIPS codes (as strings or ints)
        puma: Array of PUMA codes (as strings or ints)
        random_state: Optional random state for reproducibility

    Returns:
        Array of 3-digit county FIPS codes (within state)
    """
    if random_state is None:
        random_state = np.random.RandomState()

    crosswalk = get_puma_county_crosswalk()

    # Ensure consistent string types
    state_fips = np.array([str(s).zfill(2) for s in state_fips])
    puma = np.array([str(p).zfill(5) for p in puma])

    n = len(state_fips)
    county_fips = np.zeros(n, dtype=object)

    # Group crosswalk by state-puma for efficient lookup
    crosswalk_dict = {}
    for (st, pm), group in crosswalk.groupby(["state_fips", "puma"]):
        counties = group["county_fips"].values
        probs = group["allocation_factor"].values
        # Normalize probabilities (should already sum to 1, but be safe)
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        crosswalk_dict[(st, pm)] = (counties, probs)

    # Assign counties
    for i in range(n):
        key = (state_fips[i], puma[i])
        if key in crosswalk_dict:
            counties, probs = crosswalk_dict[key]
            if len(counties) == 1:
                county_fips[i] = counties[0]
            else:
                county_fips[i] = random_state.choice(counties, p=probs)
        else:
            # PUMA not found - assign 0 (unknown)
            county_fips[i] = "000"

    return county_fips


def get_state_county_crosswalk() -> pd.DataFrame:
    """
    Creates a state-to-county crosswalk with population-weighted allocation
    factors.

    This is derived from the PUMA-county crosswalk by aggregating to state
    level. Used for probabilistic county assignment when only state is known.

    Returns:
        DataFrame with columns:
            - state_fips: 2-digit state FIPS
            - county_fips: 3-digit county FIPS within state
            - county_fips_full: 5-digit full county FIPS
            - population: county population
            - allocation_factor: proportion of state population in county
    """
    puma_county = get_puma_county_crosswalk()

    # Aggregate to state-county level
    state_county = (
        puma_county.groupby(["state_fips", "county_fips", "county_fips_full"])[
            "population"
        ]
        .sum()
        .reset_index()
    )

    # Calculate state-level allocation factors
    state_county["state_total_pop"] = state_county.groupby("state_fips")[
        "population"
    ].transform("sum")

    state_county["allocation_factor"] = (
        state_county["population"] / state_county["state_total_pop"]
    )
    state_county["allocation_factor"] = state_county[
        "allocation_factor"
    ].fillna(0)

    # Sort by state, allocation factor (descending)
    state_county = state_county.sort_values(
        ["state_fips", "allocation_factor"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return state_county[
        [
            "state_fips",
            "county_fips",
            "county_fips_full",
            "population",
            "allocation_factor",
        ]
    ]


def assign_county_from_state(
    state_fips: np.ndarray,
    random_state: np.random.RandomState = None,
) -> np.ndarray:
    """
    Probabilistically assigns county FIPS codes based on state only.

    For each household, uses state-to-county population weights to
    randomly assign a county.

    Args:
        state_fips: Array of 2-digit state FIPS codes (as ints)
        random_state: Optional random state for reproducibility

    Returns:
        Array of 3-digit county FIPS codes (as ints)
    """
    if random_state is None:
        random_state = np.random.RandomState()

    crosswalk = get_state_county_crosswalk()

    # Ensure consistent string types for lookup
    state_fips_str = np.array([str(int(s)).zfill(2) for s in state_fips])

    n = len(state_fips)
    county_fips = np.zeros(n, dtype=int)

    # Group crosswalk by state for efficient lookup
    crosswalk_dict = {}
    for st, group in crosswalk.groupby("state_fips"):
        counties = group["county_fips"].astype(int).values
        probs = group["allocation_factor"].values
        # Normalize probabilities
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        crosswalk_dict[st] = (counties, probs)

    # Assign counties
    for i in range(n):
        st = state_fips_str[i]
        if st in crosswalk_dict:
            counties, probs = crosswalk_dict[st]
            if len(counties) == 1:
                county_fips[i] = counties[0]
            else:
                county_fips[i] = random_state.choice(counties, p=probs)
        else:
            # State not found - assign 0 (unknown)
            county_fips[i] = 0

    return county_fips


if __name__ == "__main__":
    print("Building PUMA-to-county crosswalk...")
    crosswalk = build_puma_county_crosswalk()
    crosswalk.to_csv(CROSSWALK_FILE, index=False)
    print(f"Saved crosswalk to {CROSSWALK_FILE}")

    # Print summary statistics
    print(f"\nTotal PUMA-county pairs: {len(crosswalk)}")
    print(f"Unique PUMAs: {crosswalk.groupby(['state_fips', 'puma']).ngroups}")
    print(f"Unique counties: {crosswalk['county_fips_full'].nunique()}")

    # Show distribution of counties per PUMA
    counties_per_puma = crosswalk.groupby(["state_fips", "puma"]).size()
    print(f"\nCounties per PUMA:")
    print(f"  Min: {counties_per_puma.min()}")
    print(f"  Max: {counties_per_puma.max()}")
    print(f"  Mean: {counties_per_puma.mean():.2f}")
    print(f"  Median: {counties_per_puma.median():.0f}")

    # Count single-county PUMAs (deterministic assignment)
    single_county = (counties_per_puma == 1).sum()
    print(
        f"\nPUMAs with single county (deterministic): "
        f"{single_county} ({100*single_county/len(counties_per_puma):.1f}%)"
    )
