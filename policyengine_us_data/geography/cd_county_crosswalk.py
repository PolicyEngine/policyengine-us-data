"""
Build a population-weighted crosswalk between Congressional Districts and counties.

This module creates allocation factors that allow probabilistic assignment of
counties to households based on their congressional district. The methodology
mirrors that used in make_district_mapping.py for consistency.

Methodology:
1. Download Census block-to-CD relationship file (119th Congress)
2. Download block-level population from the 2020 Census PL-94-171 data
3. Derive county FIPS from block GEOID (first 5 digits)
4. Calculate allocation factor = pop_in_county_CD_intersection / total_CD_pop
"""

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import us

from policyengine_us_data.storage import STORAGE_FOLDER


CD_COUNTY_CROSSWALK_FILE = STORAGE_FOLDER / "cd_county_crosswalk.csv"
CD_COUNTY_MAPPINGS_JSON = STORAGE_FOLDER / "cd_county_mappings.json"


def fetch_block_to_district_map(congress: int = 119) -> pd.DataFrame:
    """
    Fetches the Census Block Equivalency File (BEF) for a given Congress.

    This file maps every 2020 census block (GEOID) to its corresponding
    congressional district.

    Args:
        congress: The congressional session number (default 119 for current).

    Returns:
        A DataFrame with columns ['GEOID', 'CD'].
    """
    if congress == 119:
        url = (
            "https://www2.census.gov/programs-surveys/decennial/rdo/"
            "mapping-files/2025/119-congressional-district-befs/cd119.zip"
        )
        zbytes = requests.get(url, timeout=120).content

        with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
            fname = "NationalCD119.txt"
            bef = pd.read_csv(z.open(fname), sep=",", dtype=str)
            bef.columns = bef.columns.str.strip()
            bef = bef.rename(columns={"CDFP": "CD"})
            return bef[["GEOID", "CD"]]
    else:
        raise ValueError(
            f"Congress {congress} is not supported. Use 119 for current."
        )


def fetch_block_population(state: str) -> pd.DataFrame:
    """
    Download & parse the 2020 PL-94-171 redistricting data for one state.

    Args:
        state: Two-letter state postal code or full state name.

    Returns:
        DataFrame with columns: GEOID (15-digit block code), population
    """
    BASE = (
        "https://www2.census.gov/programs-surveys/decennial/2020/data/"
        "01-Redistricting_File--PL_94-171/{dir}/{abbr}2020.pl.zip"
    )

    # Handle DC specially
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

    # GEO file: keep blocks (SUMLEV 750)
    geo_records = [
        (parts[7], parts[8][-15:])  # LOGRECNO, 15-digit block GEOID
        for ln in geo_lines
        if (parts := ln.split("|"))[2] == "750"
    ]
    geo_df = pd.DataFrame(geo_records, columns=["LOGRECNO", "GEOID"])

    # P-file: pull total-population
    p1_records = [
        (p[4], int(p[5])) for p in map(lambda x: x.split("|"), p1_lines)
    ]
    p1_df = pd.DataFrame(p1_records, columns=["LOGRECNO", "population"])

    return (
        geo_df.merge(p1_df, on="LOGRECNO", how="left")
        .assign(population=lambda d: d["population"].fillna(0).astype(int))
        .loc[:, ["GEOID", "population"]]
    )


def build_cd_county_crosswalk() -> pd.DataFrame:
    """
    Builds the CD-to-county crosswalk with population-weighted allocation
    factors.

    Returns:
        DataFrame with columns:
            - state_fips: 2-digit state FIPS
            - cd: 2-digit congressional district code
            - county_fips: 3-digit county FIPS within state
            - county_fips_full: 5-digit full county FIPS (state + county)
            - cd_code: Full CD code (5001800US + state + cd)
            - population: population in this CD-county intersection
            - allocation_factor: proportion of CD population in this county
    """
    print("Fetching block-to-CD relationship file (119th Congress)...")
    block_cd = fetch_block_to_district_map(119)

    # Remove blocks not assigned to any congressional district
    block_cd = block_cd[block_cd["CD"] != "ZZ"]

    # Fetch block populations for all states + DC
    print("Fetching block-level population data...")
    state_pops = []

    # Get all states
    states_to_fetch = [
        s
        for s in us.states.STATES_AND_TERRITORIES
        if not s.is_territory and s.abbr not in ["ZZ"]
    ]

    for s in states_to_fetch:
        print(f"  {s.name}")
        try:
            state_pops.append(fetch_block_population(s.abbr))
        except Exception as e:
            print(f"    Warning: Could not fetch {s.name}: {e}")

    # Add DC
    print("  District of Columbia")
    try:
        state_pops.append(fetch_block_population("DC"))
    except Exception as e:
        print(f"    Warning: Could not fetch DC: {e}")

    block_pop = pd.concat(state_pops, ignore_index=True)

    # Merge block population with CD mapping
    print("Merging block data...")
    merged = block_cd.merge(block_pop, on="GEOID", how="left")
    merged["population"] = merged["population"].fillna(0).astype(int)

    # Derive geography from GEOID
    # GEOID is 15 digits: SSCCCTTTTTTBBBB
    # SS = state FIPS (2 digits)
    # CCC = county FIPS (3 digits)
    # TTTTTT = tract (6 digits)
    # BBBB = block (4 digits)
    merged["state_fips"] = merged["GEOID"].str[:2]
    merged["county_fips"] = merged["GEOID"].str[2:5]
    merged["county_fips_full"] = merged["GEOID"].str[:5]

    # Create full CD code (matching format used in calibration)
    merged["cd_code"] = "5001800US" + merged["state_fips"] + merged["CD"]

    # Aggregate to CD-county level
    print("Aggregating to CD-county level...")
    cd_county = (
        merged.groupby(
            ["state_fips", "CD", "county_fips", "county_fips_full", "cd_code"]
        )["population"]
        .sum()
        .reset_index()
    )

    # Rename CD column for clarity
    cd_county = cd_county.rename(columns={"CD": "cd"})

    # Calculate allocation factors (proportion of CD pop in each county)
    cd_county["cd_total_pop"] = cd_county.groupby(["state_fips", "cd"])[
        "population"
    ].transform("sum")

    cd_county["allocation_factor"] = (
        cd_county["population"] / cd_county["cd_total_pop"]
    )

    # Handle edge case where CD has zero population
    cd_county["allocation_factor"] = cd_county["allocation_factor"].fillna(0)

    # Sort by state, cd, allocation factor (descending)
    cd_county = cd_county.sort_values(
        ["state_fips", "cd", "allocation_factor"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    # Select final columns
    result = cd_county[
        [
            "state_fips",
            "cd",
            "county_fips",
            "county_fips_full",
            "cd_code",
            "population",
            "allocation_factor",
        ]
    ]

    return result


def get_cd_county_crosswalk() -> pd.DataFrame:
    """
    Returns the CD-to-county crosswalk, loading from cache if available.

    Returns:
        DataFrame with CD-county allocation factors.
    """
    if CD_COUNTY_CROSSWALK_FILE.exists():
        return pd.read_csv(CD_COUNTY_CROSSWALK_FILE, dtype=str).assign(
            population=lambda df: df["population"].astype(int),
            allocation_factor=lambda df: df["allocation_factor"].astype(float),
        )

    crosswalk = build_cd_county_crosswalk()
    crosswalk.to_csv(CD_COUNTY_CROSSWALK_FILE, index=False)
    return crosswalk


def assign_county_from_cd(
    cd_codes: np.ndarray,
    random_state: np.random.RandomState = None,
) -> np.ndarray:
    """
    Probabilistically assigns county FIPS codes based on congressional
    district.

    For each household, uses the CD-to-county allocation factors to
    randomly assign a county, weighted by population.

    Args:
        cd_codes: Array of CD codes (format: "5001800US" + state + cd)
        random_state: Optional random state for reproducibility

    Returns:
        Array of 5-digit full county FIPS codes (state + county)
    """
    if random_state is None:
        random_state = np.random.RandomState()

    crosswalk = get_cd_county_crosswalk()

    n = len(cd_codes)
    county_fips_full = np.zeros(n, dtype=object)

    # Group crosswalk by CD code for efficient lookup
    crosswalk_dict = {}
    for cd_code, group in crosswalk.groupby("cd_code"):
        counties = group["county_fips_full"].values
        probs = group["allocation_factor"].values
        # Normalize probabilities (should already sum to 1, but be safe)
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        crosswalk_dict[cd_code] = (counties, probs)

    # Assign counties
    for i in range(n):
        cd = cd_codes[i]
        if cd in crosswalk_dict:
            counties, probs = crosswalk_dict[cd]
            if len(counties) == 1:
                county_fips_full[i] = counties[0]
            else:
                county_fips_full[i] = random_state.choice(counties, p=probs)
        else:
            # CD not found - assign "00000" (unknown)
            county_fips_full[i] = "00000"

    return county_fips_full


def assign_county_from_state_and_cd(
    state_fips: np.ndarray,
    cd: np.ndarray,
    random_state: np.random.RandomState = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Probabilistically assigns county FIPS codes based on state and CD.

    Args:
        state_fips: Array of 2-digit state FIPS codes
        cd: Array of 2-digit congressional district codes
        random_state: Optional random state for reproducibility

    Returns:
        Tuple of (state_fips array, county_fips array) where county_fips
        is the 3-digit county code within the state.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    # Build CD codes
    state_fips_str = np.array([str(s).zfill(2) for s in state_fips])
    cd_str = np.array([str(c).zfill(2) for c in cd])
    cd_codes = np.array(
        ["5001800US" + s + c for s, c in zip(state_fips_str, cd_str)]
    )

    # Get full county FIPS
    county_fips_full = assign_county_from_cd(cd_codes, random_state)

    # Extract 3-digit county FIPS
    county_fips = np.array(
        [c[2:5] if len(c) == 5 else "000" for c in county_fips_full]
    )

    return state_fips_str, county_fips


def export_to_json(
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Export the CD-county crosswalk to JSON format.

    This format is compatible with the geo-stacking calibration workflow.
    The JSON structure is: {cd_geoid: {county_fips: proportion, ...}, ...}

    Args:
        output_path: Path to save JSON file. If None, uses default location.

    Returns:
        Dictionary mapping CD GEOID -> {county_fips: proportion}
    """
    crosswalk = get_cd_county_crosswalk()

    # Build the JSON structure
    # CD GEOID format: state_fips + cd (e.g., "601" for CA-01, "3601" for NY-01)
    cd_county_map = {}

    for (state_fips, cd), group in crosswalk.groupby(["state_fips", "cd"]):
        # Create CD GEOID (state + cd without leading zeros on state for
        # single-digit states)
        cd_geoid = f"{int(state_fips)}{cd.zfill(2)}"

        # Build county proportions dict
        county_props = {}
        for _, row in group.iterrows():
            county_props[row["county_fips_full"]] = row["allocation_factor"]

        cd_county_map[cd_geoid] = county_props

    # Save to JSON
    if output_path is None:
        output_path = CD_COUNTY_MAPPINGS_JSON

    with open(output_path, "w") as f:
        json.dump(cd_county_map, f, indent=2)

    print(f"Exported CD-county mappings to {output_path}")
    return cd_county_map


def load_cd_county_mappings(
    path: Optional[Path] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load CD-county mappings from JSON file.

    Compatible with geo-stacking calibration workflow.

    Args:
        path: Path to JSON file. If None, uses default location.

    Returns:
        Dictionary mapping CD GEOID -> {county_fips: proportion},
        or None if file not found.
    """
    if path is None:
        path = CD_COUNTY_MAPPINGS_JSON

    if not path.exists():
        print(f"WARNING: {path} not found. Run export_to_json() first.")
        return None

    with open(path, "r") as f:
        return json.load(f)


def get_county_for_cd(
    cd_geoid: str,
    cd_county_mappings: Optional[Dict[str, Dict[str, float]]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Optional[str]:
    """
    Get a county FIPS code for a given congressional district.

    Uses weighted random selection based on county proportions.
    Compatible with geo-stacking calibration workflow.

    Args:
        cd_geoid: Congressional district GEOID (e.g., "601" for CA-01)
        cd_county_mappings: Optional pre-loaded mappings dict. If None,
            loads from default JSON file.
        random_state: Optional random state for reproducibility.

    Returns:
        5-digit county FIPS code, or None if CD not found.
    """
    if cd_county_mappings is None:
        cd_county_mappings = load_cd_county_mappings()

    if cd_county_mappings is None or str(cd_geoid) not in cd_county_mappings:
        return None

    county_props = cd_county_mappings[str(cd_geoid)]
    if not county_props:
        return None

    counties = list(county_props.keys())
    weights = list(county_props.values())

    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]

    if random_state is None:
        random_state = np.random.RandomState()

    return random_state.choice(counties, p=weights)


def get_state_county_crosswalk_from_cd() -> pd.DataFrame:
    """
    Creates a state-to-county crosswalk by aggregating the CD-county data.

    This is useful when CD is not known and we need to assign county based
    only on state. The allocation factors are population-weighted across
    all CDs in the state.

    Returns:
        DataFrame with columns:
            - state_fips: 2-digit state FIPS
            - county_fips: 3-digit county FIPS within state
            - county_fips_full: 5-digit full county FIPS
            - population: county population
            - allocation_factor: proportion of state population in county
    """
    cd_crosswalk = get_cd_county_crosswalk()

    # Aggregate to state-county level
    state_county = (
        cd_crosswalk.groupby(
            ["state_fips", "county_fips", "county_fips_full"]
        )["population"]
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

    Uses state-to-county population weights derived from CD-county data.
    This is used when CD is not known.

    Args:
        state_fips: Array of 2-digit state FIPS codes (as ints)
        random_state: Optional random state for reproducibility

    Returns:
        Array of 3-digit county FIPS codes (as ints)
    """
    if random_state is None:
        random_state = np.random.RandomState()

    crosswalk = get_state_county_crosswalk_from_cd()

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
    print("Building CD-to-county crosswalk...")
    crosswalk = build_cd_county_crosswalk()
    crosswalk.to_csv(CD_COUNTY_CROSSWALK_FILE, index=False)
    print(f"Saved crosswalk to {CD_COUNTY_CROSSWALK_FILE}")

    # Export to JSON for geo-stacking compatibility
    print("\nExporting to JSON format...")
    export_to_json()

    # Print summary statistics
    print(f"\nTotal CD-county pairs: {len(crosswalk)}")
    print(f"Unique CDs: {crosswalk['cd_code'].nunique()}")
    print(f"Unique counties: {crosswalk['county_fips_full'].nunique()}")

    # Show distribution of counties per CD
    counties_per_cd = crosswalk.groupby("cd_code").size()
    print(f"\nCounties per CD:")
    print(f"  Min: {counties_per_cd.min()}")
    print(f"  Max: {counties_per_cd.max()}")
    print(f"  Mean: {counties_per_cd.mean():.2f}")
    print(f"  Median: {counties_per_cd.median():.0f}")

    # Count single-county CDs (deterministic assignment)
    single_county = (counties_per_cd == 1).sum()
    print(
        f"\nCDs with single county (deterministic): "
        f"{single_county} ({100*single_county/len(counties_per_cd):.1f}%)"
    )
