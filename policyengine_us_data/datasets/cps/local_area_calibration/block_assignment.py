"""
Census block assignment for congressional districts.

Provides population-weighted random census block assignment for households
within each congressional district. This enables consistent lookup of all
geographic variables from a single block GEOID:

- State, County, Tract (derived from block GEOID structure)
- CBSA/Metro area (via county crosswalk)
- SLDU/SLDL (State Legislative Districts)
- Place/City (via Census BAF)
- PUMA (via tract crosswalk)
- VTD (Voting Tabulation District)

The distributions are computed from Census block-level population data and
stored in storage/block_cd_distributions.csv.gz. Block GEOIDs are 15-digit
strings in format SSCCCTTTTTTBBBB (state, county, tract, block).

Additional geography lookups use storage/block_crosswalk.csv.gz which maps
blocks to SLDU, SLDL, Place, VTD, and PUMA.
"""

import random
import re
from functools import lru_cache
from io import StringIO
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)
from policyengine_us_data.storage import STORAGE_FOLDER


# === GEOID Parsing Functions ===
# Block GEOID format: SSCCCTTTTTTBBBB (15 chars)
# SS = State FIPS (2 digits)
# CCC = County FIPS (3 digits)
# TTTTTT = Tract (6 digits)
# BBBB = Block (4 digits)


def get_state_fips_from_block(block_geoid: str) -> str:
    """Extract 2-digit state FIPS from block GEOID."""
    return block_geoid[:2]


def get_county_fips_from_block(block_geoid: str) -> str:
    """Extract 5-digit county FIPS (state + county) from block GEOID."""
    return block_geoid[:5]


def get_tract_geoid_from_block(block_geoid: str) -> str:
    """Extract 11-digit tract GEOID (state + county + tract) from block GEOID."""
    return block_geoid[:11]


# === County FIPS to Enum Mapping ===


@lru_cache(maxsize=1)
def _build_county_fips_to_enum() -> Dict[str, str]:
    """
    Build mapping from 5-digit county FIPS to County enum name.

    Downloads Census county FIPS file and matches to County enum names.
    Cached to avoid repeated downloads.
    """
    url = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"
    response = requests.get(url, timeout=60)
    df = pd.read_csv(
        StringIO(response.text),
        delimiter="|",
        dtype=str,
        usecols=["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"],
    )

    valid_enum_names = set(County._member_names_)
    fips_to_enum = {}

    for _, row in df.iterrows():
        county_fips = row["STATEFP"] + row["COUNTYFP"]
        state_code = row["STATE"]
        county_name = row["COUNTYNAME"]

        # Transform to enum name format
        enum_name = county_name.upper()
        enum_name = re.sub(r"[.'\"]", "", enum_name)
        enum_name = enum_name.replace("-", "_")
        enum_name = enum_name.replace(" ", "_")
        enum_name = f"{enum_name}_{state_code}"

        if enum_name in valid_enum_names:
            fips_to_enum[county_fips] = enum_name

    return fips_to_enum


def get_county_enum_index_from_block(block_geoid: str) -> int:
    """
    Get County enum index from block GEOID.

    Args:
        block_geoid: 15-digit census block GEOID

    Returns:
        Integer index into County enum, or UNKNOWN index if not found
    """
    county_fips = get_county_fips_from_block(block_geoid)
    fips_to_enum = _build_county_fips_to_enum()
    enum_name = fips_to_enum.get(county_fips, "UNKNOWN")
    return County._member_names_.index(enum_name)


# === CBSA Lookup ===


@lru_cache(maxsize=1)
def _load_cbsa_crosswalk() -> Dict[str, str]:
    """
    Load county FIPS to CBSA code crosswalk from NBER.

    Returns:
        Dict mapping 5-digit county FIPS to CBSA code (or None if not in CBSA)
    """
    url = "https://data.nber.org/cbsa-csa-fips-county-crosswalk/2023/cbsa2fipsxw_2023.csv"
    try:
        df = pd.read_csv(url, dtype=str)
        # Build 5-digit county FIPS from state + county codes
        df["county_fips"] = df["fipsstatecode"] + df["fipscountycode"]
        # Only include rows with valid CBSA codes (not blank/NA)
        df = df.dropna(subset=["cbsacode"])
        df = df[df["cbsacode"].str.strip() != ""]

        return dict(zip(df["county_fips"], df["cbsacode"]))
    except Exception:
        # Return empty dict if download fails - rural areas will return None
        return {}


def get_cbsa_from_county(county_fips: str) -> Optional[str]:
    """
    Get CBSA code for a county.

    Args:
        county_fips: 5-digit county FIPS code

    Returns:
        CBSA code (e.g., "35620" for NYC metro) or None if not in CBSA
    """
    crosswalk = _load_cbsa_crosswalk()
    return crosswalk.get(county_fips)


# === Block Crosswalk for Additional Geographies ===


@lru_cache(maxsize=1)
def _load_block_crosswalk() -> pd.DataFrame:
    """
    Load block-level crosswalk for SLDU, SLDL, Place, VTD, PUMA.

    Returns:
        DataFrame indexed by block_geoid with columns for each geography.
    """
    csv_path = STORAGE_FOLDER / "block_crosswalk.csv.gz"

    if not csv_path.exists():
        print(
            f"Warning: {csv_path} not found. "
            "Run make_block_crosswalk.py to generate."
        )
        return pd.DataFrame()

    # Load all columns as strings, then set index
    # (using index_col directly can convert to int, dropping leading zeros)
    df = pd.read_csv(csv_path, dtype=str)
    df = df.set_index("block_geoid")
    return df


def get_sldu_from_block(block_geoid: str) -> Optional[str]:
    """Get State Legislative District Upper from block GEOID."""
    crosswalk = _load_block_crosswalk()
    if block_geoid in crosswalk.index:
        val = crosswalk.loc[block_geoid, "sldu"]
        return val if pd.notna(val) else None
    return None


def get_sldl_from_block(block_geoid: str) -> Optional[str]:
    """Get State Legislative District Lower from block GEOID."""
    crosswalk = _load_block_crosswalk()
    if block_geoid in crosswalk.index:
        val = crosswalk.loc[block_geoid, "sldl"]
        return val if pd.notna(val) else None
    return None


def get_place_fips_from_block(block_geoid: str) -> Optional[str]:
    """Get Place/City FIPS from block GEOID."""
    crosswalk = _load_block_crosswalk()
    if block_geoid in crosswalk.index:
        val = crosswalk.loc[block_geoid, "place_fips"]
        return val if pd.notna(val) else None
    return None


def get_vtd_from_block(block_geoid: str) -> Optional[str]:
    """Get Voting Tabulation District from block GEOID."""
    crosswalk = _load_block_crosswalk()
    if block_geoid in crosswalk.index:
        val = crosswalk.loc[block_geoid, "vtd"]
        return val if pd.notna(val) else None
    return None


def get_puma_from_block(block_geoid: str) -> Optional[str]:
    """Get PUMA (Public Use Microdata Area) from block GEOID."""
    crosswalk = _load_block_crosswalk()
    if block_geoid in crosswalk.index:
        val = crosswalk.loc[block_geoid, "puma"]
        return val if pd.notna(val) else None
    return None


def get_all_geography_from_block(block_geoid: str) -> Dict[str, Optional[str]]:
    """
    Get all geographic variables from a single block GEOID lookup.

    More efficient than calling individual functions when you need multiple
    geographies, as it only does one crosswalk lookup.

    Args:
        block_geoid: 15-digit census block GEOID

    Returns:
        Dict with keys: sldu, sldl, place_fips, vtd, puma
        Values are strings or None if not available.
    """
    crosswalk = _load_block_crosswalk()
    if block_geoid in crosswalk.index:
        row = crosswalk.loc[block_geoid]
        return {
            "sldu": row["sldu"] if pd.notna(row["sldu"]) else None,
            "sldl": row["sldl"] if pd.notna(row["sldl"]) else None,
            "place_fips": row["place_fips"] if pd.notna(row["place_fips"]) else None,
            "vtd": row["vtd"] if pd.notna(row["vtd"]) else None,
            "puma": row["puma"] if pd.notna(row["puma"]) else None,
        }
    return {
        "sldu": None,
        "sldl": None,
        "place_fips": None,
        "vtd": None,
        "puma": None,
    }


# === Block Distribution Loading/Generation ===


def _load_block_distributions() -> Dict[str, Dict[str, float]]:
    """
    Load pre-computed P(block|CD) distributions from CSV.

    Returns:
        Dict mapping CD GEOID to Dict[block_geoid, probability]
    """
    csv_path = STORAGE_FOLDER / "block_cd_distributions.csv.gz"

    if not csv_path.exists():
        print(
            f"Warning: {csv_path} not found. "
            "Run make_block_cd_distributions.py to generate."
        )
        return {}

    df = pd.read_csv(csv_path, dtype={"block_geoid": str})
    distributions = {}
    for cd_geoid, group in df.groupby("cd_geoid"):
        distributions[str(int(cd_geoid))] = dict(
            zip(group["block_geoid"], group["probability"])
        )
    return distributions


# Lazy-load distributions at module import
_BLOCK_DISTRIBUTIONS: Dict[str, Dict[str, float]] = {}


def _get_block_distributions() -> Dict[str, Dict[str, float]]:
    """Get block distributions, loading if not already loaded."""
    global _BLOCK_DISTRIBUTIONS
    if not _BLOCK_DISTRIBUTIONS:
        _BLOCK_DISTRIBUTIONS = _load_block_distributions()
    return _BLOCK_DISTRIBUTIONS


# === Assignment Functions ===


def _generate_fallback_blocks(cd_geoid: str, n_households: int) -> np.ndarray:
    """
    Generate fallback block GEOIDs for CDs not in pre-computed data.

    Uses county assignment as a fallback and generates synthetic but
    structurally valid block GEOIDs. Used primarily for testing.

    Args:
        cd_geoid: Congressional district geoid
        n_households: Number of blocks to generate

    Returns:
        Array of 15-character block GEOID strings
    """
    # Import here to avoid circular dependency
    from policyengine_us_data.datasets.cps.local_area_calibration.county_assignment import (
        assign_counties_for_cd,
    )

    # Fall back to county assignment
    county_indices = assign_counties_for_cd(
        cd_geoid, n_households, seed=hash(cd_geoid) % (2**31)
    )

    # Convert county indices to block GEOIDs
    fips_to_enum = _build_county_fips_to_enum()
    enum_to_fips = {v: k for k, v in fips_to_enum.items()}

    blocks = []
    for idx in county_indices:
        county_name = County._member_names_[idx]
        county_fips = enum_to_fips.get(county_name, "00000")
        # Generate synthetic block: county_fips + tract (000100) + block (1000)
        block_geoid = county_fips + "0001001000"
        blocks.append(block_geoid)

    return np.array(blocks)


def assign_blocks_for_cd(
    cd_geoid: str,
    n_households: int,
    seed: int,
    distributions: Dict[str, Dict[str, float]] = None,
) -> np.ndarray:
    """
    Assign census block GEOIDs to households in a CD using weighted random selection.

    Uses pre-computed P(block|CD) distributions from Census population data.
    Falls back to county-based synthetic blocks for CDs not in pre-computed data.

    Args:
        cd_geoid: Congressional district geoid (e.g., "3610")
        n_households: Number of households to assign
        seed: Random seed for reproducibility
        distributions: Optional override distributions. If None, uses
            pre-computed distributions from CSV.

    Returns:
        Array of 15-character block GEOID strings
    """
    random.seed(seed)

    if distributions is None:
        distributions = _get_block_distributions()

    cd_key = str(int(cd_geoid))

    if cd_key not in distributions:
        # Fall back to county-based assignment for unknown CDs (e.g., in tests)
        return _generate_fallback_blocks(cd_geoid, n_households)

    dist = distributions[cd_key]
    blocks = list(dist.keys())
    weights = list(dist.values())
    selected = random.choices(blocks, weights=weights, k=n_households)
    return np.array(selected)


def assign_geography_for_cd(
    cd_geoid: str,
    n_households: int,
    seed: int,
    distributions: Dict[str, Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Assign all geographic variables for households in a CD.

    This is the main entry point that assigns a census block and then
    derives all other geographic variables from it, ensuring consistency.

    Args:
        cd_geoid: Congressional district geoid (e.g., "3610")
        n_households: Number of households to assign
        seed: Random seed for reproducibility
        distributions: Optional override distributions

    Returns:
        Dict with arrays for each geography:
        - block_geoid: 15-char block GEOID strings
        - county_fips: 5-digit county FIPS strings
        - tract_geoid: 11-digit tract GEOID strings
        - state_fips: 2-digit state FIPS strings
        - cbsa_code: CBSA code strings (or "" if not in CBSA)
        - sldu: State Legislative District Upper (or "" if not available)
        - sldl: State Legislative District Lower (or "" if not available)
        - place_fips: Place/City FIPS (or "" if not in a place)
        - vtd: Voting Tabulation District (or "" if not available)
        - puma: Public Use Microdata Area (or "" if not available)
        - county_index: int32 indices into County enum (for backwards compat)
    """
    # Assign blocks first
    block_geoids = assign_blocks_for_cd(
        cd_geoid, n_households, seed, distributions
    )

    # Derive geography directly from block GEOID structure
    county_fips = np.array(
        [get_county_fips_from_block(b) for b in block_geoids]
    )
    tract_geoids = np.array(
        [get_tract_geoid_from_block(b) for b in block_geoids]
    )
    state_fips = np.array([get_state_fips_from_block(b) for b in block_geoids])

    # CBSA lookup via county (may be None for rural areas)
    cbsa_codes = np.array(
        [get_cbsa_from_county(c) or "" for c in county_fips]
    )

    # County enum indices for backwards compatibility
    county_indices = np.array(
        [get_county_enum_index_from_block(b) for b in block_geoids],
        dtype=np.int32,
    )

    # Lookup additional geographies from block crosswalk
    # Do batch lookup for efficiency
    crosswalk = _load_block_crosswalk()

    sldu_list = []
    sldl_list = []
    place_fips_list = []
    vtd_list = []
    puma_list = []

    for b in block_geoids:
        if not crosswalk.empty and b in crosswalk.index:
            row = crosswalk.loc[b]
            sldu_list.append(row["sldu"] if pd.notna(row["sldu"]) else "")
            sldl_list.append(row["sldl"] if pd.notna(row["sldl"]) else "")
            place_fips_list.append(row["place_fips"] if pd.notna(row["place_fips"]) else "")
            vtd_list.append(row["vtd"] if pd.notna(row["vtd"]) else "")
            puma_list.append(row["puma"] if pd.notna(row["puma"]) else "")
        else:
            sldu_list.append("")
            sldl_list.append("")
            place_fips_list.append("")
            vtd_list.append("")
            puma_list.append("")

    return {
        "block_geoid": block_geoids,
        "county_fips": county_fips,
        "tract_geoid": tract_geoids,
        "state_fips": state_fips,
        "cbsa_code": cbsa_codes,
        "sldu": np.array(sldu_list),
        "sldl": np.array(sldl_list),
        "place_fips": np.array(place_fips_list),
        "vtd": np.array(vtd_list),
        "puma": np.array(puma_list),
        "county_index": county_indices,
    }


# === County Filter Functions (for city-level datasets) ===


def get_county_filter_probability(
    cd_geoid: str,
    county_filter: set,
) -> float:
    """
    Calculate P(county in filter | CD) using block-level data.

    Returns the probability that a household in this CD would be in the
    target area (e.g., NYC). Used for weight scaling when building
    city-level datasets.

    Args:
        cd_geoid: Congressional district geoid (e.g., "3610")
        county_filter: Set of county enum names that define the target area

    Returns:
        Probability between 0 and 1
    """
    distributions = _get_block_distributions()
    cd_key = str(int(cd_geoid))

    if cd_key not in distributions:
        return 0.0

    dist = distributions[cd_key]

    # Convert county enum names to FIPS codes for comparison
    fips_to_enum = _build_county_fips_to_enum()
    enum_to_fips = {v: k for k, v in fips_to_enum.items()}
    target_fips = {enum_to_fips.get(name) for name in county_filter}
    target_fips.discard(None)

    # Sum probabilities of blocks in target counties
    return sum(
        prob
        for block, prob in dist.items()
        if get_county_fips_from_block(block) in target_fips
    )


def get_filtered_block_distribution(
    cd_geoid: str,
    county_filter: set,
) -> Dict[str, float]:
    """
    Get normalized distribution over blocks in target counties only.

    Used when building city-level datasets to assign only blocks in valid
    counties while maintaining relative proportions within the target area.

    Args:
        cd_geoid: Congressional district geoid (e.g., "3610")
        county_filter: Set of county enum names that define the target area

    Returns:
        Dictionary mapping block GEOIDs to normalized probabilities.
        Empty dict if CD has no overlap with target area.
    """
    distributions = _get_block_distributions()
    cd_key = str(int(cd_geoid))

    if cd_key not in distributions:
        return {}

    dist = distributions[cd_key]

    # Convert county enum names to FIPS codes for comparison
    fips_to_enum = _build_county_fips_to_enum()
    enum_to_fips = {v: k for k, v in fips_to_enum.items()}
    target_fips = {enum_to_fips.get(name) for name in county_filter}
    target_fips.discard(None)

    # Filter to blocks in target counties
    filtered = {
        block: prob
        for block, prob in dist.items()
        if get_county_fips_from_block(block) in target_fips
    }

    # Normalize
    total = sum(filtered.values())
    if total > 0:
        return {block: prob / total for block, prob in filtered.items()}
    return {}
