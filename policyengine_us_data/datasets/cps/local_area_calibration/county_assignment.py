"""
County assignment for congressional districts.

Provides conditional probability distributions P(county | CD) for assigning
counties to households within each congressional district.

The distributions are pre-computed from Census block-level data and stored
in storage/county_cd_distributions.csv. If the CSV is not available, falls
back to uniform distributions across all counties in the CD's state.
"""

import random
from typing import Dict, List
import numpy as np
import pandas as pd

from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
    County,
)
from policyengine_us_data.storage import STORAGE_FOLDER


# Invalid county entries in policyengine-us County enum.
# These are counties assigned to wrong states, non-existent combinations,
# or encoding mismatches. Validated against Census 2020 county reference.
# See audit_county_enum.py for details.
# TODO: Remove this workaround when fixed upstream in policyengine-us
INVALID_COUNTY_NAMES = {
    "APACHE_COUNTY_NM",
    "APACHE_COUNTY_UT",
    "ATCHISON_COUNTY_IA",
    "BAYAMÓN_MUNICIPIO_PR",
    "BENEWAH_COUNTY_WA",
    "BONNEVILLE_COUNTY_WY",
    "CARTER_COUNTY_SD",
    "CLARK_COUNTY_IA",
    "CLINTON_COUNTY_TN",
    "COLBERT_COUNTY_MS",
    "CUSTER_COUNTY_WY",
    "DECATUR_COUNTY_NE",
    "DESHA_COUNTY_MS",
    "DORCHESTER_COUNTY_DE",
    "DOÑA_ANA_COUNTY_NM",
    "DOÑA_ANA_COUNTY_TX",
    "EMMONS_COUNTY_SD",
    "FULTON_COUNTY_TN",
    "GREGORY_COUNTY_NE",
    "GUÁNICA_MUNICIPIO_PR",
    "HARDING_COUNTY_ND",
    "INYO_COUNTY_NV",
    "JEFFERSON_COUNTY_VA",
    "JEWELL_COUNTY_NE",
    "JUANA_DÍAZ_MUNICIPIO_PR",
    "KIMBALL_COUNTY_WY",
    "KOSSUTH_COUNTY_MN",
    "LARIMER_COUNTY_WY",
    "LAS_MARÍAS_MUNICIPIO_PR",
    "LEE_COUNTY_TN",
    "LE_FLORE_COUNTY_AR",
    "LOÍZA_MUNICIPIO_PR",
    "MANATÍ_MUNICIPIO_PR",
    "MARSHALL_COUNTY_ND",
    "MAYAGÜEZ_MUNICIPIO_PR",
    "MCDOWELL_COUNTY_VA",
    "MCKENZIE_COUNTY_MT",
    "MCKINLEY_COUNTY_AZ",
    "MILLER_COUNTY_TX",
    "NEW_CASTLE_COUNTY_MD",
    "OGLALA_LAKOTA_COUNTY_NE",
    "OLDHAM_COUNTY_NM",
    "O_BRIEN_COUNTY_IA",
    "PEND_OREILLE_COUNTY_ID",
    "PERKINS_COUNTY_ND",
    "PEÑUELAS_MUNICIPIO_PR",
    "PRINCE_GEORGE_S_COUNTY_MD",
    "QUEEN_ANNE_S_COUNTY_MD",
    "RICHLAND_COUNTY_SD",
    "RIO_ARRIBA_COUNTY_CO",
    "ROBERTS_COUNTY_MN",
    "ROCK_COUNTY_SD",
    "RÍO_GRANDE_MUNICIPIO_PR",
    "SAN_GERMÁN_MUNICIPIO_PR",
    "SAN_JUAN_COUNTY_AZ",
    "SCOTLAND_COUNTY_IA",
    "SHERMAN_COUNTY_OK",
    "SIOUX_COUNTY_SD",
    "ST_MARY_S_COUNTY_MD",
    "SUFFOLK_COUNTY_CT",
    "SUMMIT_COUNTY_WY",
    "TIPTON_COUNTY_AR",
    "TODD_COUNTY_NE",
    "TROUP_COUNTY_AL",
    "WHITE_PINE_COUNTY_UT",
}


def _build_state_counties() -> Dict[str, List[str]]:
    """Build mapping from state code to list of county enum names."""
    state_counties = {}
    for name in County._member_names_:
        if name == "UNKNOWN":
            continue
        if name in INVALID_COUNTY_NAMES:
            continue
        state_code = name.split("_")[-1]
        if state_code not in state_counties:
            state_counties[state_code] = []
        state_counties[state_code].append(name)
    return state_counties


_STATE_COUNTIES = _build_state_counties()


def _generate_uniform_distribution(cd_geoid: str) -> Dict[str, float]:
    """Generate uniform distribution across counties in CD's state."""
    from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
        STATE_CODES,
    )

    fips_to_code = {fips: code for fips, code in STATE_CODES.items()}
    state_fips = int(cd_geoid) // 100
    state_code = fips_to_code.get(state_fips)

    if state_code and state_code in _STATE_COUNTIES:
        counties = _STATE_COUNTIES[state_code]
        prob = 1.0 / len(counties)
        return {c: prob for c in counties}
    return {"UNKNOWN": 1.0}


def _load_county_distributions() -> Dict[str, Dict[str, float]]:
    """Load pre-computed P(county|CD) distributions from CSV."""
    csv_path = STORAGE_FOLDER / "county_cd_distributions.csv"

    if not csv_path.exists():
        print(
            f"Warning: {csv_path} not found. "
            "Using uniform distributions. "
            "Run make_county_cd_distributions.py to generate."
        )
        return {}

    df = pd.read_csv(csv_path)
    distributions = {}
    for cd_geoid, group in df.groupby("cd_geoid"):
        distributions[str(int(cd_geoid))] = dict(
            zip(group["county_name"], group["probability"])
        )
    return distributions


# Load distributions at module import
_CD_COUNTY_DISTRIBUTIONS = _load_county_distributions()


def get_county_index(county_name: str) -> int:
    """Convert county enum name to integer index."""
    return County._member_names_.index(county_name)


def assign_counties_for_cd(
    cd_geoid: str,
    n_households: int,
    seed: int,
    distributions: Dict[str, Dict[str, float]] = None,
) -> np.ndarray:
    """
    Assign county indices to households in a CD using weighted random selection.

    Uses pre-computed P(county|CD) distributions from Census block data.
    Falls back to uniform distribution if pre-computed data unavailable.

    Args:
        cd_geoid: Congressional district geoid (e.g., "3610")
        n_households: Number of households to assign
        seed: Random seed for reproducibility
        distributions: Optional override distributions. If None, uses
            pre-computed distributions from CSV.

    Returns:
        Array of county enum indices (integers)
    """
    random.seed(seed)

    # Use pre-computed distributions by default
    if distributions is None:
        distributions = _CD_COUNTY_DISTRIBUTIONS

    cd_key = str(int(cd_geoid))

    if cd_key in distributions:
        dist = distributions[cd_key]
    else:
        # Fall back to uniform distribution for this CD
        dist = _generate_uniform_distribution(cd_key)

    counties = list(dist.keys())
    weights = list(dist.values())
    selected = random.choices(counties, weights=weights, k=n_households)
    return np.array([get_county_index(c) for c in selected], dtype=np.int32)


def get_county_filter_probability(
    cd_geoid: str,
    county_filter: set,
) -> float:
    """
    Calculate P(county in filter | CD).

    Returns the probability that a household in this CD would be in the
    target area (e.g., NYC). Used for weight scaling when building
    city-level datasets.

    Args:
        cd_geoid: Congressional district geoid (e.g., "3610")
        county_filter: Set of county names that define the target area

    Returns:
        Probability between 0 and 1
    """
    cd_key = str(int(cd_geoid))

    if cd_key in _CD_COUNTY_DISTRIBUTIONS:
        dist = _CD_COUNTY_DISTRIBUTIONS[cd_key]
    else:
        dist = _generate_uniform_distribution(cd_key)

    return sum(
        prob for county, prob in dist.items() if county in county_filter
    )


def get_filtered_county_distribution(
    cd_geoid: str,
    county_filter: set,
) -> Dict[str, float]:
    """
    Get normalized distribution over target counties only.

    Used when building city-level datasets to assign only valid counties
    while maintaining relative proportions within the target area.

    Args:
        cd_geoid: Congressional district geoid (e.g., "3610")
        county_filter: Set of county names that define the target area

    Returns:
        Dictionary mapping county names to normalized probabilities.
        Empty dict if CD has no overlap with target area.
    """
    cd_key = str(int(cd_geoid))

    if cd_key in _CD_COUNTY_DISTRIBUTIONS:
        dist = _CD_COUNTY_DISTRIBUTIONS[cd_key]
    else:
        dist = _generate_uniform_distribution(cd_key)

    filtered = {c: p for c, p in dist.items() if c in county_filter}
    total = sum(filtered.values())

    if total > 0:
        return {c: p / total for c, p in filtered.items()}
    return {}
