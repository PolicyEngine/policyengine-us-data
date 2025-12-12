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


def _build_state_counties() -> Dict[str, List[str]]:
    """Build mapping from state code to list of county enum names."""
    state_counties = {}
    for name in County._member_names_:
        if name == "UNKNOWN":
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
