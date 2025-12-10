"""
Geographic adjustment (GEOADJ) lookup for congressional districts.

GEOADJ adjusts SPM thresholds for local housing costs using the formula:
    GEOADJ = (local_median_rent / national_median_rent) × 0.492 + 0.508

Where 0.492 is the housing share of the SPM threshold for renters.

Data source: ACS Table B25031 (Median Gross Rent by Bedrooms)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Housing portion of SPM threshold (for renters)
HOUSING_SHARE = 0.492

# Path to cached GEOADJ data
STORAGE_FOLDER = Path(__file__).parent.parent / "storage"


def calculate_geoadj_from_rent(
    local_rent: float | np.ndarray,
    national_rent: float,
) -> float | np.ndarray:
    """
    Calculate GEOADJ from local and national median rents.

    Formula: GEOADJ = (local_rent / national_rent) × 0.492 + 0.508

    Args:
        local_rent: Local area median rent (scalar or array)
        national_rent: National median rent

    Returns:
        GEOADJ value(s)
    """
    rent_ratio = np.asarray(local_rent) / national_rent
    return rent_ratio * HOUSING_SHARE + (1 - HOUSING_SHARE)


@lru_cache(maxsize=16)
def _load_district_geoadj(year: int) -> pd.DataFrame:
    """Load or create district GEOADJ lookup table."""
    cache_file = STORAGE_FOLDER / f"district_geoadj_{year}.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file, dtype={"district_code": str})

    # Create from ACS data
    df = _create_district_geoadj_from_acs(year)
    df.to_csv(cache_file, index=False)
    return df


def _fetch_acs_district_rents(year: int) -> pd.DataFrame:
    """
    Fetch median 2-bedroom rent by congressional district from ACS.

    Uses Census API to get ACS 5-year estimates, Table B25031.
    """
    try:
        from census import Census
    except ImportError:
        raise ImportError(
            "census package required. Install with: pip install census"
        )

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        raise ValueError(
            "CENSUS_API_KEY environment variable not set. "
            "Get a free key at https://api.census.gov/data/key_signup.html"
        )

    c = Census(api_key)

    # B25031_004E = Median gross rent, 2 bedrooms
    variable = "B25031_004E"

    all_data = []
    for state_fips in range(1, 57):  # State FIPS codes
        try:
            data = c.acs5.get(
                [variable],
                {
                    "for": "congressional district:*",
                    "in": f"state:{state_fips:02d}",
                },
                year=year,
            )
            all_data.extend(data)
        except Exception:
            pass

    df = pd.DataFrame(all_data)
    df["district_code"] = df["state"].str.zfill(2) + df[
        "congressional district"
    ].str.zfill(2)
    df["median_rent"] = pd.to_numeric(df[variable], errors="coerce")

    return df[["district_code", "median_rent"]].dropna()


def _get_national_median_rent(year: int) -> float:
    """Get national median 2-bedroom rent for a year."""
    try:
        from census import Census
    except ImportError:
        raise ImportError("census package required")

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        raise ValueError("CENSUS_API_KEY not set")

    c = Census(api_key)
    data = c.acs5.get(["B25031_004E"], {"for": "us:*"}, year=year)
    return float(data[0]["B25031_004E"])


def _create_district_geoadj_from_acs(year: int) -> pd.DataFrame:
    """
    Create GEOADJ lookup table for all congressional districts.

    Args:
        year: ACS 5-year end year

    Returns:
        DataFrame with district_code, median_rent, and geoadj columns
    """
    # Get district rents
    df = _fetch_acs_district_rents(year)

    # Get national rent
    national_rent = _get_national_median_rent(year)

    # Calculate GEOADJ
    df["geoadj"] = calculate_geoadj_from_rent(df["median_rent"], national_rent)

    # Clamp to reasonable range (0.70 to 1.50)
    df["geoadj"] = df["geoadj"].clip(0.70, 1.50)

    return df


def create_district_geoadj_lookup(
    year: int = 2022,
    from_cache: bool = True,
) -> pd.DataFrame:
    """
    Create or load GEOADJ lookup table for congressional districts.

    Args:
        year: ACS 5-year end year (use year - 1 from target year)
        from_cache: If True, use cached data if available

    Returns:
        DataFrame with columns:
        - district_code: 4-digit code (state FIPS + district number)
        - median_rent: Median 2-bedroom rent from ACS
        - geoadj: Geographic adjustment factor
    """
    if from_cache:
        return _load_district_geoadj(year)
    return _create_district_geoadj_from_acs(year)


def get_district_geoadj(district_code: str, year: int = 2022) -> float:
    """
    Get GEOADJ for a specific congressional district.

    Args:
        district_code: 4-digit district code (e.g., "0612" for CA-12)
        year: ACS year for rent data

    Returns:
        GEOADJ value (typically 0.84 to 1.27)
    """
    lookup = create_district_geoadj_lookup(year)
    match = lookup[lookup["district_code"] == district_code]

    if len(match) == 0:
        # Return national average if district not found
        return 1.0

    return match["geoadj"].iloc[0]
