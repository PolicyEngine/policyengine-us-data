"""
Calculate local SPM thresholds for congressional districts.

The full SPM threshold formula is:
    threshold = base_threshold[tenure] × equivalence_scale × geoadj

This module replaces the original CPS SPM thresholds with properly
calculated local thresholds based on:
- District-specific GEOADJ from ACS median rents
- Tenure type (renter, owner with mortgage, owner without mortgage)
- Family composition via equivalence scale
"""

from typing import Optional

import numpy as np
import pandas as pd

# Use absolute imports that work both in package and standalone contexts
try:
    from .ce_threshold import calculate_base_thresholds
    from .district_geoadj import (
        create_district_geoadj_lookup,
        get_district_geoadj,
    )
except ImportError:
    from ce_threshold import calculate_base_thresholds
    from district_geoadj import (
        create_district_geoadj_lookup,
        get_district_geoadj,
    )


# SPM three-parameter equivalence scale
# Reference: https://www.census.gov/topics/income-poverty/poverty/guidance/poverty-measures.html
def spm_equivalence_scale(
    num_adults: int | np.ndarray,
    num_children: int | np.ndarray,
    normalize: bool = True,
) -> float | np.ndarray:
    """
    Calculate SPM three-parameter equivalence scale.

    Formula:
    - First adult: 1.0
    - Additional adults: 0.5 each
    - Children: 0.3 each

    Args:
        num_adults: Number of adults (18+) in the SPM unit
        num_children: Number of children (under 18) in the SPM unit
        normalize: If True, normalize to reference family (2A2C = 1.0)

    Returns:
        Equivalence scale value(s)
    """
    num_adults = np.asarray(num_adults)
    num_children = np.asarray(num_children)

    # First adult = 1.0, additional adults = 0.5 each
    adult_scale = np.where(
        num_adults >= 1, 1.0 + 0.5 * np.maximum(num_adults - 1, 0), 0.0
    )
    child_scale = 0.3 * num_children

    raw_scale = adult_scale + child_scale

    if normalize:
        # Reference family: 2 adults, 2 children = 1.0 + 0.5 + 0.6 = 2.1
        reference_scale = 2.1
        return raw_scale / reference_scale

    return raw_scale


def calculate_local_spm_thresholds(
    district_codes: np.ndarray | pd.Series,
    tenure_types: np.ndarray | pd.Series,
    num_adults: np.ndarray | pd.Series,
    num_children: np.ndarray | pd.Series,
    year: int = 2024,
    acs_year: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate SPM thresholds for SPM units based on their district.

    This replaces the original CPS SPM thresholds with properly
    calculated local thresholds using district-specific GEOADJ.

    Args:
        district_codes: 4-digit congressional district codes for each unit
        tenure_types: Tenure type for each unit
            ('renter', 'owner_with_mortgage', 'owner_without_mortgage')
        num_adults: Number of adults in each SPM unit
        num_children: Number of children in each SPM unit
        year: Target year for base thresholds
        acs_year: ACS year for GEOADJ (defaults to year - 1)

    Returns:
        Array of SPM thresholds in dollars
    """
    if acs_year is None:
        acs_year = min(year - 1, 2022)  # Latest available ACS

    # Convert inputs to arrays
    district_codes = np.asarray(district_codes, dtype=str)
    tenure_types = np.asarray(tenure_types, dtype=str)
    num_adults = np.asarray(num_adults)
    num_children = np.asarray(num_children)

    n = len(district_codes)

    # Get base thresholds
    base_thresholds = calculate_base_thresholds(year)

    # Get GEOADJ lookup
    geoadj_lookup = create_district_geoadj_lookup(acs_year)
    geoadj_dict = dict(
        zip(geoadj_lookup["district_code"], geoadj_lookup["geoadj"])
    )

    # Calculate equivalence scales
    equiv_scales = spm_equivalence_scale(num_adults, num_children)

    # Map tenure types to base thresholds
    tenure_map = {
        "renter": base_thresholds["renter"],
        "RENTED": base_thresholds["renter"],
        "owner_with_mortgage": base_thresholds["owner_with_mortgage"],
        "OWNED_WITH_MORTGAGE": base_thresholds["owner_with_mortgage"],
        "owner_without_mortgage": base_thresholds["owner_without_mortgage"],
        "OWNED_OUTRIGHT": base_thresholds["owner_without_mortgage"],
        "NONE": base_thresholds["renter"],  # Default to renter
    }

    # Calculate thresholds
    thresholds = np.zeros(n)
    for i in range(n):
        district = district_codes[i]
        tenure = tenure_types[i]

        # Get components
        base = tenure_map.get(tenure, base_thresholds["renter"])
        geoadj = geoadj_dict.get(district, 1.0)  # Default to 1.0 if not found
        equiv = (
            equiv_scales[i]
            if hasattr(equiv_scales, "__getitem__")
            else equiv_scales
        )

        thresholds[i] = base * equiv * geoadj

    return thresholds


def update_spm_thresholds_for_districts(
    data: dict,
    district_codes: np.ndarray,
    year: int = 2024,
) -> dict:
    """
    Update SPM thresholds in a dataset for district-level analysis.

    This function modifies the 'spm_unit_spm_threshold' variable to use
    properly calculated local thresholds instead of the original CPS values.

    Args:
        data: Dataset dictionary with arrays
        district_codes: Congressional district codes for each household
        year: Target year

    Returns:
        Updated dataset dictionary
    """
    # Get tenure types - need to expand from household to SPM unit level
    tenure_types = data.get(
        "tenure_type", np.full(len(district_codes), "renter")
    )
    if isinstance(tenure_types[0], bytes):
        tenure_types = np.array([t.decode() for t in tenure_types])

    # For now, use simple approach - need to map household districts to SPM units
    # This is a placeholder that should be refined based on actual data structure
    spm_unit_ids = data.get("spm_unit_id", np.arange(len(district_codes)))
    unique_spm_units = np.unique(spm_unit_ids)

    # Count adults and children per SPM unit
    ages = data.get("age", np.zeros(len(district_codes)))
    person_spm_ids = data.get("person_spm_unit_id", spm_unit_ids)

    spm_adults = {}
    spm_children = {}
    spm_districts = {}
    spm_tenure = {}

    for spm_id in unique_spm_units:
        mask = person_spm_ids == spm_id
        unit_ages = ages[mask]
        spm_adults[spm_id] = np.sum(unit_ages >= 18)
        spm_children[spm_id] = np.sum(unit_ages < 18)
        # Use first household's district for SPM unit
        spm_districts[spm_id] = (
            district_codes[mask][0] if np.any(mask) else "0000"
        )
        spm_tenure[spm_id] = (
            tenure_types[mask][0] if np.any(mask) else "renter"
        )

    # Calculate thresholds
    new_thresholds = calculate_local_spm_thresholds(
        district_codes=np.array(
            [spm_districts[uid] for uid in unique_spm_units]
        ),
        tenure_types=np.array([spm_tenure[uid] for uid in unique_spm_units]),
        num_adults=np.array([spm_adults[uid] for uid in unique_spm_units]),
        num_children=np.array([spm_children[uid] for uid in unique_spm_units]),
        year=year,
    )

    # Map back to SPM unit array
    threshold_dict = dict(zip(unique_spm_units, new_thresholds))
    data["spm_unit_spm_threshold"] = np.array(
        [threshold_dict.get(uid, 0) for uid in spm_unit_ids]
    )

    return data
