"""SPM threshold calculation utilities using the spm-calculator package."""

import numpy as np
from spm_calculator import SPMCalculator, spm_equivalence_scale


# Census CPS SPM_TENMORTSTATUS codes to spm-calculator tenure mapping
# Based on IPUMS SPMMORT documentation:
# 1 = Owner with mortgage
# 2 = Owner without mortgage
# 3 = Renter
TENURE_CODE_MAP = {
    1: "owner_with_mortgage",
    2: "owner_without_mortgage",
    3: "renter",
}


def calculate_spm_thresholds_with_geoadj(
    num_adults: np.ndarray,
    num_children: np.ndarray,
    tenure_codes: np.ndarray,
    geoadj: np.ndarray,
    year: int,
) -> np.ndarray:
    """
    Calculate SPM thresholds using Census-provided geographic adjustments.

    This function uses the SPM_GEOADJ values already computed by the Census
    Bureau, combined with spm-calculator's base thresholds and equivalence
    scale formula. This avoids the need for a Census API key.

    Args:
        num_adults: Array of number of adults (18+) in each SPM unit.
        num_children: Array of number of children (<18) in each SPM unit.
        tenure_codes: Array of Census tenure/mortgage status codes.
            1 = owner with mortgage, 2 = owner without mortgage, 3 = renter.
        geoadj: Array of Census SPM_GEOADJ geographic adjustment factors.
        year: The year for which to calculate thresholds.

    Returns:
        Array of SPM threshold values.
    """
    calc = SPMCalculator(year=year)
    base_thresholds = calc.get_base_thresholds()

    n = len(num_adults)
    thresholds = np.zeros(n)

    for i in range(n):
        tenure_str = TENURE_CODE_MAP.get(int(tenure_codes[i]), "renter")
        base = base_thresholds[tenure_str]
        equiv_scale = spm_equivalence_scale(
            int(num_adults[i]), int(num_children[i])
        )
        thresholds[i] = base * equiv_scale * geoadj[i]

    return thresholds


def calculate_spm_thresholds_national(
    num_adults: np.ndarray,
    num_children: np.ndarray,
    tenure_codes: np.ndarray,
    year: int,
) -> np.ndarray:
    """
    Calculate SPM thresholds using national-level thresholds (no geoadj).

    This is used for datasets like ACS that don't have pre-computed
    geographic adjustment factors.

    Args:
        num_adults: Array of number of adults (18+) in each SPM unit.
        num_children: Array of number of children (<18) in each SPM unit.
        tenure_codes: Array of Census tenure/mortgage status codes.
            1 = owner with mortgage, 2 = owner without mortgage, 3 = renter.
        year: The year for which to calculate thresholds.

    Returns:
        Array of SPM threshold values using national averages.
    """
    calc = SPMCalculator(year=year)
    base_thresholds = calc.get_base_thresholds()

    n = len(num_adults)
    thresholds = np.zeros(n)

    for i in range(n):
        tenure_str = TENURE_CODE_MAP.get(int(tenure_codes[i]), "renter")
        base = base_thresholds[tenure_str]
        equiv_scale = spm_equivalence_scale(
            int(num_adults[i]), int(num_children[i])
        )
        # No geographic adjustment for national-level thresholds
        thresholds[i] = base * equiv_scale

    return thresholds


def map_tenure_acs_to_spm(tenure_type: np.ndarray) -> np.ndarray:
    """
    Map ACS tenure type values to spm-calculator tenure codes.

    Args:
        tenure_type: Array of ACS TEN values.
            1 = Owned with mortgage/loan
            2 = Owned free and clear
            3 = Rented

    Returns:
        Array of tenure code integers matching Census SPM format.
    """
    # ACS TEN codes map directly to Census SPM codes:
    # ACS 1 (owned with mortgage) -> Census 1 (owner_with_mortgage)
    # ACS 2 (owned outright) -> Census 2 (owner_without_mortgage)
    # ACS 3 (rented) -> Census 3 (renter)
    return np.where(
        np.isin(tenure_type, [1, 2, 3]),
        tenure_type,
        3,  # Default to renter for unknown values
    )
