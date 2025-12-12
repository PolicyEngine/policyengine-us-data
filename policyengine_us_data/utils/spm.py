"""SPM threshold calculation utilities using the spm-calculator package."""

import numpy as np
from spm_calculator import SPMCalculator, spm_equivalence_scale


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


def calculate_spm_thresholds_by_puma(
    num_adults: np.ndarray,
    num_children: np.ndarray,
    tenure_codes: np.ndarray,
    state_fips: np.ndarray,
    puma_codes: np.ndarray,
    year: int,
) -> np.ndarray:
    """
    Calculate SPM thresholds using PUMA-level geographic adjustments.

    This function uses the spm-calculator's PUMA-level geographic adjustment
    lookup to compute thresholds with local housing cost adjustments.
    Requires CENSUS_API_KEY environment variable to be set.

    Args:
        num_adults: Array of number of adults (18+) in each SPM unit.
        num_children: Array of number of children (<18) in each SPM unit.
        tenure_codes: Array of Census tenure/mortgage status codes.
            1 = owner with mortgage, 2 = owner without mortgage, 3 = renter.
        state_fips: Array of 2-digit state FIPS codes (integers).
        puma_codes: Array of 5-digit PUMA codes (integers).
        year: The year for which to calculate thresholds.

    Returns:
        Array of SPM threshold values with PUMA-level geographic adjustments.
    """
    calc = SPMCalculator(year=year)

    # Build 7-digit PUMA identifiers: 2-digit state + 5-digit PUMA
    # Format as strings with proper zero-padding
    puma_ids = np.array(
        [
            f"{int(state):02d}{int(puma):05d}"
            for state, puma in zip(state_fips, puma_codes)
        ]
    )

    # Map tenure codes to strings
    tenure_strs = np.array(
        [TENURE_CODE_MAP.get(int(t), "renter") for t in tenure_codes]
    )

    # Use vectorized calculation from spm-calculator
    thresholds = calc.calculate_thresholds(
        num_adults=num_adults.astype(int),
        num_children=num_children.astype(int),
        tenure=tenure_strs,
        geography_type="puma",
        geography_ids=puma_ids,
    )

    return thresholds
