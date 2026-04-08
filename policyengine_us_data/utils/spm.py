"""SPM threshold calculation utilities aligned with PolicyEngine US."""

from functools import lru_cache
import numpy as np
from policyengine_us import CountryTaxBenefitSystem

PUBLISHED_SPM_REFERENCE_THRESHOLDS = {
    2015: {
        "renter": 25_155.0,
        "owner_with_mortgage": 24_859.0,
        "owner_without_mortgage": 20_639.0,
    },
    2016: {
        "renter": 25_558.0,
        "owner_with_mortgage": 25_248.0,
        "owner_without_mortgage": 20_943.0,
    },
    2017: {
        "renter": 26_213.0,
        "owner_with_mortgage": 25_897.0,
        "owner_without_mortgage": 21_527.0,
    },
    2018: {
        "renter": 26_905.0,
        "owner_with_mortgage": 26_565.0,
        "owner_without_mortgage": 22_095.0,
    },
    2019: {
        "renter": 27_515.0,
        "owner_with_mortgage": 27_172.0,
        "owner_without_mortgage": 22_600.0,
    },
    2020: {
        "renter": 28_881.0,
        "owner_with_mortgage": 28_533.0,
        "owner_without_mortgage": 23_948.0,
    },
    2021: {
        "renter": 31_453.0,
        "owner_with_mortgage": 31_089.0,
        "owner_without_mortgage": 26_022.0,
    },
    2022: {
        "renter": 33_402.0,
        "owner_with_mortgage": 32_949.0,
        "owner_without_mortgage": 27_679.0,
    },
    2023: {
        "renter": 36_606.0,
        "owner_with_mortgage": 36_192.0,
        "owner_without_mortgage": 30_347.0,
    },
    2024: {
        "renter": 39_430.0,
        "owner_with_mortgage": 39_068.0,
        "owner_without_mortgage": 32_586.0,
    },
}

LATEST_PUBLISHED_SPM_THRESHOLD_YEAR = max(PUBLISHED_SPM_REFERENCE_THRESHOLDS)
REFERENCE_RAW_SCALE = 3**0.7
TENURE_HOUSING_SHARES = {
    "owner_with_mortgage": 0.434,
    "owner_without_mortgage": 0.323,
    "renter": 0.443,
}

TENURE_CODE_MAP = {
    1: "owner_with_mortgage",
    2: "owner_without_mortgage",
    3: "renter",
}


@lru_cache(maxsize=1)
def _get_cpi_u_parameter():
    system = CountryTaxBenefitSystem()
    return system.parameters.gov.bls.cpi.cpi_u


def spm_equivalence_scale(num_adults, num_children):
    adults, children = np.broadcast_arrays(
        np.asarray(num_adults, dtype=float),
        np.asarray(num_children, dtype=float),
    )

    raw = np.zeros_like(adults, dtype=float)
    has_people = (adults + children) > 0
    with_children = has_people & (children > 0)

    single_adult_with_children = with_children & (adults <= 1)
    raw[single_adult_with_children] = (
        1.0 + 0.8 + 0.5 * np.maximum(children[single_adult_with_children] - 1, 0)
    ) ** 0.7

    multi_adult_with_children = with_children & ~single_adult_with_children
    raw[multi_adult_with_children] = (
        adults[multi_adult_with_children] + 0.5 * children[multi_adult_with_children]
    ) ** 0.7

    no_children = has_people & ~with_children
    one_adult = no_children & (adults <= 1)
    two_adults = no_children & (adults == 2)
    larger_adult_units = no_children & (adults > 2)

    raw[one_adult] = 1.0
    raw[two_adults] = 1.41
    raw[larger_adult_units] = adults[larger_adult_units] ** 0.7

    return raw / REFERENCE_RAW_SCALE


def get_spm_reference_thresholds(year: int) -> dict[str, float]:
    if year in PUBLISHED_SPM_REFERENCE_THRESHOLDS:
        return PUBLISHED_SPM_REFERENCE_THRESHOLDS[year].copy()

    if year < min(PUBLISHED_SPM_REFERENCE_THRESHOLDS):
        raise ValueError(
            f"No published SPM reference thresholds for {year}. "
            f"Earliest available year is {min(PUBLISHED_SPM_REFERENCE_THRESHOLDS)}."
        )

    cpi_u = _get_cpi_u_parameter()
    factor = float(
        cpi_u(f"{year}-02-01") / cpi_u(f"{LATEST_PUBLISHED_SPM_THRESHOLD_YEAR}-02-01")
    )
    latest_thresholds = PUBLISHED_SPM_REFERENCE_THRESHOLDS[
        LATEST_PUBLISHED_SPM_THRESHOLD_YEAR
    ]
    return {tenure: value * factor for tenure, value in latest_thresholds.items()}


def calculate_geoadj_from_rent(
    local_rent,
    national_rent: float,
    tenure: str = "renter",
):
    share = TENURE_HOUSING_SHARES[tenure]
    rent_ratio = np.asarray(local_rent, dtype=float) / float(national_rent)
    return rent_ratio * share + (1.0 - share)


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
    base_thresholds = get_spm_reference_thresholds(year)

    n = len(num_adults)
    thresholds = np.zeros(n)

    for i in range(n):
        tenure_str = TENURE_CODE_MAP.get(int(tenure_codes[i]), "renter")
        base = base_thresholds[tenure_str]
        equiv_scale = spm_equivalence_scale(int(num_adults[i]), int(num_children[i]))
        thresholds[i] = base * equiv_scale * geoadj[i]

    return thresholds
