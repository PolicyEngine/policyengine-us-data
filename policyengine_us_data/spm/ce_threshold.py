"""
Base SPM thresholds from Consumer Expenditure Survey.

Wraps spm-calculator for use in policyengine-us-data.
"""

from typing import Optional

# Published BLS thresholds by year and tenure type
# Source: https://www.bls.gov/pir/spm/spm_thresholds.htm
BLS_PUBLISHED_THRESHOLDS = {
    2024: {
        "renter": 39430,
        "owner_with_mortgage": 39068,
        "owner_without_mortgage": 32586,
    },
    2023: {
        "renter": 36606,
        "owner_with_mortgage": 36192,
        "owner_without_mortgage": 30347,
    },
    2022: {
        "renter": 33402,
        "owner_with_mortgage": 32949,
        "owner_without_mortgage": 27679,
    },
}


def calculate_base_thresholds(
    year: int = 2024,
    use_published: bool = True,
) -> dict[str, float]:
    """
    Get base SPM thresholds for the reference family (2A2C) by tenure type.

    Args:
        year: Target year for thresholds
        use_published: If True, use published BLS thresholds when available.
                      If False or year not available, forecast from latest.

    Returns:
        Dict with keys 'renter', 'owner_with_mortgage', 'owner_without_mortgage'
        and threshold values in dollars.
    """
    if use_published and year in BLS_PUBLISHED_THRESHOLDS:
        return BLS_PUBLISHED_THRESHOLDS[year].copy()

    # Forecast from latest available year
    latest_year = max(BLS_PUBLISHED_THRESHOLDS.keys())
    latest_thresholds = BLS_PUBLISHED_THRESHOLDS[latest_year]

    # Use approximately 3% annual inflation for forecasting
    # TODO: Use actual CPI-U or better methodology
    years_ahead = year - latest_year
    inflation_factor = 1.03**years_ahead

    return {k: v * inflation_factor for k, v in latest_thresholds.items()}
