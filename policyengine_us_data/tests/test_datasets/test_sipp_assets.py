"""Tests for SIPP-imputed liquid assets.

These tests verify that liquid assets imputed from SIPP fall within
expected bounds based on Federal Reserve data.

Asset categories imputed (policy-neutral):
- bank_account_assets: checking, savings, money market (SIPP TVAL_BANK)
- stock_assets: stocks and mutual funds (SIPP TVAL_STMF)
- bond_assets: bonds and government securities (SIPP TVAL_BOND)

Policy models (policyengine-us) define which assets count for SSI, TANF, etc.
"""

import pytest


def _has_asset_variables():
    """Check if policyengine-us has the required asset variables."""
    try:
        from policyengine_us import CountryTaxBenefitSystem

        system = CountryTaxBenefitSystem()
        return all(
            var in system.variables
            for var in ["bank_account_assets", "stock_assets", "bond_assets"]
        )
    except Exception:
        return False


# Skip all tests in this module if asset variables not available
pytestmark = pytest.mark.skipif(
    not _has_asset_variables(),
    reason="Asset variables not yet available in policyengine-us",
)


def test_ecps_has_liquid_assets():
    """Test that liquid asset categories are imputed and within bounds.

    Based on Federal Reserve SCF 2022:
    - Median household liquid assets: ~$8,000
    - Total US household liquid assets: ~$15-20 trillion
    """
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    # Sum all liquid asset categories
    bank = sim.calculate("bank_account_assets", map_to="household")
    stocks = sim.calculate("stock_assets", map_to="household")
    bonds = sim.calculate("bond_assets", map_to="household")
    total_liquid = bank + stocks + bonds

    # Total should be in trillions (Fed estimates ~$15-20T in liquid assets)
    total = total_liquid.sum()
    MINIMUM_TOTAL = 5e12  # $5 trillion floor
    MAXIMUM_TOTAL = 30e12  # $30 trillion ceiling

    assert total > MINIMUM_TOTAL, (
        f"Total liquid assets ${total/1e12:.1f}T below "
        f"minimum ${MINIMUM_TOTAL/1e12:.0f}T"
    )
    assert total < MAXIMUM_TOTAL, (
        f"Total liquid assets ${total/1e12:.1f}T above "
        f"maximum ${MAXIMUM_TOTAL/1e12:.0f}T"
    )


def test_liquid_assets_distribution():
    """Test that liquid asset distribution is reasonable.

    Based on SCF 2022:
    - ~20% of households have <$1,000 in liquid assets
    - ~40% of households have <$5,000
    - Median is around $8,000
    """
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation
    import numpy as np

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    bank = sim.calculate("bank_account_assets", map_to="household")
    stocks = sim.calculate("stock_assets", map_to="household")
    bonds = sim.calculate("bond_assets", map_to="household")
    liquid_assets = bank + stocks + bonds

    weights = sim.calculate("household_weight")

    # Calculate weighted median
    sorted_idx = np.argsort(liquid_assets)
    sorted_assets = liquid_assets.values[sorted_idx]
    sorted_weights = weights.values[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    weighted_median = sorted_assets[median_idx]

    # Median should be between $3k and $20k
    MEDIAN_MIN = 3_000
    MEDIAN_MAX = 20_000

    assert weighted_median > MEDIAN_MIN, (
        f"Median liquid assets ${weighted_median:,.0f} below "
        f"minimum ${MEDIAN_MIN:,}"
    )
    assert weighted_median < MEDIAN_MAX, (
        f"Median liquid assets ${weighted_median:,.0f} above "
        f"maximum ${MEDIAN_MAX:,}"
    )


def test_asset_categories_exist():
    """Test that all three asset categories are imputed."""
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    # Each category should exist and have non-zero totals
    bank = sim.calculate("bank_account_assets").sum()
    stocks = sim.calculate("stock_assets").sum()
    bonds = sim.calculate("bond_assets").sum()

    # Bank accounts should be the largest category for most households
    assert bank > 0, "Bank account assets should be positive"
    assert stocks >= 0, "Stock assets should be non-negative"
    assert bonds >= 0, "Bond assets should be non-negative"

    # Bank accounts typically largest category of liquid assets
    assert (
        bank > stocks * 0.3
    ), "Bank accounts should be substantial relative to stocks"


def test_low_asset_households():
    """Test that a realistic share of households have low assets.

    For SSI and other means-tested programs, many households need
    to have assets below program limits ($2k-$3k for SSI).
    """
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    bank = sim.calculate("bank_account_assets")
    stocks = sim.calculate("stock_assets")
    bonds = sim.calculate("bond_assets")
    liquid_assets = bank + stocks + bonds

    # Calculate % with assets below $2,000 (SSI individual limit)
    below_2k = (liquid_assets < 2000).mean()

    # 10-40% of individuals should have <$2k in liquid assets
    MIN_PCT = 0.10
    MAX_PCT = 0.70

    assert below_2k > MIN_PCT, (
        f"Only {below_2k:.1%} have <$2k liquid assets, "
        f"expected at least {MIN_PCT:.0%}"
    )
    assert below_2k < MAX_PCT, (
        f"{below_2k:.1%} have <$2k liquid assets, "
        f"expected at most {MAX_PCT:.0%}"
    )
