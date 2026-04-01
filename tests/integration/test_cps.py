"""Integration tests for CPS dataset (requires cps_2024.h5)."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cps_sim():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    return Microsimulation(dataset=CPS_2024)


# ── Sanity checks ─────────────────────────────────────────────


def test_cps_employment_income_positive(cps_sim):
    total = cps_sim.calculate("employment_income").sum()
    assert total > 5e12, f"CPS employment_income sum is {total:.2e}, expected > 5T."


def test_cps_household_count(cps_sim):
    total_hh = cps_sim.calculate("household_weight").values.sum()
    assert 100e6 < total_hh < 200e6, f"CPS total households = {total_hh:.2e}."


# ── Calibration checks ────────────────────────────────────────


def test_cps_has_auto_loan_interest():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2024)
    # Ensure we impute around $85 billion in overtime premium with 25% error bounds.
    AUTO_LOAN_INTEREST_TARGET = 85e9
    AUTO_LOAN_BALANCE_TARGET = 1550e9
    RELATIVE_TOLERANCE = 0.4

    assert (
        abs(sim.calculate("auto_loan_interest").sum() / AUTO_LOAN_INTEREST_TARGET - 1)
        < RELATIVE_TOLERANCE
    )
    assert (
        abs(sim.calculate("auto_loan_balance").sum() / AUTO_LOAN_BALANCE_TARGET - 1)
        < RELATIVE_TOLERANCE
    )


def test_cps_has_fsla_overtime_premium():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2024)
    # ORG-backed hourly-pay data materially increases modeled overtime premium.
    # Keep a broad sanity band around the new CPS aggregate level.
    OVERTIME_PREMIUM_TARGET = 130e9
    RELATIVE_TOLERANCE = 0.2
    assert (
        abs(sim.calculate("fsla_overtime_premium").sum() / OVERTIME_PREMIUM_TARGET - 1)
        < RELATIVE_TOLERANCE
    )


def test_cps_has_net_worth():
    from policyengine_us_data.datasets.cps import CPS_2022
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2022)
    # Ensure we impute around 160 trillion in net worth with 25% error bounds.
    # https://fred.stlouisfed.org/series/BOGZ1FL192090005Q
    NET_WORTH_TARGET = 160e12
    RELATIVE_TOLERANCE = 0.25
    np.random.seed(42)
    assert (
        abs(sim.calculate("net_worth").sum() / NET_WORTH_TARGET - 1)
        < RELATIVE_TOLERANCE
    )
