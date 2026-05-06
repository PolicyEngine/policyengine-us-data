"""Validate the built Stage 1 CPS artifact."""

import pytest


@pytest.fixture(scope="module")
def cps_sim():
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import CPS_2024

    if not CPS_2024.file_path.exists():
        pytest.skip("cps_2024.h5 not built locally")
    return Microsimulation(dataset=CPS_2024)


def test_cps_employment_income_positive(cps_sim):
    total = cps_sim.calculate("employment_income").sum()
    assert total > 5e12, f"CPS employment_income sum is {total:.2e}, expected > 5T."


def test_cps_household_count(cps_sim):
    total_hh = cps_sim.calculate("household_weight").values.sum()
    assert 100e6 < total_hh < 200e6, f"CPS total households = {total_hh:.2e}."


def test_cps_has_auto_loan_interest(cps_sim):
    auto_loan_interest_target = 85e9
    auto_loan_balance_target = 1550e9
    relative_tolerance = 0.4

    assert (
        abs(
            cps_sim.calculate("auto_loan_interest").sum() / auto_loan_interest_target
            - 1
        )
        < relative_tolerance
    )
    assert (
        abs(cps_sim.calculate("auto_loan_balance").sum() / auto_loan_balance_target - 1)
        < relative_tolerance
    )


def test_cps_has_fsla_overtime_premium(cps_sim):
    overtime_premium_target = 130e9
    relative_tolerance = 0.2

    assert (
        abs(
            cps_sim.calculate("fsla_overtime_premium").sum() / overtime_premium_target
            - 1
        )
        < relative_tolerance
    )
