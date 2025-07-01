import pytest
import numpy as np


def test_cps_has_auto_loan_interest():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2024)
    # Ensure we impute around $85 billion in overtime premium with 25% error bounds.
    AUTO_LOAN_INTEREST_TARGET = 85e9
    AUTO_LOAN_BALANCE_TARGET = 1550e9
    RELATIVE_TOLERANCE = 0.4

    assert (
        abs(
            sim.calculate("auto_loan_interest").sum()
            / AUTO_LOAN_INTEREST_TARGET
            - 1
        )
        < RELATIVE_TOLERANCE
    )
    assert (
        abs(
            sim.calculate("auto_loan_balance").sum() / AUTO_LOAN_BALANCE_TARGET
            - 1
        )
        < RELATIVE_TOLERANCE
    )


def test_cps_has_fsla_overtime_premium():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2024)
    # Ensure we impute around 70 billion in overtime premium with 20% error bounds.
    OVERTIME_PREMIUM_TARGET = 70e9
    RELATIVE_TOLERANCE = 0.2
    assert (
        abs(
            sim.calculate("fsla_overtime_premium").sum()
            / OVERTIME_PREMIUM_TARGET
            - 1
        )
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
