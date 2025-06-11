import pytest


@pytest.mark.parametrize("year", [2022])
def test_policyengine_cps_generates(year: int):
    from policyengine_us_data.datasets.cps.cps import CPS_2022

    dataset_by_year = {
        2022: CPS_2022,
    }

    dataset_by_year[year](require=True)


@pytest.mark.parametrize("year", [2022])
def test_policyengine_cps_loads(year: int):
    from policyengine_us_data.datasets.cps.cps import CPS_2022

    dataset_by_year = {
        2022: CPS_2022,
    }

    dataset = dataset_by_year[year]

    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=dataset)

    assert not sim.calculate("household_net_income").isna().any()


def test_cps_has_auto_loan_interest():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2024)
    # Ensure we impute around $85 billion in overtime premium with 20% error bounds.
    AUTO_LOAN_INTEREST_TARGET = 85e9
    AUTO_LOAN_BALANCE_TARGET = 1550e9
    RELATIVE_TOLERANCE = 0.25
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
    # Ensure we impute around 200 trillion in net worth with 20% error bounds.
    # https://www.cbo.gov/publication/60807
    NET_WORTH_TARGET = 200e12
    RELATIVE_TOLERANCE = 0.25
    assert (
        abs(sim.calculate("net_worth").sum() / NET_WORTH_TARGET - 1)
        < RELATIVE_TOLERANCE
    )
