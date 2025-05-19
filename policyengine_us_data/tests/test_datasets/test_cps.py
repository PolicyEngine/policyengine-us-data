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
    # Ensure we impute at least $85 billion in auto loan interest.
    # We currently target $270 billion.
    AUTO_LOAN_INTEREST_MINIMUM = 85e9
    assert (
        sim.calculate("auto_loan_interest").sum() > AUTO_LOAN_INTEREST_MINIMUM
    )
