import pytest


@pytest.mark.parametrize("year", [2024])
def test_policyengine_cps_generates(year: int):
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    dataset_by_year = {
        2024: EnhancedCPS_2024,
    }

    dataset_by_year[year](require=True)


@pytest.mark.parametrize("year", [2024])
def test_policyengine_cps_loads(year: int):
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    dataset_by_year = {
        2024: EnhancedCPS_2024,
    }

    dataset = dataset_by_year[year]

    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=dataset)

    assert not sim.calculate("household_net_income").isna().any()


def test_ecps_has_mortgage_interest():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    assert sim.calculate("deductible_mortgage_interest").sum() > 1
    assert sim.calculate("deductible_interest_expense").sum() > 1

def test_newborns_and_pregnancies():
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=EnhancedCPS_2024)

    # Test for unborn children (age < 0)
    unborn = sim.calculate("age") < 0
    unborn_count = unborn.sum()
    assert unborn_count > 0

    # Test for newborns (0 <= age < 1)
    newborns = (sim.calculate("age") >= 0) & (sim.calculate("age") < 1)
    newborn_count = newborns.sum()
    assert newborn_count > 0