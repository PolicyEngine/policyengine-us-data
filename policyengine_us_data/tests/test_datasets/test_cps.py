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
