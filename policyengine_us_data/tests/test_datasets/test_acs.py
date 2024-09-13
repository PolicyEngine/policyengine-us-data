import pytest
from policyengine_us import Microsimulation


@pytest.mark.parametrize("year", [2022])
def test_acs_generates(year: int):
    from policyengine_us_data.datasets.acs.acs import ACS_2022

    dataset_by_year = {
        2022: ACS_2022,
    }

    dataset = dataset_by_year[year]()
    dataset.generate()  # This will generate the dataset


@pytest.mark.parametrize("year", [2022])
def test_acs_loads(year: int):
    from policyengine_us_data.datasets.acs.acs import ACS_2022

    dataset_by_year = {
        2022: ACS_2022,
    }

    dataset = dataset_by_year[year]()
    dataset.generate()  # Ensure the dataset is generated before loading

    sim = Microsimulation(dataset=dataset)

    assert not sim.calculate("household_net_income").isna().any()


@pytest.mark.parametrize("year", [2022])
def test_acs_has_all_tables(year: int):
    from policyengine_us_data.datasets.acs.acs import ACS_2022

    dataset_by_year = {
        2022: ACS_2022,
    }

    dataset = dataset_by_year[year]()
    dataset.generate()  # Ensure the dataset is generated before checking tables

    TABLES = [
        "person",
        "household",
        "spm_unit",
    ]

    for table in TABLES:
        df = dataset.load(table)
        assert len(df) > 0
