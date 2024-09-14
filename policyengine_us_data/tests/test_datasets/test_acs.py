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
