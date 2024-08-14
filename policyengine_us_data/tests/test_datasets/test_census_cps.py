import pytest


@pytest.mark.parametrize("year", [2022])
def test_census_cps_generates(year: int):
    from policyengine_us_data.datasets.cps.census_cps import CensusCPS_2022

    dataset_by_year = {
        2022: CensusCPS_2022,
    }

    dataset_by_year[year](require=True)


@pytest.mark.parametrize("year", [2022])
def test_census_cps_has_all_tables(year: int):
    from policyengine_us_data.datasets.cps.census_cps import CensusCPS_2022

    dataset_by_year = {
        2022: CensusCPS_2022,
    }

    dataset = dataset_by_year[year](require=True)
    TABLES = [
        "person",
        "family",
        "tax_unit",
        "spm_unit",
    ]
    # Try loading each as a dataframe

    for table in TABLES:
        df = dataset.load(table)
        assert len(df) > 0
