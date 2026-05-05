"""Validate built Census CPS artifacts used by the Stage 1 CPS build."""

import pandas as pd
import pytest

from policyengine_us_data.datasets.cps.census_cps import (
    CensusCPS_2023,
    CensusCPS_2024,
)


DATASETS_BY_YEAR = {
    2023: CensusCPS_2023,
    2024: CensusCPS_2024,
}
TABLES = ("person", "family", "tax_unit", "spm_unit", "household")


@pytest.fixture(scope="module", params=sorted(DATASETS_BY_YEAR))
def census_cps_store(request):
    year = request.param
    dataset_cls = DATASETS_BY_YEAR[year]
    path = dataset_cls.file_path
    if not path.exists():
        pytest.skip(f"{path.name} not built locally")

    with pd.HDFStore(path, mode="r") as store:
        yield year, store


def test_census_cps_has_stage_1_input_tables(census_cps_store):
    _year, store = census_cps_store
    keys = {key.removeprefix("/") for key in store.keys()}

    assert set(TABLES) <= keys


def test_census_cps_tables_are_populated(census_cps_store):
    _year, store = census_cps_store

    for table in TABLES:
        assert len(store[table]) > 0


def test_census_cps_person_table_has_current_schema(census_cps_store):
    year, store = census_cps_store
    person = store["person"]
    required_columns = {
        "CENSUS_TAX_ID",
        "TAX_ID",
        "PH_SEQ",
        "SPM_ID",
        "A_LINENO",
        "A_FNLWGT",
    }

    assert required_columns <= set(person.columns), (
        f"Census CPS {year} person table is missing columns: "
        f"{sorted(required_columns - set(person.columns))}"
    )
    assert person["CENSUS_TAX_ID"].notna().all()
    assert person["TAX_ID"].notna().all()


def test_census_cps_tax_unit_table_matches_constructed_tax_ids(census_cps_store):
    _year, store = census_cps_store
    person_tax_ids = set(store["person"]["TAX_ID"])
    tax_unit_ids = set(store["tax_unit"]["TAX_ID"])

    assert tax_unit_ids
    assert tax_unit_ids <= person_tax_ids
