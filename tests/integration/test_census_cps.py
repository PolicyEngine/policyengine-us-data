import pytest


@pytest.mark.parametrize("year", [2022, 2024])
def test_census_cps_generates(year: int):
    from policyengine_us_data.datasets.cps.census_cps import (
        CensusCPS_2022,
        CensusCPS_2024,
    )

    dataset_by_year = {
        2022: CensusCPS_2022,
        2024: CensusCPS_2024,
    }

    dataset_by_year[year](require=True)


@pytest.mark.parametrize("year", [2022, 2024])
def test_census_cps_has_all_tables(year: int):
    from policyengine_us_data.datasets.cps.census_cps import (
        CensusCPS_2022,
        CensusCPS_2024,
    )

    dataset_by_year = {
        2022: CensusCPS_2022,
        2024: CensusCPS_2024,
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


def test_resolve_person_usecols_allows_missing_optional_now_columns():
    from policyengine_us_data.datasets.cps.census_cps import (
        PERSON_COLUMNS,
        SPM_UNIT_COLUMNS,
        TAX_UNIT_COLUMNS,
        _resolve_person_usecols,
    )

    missing_optional = {"NOW_MRKS", "NOW_MRKUN"}
    available_columns = [
        column
        for column in PERSON_COLUMNS + SPM_UNIT_COLUMNS + TAX_UNIT_COLUMNS
        if column not in missing_optional
    ]

    usecols = _resolve_person_usecols(available_columns, SPM_UNIT_COLUMNS)

    assert "NOW_MRKS" not in usecols
    assert "NOW_MRKUN" not in usecols
    assert "PH_SEQ" in usecols
    assert "A_FNLWGT" in usecols


def test_fill_missing_optional_person_columns_backfills_zeroes():
    import pandas as pd

    from policyengine_us_data.datasets.cps.census_cps import (
        OPTIONAL_PERSON_COLUMNS,
        _fill_missing_optional_person_columns,
    )

    person = pd.DataFrame({"PH_SEQ": [1], "NOW_COV": [1]})

    filled = _fill_missing_optional_person_columns(person)

    for column in OPTIONAL_PERSON_COLUMNS:
        assert column in filled.columns
    assert filled.loc[0, "NOW_COV"] == 1
    assert filled.loc[0, "NOW_MRKS"] == 0
