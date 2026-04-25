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


def test_create_tax_unit_table_preserves_census_tax_id_and_replaces_tax_id():
    import pandas as pd

    from policyengine_us_data.datasets.cps.census_cps import CensusCPS_2024

    dataset = object.__new__(CensusCPS_2024)
    person = pd.DataFrame(
        {
            "PH_SEQ": [1, 1],
            "TAX_ID": [501, 999],
            "A_LINENO": [1, 2],
            "A_AGE": [40, 10],
            "A_MARITL": [7, 7],
            "A_SPOUSE": [0, 0],
            "A_EXPRRP": [1, 5],
            "PEPAR1": [-1, 1],
            "PEPAR2": [-1, -1],
            "PECOHAB": [-1, -1],
            "A_ENRLW": [0, 0],
            "A_FTPT": [0, 0],
            "WSAL_VAL": [50_000, 0],
            "SEMP_VAL": [0, 0],
            "FRSE_VAL": [0, 0],
            "INT_VAL": [0, 0],
            "DIV_VAL": [0, 0],
            "RNT_VAL": [0, 0],
            "CAP_VAL": [0, 0],
            "UC_VAL": [0, 0],
            "OI_VAL": [0, 0],
            "ANN_VAL": [0, 0],
            "PNSN_VAL": [0, 0],
            "PTOTVAL": [50_000, 0],
            "SS_VAL": [0, 0],
            "PEDISDRS": [0, 0],
            "PEDISEAR": [0, 0],
            "PEDISEYE": [0, 0],
            "PEDISOUT": [0, 0],
            "PEDISPHY": [0, 0],
            "PEDISREM": [0, 0],
        }
    )

    tax_unit = dataset._create_tax_unit_table(person)

    assert person["CENSUS_TAX_ID"].tolist() == [501, 999]
    assert person["TAX_ID"].tolist() == [1, 1]
    assert "tax_unit_role_input" not in person
    assert "is_related_to_head_or_spouse" not in person
    assert tax_unit["TAX_ID"].tolist() == [1]
    assert "filing_status_input" not in tax_unit


def test_create_tax_unit_table_accepts_census_documented_mode():
    import pandas as pd

    from policyengine_us_data.datasets.cps.census_cps import CensusCPS_2024

    dataset = object.__new__(CensusCPS_2024)
    person = pd.DataFrame(
        {
            "PH_SEQ": [1, 1],
            "TAX_ID": [501, 999],
            "A_LINENO": [1, 2],
            "A_AGE": [40, 12],
            "A_MARITL": [7, 7],
            "A_SPOUSE": [0, 0],
            "A_EXPRRP": [1, 14],
            "PEPAR1": [-1, -1],
            "PEPAR2": [-1, -1],
            "PECOHAB": [-1, -1],
            "A_ENRLW": [0, 0],
            "A_FTPT": [0, 0],
            "WSAL_VAL": [50_000, 0],
            "SEMP_VAL": [0, 0],
            "FRSE_VAL": [0, 0],
            "INT_VAL": [0, 0],
            "DIV_VAL": [0, 0],
            "RNT_VAL": [0, 0],
            "CAP_VAL": [0, 0],
            "UC_VAL": [0, 0],
            "OI_VAL": [0, 0],
            "ANN_VAL": [0, 0],
            "PNSN_VAL": [0, 0],
            "PTOTVAL": [50_000, 0],
            "SS_VAL": [0, 0],
            "PEDISDRS": [0, 0],
            "PEDISEAR": [0, 0],
            "PEDISEYE": [0, 0],
            "PEDISOUT": [0, 0],
            "PEDISPHY": [0, 0],
            "PEDISREM": [0, 0],
        }
    )

    tax_unit = dataset._create_tax_unit_table(person, mode="census_documented")

    assert person["CENSUS_TAX_ID"].tolist() == [501, 999]
    assert person["TAX_ID"].tolist() == [1, 1]
    assert "tax_unit_role_input" not in person
    assert "is_related_to_head_or_spouse" not in person
    assert "filing_status_input" not in tax_unit
