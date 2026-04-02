import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "policyengine_us_data"
    / "storage"
    / "calibration_targets"
    / "refresh_soi_table_targets.py"
)

TARGET_COLUMNS = [
    "Year",
    "SOI table",
    "XLSX column",
    "XLSX row",
    "Variable",
    "Filing status",
    "AGI lower bound",
    "AGI upper bound",
    "Count",
    "Taxable only",
    "Full population",
    "Value",
]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "refresh_soi_table_targets", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_target_row(**kwargs):
    row = {
        "Year": 2021,
        "SOI table": "Table 1.1",
        "XLSX column": "D",
        "XLSX row": 10,
        "Variable": "adjusted_gross_income",
        "Filing status": "All",
        "AGI lower bound": float("-inf"),
        "AGI upper bound": float("inf"),
        "Count": False,
        "Taxable only": False,
        "Full population": True,
        "Value": 0.0,
    }
    row.update(kwargs)
    return row


def make_workbook(rows=20, cols=70):
    return pd.DataFrame(0.0, index=range(rows), columns=range(cols))


def test_build_target_year_rows_reads_standard_table_cells(monkeypatch):
    module = load_module()
    workbook = make_workbook()
    workbook.iat[9, module._column_index("D")] = 123.0
    workbook.iat[9, module._column_index("B")] = 789.0

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "SOI table": "Table 1.1",
                    "XLSX column": "D",
                    "XLSX row": 10,
                    "Variable": "adjusted_gross_income",
                    "Count": False,
                }
            ),
            make_target_row(
                **{
                    "SOI table": "Table 1.1",
                    "XLSX column": "B",
                    "XLSX row": 10,
                    "Variable": "count",
                    "Count": True,
                }
            ),
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook)

    refreshed = module.build_target_year_rows(
        targets, source_year=2021, target_year=2023
    )

    assert refreshed["Year"].tolist() == [2023, 2023]
    assert refreshed["Value"].tolist() == [123_000.0, 789.0]


def test_build_target_year_rows_uses_semantic_table_1_4_columns(monkeypatch):
    module = load_module()
    workbook = make_workbook(cols=80)
    row_index = 9  # Excel row 10

    workbook.iat[row_index, module._column_index("BP")] = 10
    workbook.iat[row_index, module._column_index("BQ")] = 30
    workbook.iat[row_index, module._column_index("BT")] = 20
    workbook.iat[row_index, module._column_index("BU")] = 40
    workbook.iat[row_index, module._column_index("BR")] = 5
    workbook.iat[row_index, module._column_index("BS")] = 7
    workbook.iat[row_index, module._column_index("BV")] = 6
    workbook.iat[row_index, module._column_index("BW")] = 8

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "SOI table": "Table 1.4",
                    "XLSX column": "BD",
                    "XLSX row": 10,
                    "Variable": "partnership_and_s_corp_income",
                    "Count": True,
                }
            ),
            make_target_row(
                **{
                    "SOI table": "Table 1.4",
                    "XLSX column": "BE",
                    "XLSX row": 10,
                    "Variable": "partnership_and_s_corp_income",
                    "Count": False,
                }
            ),
            make_target_row(
                **{
                    "SOI table": "Table 1.4",
                    "XLSX column": "BF",
                    "XLSX row": 10,
                    "Variable": "partnership_and_s_corp_losses",
                    "Count": True,
                }
            ),
            make_target_row(
                **{
                    "SOI table": "Table 1.4",
                    "XLSX column": "BG",
                    "XLSX row": 10,
                    "Variable": "partnership_and_s_corp_losses",
                    "Count": False,
                }
            ),
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook)

    refreshed = module.build_target_year_rows(
        targets, source_year=2021, target_year=2023
    )

    assert refreshed["Value"].tolist() == [30.0, 70_000.0, 11.0, 15_000.0]
    assert refreshed["XLSX column"].tolist() == [
        "BP+BT",
        "BQ+BU",
        "BR+BV",
        "BS+BW",
    ]


def test_build_target_year_rows_maps_table_2_1_rows_and_columns(monkeypatch):
    module = load_module()
    workbook = make_workbook(rows=40, cols=110)
    row_index = 32  # Excel row 33

    workbook.iat[row_index, module._column_index("CU")] = 321
    workbook.iat[row_index, module._column_index("CV")] = 654

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "SOI table": "Table 2.1",
                    "XLSX column": "CH",
                    "XLSX row": 29,
                    "Variable": "mortgage_interest_deductions",
                    "Count": True,
                    "Taxable only": True,
                    "Full population": False,
                }
            ),
            make_target_row(
                **{
                    "SOI table": "Table 2.1",
                    "XLSX column": "CJ",
                    "XLSX row": 29,
                    "Variable": "mortgage_interest_deductions",
                    "Count": False,
                    "Taxable only": True,
                    "Full population": False,
                }
            ),
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook)

    refreshed = module.build_target_year_rows(
        targets, source_year=2021, target_year=2023
    )

    assert refreshed["XLSX row"].tolist() == [33, 33]
    assert refreshed["XLSX column"].tolist() == ["CU", "CV"]
    assert refreshed["Value"].tolist() == [321.0, 654_000.0]


def test_build_target_year_rows_preserves_source_year_layout_for_validation(
    monkeypatch,
):
    module = load_module()
    workbook_2021 = make_workbook(rows=40, cols=110)
    workbook_2021.iat[28, module._column_index("CI")] = 123
    workbook_2021.iat[28, module._column_index("CJ")] = 456
    workbook_2021.iat[32, module._column_index("CU")] = 999
    workbook_2021.iat[32, module._column_index("CV")] = 999

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 2.1",
                    "XLSX column": "CI",
                    "XLSX row": 29,
                    "Variable": "mortgage_interest_deductions",
                    "Count": True,
                    "Taxable only": True,
                    "Full population": False,
                }
            ),
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 2.1",
                    "XLSX column": "CJ",
                    "XLSX row": 29,
                    "Variable": "mortgage_interest_deductions",
                    "Count": False,
                    "Taxable only": True,
                    "Full population": False,
                }
            ),
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook_2021)

    refreshed = module.build_target_year_rows(
        targets, source_year=2021, target_year=2021
    )

    assert refreshed["XLSX row"].tolist() == [29, 29]
    assert refreshed["XLSX column"].tolist() == ["CI", "CJ"]
    assert refreshed["Value"].tolist() == [123.0, 456_000.0]


def test_build_target_year_rows_uses_legacy_table_1_4_combined_columns_for_2021(
    monkeypatch,
):
    module = load_module()
    workbook_2021 = make_workbook(rows=40, cols=80)
    workbook_2021.iat[8, module._column_index("BD")] = 30
    workbook_2021.iat[8, module._column_index("BE")] = 400
    workbook_2021.iat[8, module._column_index("BH")] = 12
    workbook_2021.iat[8, module._column_index("BI")] = 600
    workbook_2021.iat[8, module._column_index("BF")] = 9
    workbook_2021.iat[8, module._column_index("BG")] = 50
    workbook_2021.iat[8, module._column_index("BJ")] = 11
    workbook_2021.iat[8, module._column_index("BK")] = 150

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BD",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_income",
                    "Count": True,
                }
            ),
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BE",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_income",
                    "Count": False,
                }
            ),
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BF",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_losses",
                    "Count": True,
                }
            ),
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BG",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_losses",
                    "Count": False,
                }
            ),
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook_2021)

    refreshed = module.build_target_year_rows(
        targets, source_year=2021, target_year=2021
    )

    assert refreshed["XLSX column"].tolist() == ["BD", "BE", "BF", "BG"]
    assert refreshed["Value"].tolist() == [42.0, 1_000_000.0, 20.0, 200_000.0]


def test_validate_source_year_round_trips_mixed_layouts(monkeypatch):
    module = load_module()
    workbook_2021 = make_workbook(rows=40, cols=110)
    workbook_2021.iat[28, module._column_index("CI")] = 123
    workbook_2021.iat[28, module._column_index("CJ")] = 456

    workbook_2022 = make_workbook(rows=40, cols=110)
    workbook_2022.iat[32, module._column_index("CU")] = 321
    workbook_2022.iat[32, module._column_index("CV")] = 654

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 2.1",
                    "XLSX column": "CI",
                    "XLSX row": 29,
                    "Variable": "mortgage_interest_deductions",
                    "Count": True,
                    "Taxable only": True,
                    "Full population": False,
                    "Value": 123.0,
                }
            ),
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 2.1",
                    "XLSX column": "CJ",
                    "XLSX row": 29,
                    "Variable": "mortgage_interest_deductions",
                    "Count": False,
                    "Taxable only": True,
                    "Full population": False,
                    "Value": 456_000.0,
                }
            ),
            make_target_row(
                **{
                    "Year": 2022,
                    "SOI table": "Table 2.1",
                    "XLSX column": "CU",
                    "XLSX row": 33,
                    "Variable": "mortgage_interest_deductions",
                    "Count": True,
                    "Taxable only": True,
                    "Full population": False,
                    "Value": 321.0,
                }
            ),
            make_target_row(
                **{
                    "Year": 2022,
                    "SOI table": "Table 2.1",
                    "XLSX column": "CV",
                    "XLSX row": 33,
                    "Variable": "mortgage_interest_deductions",
                    "Count": False,
                    "Taxable only": True,
                    "Full population": False,
                    "Value": 654_000.0,
                }
            ),
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(
        module,
        "_load_workbook",
        lambda table, year: workbook_2021 if year == 2021 else workbook_2022,
    )

    module._validate_source_year(targets, 2021)
    module._validate_source_year(targets, 2022)


def test_validate_source_year_round_trips_legacy_table_1_4_layouts(monkeypatch):
    module = load_module()
    workbook_2021 = make_workbook(rows=40, cols=80)
    workbook_2021.iat[8, module._column_index("BD")] = 30
    workbook_2021.iat[8, module._column_index("BE")] = 400
    workbook_2021.iat[8, module._column_index("BH")] = 12
    workbook_2021.iat[8, module._column_index("BI")] = 600
    workbook_2021.iat[8, module._column_index("BF")] = 9
    workbook_2021.iat[8, module._column_index("BG")] = 50
    workbook_2021.iat[8, module._column_index("BJ")] = 11
    workbook_2021.iat[8, module._column_index("BK")] = 150

    workbook_2022 = make_workbook(rows=40, cols=80)
    workbook_2022.iat[8, module._column_index("BP")] = 30
    workbook_2022.iat[8, module._column_index("BQ")] = 400
    workbook_2022.iat[8, module._column_index("BT")] = 12
    workbook_2022.iat[8, module._column_index("BU")] = 600

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BD",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_income",
                    "Count": True,
                    "Value": 42.0,
                }
            ),
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BE",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_income",
                    "Count": False,
                    "Value": 1_000_000.0,
                }
            ),
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BF",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_losses",
                    "Count": True,
                    "Value": 20.0,
                }
            ),
            make_target_row(
                **{
                    "Year": 2021,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BG",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_losses",
                    "Count": False,
                    "Value": 200_000.0,
                }
            ),
            make_target_row(
                **{
                    "Year": 2022,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BP+BT",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_income",
                    "Count": True,
                    "Value": 42.0,
                }
            ),
            make_target_row(
                **{
                    "Year": 2022,
                    "SOI table": "Table 1.4",
                    "XLSX column": "BQ+BU",
                    "XLSX row": 9,
                    "Variable": "partnership_and_s_corp_income",
                    "Count": False,
                    "Value": 1_000_000.0,
                }
            ),
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(
        module,
        "_load_workbook",
        lambda table, year: workbook_2021 if year == 2021 else workbook_2022,
    )

    module._validate_source_year(targets, 2021)
    module._validate_source_year(targets, 2022)


def test_validate_source_year_matches_serialized_rounding(monkeypatch):
    module = load_module()
    workbook = make_workbook(rows=20, cols=20)
    workbook.iat[10, module._column_index("N")] = 3843.13

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "Year": 2022,
                    "SOI table": "Table 1.1",
                    "XLSX column": "N",
                    "XLSX row": 11,
                    "Variable": "income_tax_after_credits",
                    "Count": True,
                    "Taxable only": True,
                    "Full population": False,
                    "Value": 3843.0,
                }
            )
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook)

    module._validate_source_year(targets, 2022)


def test_build_target_year_rows_rejects_unaudited_target_year(monkeypatch):
    module = load_module()
    workbook = make_workbook(rows=40, cols=110)
    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook)

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "Year": 2023,
                    "SOI table": "Table 2.1",
                    "XLSX column": "CU",
                    "XLSX row": 33,
                    "Variable": "mortgage_interest_deductions",
                    "Count": True,
                    "Taxable only": True,
                    "Full population": False,
                }
            )
        ],
        columns=TARGET_COLUMNS,
    )

    try:
        module.build_target_year_rows(targets, source_year=2023, target_year=2024)
    except ValueError as exc:
        assert "No audited workbook layout mapping" in str(exc)
    else:
        raise AssertionError("Expected an unaudited target year to fail")


def test_build_target_year_rows_skips_unsupported_rows(monkeypatch):
    module = load_module()
    workbook = make_workbook()

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "SOI table": "Table 1.4",
                    "XLSX column": "DX",
                    "XLSX row": 9,
                    "Variable": "count_of_exemptions",
                    "Count": True,
                }
            )
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook)

    refreshed = module.build_target_year_rows(
        targets, source_year=2021, target_year=2023
    )

    assert refreshed.empty


def test_build_target_year_rows_differences_top_tail_rows_and_updates_bounds(
    monkeypatch,
):
    module = load_module()
    workbook = make_workbook(cols=25)

    workbook.iat[9, module._column_index("B")] = 10
    workbook.iat[9, module._column_index("3")] = 200
    workbook.iat[9, module.TOP_TAIL_FLOOR_COLUMN] = 1_000

    workbook.iat[10, module._column_index("B")] = 60
    workbook.iat[10, module._column_index("3")] = 1_000
    workbook.iat[10, module.TOP_TAIL_FLOOR_COLUMN] = 400

    targets = pd.DataFrame(
        [
            make_target_row(
                **{
                    "SOI table": "Table 4.3",
                    "XLSX column": "B",
                    "XLSX row": 10,
                    "Variable": "count",
                    "Count": True,
                    "Taxable only": True,
                    "Full population": False,
                }
            ),
            make_target_row(
                **{
                    "SOI table": "Table 4.3",
                    "XLSX column": "B",
                    "XLSX row": 11,
                    "Variable": "count",
                    "Count": True,
                    "Taxable only": True,
                    "Full population": False,
                }
            ),
            make_target_row(
                **{
                    "SOI table": "Table 4.3",
                    "XLSX column": "3",
                    "XLSX row": 10,
                    "Variable": "adjusted_gross_income",
                    "Count": False,
                    "Taxable only": True,
                    "Full population": False,
                }
            ),
            make_target_row(
                **{
                    "SOI table": "Table 4.3",
                    "XLSX column": "3",
                    "XLSX row": 11,
                    "Variable": "adjusted_gross_income",
                    "Count": False,
                    "Taxable only": True,
                    "Full population": False,
                }
            ),
        ],
        columns=TARGET_COLUMNS,
    )

    monkeypatch.setattr(module, "_load_workbook", lambda table, year: workbook)

    refreshed = module.build_target_year_rows(
        targets, source_year=2021, target_year=2023
    )

    assert refreshed["Value"].tolist() == [10.0, 50.0, 200_000.0, 800_000.0]
    assert refreshed["AGI lower bound"].tolist() == [1_000.0, 400.0, 1_000.0, 400.0]
    assert refreshed["AGI upper bound"].tolist() == [
        float("inf"),
        1_000.0,
        float("inf"),
        1_000.0,
    ]
