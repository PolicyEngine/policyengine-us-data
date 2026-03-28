import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parent.parent
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
    assert refreshed["XLSX column"].tolist() == ["BT", "BU", "BV", "BW"]


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
