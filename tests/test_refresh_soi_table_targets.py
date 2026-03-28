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

    refreshed = module.build_target_year_rows(targets, source_year=2021, target_year=2023)

    assert refreshed["Year"].tolist() == [2023, 2023]
    assert refreshed["Value"].tolist() == [123_000.0, 789.0]


def test_build_target_year_rows_sums_partnership_and_s_corp_components(monkeypatch):
    module = load_module()
    workbook = make_workbook(cols=80)
    row_index = 9  # Excel row 10

    workbook.iat[row_index, module._column_index("BD")] = 10
    workbook.iat[row_index, module._column_index("BE")] = 30
    workbook.iat[row_index, module._column_index("BF")] = 5
    workbook.iat[row_index, module._column_index("BG")] = 7
    workbook.iat[row_index, module._column_index("BH")] = 20
    workbook.iat[row_index, module._column_index("BI")] = 40
    workbook.iat[row_index, module._column_index("BJ")] = 6
    workbook.iat[row_index, module._column_index("BK")] = 8

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

    refreshed = module.build_target_year_rows(targets, source_year=2021, target_year=2023)

    assert refreshed["Value"].tolist() == [30.0, 70_000.0, 11.0, 15_000.0]


def test_build_target_year_rows_differences_top_tail_rows_and_updates_bounds(monkeypatch):
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

    refreshed = module.build_target_year_rows(targets, source_year=2021, target_year=2023)

    assert refreshed["Value"].tolist() == [10.0, 50.0, 200_000.0, 800_000.0]
    assert refreshed["AGI lower bound"].tolist() == [1_000.0, 400.0, 1_000.0, 400.0]
    assert refreshed["AGI upper bound"].tolist() == [
        float("inf"),
        1_000.0,
        float("inf"),
        1_000.0,
    ]
