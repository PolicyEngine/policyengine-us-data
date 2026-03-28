from __future__ import annotations

"""Refresh tracked SOI table targets from IRS Publication 1304 workbooks.

This script updates the workbook-backed national SOI targets stored in
``soi_targets.csv``. It does not touch the separate state/district AGI
pulls, which depend on the ``in54``, ``in55cm``, and ``incd`` IRS files.
"""

import argparse
import csv
import math
from functools import lru_cache
from io import StringIO
from pathlib import Path

import pandas as pd


IRS_SOI_ROOT = "https://www.irs.gov/pub/irs-soi"
TARGETS_PATH = Path(__file__).with_name("soi_targets.csv")

TABLE_FILE_SUFFIX = {
    "Table 1.1": "in11si.xls",
    "Table 1.2": "in12ms.xls",
    "Table 1.4": "in14ar.xls",
    "Table 2.1": "in21id.xls",
    "Table 4.3": "in43ts.xls",
}

TABLE_1_4_COLUMNS = {
    "alternative_minimum_tax": {True: ("EN",), False: ("EO",)},
    "business_net_losses": {True: ("AH",), False: ("AI",)},
    "business_net_profits": {True: ("AF",), False: ("AG",)},
    "capital_gains_distributions": {True: ("AJ",), False: ("AK",)},
    "capital_gains_gross": {True: ("AL",), False: ("AM",)},
    "capital_gains_losses": {True: ("AN",), False: ("AO",)},
    "employment_income": {True: ("F",), False: ("G",)},
    "estate_income": {True: ("BX",), False: ("BY",)},
    "estate_losses": {True: ("BZ",), False: ("CA",)},
    "exempt_interest": {True: ("V",), False: ("W",)},
    "income_tax_before_credits": {True: ("ER",), False: ("ES",)},
    "ira_distributions": {True: ("AT",), False: ("AU",)},
    "ordinary_dividends": {True: ("X",), False: ("Y",)},
    "partnership_and_s_corp_income": {
        True: ("BP", "BT"),
        False: ("BQ", "BU"),
    },
    "partnership_and_s_corp_losses": {
        True: ("BR", "BV"),
        False: ("BS", "BW"),
    },
    "qualified_business_income_deduction": {
        True: ("EH",),
        False: ("EI",),
    },
    "qualified_dividends": {True: ("Z",), False: ("AA",)},
    "rent_and_royalty_net_income": {True: ("BL",), False: ("BM",)},
    "rent_and_royalty_net_losses": {True: ("BN",), False: ("BO",)},
    "s_corporation_net_income": {True: ("BT",), False: ("BU",)},
    "s_corporation_net_losses": {True: ("BV",), False: ("BW",)},
    "taxable_interest_income": {True: ("T",), False: ("U",)},
    "taxable_pension_income": {True: ("AX",), False: ("AY",)},
    "taxable_social_security": {True: ("CJ",), False: ("CK",)},
    "total_pension_income": {True: ("AV",), False: ("AW",)},
    "total_social_security": {True: ("CH",), False: ("CI",)},
    "unemployment_compensation": {True: ("CF",), False: ("CG",)},
}

TABLE_2_1_COLUMNS = {
    "charitable_contributions_deductions": {True: ("DG",), False: ("DH",)},
    "idpitgst": {True: ("CE",), False: ("CF",)},
    "interest_paid_deductions": {True: ("CS",), False: ("CT",)},
    "itemized_deductions": {True: ("B",), False: ("BT",)},
    "itemized_general_sales_tax_deduction": {True: ("CI",), False: ("CJ",)},
    "itemized_real_estate_tax_deductions": {True: ("CK",), False: ("CL",)},
    "itemized_state_income_tax_deductions": {True: ("CG",), False: ("CH",)},
    "itemized_taxes_paid_deductions": {True: ("CA",), False: ("CB",)},
    "medical_expense_deductions_capped": {True: ("BU",), False: ("BV",)},
    "medical_expense_deductions_uncapped": {True: ("BW",), False: ("BX",)},
    "mortgage_interest_deductions": {True: ("CU",), False: ("CV",)},
    "state_and_local_tax_deductions": {True: ("CO",), False: ("CP",)},
}

TABLE_2_1_ROW_BY_BOUNDS = {
    (float("-inf"), float("inf")): 10,
    (0.0, 5_000.0): 11,
    (5_000.0, 10_000.0): 12,
    (10_000.0, 15_000.0): 13,
    (15_000.0, 20_000.0): 14,
    (20_000.0, 25_000.0): 15,
    (25_000.0, 30_000.0): 16,
    (30_000.0, 35_000.0): 17,
    (35_000.0, 40_000.0): 18,
    (40_000.0, 45_000.0): 19,
    (45_000.0, 50_000.0): 20,
    (50_000.0, 55_000.0): 21,
    (55_000.0, 60_000.0): 22,
    (60_000.0, 75_000.0): 23,
    (75_000.0, 100_000.0): 24,
    (100_000.0, 200_000.0): 25,
    (200_000.0, 500_000.0): 26,
    (500_000.0, 1_000_000.0): 27,
    (1_000_000.0, 1_500_000.0): 28,
    (1_500_000.0, 2_000_000.0): 29,
    (2_000_000.0, 5_000_000.0): 30,
    (5_000_000.0, 10_000_000.0): 31,
    (10_000_000.0, float("inf")): 32,
}
TABLE_2_1_TAXABLE_TOTAL_ROW = 33

TOP_TAIL_FLOOR_COLUMN = 2
TOP_TAIL_FIRST_ROW = 10


def _column_index(column: str) -> int:
    column = str(column)
    if column.isdigit():
        return int(column)

    result = 0
    for char in column.upper():
        result = result * 26 + (ord(char) - 64)
    return result - 1


def _numeric_cell(workbook: pd.DataFrame, excel_row: int, column: str | int) -> float:
    value = workbook.iat[excel_row - 1, _column_index(column)]
    if isinstance(value, str):
        value = value.split("(")[0].replace(",", "").strip()
    return float(value)


def _scaled_cell(
    workbook: pd.DataFrame,
    excel_row: int,
    column: str | int,
    is_count: bool,
) -> float:
    value = _numeric_cell(workbook, excel_row, column)
    return value if is_count else value * 1_000


def _format_year_prefix(year: int) -> str:
    return f"{year % 100:02d}"


@lru_cache(maxsize=None)
def _load_workbook(table_name: str, year: int) -> pd.DataFrame:
    suffix = TABLE_FILE_SUFFIX[table_name]
    year_prefix = _format_year_prefix(year)
    return pd.read_excel(f"{IRS_SOI_ROOT}/{year_prefix}{suffix}", header=None)


def _table_2_1_excel_row(row: pd.Series) -> int | None:
    lower = float(row["AGI lower bound"])
    upper = float(row["AGI upper bound"])
    if not bool(row["Taxable only"]):
        return TABLE_2_1_ROW_BY_BOUNDS.get((lower, upper))

    if math.isinf(lower) and lower < 0 and math.isinf(upper) and upper > 0:
        return TABLE_2_1_TAXABLE_TOTAL_ROW

    return None


def _semantic_columns(row: pd.Series) -> tuple[str, ...] | None:
    table_name = row["SOI table"]
    variable = row["Variable"]
    is_count = bool(row["Count"])
    if table_name == "Table 1.4":
        table_map = TABLE_1_4_COLUMNS
    elif table_name == "Table 2.1":
        table_map = TABLE_2_1_COLUMNS
    else:
        return None

    column_map = table_map.get(variable)
    if column_map is None:
        return None
    return column_map[is_count]


def _refresh_excel_row(row: pd.Series) -> int | None:
    if row["SOI table"] == "Table 2.1":
        return _table_2_1_excel_row(row)
    return int(row["XLSX row"])


def _sum_scaled_cells(
    workbook: pd.DataFrame,
    excel_row: int,
    columns: tuple[str, ...],
    is_count: bool,
) -> float:
    return sum(
        _scaled_cell(workbook, excel_row, column, is_count) for column in columns
    )


def _table_4_3_value(row: pd.Series, workbook: pd.DataFrame) -> float:
    excel_row = int(row["XLSX row"])
    column = row["XLSX column"]
    is_count = bool(row["Count"])

    current_value = _scaled_cell(workbook, excel_row, column, is_count)
    if excel_row == TOP_TAIL_FIRST_ROW:
        return current_value

    previous_value = _scaled_cell(workbook, excel_row - 1, column, is_count)
    return current_value - previous_value


def _table_4_3_bounds(excel_row: int, workbook: pd.DataFrame) -> tuple[float, float]:
    lower = _numeric_cell(workbook, excel_row, TOP_TAIL_FLOOR_COLUMN)
    if excel_row == TOP_TAIL_FIRST_ROW:
        return lower, float("inf")

    upper = _numeric_cell(workbook, excel_row - 1, TOP_TAIL_FLOOR_COLUMN)
    return lower, upper


def _compute_value(row: pd.Series, workbook: pd.DataFrame) -> float:
    table_name = row["SOI table"]
    semantic_columns = _semantic_columns(row)
    excel_row = _refresh_excel_row(row)
    if semantic_columns is not None and excel_row is not None:
        return _sum_scaled_cells(
            workbook,
            excel_row,
            semantic_columns,
            bool(row["Count"]),
        )
    if table_name == "Table 4.3":
        return _table_4_3_value(row, workbook)

    if excel_row is None:
        raise ValueError(
            f"Unsupported SOI refresh row for {row['SOI table']} / {row['Variable']}"
        )

    return _scaled_cell(
        workbook,
        excel_row,
        row["XLSX column"],
        bool(row["Count"]),
    )


def build_target_year_rows(
    all_targets: pd.DataFrame, source_year: int, target_year: int
) -> pd.DataFrame:
    template_rows = all_targets[all_targets["Year"] == source_year].copy()

    refreshed_rows = []
    skipped_rows = []
    for _, row in template_rows.iterrows():
        refreshed = row.copy()
        refreshed["Year"] = target_year

        semantic_columns = _semantic_columns(refreshed)
        if refreshed["SOI table"] in {"Table 1.4", "Table 2.1"} and semantic_columns is None:
            skipped_rows.append((refreshed["SOI table"], refreshed["Variable"]))
            continue

        excel_row = _refresh_excel_row(refreshed)
        if excel_row is None:
            skipped_rows.append((refreshed["SOI table"], refreshed["Variable"]))
            continue
        refreshed["XLSX row"] = excel_row

        if semantic_columns is not None:
            refreshed["XLSX column"] = semantic_columns[-1]

        workbook = _load_workbook(refreshed["SOI table"], target_year)
        refreshed["Value"] = _compute_value(refreshed, workbook)

        if refreshed["SOI table"] == "Table 4.3":
            lower, upper = _table_4_3_bounds(int(refreshed["XLSX row"]), workbook)
            refreshed["AGI lower bound"] = lower
            refreshed["AGI upper bound"] = upper

        refreshed_rows.append(refreshed)

    if skipped_rows:
        skipped = ", ".join(sorted({f"{table}/{variable}" for table, variable in skipped_rows}))
        print(f"Skipped unsupported SOI rows for {target_year}: {skipped}")

    return pd.DataFrame(refreshed_rows, columns=all_targets.columns)


def _validate_source_year(all_targets: pd.DataFrame, source_year: int) -> None:
    expected = all_targets[all_targets["Year"] == source_year].reset_index(drop=True)
    actual = build_target_year_rows(all_targets, source_year, source_year).reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(
        expected, actual, check_dtype=False, check_exact=False
    )


def _serialize_bound(value: float) -> str:
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value.is_integer():
        return f"{value:.1f}"
    return repr(value)


def _serialize_row(row: pd.Series) -> str:
    formatted = [
        str(int(row["Year"])),
        str(row["SOI table"]),
        str(row["XLSX column"]),
        str(int(row["XLSX row"])),
        str(row["Variable"]),
        str(row["Filing status"]),
        _serialize_bound(row["AGI lower bound"]),
        _serialize_bound(row["AGI upper bound"]),
        "True" if bool(row["Count"]) else "False",
        "True" if bool(row["Taxable only"]) else "False",
        "True" if bool(row["Full population"]) else "False",
        str(int(round(float(row["Value"])))),
    ]

    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="")
    writer.writerow(formatted)
    return buffer.getvalue()


def write_target_year_rows(
    file_path: Path, target_year: int, refreshed_rows: pd.DataFrame
) -> None:
    existing_lines = file_path.read_text().splitlines()
    header, *body = existing_lines
    retained_lines = [
        line for line in body if not line.startswith(f"{int(target_year)},")
    ]
    appended_lines = [_serialize_row(row) for _, row in refreshed_rows.iterrows()]

    updated_lines = [header, *retained_lines, *appended_lines]
    file_path.write_text("\n".join(updated_lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh policyengine-us-data SOI table targets from IRS workbooks."
    )
    parser.add_argument(
        "--source-year",
        type=int,
        default=2021,
        help="Template year already present in soi_targets.csv.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        required=True,
        help="IRS tax year to append or replace in soi_targets.csv.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=TARGETS_PATH,
        help="Path to soi_targets.csv.",
    )
    parser.add_argument(
        "--validate-source-year",
        action="store_true",
        help="Regenerate the template year and assert it matches the current CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_targets = pd.read_csv(args.file)

    if args.validate_source_year:
        _validate_source_year(all_targets, args.source_year)

    refreshed_rows = build_target_year_rows(
        all_targets,
        source_year=args.source_year,
        target_year=args.target_year,
    )

    write_target_year_rows(args.file, args.target_year, refreshed_rows)

    print(
        f"Refreshed {len(refreshed_rows)} SOI rows for {args.target_year} in {args.file}"
    )


if __name__ == "__main__":
    main()
