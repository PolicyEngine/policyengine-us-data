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

TABLE_1_4_AGGREGATES = {
    "partnership_and_s_corp_income": {
        True: ["BD", "BH"],
        False: ["BE", "BI"],
    },
    "partnership_and_s_corp_losses": {
        True: ["BF", "BJ"],
        False: ["BG", "BK"],
    },
}

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


def _table_1_4_value(row: pd.Series, workbook: pd.DataFrame) -> float:
    variable = row["Variable"]
    if variable in TABLE_1_4_AGGREGATES:
        columns = TABLE_1_4_AGGREGATES[variable][bool(row["Count"])]
        return sum(
            _scaled_cell(workbook, int(row["XLSX row"]), column, bool(row["Count"]))
            for column in columns
        )

    return _scaled_cell(
        workbook,
        int(row["XLSX row"]),
        row["XLSX column"],
        bool(row["Count"]),
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
    if table_name == "Table 1.4":
        return _table_1_4_value(row, workbook)
    if table_name == "Table 4.3":
        return _table_4_3_value(row, workbook)

    return _scaled_cell(
        workbook,
        int(row["XLSX row"]),
        row["XLSX column"],
        bool(row["Count"]),
    )


def build_target_year_rows(
    all_targets: pd.DataFrame, source_year: int, target_year: int
) -> pd.DataFrame:
    template_rows = all_targets[all_targets["Year"] == source_year].copy()

    refreshed_rows = []
    for _, row in template_rows.iterrows():
        refreshed = row.copy()
        refreshed["Year"] = target_year

        workbook = _load_workbook(refreshed["SOI table"], target_year)
        refreshed["Value"] = _compute_value(refreshed, workbook)

        if refreshed["SOI table"] == "Table 4.3":
            lower, upper = _table_4_3_bounds(int(refreshed["XLSX row"]), workbook)
            refreshed["AGI lower bound"] = lower
            refreshed["AGI upper bound"] = upper

        refreshed_rows.append(refreshed)

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
