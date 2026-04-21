"""Refresh EITC state-level and AGI-cross-tab calibration targets.

Two output CSVs are produced:

* ``eitc_state.csv`` - per-state EITC returns and dollar amounts, sourced from
  IRS SOI Historical Table 2 (the ``{YY}in55cmcsv.csv`` consolidated file).
* ``eitc_by_agi_and_children.csv`` - per-(qualifying-children x AGI bucket)
  EITC returns and amounts, sourced from IRS SOI Publication 1304 Table 2.5
  (``{YY}in25ic.xls``).

Both sources publish money amounts in thousands of dollars; this script
converts those to whole dollars before writing the CSVs. The scripts mirrors
the spirit of ``refresh_soi_table_targets.py`` - keep logic purely in
pandas/stdlib so future-year refreshes stay a one-command operation.

Usage::

    uv run python -m policyengine_us_data.storage.calibration_targets.refresh_eitc_state_and_agi_targets \
        --year 2022
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from policyengine_us_data.storage.calibration_targets.pull_soi_targets import (
    STATE_ABBR_TO_FIPS,
)


IRS_SOI_ROOT = "https://www.irs.gov/pub/irs-soi"

STATE_CSV_PATH = Path(__file__).with_name("eitc_state.csv")
AGI_CSV_PATH = Path(__file__).with_name("eitc_by_agi_and_children.csv")


# Columns in ``{YY}in25ic.xls`` that hold "Total earned income credit".
# Mapping: qualifying-children count -> (count_column, amount_column).
# Verified against the 2022 and 2023 Tax Year workbooks on 2026-04-19.
# Row 8 holds totals, rows 9..36 hold the AGI size-of-income rows, and rows
# 37+ are footnotes.
TABLE_2_5_CHILD_COLUMNS: dict[int, tuple[int, int]] = {
    0: (25, 26),
    1: (41, 42),
    2: (57, 58),
    3: (73, 74),
}

TABLE_2_5_HEADER_ROWS = 8  # skip header; totals land on row 8 (0-indexed)
TABLE_2_5_FIRST_BIN_ROW = 9
TABLE_2_5_LAST_BIN_ROW = 36  # inclusive

# The AGI "Size of adjusted gross income" column labels in ``{YY}in25ic.xls``
# use the SOI small-bin structure (one $1k bin for each of $1-$20k, then
# $5k bins up to $50k, then $50k+). ``_parse_agi_label`` converts these
# strings into half-open [lower, upper) numeric bounds.
AGI_LABEL_NEGATIVE = "No adjusted gross income"
AGI_LABEL_OPEN_TOP = "$50,000 and over"


def _soi_year_prefix(year: int) -> str:
    return f"{year % 100:02d}"


def _state_file_url(year: int) -> str:
    return f"{IRS_SOI_ROOT}/{_soi_year_prefix(year)}in55cmcsv.csv"


def _table_2_5_url(year: int) -> str:
    return f"{IRS_SOI_ROOT}/{_soi_year_prefix(year)}in25ic.xls"


def extract_state_eitc(year: int) -> pd.DataFrame:
    """Build the per-state EITC CSV rows.

    Columns follow the ``snap_state.csv`` convention: ``GEO_ID, Returns,
    Amount``. ``Returns`` is the raw SOI count; ``Amount`` is whole dollars
    (the SOI file publishes thousands of dollars).
    """

    df = pd.read_csv(_state_file_url(year), thousands=",")
    # ``AGI_STUB == 0`` rows are the all-AGI totals per state. ``STATE == 'US'``
    # is the national aggregate; filter to the 51 voting jurisdictions (50
    # states + DC) that have FIPS codes in STATE_ABBR_TO_FIPS.
    mask = (df["AGI_STUB"] == 0) & (df["STATE"].isin(STATE_ABBR_TO_FIPS))
    state_rows = df.loc[mask].copy()
    state_rows["FIPS"] = state_rows["STATE"].map(STATE_ABBR_TO_FIPS)
    state_rows["GEO_ID"] = "0400000US" + state_rows["FIPS"]
    state_rows["Returns"] = state_rows["N59660"].astype(int)
    # IRS SOI Historical Table 2 publishes amounts in thousands of dollars.
    state_rows["Amount"] = (state_rows["A59660"].astype("int64") * 1_000).astype(
        "int64"
    )

    out = (
        state_rows[["GEO_ID", "Returns", "Amount"]]
        .sort_values("GEO_ID")
        .reset_index(drop=True)
    )

    national_n = int(df.loc[df["STATE"] == "US", "N59660"].iloc[0])
    national_a = int(df.loc[df["STATE"] == "US", "A59660"].iloc[0]) * 1_000

    # Small-state SOI disclosure rounding produces aggregate differences of
    # well under 1% from the national row. Anything larger means the source
    # file changed shape; fail loudly so a refresh run can be audited.
    rel_returns = abs(out["Returns"].sum() - national_n) / max(national_n, 1)
    rel_amount = abs(out["Amount"].sum() - national_a) / max(national_a, 1)
    if rel_returns > 0.01 or rel_amount > 0.01:
        raise ValueError(
            "State sum diverges from published US total by more than 1%%: "
            f"returns diff={rel_returns:.4%}, amount diff={rel_amount:.4%}"
        )

    return out


def _parse_agi_label(label: str) -> tuple[float, float]:
    """Convert an SOI "Size of adjusted gross income" label to bounds.

    Returns a half-open [lower, upper) pair in whole dollars using
    ``float('inf')`` / ``float('-inf')`` as open endpoints.
    """

    if not isinstance(label, str):
        raise ValueError(f"Unexpected non-string AGI label: {label!r}")

    text = label.strip()
    if text == AGI_LABEL_NEGATIVE:
        return float("-inf"), 1.0
    if text == AGI_LABEL_OPEN_TOP:
        return 50_000.0, float("inf")

    # Patterns like "$1 under $1,000" or "$20,000 under $25,000".
    parts = text.replace("$", "").replace(",", "").split(" under ")
    if len(parts) != 2:
        raise ValueError(f"Could not parse SOI AGI label: {label!r}")
    return float(parts[0]), float(parts[1])


def extract_eitc_by_agi_and_children(year: int) -> pd.DataFrame:
    """Build the per-(child-count x AGI) EITC CSV rows.

    Columns: ``count_children, agi_lower, agi_upper, returns, amount``.
    ``count_children == 3`` means "three or more" (the SOI bucket).
    ``amount`` is whole dollars.
    """

    workbook = pd.read_excel(_table_2_5_url(year), header=None)

    agi_labels = workbook.iloc[
        TABLE_2_5_FIRST_BIN_ROW : TABLE_2_5_LAST_BIN_ROW + 1, 0
    ].tolist()

    rows: list[dict] = []
    for count_children, (n_col, a_col) in TABLE_2_5_CHILD_COLUMNS.items():
        counts = pd.to_numeric(
            workbook.iloc[TABLE_2_5_FIRST_BIN_ROW : TABLE_2_5_LAST_BIN_ROW + 1, n_col],
            errors="coerce",
        ).fillna(0)
        amounts_thousands = pd.to_numeric(
            workbook.iloc[TABLE_2_5_FIRST_BIN_ROW : TABLE_2_5_LAST_BIN_ROW + 1, a_col],
            errors="coerce",
        ).fillna(0)

        for label, returns, amount_k in zip(agi_labels, counts, amounts_thousands):
            lower, upper = _parse_agi_label(label)
            rows.append(
                {
                    "count_children": int(count_children),
                    "agi_lower": lower,
                    "agi_upper": upper,
                    "returns": int(round(returns)),
                    "amount": int(round(amount_k * 1_000)),
                }
            )

    df = pd.DataFrame(rows)

    # Sanity-check: per-child-count totals should match the row-8 totals.
    for count_children, (n_col, a_col) in TABLE_2_5_CHILD_COLUMNS.items():
        expected_n = int(workbook.iat[TABLE_2_5_HEADER_ROWS, n_col])
        expected_a = int(workbook.iat[TABLE_2_5_HEADER_ROWS, a_col]) * 1_000
        got = df[df["count_children"] == count_children]
        got_n = int(got["returns"].sum())
        got_a = int(got["amount"].sum())
        # Allow 0.5% slack for rounding in disclosure-rounded bins.
        if abs(got_n - expected_n) > max(100, 0.005 * expected_n):
            raise ValueError(
                f"Child-count {count_children} returns sum {got_n} "
                f"differs from published total {expected_n}"
            )
        if abs(got_a - expected_a) > max(1_000_000, 0.005 * expected_a):
            raise ValueError(
                f"Child-count {count_children} amount sum {got_a} "
                f"differs from published total {expected_a}"
            )

    return df


def _write_state_csv(state_df: pd.DataFrame, year: int) -> None:
    header_comment = (
        f"# IRS SOI Historical Table 2 ({year}in55cmcsv.csv), "
        f"EIC columns N59660 (returns) and A59660 (amount, thousands USD). "
        f"Pulled from {_state_file_url(year)}. Amount converted to dollars.\n"
    )
    with STATE_CSV_PATH.open("w", newline="") as fh:
        fh.write(header_comment)
        state_df.to_csv(fh, index=False)


def _write_agi_csv(agi_df: pd.DataFrame, year: int) -> None:
    header_comment = (
        f"# IRS SOI Publication 1304 Table 2.5, Tax Year {year} "
        f"({year}in25ic.xls). 'Total earned income credit' columns by "
        f"qualifying-children bucket. count_children=3 means 'three or "
        f"more'. Amount converted from thousands of dollars to dollars. "
        f"Pulled from {_table_2_5_url(year)}.\n"
    )

    def _format_bound(value: float) -> str:
        if value == float("-inf"):
            return "-inf"
        if value == float("inf"):
            return "inf"
        return str(int(value))

    agi_df = agi_df.copy()
    agi_df["agi_lower"] = agi_df["agi_lower"].map(_format_bound)
    agi_df["agi_upper"] = agi_df["agi_upper"].map(_format_bound)

    with AGI_CSV_PATH.open("w", newline="") as fh:
        fh.write(header_comment)
        agi_df.to_csv(fh, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh policyengine-us-data EITC state and AGI-cross-tab "
            "calibration targets from IRS SOI workbooks."
        )
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help=(
            "IRS tax year to pull. 2022 is the most recent year with both "
            "Historical Table 2 and Publication 1304 Table 2.5 published."
        ),
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Only refresh eitc_state.csv (skip Publication 1304 pull).",
    )
    parser.add_argument(
        "--agi-only",
        action="store_true",
        help="Only refresh eitc_by_agi_and_children.csv (skip Table 2 pull).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.agi_only:
        state_df = extract_state_eitc(args.year)
        _write_state_csv(state_df, args.year)
        print(
            f"Wrote {len(state_df)} state rows to {STATE_CSV_PATH} "
            f"(sum returns={state_df['Returns'].sum():,}, "
            f"sum amount=${state_df['Amount'].sum():,})"
        )

    if not args.state_only:
        agi_df = extract_eitc_by_agi_and_children(args.year)
        _write_agi_csv(agi_df, args.year)
        print(
            f"Wrote {len(agi_df)} (child x AGI) rows to {AGI_CSV_PATH} "
            f"(sum amount=${agi_df['amount'].sum():,})"
        )


if __name__ == "__main__":
    main()
