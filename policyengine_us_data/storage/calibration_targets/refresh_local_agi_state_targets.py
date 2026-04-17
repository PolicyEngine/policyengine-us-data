"""Refresh tracked SOI targets used by legacy local calibration.

This regenerates ``agi_state.csv`` from the IRS geographic SOI state file while
preserving the legacy schema consumed by ``utils/loss.py``:

- ``GEO_NAME`` is the two-letter state abbreviation
- ``VARIABLE`` is ``adjusted_gross_income/count`` or ``.../amount``
- AGI bounds live in ``AGI_LOWER_BOUND`` / ``AGI_UPPER_BOUND``

This file intentionally remains separate from the national workbook-backed
``soi_targets.csv`` refresh path because IRS geographic releases lag the
national Publication 1304 tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CALIBRATION_FOLDER = Path(__file__).resolve().parent
TARGETS_PATH = CALIBRATION_FOLDER / "agi_state.csv"
STATE_SOI_TAX_YEAR = 2022
LOCAL_STATE_SOI_TAX_YEAR = STATE_SOI_TAX_YEAR

AGI_STUB_TO_BAND = {
    1: "Under $1",
    2: "$1 under $10,000",
    3: "$10,000 under $25,000",
    4: "$25,000 under $50,000",
    5: "$50,000 under $75,000",
    6: "$75,000 under $100,000",
    7: "$100,000 under $200,000",
    8: "$200,000 under $500,000",
    9: "$500,000 under $1,000,000",
    10: "$1,000,000 or more",
}

AGI_BOUNDS = {
    "Under $1": (-np.inf, 1),
    "$1 under $10,000": (1, 10_000),
    "$10,000 under $25,000": (10_000, 25_000),
    "$25,000 under $50,000": (25_000, 50_000),
    "$50,000 under $75,000": (50_000, 75_000),
    "$75,000 under $100,000": (75_000, 100_000),
    "$100,000 under $200,000": (100_000, 200_000),
    "$200,000 under $500,000": (200_000, 500_000),
    "$500,000 under $1,000,000": (500_000, 1_000_000),
    "$1,000,000 or more": (1_000_000, np.inf),
}

STATE_ABBR_TO_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}

NON_VOTING_STATES = {"US", "AS", "GU", "MP", "PR", "VI", "OA"}
VARIABLE_SPECS = (
    ("N1", "adjusted_gross_income/count", True),
    ("A00100", "adjusted_gross_income/amount", False),
)


def _state_soi_url(tax_year: int) -> str:
    return f"https://www.irs.gov/pub/irs-soi/{tax_year % 100:02d}in55cmcsv.csv"


def _load_state_soi_raw(tax_year: int = LOCAL_STATE_SOI_TAX_YEAR) -> pd.DataFrame:
    return pd.read_csv(_state_soi_url(tax_year), thousands=",")


def _base_state_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    df = source_df.copy()
    df = df[df["AGI_STUB"] != 0].copy()
    df = df.loc[~df["STATE"].isin(NON_VOTING_STATES.union({"US"}))].copy()
    df["agi_bracket"] = df["AGI_STUB"].map(AGI_STUB_TO_BAND)
    df["GEO_NAME"] = df["STATE"]
    df["GEO_ID"] = "0400000US" + df["GEO_NAME"].map(STATE_ABBR_TO_FIPS)
    df["AGI_LOWER_BOUND"] = df["agi_bracket"].map(lambda band: AGI_BOUNDS[band][0])
    df["AGI_UPPER_BOUND"] = df["agi_bracket"].map(lambda band: AGI_BOUNDS[band][1])
    return df


def build_local_agi_state_targets(
    source_df: pd.DataFrame | None = None,
    tax_year: int = LOCAL_STATE_SOI_TAX_YEAR,
) -> pd.DataFrame:
    base = _base_state_frame(
        _load_state_soi_raw(tax_year=tax_year) if source_df is None else source_df
    )
    frames = []

    for column, variable, is_count in VARIABLE_SPECS:
        frame = base[
            ["GEO_ID", "GEO_NAME", "AGI_LOWER_BOUND", "AGI_UPPER_BOUND", column]
        ].rename(columns={column: "VALUE"})
        frame["IS_COUNT"] = int(is_count)
        frame["VARIABLE"] = variable
        if not is_count:
            frame["VALUE"] = frame["VALUE"] * 1_000
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def build_agi_state_targets(
    tax_year: int = LOCAL_STATE_SOI_TAX_YEAR,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_local_agi_state_targets(
        source_df=source_df,
        tax_year=tax_year,
    )


def refresh_local_agi_state_targets(
    out_path: Path = TARGETS_PATH,
) -> Path:
    targets = build_local_agi_state_targets()
    targets.to_csv(out_path, index=False)
    return out_path


def refresh_agi_state_targets(
    tax_year: int = LOCAL_STATE_SOI_TAX_YEAR,
    out_path: Path = TARGETS_PATH,
) -> pd.DataFrame:
    targets = build_local_agi_state_targets(tax_year=tax_year)
    targets.to_csv(out_path, index=False)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh agi_state.csv for local calibration"
    )
    parser.add_argument(
        "--tax-year",
        type=int,
        default=LOCAL_STATE_SOI_TAX_YEAR,
        help="IRS geographic SOI tax year to pull",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=TARGETS_PATH,
        help="Output CSV path",
    )
    args = parser.parse_args()
    refresh_agi_state_targets(tax_year=args.tax_year, out_path=args.out)


if __name__ == "__main__":
    main()
