"""Refresh state ACA premium tax credit targets from IRS SOI Table 2.

Output:
* ``aca_ptc_state.csv`` - per-state total PTC returns and amounts, sourced from
  IRS SOI Historical Table 2 (``{YY}in55cmcsv.csv``).

The source publishes money amounts in thousands of dollars; this script writes
whole dollars.

Usage::

    uv run python -m policyengine_us_data.storage.calibration_targets.refresh_aca_ptc_state_targets \
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
STATE_CSV_PATH = Path(__file__).with_name("aca_ptc_state.csv")


def _soi_year_prefix(year: int) -> str:
    return f"{year % 100:02d}"


def _state_file_url(year: int) -> str:
    return f"{IRS_SOI_ROOT}/{_soi_year_prefix(year)}in55cmcsv.csv"


def extract_state_aca_ptc(
    year: int, source_file: str | Path | None = None
) -> pd.DataFrame:
    """Build per-state ACA PTC rows from SOI Historical Table 2.

    ``A85770`` is total premium tax credit claimed, in thousands of dollars.
    ``N85770`` is returns with total premium tax credit. This matches the
    model's gross ``aca_ptc`` concept more closely than APTC outlays or net PTC
    after reconciliation.
    """
    source = Path(source_file) if source_file is not None else _state_file_url(year)
    df = pd.read_csv(source, thousands=",")
    mask = (df["AGI_STUB"] == 0) & (df["STATE"].isin(STATE_ABBR_TO_FIPS))
    state_rows = df.loc[mask].copy()
    state_rows["FIPS"] = state_rows["STATE"].map(STATE_ABBR_TO_FIPS)
    state_rows["GEO_ID"] = "0400000US" + state_rows["FIPS"]
    state_rows["Returns"] = state_rows["N85770"].astype(int)
    state_rows["TotalPTCAmount"] = (
        state_rows["A85770"].astype("int64") * 1_000
    ).astype("int64")

    out = (
        state_rows[["GEO_ID", "Returns", "TotalPTCAmount"]]
        .sort_values("GEO_ID")
        .reset_index(drop=True)
    )

    national_row = df[(df["STATE"] == "US") & (df["AGI_STUB"] == 0)].iloc[0]
    national_returns = int(national_row["N85770"])
    national_amount = int(national_row["A85770"]) * 1_000
    rel_returns = abs(out["Returns"].sum() - national_returns) / max(
        national_returns, 1
    )
    rel_amount = abs(out["TotalPTCAmount"].sum() - national_amount) / max(
        national_amount, 1
    )
    if rel_returns > 0.01 or rel_amount > 0.01:
        raise ValueError(
            "State sum diverges from published US total by more than 1%%: "
            f"returns diff={rel_returns:.4%}, amount diff={rel_amount:.4%}"
        )

    return out


def _write_state_csv(state_df: pd.DataFrame, year: int) -> None:
    header_comment = (
        f"# IRS SOI Historical Table 2 ({_soi_year_prefix(year)}in55cmcsv.csv), total premium "
        "tax credit columns N85770 (returns) and A85770 (amount, thousands "
        f"USD). Pulled from {_state_file_url(year)}. Amount converted to "
        "dollars.\n"
    )
    with STATE_CSV_PATH.open("w", newline="") as file:
        file.write(header_comment)
        state_df.to_csv(file, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh ACA PTC state calibration targets from IRS SOI."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="IRS tax year to pull. 2022 is the latest state geography file.",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="Optional local IRS SOI CSV path for offline refreshes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_df = extract_state_aca_ptc(args.year, source_file=args.source_file)
    _write_state_csv(state_df, args.year)
    print(
        f"Wrote {len(state_df)} state rows to {STATE_CSV_PATH} "
        f"(sum returns={state_df['Returns'].sum():,}, "
        f"sum amount=${state_df['TotalPTCAmount'].sum():,})"
    )


if __name__ == "__main__":
    main()
