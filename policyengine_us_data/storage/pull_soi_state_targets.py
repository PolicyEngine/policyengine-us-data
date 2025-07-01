from pathlib import Path

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from policyengine_us_data.storage import STORAGE_FOLDER


"""Utilities to pull AGI targets from the IRS SOI data files."""

SOI_COLUMNS = [
    "Under $1",
    "$1 under $10,000",
    "$10,000 under $25,000",
    "$25,000 under $50,000",
    "$50,000 under $75,000",
    "$75,000 under $100,000",
    "$100,000 under $200,000",
    "$200,000 under $500,000",
    "$500,000 or more",
]

AGI_STUB_TO_BAND = {i + 1: band for i, band in enumerate(SOI_COLUMNS)}

AGI_BOUNDS = {
    "Under $1": (-np.inf, 1),
    "$1 under $10,000": (1, 10_000),
    "$10,000 under $25,000": (10_000, 25_000),
    "$25,000 under $50,000": (25_000, 50_000),
    "$50,000 under $75,000": (50_000, 75_000),
    "$75,000 under $100,000": (75_000, 100_000),
    "$100,000 under $200,000": (100_000, 200_000),
    "$200,000 under $500,000": (200_000, 500_000),
    "$500,000 or more": (500_000, np.inf),
}

NON_VOTING_STATES = {"US", "AS", "GU", "MP", "PR", "VI", "OA"}


# the state and district SOI file have targets as column names:
GEOGRAPHY_VARIABLES = {
    "adjusted_gross_income/count": "N1",
    "adjusted_gross_income/amount": "A00100",
    # "real_estate_taxes/count": "N18500",
    # "real_estate_taxes/amount": "A18500",
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


def pull_state_soi_variable(
    soi_variable_ident: str,  # the state SOI csv file has a column for each target variable
    variable_name: Union[str, None],
    is_count: bool,
    state_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Download and save state AGI totals."""
    df = pd.read_csv(
        "https://www.irs.gov/pub/irs-soi/22in55cmcsv.csv", thousands=","
    )

    merged = (
        df[df["AGI_STUB"].isin([9, 10])]
        .groupby("STATE", as_index=False)
        .agg({soi_variable_ident: "sum"})
        .assign(AGI_STUB=9)
    )
    df = df[~df["AGI_STUB"].isin([9, 10])]
    df = pd.concat([df, merged], ignore_index=True)
    df = df[df["AGI_STUB"] != 0]

    df["agi_bracket"] = df["AGI_STUB"].map(AGI_STUB_TO_BAND)

    df["state_abbr"] = df["STATE"]
    df["GEO_ID"] = "0400000US" + df["state_abbr"].map(STATE_ABBR_TO_FIPS)
    df["GEO_NAME"] = df["state_abbr"]

    result = df.loc[
        ~df["STATE"].isin(NON_VOTING_STATES.union({"US"})),
        ["GEO_ID", "GEO_NAME", "agi_bracket", soi_variable_ident],
    ].rename(columns={soi_variable_ident: "VALUE"})

    result["AGI_LOWER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][0]
    )
    result["AGI_UPPER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][1]
    )

    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "AGI_LOWER_BOUND", "AGI_UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    result["VALUE"] = np.where(
        result["IS_COUNT"] == 0, result["VALUE"] * 1_000, result["VALUE"]
    )

    if state_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([state_df, result], ignore_index=True)
        return df

    return result


def create_targets(
    var_indices: dict[str : Union[int, str]],
    variable_pull: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    """Create a DataFrame with AGI targets."""
    df = pd.DataFrame()
    for variable, identifyer in var_indices.items():
        variable_df = variable_pull(
            soi_variable_ident=identifyer,
            variable_name=variable,
            is_count=1 if variable.endswith("count") else 0,
        )
        df = pd.concat([df, variable_df], ignore_index=True)
    return df


def main() -> None:
    out_dir = STORAGE_FOLDER
    state_df = create_targets(GEOGRAPHY_VARIABLES, pull_state_soi_variable)
    state_df.to_csv(out_dir / "agi_state.csv", index=False)


if __name__ == "__main__":
    main()
