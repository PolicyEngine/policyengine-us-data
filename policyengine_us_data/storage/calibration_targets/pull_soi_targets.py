from pathlib import Path

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import logging

from policyengine_us_data.storage import CALIBRATION_FOLDER
from policyengine_us_data.storage.calibration_targets.district_mapping import (
    get_district_mapping_matrix,
)

logger = logging.getLogger(__name__)

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
NON_VOTING_GEO_IDS = {
    "0400000US72",  # Puerto Rico (state level)
    "5001800US7298",  # Puerto Rico
    "5001800US6098",  # American Samoa
    "5001800US6698",  # Guam
    "5001800US6998",  # Northern Mariana Islands
    "5001800US7898",  # U.S. Virgin Islands
}

# after skipping the first 7 rows, the national SOI file has targets as row indices [COUNT_INDEX, AMOUNT_INDEX]
NATIONAL_VARIABLES = {
    "adjusted_gross_income": [0, 17],
}

# the state and district SOI file have targets as column names [COUNT_COL_NAME, AMOUNT_COL_NAME]
GEOGRAPHY_VARIABLES = {"adjusted_gross_income": ["N1", "A00100"]}

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
FIPS_TO_STATE_ABBR = {v: k for k, v in STATE_ABBR_TO_FIPS.items()}


def pull_national_soi_variable(
    soi_variable_ident: int,  # the national SOI xlsx file has a row for each target variable
    variable_name: Union[str, None],
    is_count: bool,
    national_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Download and save national AGI totals."""
    df = pd.read_excel(
        "https://www.irs.gov/pub/irs-soi/22in54us.xlsx", skiprows=7
    )

    assert (
        np.abs(
            df.iloc[soi_variable_ident, 1]
            - df.iloc[soi_variable_ident, 2:12].sum()
        )
        < 100
    ), "Row 0 doesn't add up — check the file."

    agi_values = df.iloc[soi_variable_ident, 2:12].astype(int).to_numpy()
    agi_values = np.concatenate(
        [agi_values[:8], [agi_values[8] + agi_values[9]]]
    )

    agi_brackets = [
        AGI_STUB_TO_BAND[i] for i in range(1, len(SOI_COLUMNS) + 1)
    ]

    result = pd.DataFrame(
        {
            "GEO_ID": ["0100000US"] * len(agi_brackets),
            "GEO_NAME": ["US"] * len(agi_brackets),
            "LOWER_BOUND": [AGI_BOUNDS[b][0] for b in agi_brackets],
            "UPPER_BOUND": [AGI_BOUNDS[b][1] for b in agi_brackets],
            "VALUE": agi_values,
        }
    )

    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "LOWER_BOUND", "UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    result["VALUE"] = np.where(
        result["IS_COUNT"] == 0, result["VALUE"] * 1_000, result["VALUE"]
    )

    if national_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([national_df, result], ignore_index=True)
        return df

    return result


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

    result["LOWER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][0]
    )
    result["UPPER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][1]
    )

    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "LOWER_BOUND", "UPPER_BOUND", "VALUE"]
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


def pull_district_soi_variable(
    soi_variable_ident: str,  # the district SOI csv file has a column for each target variable
    variable_name: Union[str, None],
    is_count: bool,
    district_df: Optional[pd.DataFrame] = None,
    redistrict: bool = True,
) -> pd.DataFrame:
    """Download and save congressional district AGI totals."""
    df = pd.read_csv("https://www.irs.gov/pub/irs-soi/22incd.csv")
    df = df[df["agi_stub"] != 0]

    df["STATEFIPS"] = df["STATEFIPS"].astype(int).astype(str).str.zfill(2)
    df["CONG_DISTRICT"] = (
        df["CONG_DISTRICT"].astype(int).astype(str).str.zfill(2)
    )
    df["GEO_ID"] = "5001800US" + df["STATEFIPS"] + df["CONG_DISTRICT"]
    df = df[~df["GEO_ID"].isin(NON_VOTING_GEO_IDS)]

    at_large_states = (
        df.groupby("STATEFIPS")["CONG_DISTRICT"]
        .nunique()
        .pipe(lambda s: s[s == 1].index)
    )
    df = df.loc[
        (df["CONG_DISTRICT"] != "00") | (df["STATEFIPS"].isin(at_large_states))
    ].reset_index(drop=True)

    df["GEO_NAME"] = (
        f"{df['STATEFIPS'].map(FIPS_TO_STATE_ABBR)}-{df['CONG_DISTRICT']}"
    )

    df["agi_bracket"] = df["agi_stub"].map(AGI_STUB_TO_BAND)
    result = df[
        [
            "GEO_ID",
            "GEO_NAME",
            "CONG_DISTRICT",
            "STATE",
            "agi_bracket",
            soi_variable_ident,
        ]
    ].rename(columns={soi_variable_ident: "VALUE"})

    result["LOWER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][0]
    )
    result["UPPER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][1]
    )

    if redistrict:
        result = apply_redistricting(result, variable_name)

    assert df["GEO_ID"].nunique() == 436

    if redistrict:
        # After redistricting, validate against the new district codes from the mapping
        mapping_df = pd.read_csv(CALIBRATION_FOLDER / "district_mapping.csv")
        valid_district_codes = set(mapping_df["code_new"].unique())

        # Check that all GEO_IDs are valid
        produced_codes = set(result["GEO_ID"])
        invalid_codes = produced_codes - valid_district_codes
        assert (
            not invalid_codes
        ), f"Invalid district codes after redistricting: {invalid_codes}"

        # Check we have exactly 436 districts
        assert (
            len(produced_codes) == 436
        ), f"Expected 436 districts after redistricting, got {len(produced_codes)}"

        # Check that all GEO_IDs successfully mapped to names
        missing_names = result[result["GEO_NAME"].isna()]["GEO_ID"].unique()
        assert (
            len(missing_names) == 0
        ), f"GEO_IDs without names in ID_TO_NAME mapping: {missing_names}"

    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "LOWER_BOUND", "UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    result["VALUE"] = np.where(
        result["IS_COUNT"] == 0, result["VALUE"] * 1_000, result["VALUE"]
    )

    if district_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([district_df, result], ignore_index=True)
        return df

    return result


def apply_redistricting(
    df: pd.DataFrame,
    variable_name: str,
) -> pd.DataFrame:
    """Apply redistricting transformation to congressional district data."""
    mapping_matrix = get_district_mapping_matrix()
    mapping_df = pd.read_csv(CALIBRATION_FOLDER / "district_mapping.csv")

    # Get sorted lists of old and new codes (to match the matrix ordering)
    old_codes = sorted(mapping_df["code_old"].unique())
    new_codes = sorted(mapping_df["code_new"].unique())

    old_to_idx = {code: i for i, code in enumerate(old_codes)}

    assert mapping_matrix.shape == (
        436,
        436,
    ), f"Expected 436x436 matrix, got {mapping_matrix.shape}"
    assert np.allclose(
        mapping_matrix.sum(axis=1), 1.0
    ), "Mapping proportions don't sum to 1"

    # Process each AGI bracket separately
    result_dfs = []

    for bracket in (
        df[["LOWER_BOUND", "UPPER_BOUND"]].drop_duplicates().itertuples()
    ):
        bracket_df = df[
            (df["LOWER_BOUND"] == bracket.LOWER_BOUND)
            & (df["UPPER_BOUND"] == bracket.UPPER_BOUND)
        ].copy()

        # Create value vector for old districts (436 elements)
        old_values = np.zeros(436)
        for _, row in bracket_df.iterrows():
            geo_id = row["GEO_ID"]

            # Handle DC special case: SOI uses 1100, current map uses 1198
            if geo_id == "5001800US1100":
                geo_id = "5001800US1198"

            if geo_id in old_to_idx:
                idx = old_to_idx[geo_id]
                old_values[idx] = row["VALUE"]

        # Apply transformation: new = matrix^T @ old
        new_values = mapping_matrix.T @ old_values

        # Create new dataframe with redistributed values
        new_rows = []
        for i, new_code in enumerate(new_codes):
            state_fips = new_code[-4:-2]
            district = new_code[-2:]

            state_abbr = FIPS_TO_STATE_ABBR.get(state_fips, state_fips)
            geo_name = f"{state_abbr}-{district}"

            new_row = {
                "GEO_ID": new_code,
                "GEO_NAME": geo_name,
                "CONG_DISTRICT": district,
                "STATE": state_fips,  # This is FIPS code, not abbreviation
                "agi_bracket": bracket_df.iloc[0]["agi_bracket"],
                "LOWER_BOUND": bracket.LOWER_BOUND,
                "UPPER_BOUND": bracket.UPPER_BOUND,
                "VALUE": new_values[i],
            }
            new_rows.append(new_row)

        if new_rows:
            result_dfs.append(pd.DataFrame(new_rows))

    # Combine all brackets
    if result_dfs:
        result = pd.concat(result_dfs, ignore_index=True)
    else:
        # If no result_dfs, create empty DataFrame with proper structure
        result = pd.DataFrame(
            columns=[
                "GEO_ID",
                "GEO_NAME",
                "CONG_DISTRICT",
                "STATE",
                "agi_bracket",
                "LOWER_BOUND",
                "UPPER_BOUND",
                "VALUE",
            ]
        )

    logger.info(f"Redistricting complete for {variable_name}")
    logger.info(
        f"Old districts: {len(old_codes)}, New districts: {len(new_codes)}"
    )

    # Verify total preservation
    old_total = df["VALUE"].sum()
    new_total = result["VALUE"].sum()
    if not np.isclose(old_total, new_total, rtol=1e-6):
        logger.error(
            f"Total value changed during redistricting: {old_total} -> {new_total}"
        )
        raise ValueError(f"Total value not preserved during redistricting")

    return result


def _get_soi_data(geo_level: str) -> pd.DataFrame:
    """
    geo_level ∈ {'National', 'State', 'District'}
    Returns a DataFrame with all SOI variables for the specified geography level
    """
    if geo_level == "National":
        var_indices = NATIONAL_VARIABLES
        variable_pull = pull_national_soi_variable
    elif geo_level == "State":
        var_indices = GEOGRAPHY_VARIABLES
        variable_pull = pull_state_soi_variable
    elif geo_level == "District":
        var_indices = GEOGRAPHY_VARIABLES
        variable_pull = pull_district_soi_variable
    else:
        raise ValueError("geo_level must be National, State or District")

    df = pd.DataFrame()
    for variable, identifiers in var_indices.items():
        count_id, amount_id = identifiers
        # Pull count data (first identifier)
        count_df = variable_pull(
            soi_variable_ident=count_id,
            variable_name=variable,
            is_count=float(True),
        )
        df = pd.concat([df, count_df], ignore_index=True)
        # Pull amount data (second identifier)
        amount_df = variable_pull(
            soi_variable_ident=amount_id,
            variable_name=variable,
            is_count=float(False),
        )
        df = pd.concat([df, amount_df], ignore_index=True)

    return df


def combine_geography_levels() -> None:
    """Combine SOI data across geography levels with validation and rescaling."""
    national = _get_soi_data("National")
    state = _get_soi_data("State")
    district = _get_soi_data("District")

    # Add state FIPS codes for validation
    state["STATEFIPS"] = state["GEO_ID"].str[-2:]
    district["STATEFIPS"] = district["GEO_ID"].str[-4:-2]

    # Get unique variables and AGI brackets for iteration
    variables = national["VARIABLE"].unique()
    agi_brackets = national[["LOWER_BOUND", "UPPER_BOUND"]].drop_duplicates()

    # Validate and rescale state totals against national totals
    for variable in variables:
        for is_count in [0.0, 1.0]:  # Process count and amount separately
            for _, bracket in agi_brackets.iterrows():
                lower, upper = (
                    bracket["LOWER_BOUND"],
                    bracket["UPPER_BOUND"],
                )

                # Get national total for this variable/bracket/type combination
                nat_mask = (
                    (national["VARIABLE"] == variable)
                    & (national["LOWER_BOUND"] == lower)
                    & (national["UPPER_BOUND"] == upper)
                    & (national["IS_COUNT"] == is_count)
                )
                us_total = national.loc[nat_mask, "VALUE"].iloc[0]

                # Get state total for this variable/bracket/type combination
                state_mask = (
                    (state["VARIABLE"] == variable)
                    & (state["LOWER_BOUND"] == lower)
                    & (state["UPPER_BOUND"] == upper)
                    & (state["IS_COUNT"] == is_count)
                )
                state_total = state.loc[state_mask, "VALUE"].sum()

                # Rescale states if they don't match national total
                if not np.isclose(state_total, us_total, rtol=1e-3):
                    count_type = "count" if is_count == 1.0 else "amount"
                    logger.warning(
                        f"States' sum does not match national total for {variable}/{count_type} "
                        f"in bracket [{lower}, {upper}]. Rescaling state targets."
                    )
                    state.loc[state_mask, "VALUE"] *= us_total / state_total

    # Validate and rescale district totals against state totals
    for variable in variables:
        for is_count in [0.0, 1.0]:  # Process count and amount separately
            for _, bracket in agi_brackets.iterrows():
                lower, upper = (
                    bracket["LOWER_BOUND"],
                    bracket["UPPER_BOUND"],
                )

                # Create masks for this variable/bracket/type combination
                state_mask = (
                    (state["VARIABLE"] == variable)
                    & (state["LOWER_BOUND"] == lower)
                    & (state["UPPER_BOUND"] == upper)
                    & (state["IS_COUNT"] == is_count)
                )
                district_mask = (
                    (district["VARIABLE"] == variable)
                    & (district["LOWER_BOUND"] == lower)
                    & (district["UPPER_BOUND"] == upper)
                    & (district["IS_COUNT"] == is_count)
                )

            # Get state totals indexed by STATEFIPS
            state_totals = state.loc[state_mask].set_index("STATEFIPS")[
                "VALUE"
            ]

            # Get district totals grouped by STATEFIPS
            district_totals = (
                district.loc[district_mask].groupby("STATEFIPS")["VALUE"].sum()
            )

            # Check and rescale districts for each state
            for fips, d_total in district_totals.items():
                s_total = state_totals.get(fips)

                if s_total is not None and not np.isclose(
                    d_total, s_total, rtol=1e-3
                ):
                    count_type = "count" if is_count == 1.0 else "amount"
                    logger.warning(
                        f"Districts' sum does not match {fips} state total for {variable}/{count_type} "
                        f"in bracket [{lower}, {upper}]. Rescaling district targets."
                    )
                    rescale_mask = district_mask & (
                        district["STATEFIPS"] == fips
                    )
                    district.loc[rescale_mask, "VALUE"] *= s_total / d_total

    # Combine all data
    combined = pd.concat(
        [
            national,
            state.drop(columns="STATEFIPS"),
            district.drop(columns="STATEFIPS"),
        ],
        ignore_index=True,
    ).sort_values(["GEO_ID", "VARIABLE", "LOWER_BOUND"])

    combined["DATA_SOURCE"] = "soi"
    combined["BREAKDOWN_VARIABLE"] = "adjusted_gross_income"

    combined = combined[
        [
            "DATA_SOURCE",
            "GEO_ID",
            "GEO_NAME",
            "VARIABLE",
            "VALUE",
            "IS_COUNT",
            "BREAKDOWN_VARIABLE",
            "LOWER_BOUND",
            "UPPER_BOUND",
        ]
    ]

    # Save combined data
    out_path = CALIBRATION_FOLDER / "soi_targets.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Combined SOI targets saved to {out_path}")


def main() -> None:
    combine_geography_levels()


if __name__ == "__main__":
    main()
