# This is the file where we actually get the SOI information that we want:

# Goal: start with raw AGI and EITC:
# Data Dictionary: https://www.irs.gov/pub/irs-soi/22incddocguide.docx
# The Data: https://www.irs.gov/pub/irs-soi/22incd.csv

from pathlib import Path
from typing import List, Optional, Sequence, Dict, Tuple, Any, Union

import numpy as np
import pandas as pd
import logging

from policyengine_us_data.storage import CALIBRATION_FOLDER

logger = logging.getLogger(__name__)

"""Utilities to pull AGI targets from the IRS SOI data files."""

# Congressional districts have one fewer level than the national and state
# They're missing the million plus category
#  ("No AGI Stub") is a specific, intentional category used by the IRS in its summary data files.
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

#NON_VOTING_STATES = {"US", "AS", "GU", "MP", "PR", "VI", "OA"}

IGNORE_GEO_IDS = {
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
    "DC": "11",
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


#def pull_national_soi_variable(
#    soi_variable_ident: int,  # the national SOI xlsx file has a row for each target variable
#    variable_name: Union[str, None],
#    is_count: bool,
#    national_df: Optional[pd.DataFrame] = None,
#) -> pd.DataFrame:
#    """Download and save national AGI totals."""
#    df = pd.read_excel(
#        "https://www.irs.gov/pub/irs-soi/22in54us.xlsx", skiprows=7
#    )
#
#    assert (
#        np.abs(
#            df.iloc[soi_variable_ident, 1]
#            - df.iloc[soi_variable_ident, 2:12].sum()
#        )
#        < 100
#    ), "Row 0 doesn't add up — check the file."
#
#    agi_values = df.iloc[soi_variable_ident, 2:12].astype(int).to_numpy()
#    agi_values = np.concatenate(
#        [agi_values[:8], [agi_values[8] + agi_values[9]]]
#    )
#
#    agi_brackets = [
#        AGI_STUB_TO_BAND[i] for i in range(1, len(SOI_COLUMNS) + 1)
#    ]
#
#    result = pd.DataFrame(
#        {
#            "GEO_ID": ["0100000US"] * len(agi_brackets),
#            "GEO_NAME": ["national"] * len(agi_brackets),
#            "LOWER_BOUND": [AGI_BOUNDS[b][0] for b in agi_brackets],
#            "UPPER_BOUND": [AGI_BOUNDS[b][1] for b in agi_brackets],
#            "VALUE": agi_values,
#        }
#    )
#
#    # final column order
#    result = result[
#        ["GEO_ID", "GEO_NAME", "LOWER_BOUND", "UPPER_BOUND", "VALUE"]
#    ]
#    result["IS_COUNT"] = int(is_count)
#    result["VARIABLE"] = variable_name
#
#    result["VALUE"] = np.where(
#        result["IS_COUNT"] == 0, result["VALUE"] * 1_000, result["VALUE"]
#    )
#
#    if national_df is not None:
#        # If a DataFrame is passed, we append the new data to it.
#        df = pd.concat([national_df, result], ignore_index=True)
#        return df
#
#    return result
#
#
#def pull_state_soi_variable(
#    soi_variable_ident: str,  # the state SOI csv file has a column for each target variable
#    variable_name: Union[str, None],
#    is_count: bool,
#    state_df: Optional[pd.DataFrame] = None,
#) -> pd.DataFrame:
#    """Download and save state AGI totals."""
#    df = pd.read_csv(
#        "https://www.irs.gov/pub/irs-soi/22in55cmcsv.csv", thousands=","
#    )
#
#    merged = (
#        df[df["AGI_STUB"].isin([9, 10])]
#        .groupby("STATE", as_index=False)
#        .agg({soi_variable_ident: "sum"})
#        .assign(AGI_STUB=9)
#    )
#    df = df[~df["AGI_STUB"].isin([9, 10])]
#    df = pd.concat([df, merged], ignore_index=True)
#    df = df[df["AGI_STUB"] != 0]
#
#    df["agi_bracket"] = df["AGI_STUB"].map(AGI_STUB_TO_BAND)
#
#    df["state_abbr"] = df["STATE"]
#    df["GEO_ID"] = "0400000US" + df["state_abbr"].map(STATE_ABBR_TO_FIPS)
#    df["GEO_NAME"] = "state_" + df["state_abbr"]
#
#    result = df.loc[
#        ~df["STATE"].isin(NON_VOTING_STATES.union({"US"})),
#        ["GEO_ID", "GEO_NAME", "agi_bracket", soi_variable_ident],
#    ].rename(columns={soi_variable_ident: "VALUE"})
#
#    result["LOWER_BOUND"] = result["agi_bracket"].map(
#        lambda b: AGI_BOUNDS[b][0]
#    )
#    result["UPPER_BOUND"] = result["agi_bracket"].map(
#        lambda b: AGI_BOUNDS[b][1]
#    )
#
#    # final column order
#    result = result[
#        ["GEO_ID", "GEO_NAME", "LOWER_BOUND", "UPPER_BOUND", "VALUE"]
#    ]
#    result["IS_COUNT"] = int(is_count)
#    result["VARIABLE"] = variable_name
#
#    result["VALUE"] = np.where(
#        result["IS_COUNT"] == 0, result["VALUE"] * 1_000, result["VALUE"]
#    )
#
#    if state_df is not None:
#        # If a DataFrame is passed, we append the new data to it.
#        df = pd.concat([state_df, result], ignore_index=True)
#        return df
#
#    return result

def create_records(df, breakdown_variable, target_variable):
    """Transforms a DataFrame subset into a standardized list of records."""
    temp_df = df[["ucgid_str"]].copy()
    temp_df["breakdown_variable"] = breakdown_variable 
    temp_df["breakdown_value"] = df[breakdown_variable]
    temp_df["target_variable"] = target_variable 
    temp_df["target_value"] = df[target_variable]
    return temp_df


def make_records(
    df: pd.DataFrame,
    *,
    count_col: str,
    amount_col: str,
    amount_name: str,
    breakdown_col: Optional[str] = None,
    multiplier: int = 1_000,
):
    df = (
        df.rename({count_col: "tax_unit_count",
                   amount_col: amount_name},
                  axis=1)
          .copy()
    )

    if breakdown_col is None:
        breakdown_col = "one"
        df[breakdown_col] = 1

    rec_counts  = create_records(df, breakdown_col, "tax_unit_count")
    rec_amounts = create_records(df, breakdown_col, amount_name)
    rec_amounts["target_value"] *= multiplier  # Only the amounts get * 1000
    rec_counts["target_variable"] = f"{amount_name}_tax_unit_count"

    return rec_counts, rec_amounts



_TARGET_COL_MAP = {
    "N1":     "agi_tax_unit_count",   # number of returns (≈ “tax units”)
    "N2":     "agi_person_count",     # number of individuals
    "A00100": "agi_total_amount",     # total Adjusted Gross Income
}

_BREAKDOWN_FIELD = "agi_stub"        # numeric AGI stub 1‑10 from IRS
_BREAKDOWN_NAME  = "agi_stub"        # what will go in `breakdown_variable`

def make_agi_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert IRS SOI AGI‑split table from wide to the long format used
    in your `records[*]` list.
    
    Parameters
    ----------
    df : DataFrame
        Must contain `ucgid_str`, `agi_stub` and the three IRS fields
        in `_TARGET_COL_MAP` (N1, N2, A00100).
    
    Returns
    -------
    DataFrame with columns:
        ucgid_str
        breakdown_variable   (always "agi_stub")
        breakdown_value      (1‑10)
        target_variable      ("agi_tax_unit_count" | "agi_person_count" | "agi_total_amount")
        target_value         (float)
    """
    # — keep only what we need and rename for clarity
    work = (
        df[["ucgid_str", _BREAKDOWN_FIELD] + list(_TARGET_COL_MAP)]
          .rename(columns=_TARGET_COL_MAP)     # N1 → agi_tax_unit_count, etc.
    )

    # — wide → long
    long = (
        work.melt(
            id_vars=["ucgid_str", _BREAKDOWN_FIELD],
            var_name="target_variable",
            value_name="target_value"
        )
        .rename(columns={_BREAKDOWN_FIELD: "breakdown_value"})
        .assign(breakdown_variable=_BREAKDOWN_NAME)
        # Optional: add a human‑readable band label if useful
        # .assign(breakdown_label=lambda d: d["breakdown_value"].map(AGI_STUB_TO_BAND))
    )

    # — final column order
    long = long[["ucgid_str",
                 "breakdown_variable",
                 "breakdown_value",
                 "target_variable",
                 "target_value"]]

    # consistently sort (purely cosmetic)
    return (
        long.sort_values(["ucgid_str", "breakdown_value", "target_variable"])
            .reset_index(drop=True)
    )


def extract_soi_data() -> pd.DataFrame:
    """Download and save congressional district AGI totals.

    In the file below, "22" is 2022, "in" is individual returns,
    "cd" is congressional districts
    """
    return pd.read_csv("https://www.irs.gov/pub/irs-soi/22incd.csv")


raw_df = extract_soi_data()
# a "stub" is a term the IRS uses for a predefined category or group, specifically an income bracket.

TARGETS = [
    dict(code="59661", name="eitc", breakdown=("eitc_children", 0)),
    dict(code="59662", name="eitc", breakdown=("eitc_children", 1)),
    dict(code="59663", name="eitc", breakdown=("eitc_children", 2)),
    dict(code="59664", name="eitc", breakdown=("eitc_children", "3+")),
    dict(code="59664", name="qbid", breakdown=None),
    dict(code="18500", name="real_estate_taxes", breakdown=None),
    dict(code="01000", name="net_capital_gain", breakdown=None),
    dict(code="03150", name="ira_payments", breakdown=None),
    dict(code="00300", name="taxable_interest", breakdown=None),
    dict(code="00400", name="tax_exempt_interest", breakdown=None),
    dict(code="00600", name="oridinary_dividends", breakdown=None),
    dict(code="00650", name="qualified_dividends", breakdown=None),
    dict(code="26270", name="partnership_and_s_crop_net_income", breakdown=None),
    dict(code="02500", name="total_social_security", breakdown=None),
    dict(code="01700", name="pension_and_annuities", breakdown=None),
    dict(code="02300", name="unemployment_compensation", breakdown=None),
    dict(code="00900", name="business_net_income", breakdown=None),
    dict(code="17000", name="medical_and_dental_deduction", breakdown=None),
    dict(code="00700", name="salt_refunds", breakdown=None),
    dict(code="18425", name="salt_amount", breakdown=None),
    dict(code="06500", name="income_tax", breakdown=None),
]



def transform_soi_data(raw_df)


    # agi_stub is only 0, so there are only agi breakdowns at the state level
    # So you can confirm summability for 0 and then forget that national exists
    # Honestly I think that's a better idea in general. If your states don't add
    # Up to your national, something's off and you should treat it as an immediate
    # problem to fix rather than something to be adjusted
    national_df = raw_df.copy().loc[
        (raw_df.STATE == "US")
    ]
    national_df["ucgid_str"] = "0100000US"

    # You've got agi_stub == 0 in here, which you want to use any time you don't want to
    # break things up by AGI
    state_df = raw_df.copy().loc[
        (raw_df.STATE != "US") &
        (raw_df.CONG_DISTRICT == 0)
    ]
    state_df["ucgid_str"] = "0400000US" + state_df["STATEFIPS"].astype(str).str.zfill(2)

    # This is going to fail because we're missing the single cong district states
    district_df = raw_df.copy().loc[
        (raw_df.CONG_DISTRICT > 0)
    ]

    max_cong_district_by_state = raw_df.groupby('STATE')['CONG_DISTRICT'].transform('max')
    district_df = raw_df.copy().loc[
        (raw_df['CONG_DISTRICT'] > 0) | (max_cong_district_by_state == 0)
    ]
    district_df = district_df.loc[district_df['STATE'] != 'US']
    district_df["STATEFIPS"] = district_df["STATEFIPS"].astype(int).astype(str).str.zfill(2)
    district_df["CONG_DISTRICT"] = (
        district_df["CONG_DISTRICT"].astype(int).astype(str).str.zfill(2)
    )
    district_df["ucgid_str"] = "5001800US" + district_df["STATEFIPS"] + district_df["CONG_DISTRICT"]
    district_df = district_df[~district_df["ucgid_str"].isin(IGNORE_GEO_IDS)]

    assert district_df.shape[0] % 436 == 0

    # And you've got everything you need for all 3 levels of targets:
    #  1. national_df
    #  2. state_df
    #  3. district_df
    
    all_df = pd.concat([national_df, state_df, district_df])

    # So I want to get 2 variable categories out of this thing, in long format
    # 1) EITC, and 2) AGI
    # There's eitc_child_count, eitc. There's person_count and tax_unit_count
    # but no household_count. That's why you're doing this though, for a great example
    # Wide (a new variable per number of children) or Long (breakdown variable is number of children)

    # Marginal in terms of AGI, which this data set is organized with respect to 
    all_marginals = all_df.copy().loc[all_df.agi_stub == 0]
    assert all_marginals.shape[0] == 436 + 51 + 1

    # Collect targets from the SOI file
    records = []
    for spec in TARGETS:
        count_col  = f"N{spec['code']}"   # e.g. 'N59661'
        amount_col = f"A{spec['code']}"   # e.g. 'A59661'
    
        df = all_marginals.copy()
    
        if spec["breakdown"] is not None:
            col, val = spec["breakdown"]
            df[col] = val
            breakdown_col = col
        else:
            breakdown_col = None
    
        rec_counts, rec_amounts = make_records(
            df,
            count_col   = count_col,
            amount_col  = amount_col,
            amount_name = spec["name"],
            breakdown_col = breakdown_col,
            multiplier  = 1_000,
        )
        records.extend([rec_counts, rec_amounts])


    # Custom AGI amount, which doesn't have a count column (it has N1 and N2)
    temp_df = df[["ucgid_str"]].copy()
    temp_df["breakdown_variable"] = "one" 
    temp_df["breakdown_value"] = 1 
    temp_df["target_variable"] = "agi"
    temp_df["target_value"] = df["A00100"] * 1_000

    records.append(temp_df)

    # It's notable that the national counts only have agi_stub = 0
    all_agi_splits = all_df.copy().loc[all_df.agi_stub != 0]
    assert all_agi_splits.shape[0] % (436 + 51 + 0) == 0

    # Still a bit of work to do at the time of loading, since the breakdown variable
    # is agi_stub
    agi_long = make_agi_long(all_agi_splits)

    # We have the distribution and the total amount, let's not go crazy here
    agi_long = agi_long.loc[agi_long.target_variable != "agi_total_amount"] 

    records.append(agi_long)

    return pd.concat(records)


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


def combine_geography_levels(districts: Optional[bool] = False) -> None:
    """Combine SOI data across geography levels with validation and rescaling."""
    national = _get_soi_data("National")
    state = _get_soi_data("State")
    if districts:
        district = _get_soi_data("District")

    # Add state FIPS codes for validation
    state["STATEFIPS"] = state["GEO_ID"].str[-2:]
    if districts:
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

    if districts:
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
                    district.loc[district_mask]
                    .groupby("STATEFIPS")["VALUE"]
                    .sum()
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
                        district.loc[rescale_mask, "VALUE"] *= (
                            s_total / d_total
                        )

    # Combine all data
    combined = pd.concat(
        [
            national,
            state.drop(columns="STATEFIPS"),
            (
                district.drop(columns="STATEFIPS")
                if districts
                else pd.DataFrame(columns=national.columns)
            ),
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
    out_path = CALIBRATION_FOLDER / "soi.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Combined SOI targets saved to {out_path}")


def main() -> None:
    combine_geography_levels()


if __name__ == "__main__":
    main()
