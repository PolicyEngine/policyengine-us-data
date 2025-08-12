from pathlib import Path
from typing import List, Optional, Sequence, Dict, Tuple, Any, Union

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)




"""Utilities to pull AGI targets from the IRS SOI data files."""

# Congressional districts have one fewer level than the national and state
# They're missing the million plus category
#  ("No AGI Stub") is a specific, intentional category used by the IRS in its summary data files.
#
#SOI_COLUMNS = [
#    "Under $1",
#    "$1 under $10,000",
#    "$10,000 under $25,000",
#    "$25,000 under $50,000",
#    "$50,000 under $75,000",
#    "$75,000 under $100,000",
#    "$100,000 under $200,000",
#    "$200,000 under $500,000",
#    "$500,000 or more",
#]
#
#AGI_STUB_TO_BAND = {i + 1: band for i, band in enumerate(SOI_COLUMNS)}
#
#AGI_BOUNDS = {
#    "Under $1": (-np.inf, 1),
#    "$1 under $10,000": (1, 10_000),
#    "$10,000 under $25,000": (10_000, 25_000),
#    "$25,000 under $50,000": (25_000, 50_000),
#    "$50,000 under $75,000": (50_000, 75_000),
#    "$75,000 under $100,000": (75_000, 100_000),
#    "$100,000 under $200,000": (100_000, 200_000),
#    "$200,000 under $500,000": (200_000, 500_000),
#    "$500,000 or more": (500_000, np.inf),
#}
#
##NON_VOTING_STATES = {"US", "AS", "GU", "MP", "PR", "VI", "OA"}
#
IGNORE_GEO_IDS = {
    "0400000US72",  # Puerto Rico (state level)
    "5001800US7298",  # Puerto Rico
    "5001800US6098",  # American Samoa
    "5001800US6698",  # Guam
    "5001800US6998",  # Northern Mariana Islands
    "5001800US7898",  # U.S. Virgin Islands
}


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


def make_agi_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert IRS SOI AGI‑split table from wide to the long format used"""
    target_col_map = {
        "N1":     "agi_tax_unit_count",
        "N2":     "agi_person_count",
        "A00100": "agi_total_amount",
    }
    work = (
        df[["ucgid_str", "agi_stub"] + list(target_col_map)]
          .rename(columns=target_col_map)
    )
    long = (
        work.melt(
            id_vars=["ucgid_str", "agi_stub"],
            var_name="target_variable",
            value_name="target_value"
        )
        .rename(columns={"agi_stub": "breakdown_value"})
        .assign(breakdown_variable="agi_stub")
    )
    long = long[["ucgid_str",
                 "breakdown_variable",
                 "breakdown_value",
                 "target_variable",
                 "target_value"]]
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


def transform_soi_data(raw_df):

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

    # National ---------------
    national_df = raw_df.copy().loc[
        (raw_df.STATE == "US")
    ]
    national_df["ucgid_str"] = "0100000US"

    # State -------------------
    # You've got agi_stub == 0 in here, which you want to use any time you don't want to
    # break things up by AGI
    state_df = raw_df.copy().loc[
        (raw_df.STATE != "US") &
        (raw_df.CONG_DISTRICT == 0)
    ]
    state_df["ucgid_str"] = "0400000US" + state_df["STATEFIPS"].astype(str).str.zfill(2)

    # District ------------------
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

    all_df = pd.concat([national_df, state_df, district_df])

    # "Marginal" over AGI bands, which this data set is organized according to 
    all_marginals = all_df.copy().loc[all_df.agi_stub == 0]
    assert all_marginals.shape[0] == 436 + 51 + 1

    # Collect targets from the SOI file
    records = []
    for spec in TARGETS:
        count_col  = f"N{spec['code']}"  # e.g. 'N59661'
        amount_col = f"A{spec['code']}"  # e.g. 'A59661'
    
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


    # AGI Processing (separate, doesn't have a count column)
    temp_df = df[["ucgid_str"]].copy()
    temp_df["breakdown_variable"] = "one" 
    temp_df["breakdown_value"] = 1 
    temp_df["target_variable"] = "agi"
    temp_df["target_value"] = df["A00100"] * 1_000

    records.append(temp_df)

    # Note: national counts only have agi_stub = 0
    all_agi_splits = all_df.copy().loc[all_df.agi_stub != 0]
    assert all_agi_splits.shape[0] % (436 + 51 + 0) == 0

    agi_long = make_agi_long(all_agi_splits)
    agi_long = agi_long.loc[agi_long.target_variable != "agi_total_amount"] 

    records.append(agi_long)

    return pd.concat(records)


def load_soi_data(long_dfs, year):

    DATABASE_URL = "sqlite:///policy_data.db"
    engine = create_engine(DATABASE_URL)

    Session = sessionmaker(bind=engine)
    session = Session()

    # Load EITC data -------------------------------------------------------- 
    # NOTE: obviously this is not especially robust ---
    eitc_data = {'0': (long_dfs[0], long_dfs[1]),
                 '1': (long_dfs[2], long_dfs[3]),
                 '2': (long_dfs[4], long_dfs[5]),
                 '3+': (long_dfs[6], long_dfs[7])}

    stratum_lookup = {"State": {}, "District": {}}
    for n_children in eitc_data.keys():
        eitc_count_i, eitc_amount_i = eitc_data[n_children]
        for i in range(eitc_count_i.shape[0]):
            ucgid_i = eitc_count_i[['ucgid_str']].iloc[i].values[0]
            note = f"Geo: {ucgid_i}, EITC received with {n_children} children"

            if len(ucgid_i) == 9:  # National.
                new_stratum = Stratum(
                    parent_stratum_id=None, stratum_group_id=0, notes=note
                )
            elif len(ucgid_i) == 11:  # State 
                new_stratum = Stratum(
                    parent_stratum_id=stratum_lookup["National"],
                    stratum_group_id=0,
                    notes=note
                )
            elif len(ucgid_i) == 13:  # District 
                new_stratum = Stratum(
                    parent_stratum_id=stratum_lookup["State"]['0400000US' + ucgid_i[9:11]],
                    stratum_group_id=0,
                    notes=note
                )

            new_stratum.constraints_rel = [
               StratumConstraint(
                   constraint_variable="ucgid_str",
                   operation="in",
                   value=ucgid_i,
               ),
            ]
            if n_children == "3+":
               new_stratum.constraints_rel.append(
                   StratumConstraint(
                       constraint_variable="eitc_children",
                       operation="greater_than_or_equal_to",
                       value='3',
                   )
               )
            else:
               new_stratum.constraints_rel.append(
                   StratumConstraint(
                       constraint_variable="eitc_children",
                       operation="equals",
                       value=f'{n_children}',
                   )
               )

            new_stratum.targets_rel = [
                Target(
                    variable="tax_unit_count",
                    period=year,
                    value=eitc_count_i.iloc[i][["target_value"]].values[0],
                    source_id=5,
                    active=True,
                ),
                Target(
                    variable="eitc",
                    period=year,
                    value=eitc_amount_i.iloc[i][["target_value"]].values[0],
                    source_id=5,
                    active=True,
                )
            ]

            session.add(new_stratum)
            session.flush()

            if len(ucgid_i) == 9:
                 stratum_lookup["National"] = new_stratum.stratum_id
            elif len(ucgid_i) == 11: 
                 stratum_lookup["State"][ucgid_i] = new_stratum.stratum_id


    # No breakdown variables in this set 
    for j in range(8, 42, 2):
        print(long_dfs[j])  # count
        print(long_dfs[j + 1])  # amount

        # Why are we making strata here? You have a lot of these to run through
        count_j, amount_j = long_dfs[j], long_dfs[j + 1] 
        for i in range(count_j.shape[0]):
            ucgid_i = count_j[['ucgid_str']].iloc[i].values[0]
            # If there's no breakdown variable, is this a new geo?
            # The problem is, it's vary difficult to search for a geography
            # That's already in existance
            note = f"Geo: {ucgid_i}"

            if len(ucgid_i) == 9:  # National.
                new_stratum = Stratum(
                    parent_stratum_id=None, stratum_group_id=0, notes=note
                )
            elif len(ucgid_i) == 11:  # State 
                new_stratum = Stratum(
                    parent_stratum_id=stratum_lookup["National"],
                    stratum_group_id=0,
                    notes=note
                )
            elif len(ucgid_i) == 13:  # District 
                new_stratum = Stratum(
                    parent_stratum_id=stratum_lookup["State"]['0400000US' + ucgid_i[9:11]],
                    stratum_group_id=0,
                    notes=note
                )

            new_stratum.constraints_rel = [
               StratumConstraint(
                   constraint_variable="ucgid_str",
                   operation="in",
                   value=ucgid_i,
               ),
            ]
            new_stratum.targets_rel = [
                Target(
                    variable="tax_unit_count",
                    period=year,
                    value=count_j.iloc[i][["target_value"]].values[0],
                    source_id=5,
                    active=True,
                ),
                Target(
                    variable=amount_j.iloc[0][["target_variable"]].values[0],
                    period=year,
                    value=amount_j.iloc[i][["target_value"]].values[0],
                    source_id=5,
                    active=True,
                )
            ]

            session.add(new_stratum)
            session.flush()

            if len(ucgid_i) == 9:
                 stratum_lookup["National"] = new_stratum.stratum_id
            elif len(ucgid_i) == 11: 
                 stratum_lookup["State"][ucgid_i] = new_stratum.stratum_id

    session.commit()



def main() -> None:
    year = 2022  # NOTE: predates the finalization of the 2020 Census redistricting
    raw_df = extract_soi_data()

    long_dfs = transform_soi_data(raw_df):


if __name__ == "__main__":
    main()
