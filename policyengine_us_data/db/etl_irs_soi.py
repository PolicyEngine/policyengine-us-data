from typing import Optional

import numpy as np
import pandas as pd

from sqlmodel import Session, create_engine, select

from policyengine_us_data.storage import STORAGE_FOLDER

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.db import (
    get_stratum_by_id,
    get_root_strata,
    get_stratum_children,
    get_stratum_parent,
    parse_ucgid,
    get_geographic_strata,
)
from policyengine_us_data.utils.census import TERRITORY_UCGIDS
from policyengine_us_data.storage.calibration_targets.make_district_mapping import (
    get_district_mapping,
)


"""See the 22incddocguide.docx manual from the IRS SOI"""
# Language in the doc: '$10,000 under $25,000' means >= $10,000 and < $25,000
AGI_STUB_TO_INCOME_RANGE = {
    1: (-np.inf, 1),          # Under $1 (negative AGI allowed)
    2: (1, 10_000),            # $1 under $10,000
    3: (10_000, 25_000),       # $10,000 under $25,000
    4: (25_000, 50_000),       # $25,000 under $50,000
    5: (50_000, 75_000),       # $50,000 under $75,000
    6: (75_000, 100_000),      # $75,000 under $100,000
    7: (100_000, 200_000),     # $100,000 under $200,000
    8: (200_000, 500_000),     # $200,000 under $500,000
    9: (500_000, np.inf),      # $500,000 or more
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
    df = df.rename(
        {count_col: "tax_unit_count", amount_col: amount_name}, axis=1
    ).copy()

    if breakdown_col is None:
        breakdown_col = "one"
        df[breakdown_col] = 1

    rec_counts = create_records(df, breakdown_col, "tax_unit_count")
    rec_amounts = create_records(df, breakdown_col, amount_name)
    rec_amounts["target_value"] *= multiplier  # Only the amounts get * 1000
    rec_counts["target_variable"] = f"{amount_name}_tax_unit_count"

    return rec_counts, rec_amounts


def make_agi_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert IRS SOI AGI‑split table from wide to the long format used"""
    target_col_map = {
        "N1": "tax_unit_count",
        "N2": "person_count",
        "A00100": "adjusted_gross_income",
    }
    work = df[["ucgid_str", "agi_stub"] + list(target_col_map)].rename(
        columns=target_col_map
    )
    long = (
        work.melt(
            id_vars=["ucgid_str", "agi_stub"],
            var_name="target_variable",
            value_name="target_value",
        )
        .rename(columns={"agi_stub": "breakdown_value"})
        .assign(breakdown_variable="agi_stub")
    )
    long = long[
        [
            "ucgid_str",
            "breakdown_variable",
            "breakdown_value",
            "target_variable",
            "target_value",
        ]
    ]

    return [
        df.sort_values(by="ucgid_str").reset_index(drop=True)
        for name, df in long.groupby(["breakdown_value", "target_variable"])
    ]


def convert_district_data(
    input_df: pd.DataFrame,
    mapping_matrix: np.ndarray,  # 436 x 436A
    new_district_codes,
) -> pd.DataFrame:
    """Transforms data from pre- to post- 2020 census districts"""
    df = input_df.copy()
    old_districts_df = df[df["ucgid_str"].str.startswith("5001800US")].copy()
    old_districts_df = old_districts_df.sort_values("ucgid_str").reset_index(
        drop=True
    )
    old_values = old_districts_df["target_value"].to_numpy()
    new_values = mapping_matrix.T @ old_values

    # Create a new DataFrame for the transformed data, preserving the original schema.
    new_districts_df = pd.DataFrame(
        {
            "ucgid_str": new_district_codes,
            "breakdown_variable": old_districts_df["breakdown_variable"],
            "breakdown_value": old_districts_df["breakdown_value"],
            "target_variable": old_districts_df["target_variable"],
            "target_value": new_values,
        }
    )

    other_geos_df = df[~df["ucgid_str"].str.startswith("5001800US")].copy()

    final_df = pd.concat([other_geos_df, new_districts_df], ignore_index=True)

    return final_df


def extract_soi_data() -> pd.DataFrame:
    """Download and save congressional district AGI totals.

    In the file below, "22" is 2022, "in" is individual returns,
    "cd" is congressional districts
    """
    return pd.read_csv("https://www.irs.gov/pub/irs-soi/22incd.csv")


def transform_soi_data(raw_df):

    TARGETS = [
        dict(code="59661", name="eitc", breakdown=("eitc_child_count", 0)),
        dict(code="59662", name="eitc", breakdown=("eitc_child_count", 1)),
        dict(code="59663", name="eitc", breakdown=("eitc_child_count", 2)),
        dict(code="59664", name="eitc", breakdown=("eitc_child_count", "3+")),
        dict(
            code="04475",
            name="qualified_business_income_deduction",
            breakdown=None,
        ),
        dict(code="18500", name="real_estate_taxes", breakdown=None),
        dict(code="01000", name="net_capital_gain", breakdown=None),
        dict(code="01400", name="taxable_ira_distributions", breakdown=None),
        dict(code="00300", name="taxable_interest_income", breakdown=None),
        dict(code="00400", name="tax_exempt_interest_income", breakdown=None),
        dict(code="00600", name="dividend_income", breakdown=None),
        dict(code="00650", name="qualified_dividend_income", breakdown=None),
        dict(
            code="26270",
            name="tax_unit_partnership_s_corp_income",
            breakdown=None,
        ),
        dict(code="02500", name="taxable_social_security", breakdown=None),
        dict(code="02300", name="unemployment_compensation", breakdown=None),
        dict(code="17000", name="medical_expense_deduction", breakdown=None),
        dict(code="01700", name="taxable_pension_income", breakdown=None),
        dict(code="11070", name="refundable_ctc", breakdown=None),
        dict(code="18425", name="salt", breakdown=None),
        dict(code="06500", name="income_tax", breakdown=None),
    ]

    # National ---------------
    national_df = raw_df.copy().loc[(raw_df.STATE == "US")]
    national_df["ucgid_str"] = "0100000US"

    # State -------------------
    # You've got agi_stub == 0 in here, which you want to use any time you don't want to
    # divide data by AGI classes (i.e., agi_stub)
    state_df = raw_df.copy().loc[
        (raw_df.STATE != "US") & (raw_df.CONG_DISTRICT == 0)
    ]
    state_df["ucgid_str"] = "0400000US" + state_df["STATEFIPS"].astype(
        str
    ).str.zfill(2)

    # District ------------------
    district_df = raw_df.copy().loc[(raw_df.CONG_DISTRICT > 0)]

    max_cong_district_by_state = raw_df.groupby("STATE")[
        "CONG_DISTRICT"
    ].transform("max")
    district_df = raw_df.copy().loc[
        (raw_df["CONG_DISTRICT"] > 0) | (max_cong_district_by_state == 0)
    ]
    district_df = district_df.loc[district_df["STATE"] != "US"]
    district_df["STATEFIPS"] = (
        district_df["STATEFIPS"].astype(int).astype(str).str.zfill(2)
    )
    district_df["CONG_DISTRICT"] = (
        district_df["CONG_DISTRICT"].astype(int).astype(str).str.zfill(2)
    )
    district_df["ucgid_str"] = (
        "5001800US" + district_df["STATEFIPS"] + district_df["CONG_DISTRICT"]
    )
    district_df = district_df[~district_df["ucgid_str"].isin(TERRITORY_UCGIDS)]

    assert district_df.shape[0] % 436 == 0

    all_df = pd.concat([national_df, state_df, district_df])

    # "Marginal" over AGI bands, which this data set is organized according to
    all_marginals = all_df.copy().loc[all_df.agi_stub == 0]
    assert all_marginals.shape[0] == 436 + 51 + 1

    # Collect targets from the SOI file
    records = []
    for spec in TARGETS:
        count_col = f"N{spec['code']}"  # e.g. 'N59661'
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
            count_col=count_col,
            amount_col=amount_col,
            amount_name=spec["name"],
            breakdown_col=breakdown_col,
            multiplier=1_000,
        )
        records.extend([rec_counts, rec_amounts])

    # AGI Processing (separate, doesn't have a count column)
    temp_df = df[["ucgid_str"]].copy()
    temp_df["breakdown_variable"] = "one"
    temp_df["breakdown_value"] = 1
    temp_df["target_variable"] = "adjusted_gross_income"
    temp_df["target_value"] = df["A00100"] * 1_000

    records.append(temp_df)

    # Note: national counts only have agi_stub = 0
    all_agi_splits = all_df.copy().loc[all_df.agi_stub != 0]
    assert all_agi_splits.shape[0] % (436 + 51 + 0) == 0

    agi_long_records = make_agi_long(all_agi_splits)

    records.extend(agi_long_records)

    # Pre- to Post- 2020 Census redisticting
    mapping = get_district_mapping()
    converted = [
        convert_district_data(
            r, mapping["mapping_matrix"], mapping["new_codes"]
        )
        for r in records
    ]

    return converted


def load_soi_data(long_dfs, year):
    """Load a list of databases into the db, critically dependent on order"""

    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    session = Session(engine)
    
    # Fetch existing geographic strata
    geo_strata = get_geographic_strata(session)

    # Load EITC data --------------------------------------------------------
    eitc_data = {
        "0": (long_dfs[0], long_dfs[1]),
        "1": (long_dfs[2], long_dfs[3]),
        "2": (long_dfs[4], long_dfs[5]),
        "3+": (long_dfs[6], long_dfs[7]),
    }

    eitc_stratum_lookup = {"national": {}, "state": {}, "district": {}}
    for n_children in eitc_data.keys():
        eitc_count_i, eitc_amount_i = eitc_data[n_children]
        for i in range(eitc_count_i.shape[0]):
            ucgid_i = eitc_count_i[["ucgid_str"]].iloc[i].values[0]
            geo_info = parse_ucgid(ucgid_i)
            
            # Determine parent stratum based on geographic level
            if geo_info["type"] == "national":
                parent_stratum_id = geo_strata["national"]
                note = f"National EITC received with {n_children} children"
                constraints = []
            elif geo_info["type"] == "state":
                parent_stratum_id = geo_strata["state"][geo_info["state_fips"]]
                note = f"State FIPS {geo_info['state_fips']} EITC received with {n_children} children"
                constraints = [
                    StratumConstraint(
                        constraint_variable="state_fips",
                        operation="==",
                        value=str(geo_info["state_fips"]),
                    )
                ]
            elif geo_info["type"] == "district":
                parent_stratum_id = geo_strata["district"][geo_info["congressional_district_geoid"]]
                note = f"Congressional District {geo_info['congressional_district_geoid']} EITC received with {n_children} children"
                constraints = [
                    StratumConstraint(
                        constraint_variable="congressional_district_geoid",
                        operation="==",
                        value=str(geo_info["congressional_district_geoid"]),
                    )
                ]

            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=0,  # IRS SOI strata group
                notes=note,
            )
            
            new_stratum.constraints_rel = constraints
            if n_children == "3+":
                new_stratum.constraints_rel.append(
                    StratumConstraint(
                        constraint_variable="eitc_child_count",
                        operation=">",
                        value="2",
                    )
                )
            else:
                new_stratum.constraints_rel.append(
                    StratumConstraint(
                        constraint_variable="eitc_child_count",
                        operation="==",
                        value=f"{n_children}",
                    )
                )

            new_stratum.targets_rel = [
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

            # Store lookup for later use
            if geo_info["type"] == "national":
                eitc_stratum_lookup["national"][n_children] = new_stratum.stratum_id
            elif geo_info["type"] == "state":
                key = (geo_info["state_fips"], n_children)
                eitc_stratum_lookup["state"][key] = new_stratum.stratum_id
            elif geo_info["type"] == "district":
                key = (geo_info["congressional_district_geoid"], n_children)
                eitc_stratum_lookup["district"][key] = new_stratum.stratum_id

    session.commit()

    # There are no breakdown variables used in the following set
    first_agi_index = [
        i
        for i in range(len(long_dfs))
        if long_dfs[i][["target_variable"]].values[0]
        == "adjusted_gross_income"
        and long_dfs[i][["breakdown_variable"]].values[0] == "one"
    ][0]
    for j in range(8, first_agi_index, 2):
        count_j, amount_j = long_dfs[j], long_dfs[j + 1]
        amount_variable_name = amount_j.iloc[0][["target_variable"]].values[0]
        print(
            f"Loading amount data for IRS SOI data on {amount_variable_name}"
        )
        for i in range(count_j.shape[0]):
            ucgid_i = count_j[["ucgid_str"]].iloc[i].values[0]
            geo_info = parse_ucgid(ucgid_i)

            # Add target to existing geographic stratum
            if geo_info["type"] == "national":
                stratum = session.get(Stratum, geo_strata["national"])
            elif geo_info["type"] == "state":
                stratum = session.get(Stratum, geo_strata["state"][geo_info["state_fips"]])
            elif geo_info["type"] == "district":
                stratum = session.get(Stratum, geo_strata["district"][geo_info["congressional_district_geoid"]])
            
            amount_value = amount_j.iloc[i][["target_value"]].values[0]

            stratum.targets_rel.append(
                Target(
                    variable=amount_variable_name,
                    period=year,
                    value=amount_value,
                    source_id=5,
                    active=True,
                )
            )

            session.add(stratum)
            session.flush()

    session.commit()

    # Adjusted Gross Income ------
    agi_values = long_dfs[first_agi_index]
    assert agi_values[["target_variable"]].values[0] == "adjusted_gross_income"

    for i in range(agi_values.shape[0]):
        ucgid_i = agi_values[["ucgid_str"]].iloc[i].values[0]
        geo_info = parse_ucgid(ucgid_i)
        
        # Add target to existing geographic stratum
        if geo_info["type"] == "national":
            stratum = session.get(Stratum, geo_strata["national"])
        elif geo_info["type"] == "state":
            stratum = session.get(Stratum, geo_strata["state"][geo_info["state_fips"]])
        elif geo_info["type"] == "district":
            stratum = session.get(Stratum, geo_strata["district"][geo_info["congressional_district_geoid"]])
        
        stratum.targets_rel.append(
            Target(
                variable="adjusted_gross_income",
                period=year,
                value=agi_values.iloc[i][["target_value"]].values[0],
                source_id=5,
                active=True,
            )
        )
        session.add(stratum)
        session.flush()

    session.commit()

    agi_person_count_dfs = [
        df
        for df in long_dfs[(first_agi_index + 1) :]
        if df["target_variable"].iloc[0] == "person_count"
    ]

    for agi_df in agi_person_count_dfs:
        agi_stub = agi_df.iloc[0][["breakdown_value"]].values[0]
        agi_income_lower, agi_income_upper = AGI_STUB_TO_INCOME_RANGE[agi_stub]

        # Make a National Stratum for each AGI Stub even w/o associated national target
        note = f"National, AGI >= {agi_income_lower}, AGI < {agi_income_upper}"
        nat_stratum = Stratum(
            parent_stratum_id=geo_strata["national"],
            stratum_group_id=0,  # IRS SOI strata group
            notes=note
        )
        nat_stratum.constraints_rel.extend(
            [
                StratumConstraint(
                    constraint_variable="adjusted_gross_income",
                    operation=">=",
                    value=str(agi_income_lower),
                ),
                StratumConstraint(
                    constraint_variable="adjusted_gross_income",
                    operation="<",
                    value=str(agi_income_upper),
                ),
            ]
        )
        session.add(nat_stratum)
        session.flush()

        agi_stratum_lookup = {
            "national": nat_stratum.stratum_id,
            "state": {},
            "district": {},
        }
        for i in range(agi_df.shape[0]):
            ucgid_i = agi_df[["ucgid_str"]].iloc[i].values[0]
            geo_info = parse_ucgid(ucgid_i)
            person_count = agi_df.iloc[i][["target_value"]].values[0]

            if geo_info["type"] == "state":
                parent_stratum_id = geo_strata["state"][geo_info["state_fips"]]
                note = f"State FIPS {geo_info['state_fips']}, AGI >= {agi_income_lower}, AGI < {agi_income_upper}"
                constraints = [
                    StratumConstraint(
                        constraint_variable="state_fips",
                        operation="==",
                        value=str(geo_info["state_fips"]),
                    )
                ]
            elif geo_info["type"] == "district":
                parent_stratum_id = geo_strata["district"][geo_info["congressional_district_geoid"]]
                note = f"Congressional District {geo_info['congressional_district_geoid']}, AGI >= {agi_income_lower}, AGI < {agi_income_upper}"
                constraints = [
                    StratumConstraint(
                        constraint_variable="congressional_district_geoid",
                        operation="==",
                        value=str(geo_info["congressional_district_geoid"]),
                    )
                ]
            else:
                continue  # Skip if not state or district (shouldn't happen, but defensive)
            
            new_stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                stratum_group_id=0,  # IRS SOI strata group
                notes=note,
            )
            new_stratum.constraints_rel = constraints
            new_stratum.constraints_rel.extend(
                [
                    StratumConstraint(
                        constraint_variable="adjusted_gross_income",
                        operation=">=",
                        value=str(agi_income_lower),
                    ),
                    StratumConstraint(
                        constraint_variable="adjusted_gross_income",
                        operation="<",
                        value=str(agi_income_upper),
                    ),
                ]
            )
            new_stratum.targets_rel.append(
                Target(
                    variable="person_count",
                    period=year,
                    value=person_count,
                    source_id=5,
                    active=True,
                )
            )

            session.add(new_stratum)
            session.flush()

            if geo_info["type"] == "state":
                agi_stratum_lookup["state"][geo_info["state_fips"]] = new_stratum.stratum_id
            elif geo_info["type"] == "district":
                agi_stratum_lookup["district"][geo_info["congressional_district_geoid"]] = new_stratum.stratum_id

    session.commit()


def main():
    # NOTE: predates the finalization of the 2020 Census redistricting
    # and there is district mapping in the Transform step
    year = 2022

    # Extract -----------------------
    raw_df = extract_soi_data()

    # Transform ---------------------
    long_dfs = transform_soi_data(raw_df)

    # Load ---------------------
    load_soi_data(long_dfs, year)


if __name__ == "__main__":
    main()
