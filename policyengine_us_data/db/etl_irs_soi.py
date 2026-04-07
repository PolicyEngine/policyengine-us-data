import logging
from typing import Optional

import numpy as np
import pandas as pd

from sqlmodel import Session, create_engine

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.db import (
    parse_ucgid,
    get_geographic_strata,
    etl_argparser,
)
from policyengine_us_data.utils.census import TERRITORY_UCGIDS
from policyengine_us_data.storage.calibration_targets.make_district_mapping import (
    get_district_mapping,
)
from policyengine_us_data.storage.calibration_targets.soi_metadata import (
    LATEST_PUBLISHED_NATIONAL_SOI_YEAR,
    LATEST_PUBLISHED_GEOGRAPHIC_SOI_YEAR,
)
from policyengine_us_data.utils.raw_cache import (
    is_cached,
    cache_path,
    save_bytes,
)
from policyengine_us_data.utils.soi import get_tracked_soi_row
from policyengine_us_data.storage.calibration_targets.pull_soi_targets import (
    STATE_ABBR_TO_FIPS,
)
from policyengine_us_data.storage.calibration_targets.refresh_soi_table_targets import (
    _load_workbook,
    _scaled_cell,
)

logger = logging.getLogger(__name__)

ITEMIZED_DEDUCTION_VARIABLES = {
    "salt",
    "real_estate_taxes",
    "medical_expense_deduction",
}

# IRS SOI data is typically available ~2 years after the tax year
IRS_SOI_LAG_YEARS = 2

"""See the 22incddocguide.docx manual from the IRS SOI"""
# Language in the doc: '$10,000 under $25,000' means >= $10,000 and < $25,000
AGI_STUB_TO_INCOME_RANGE = {
    1: (-np.inf, 1),  # Under $1 (negative AGI allowed)
    2: (1, 10_000),  # $1 under $10,000
    3: (10_000, 25_000),  # $10,000 under $25,000
    4: (25_000, 50_000),  # $25,000 under $50,000
    5: (50_000, 75_000),  # $50,000 under $75,000
    6: (75_000, 100_000),  # $75,000 under $100,000
    7: (100_000, 200_000),  # $100,000 under $200,000
    8: (200_000, 500_000),  # $200,000 under $500,000
    9: (500_000, np.inf),  # $500,000 or more
}

STATE_FINE_AGI_STUBS = {
    9: (500_000, 1_000_000),  # $500,000 under $1,000,000
    10: (1_000_000, np.inf),  # $1,000,000 or more
}

NATIONAL_FINE_AGI_BRACKETS = {
    23: (500_000, 1_000_000),  # Table 1.4 row 23
    24: (1_000_000, 1_500_000),  # row 24
    25: (1_500_000, 2_000_000),  # row 25
    26: (2_000_000, 5_000_000),  # row 26
    27: (5_000_000, 10_000_000),  # row 27
    28: (10_000_000, np.inf),  # row 28
}


def _skip_coarse_state_agi_person_count_target(geo_type: str, agi_stub: int) -> bool:
    """Skip the coarse state 500k+ count target when fine state bins are loaded.

    The standard geography-file SOI feed only has a top-coded state AGI stub 9
    (500k+). We separately load `in55cmcsv`, which splits that state tail into
    500k-1m and 1m+. Keeping the coarse state count target alongside the fine
    rows would double-constrain the same top-tail population in calibration.
    """

    return geo_type == "state" and agi_stub == 9


# These variables map cleanly from Publication 1304 aggregate tables to the
# existing national IRS-SOI domain strata. We intentionally leave `aca_ptc`
# and `refundable_ctc` on the geography-file path for now because the
# published 2023 workbook tables do not line up one-for-one with the current
# `incd` national codes.
WORKBOOK_NATIONAL_DOMAIN_TARGETS = {
    "dividend_income": "ordinary_dividends",
    "income_tax_before_credits": "income_tax_before_credits",
    "net_capital_gains": "capital_gains_gross",
    "qualified_dividend_income": "qualified_dividends",
    "rental_income": "rent_and_royalty_net_income",
    "self_employment_income": "business_net_profits",
    "tax_exempt_interest_income": "exempt_interest",
    "tax_unit_partnership_s_corp_income": "partnership_and_s_corp_income",
    "taxable_interest_income": "taxable_interest_income",
    "taxable_ira_distributions": "ira_distributions",
    "taxable_pension_income": "taxable_pension_income",
    "taxable_social_security": "taxable_social_security",
    "unemployment_compensation": "unemployment_compensation",
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
    """
    Create standardized records from IRS SOI data.

    IMPORTANT DATA INCONSISTENCY (discovered 2024-12):
    The IRS SOI documentation states "money amounts are reported in thousands of dollars."
    This is true for almost all columns EXCEPT A59664 (EITC with 3+ children amount),
    which is already in dollars, not thousands. This appears to be a data quality issue
    in the IRS SOI file itself. We handle this special case below.
    """
    df = df.rename(
        {count_col: "tax_unit_count", amount_col: amount_name}, axis=1
    ).copy()

    if breakdown_col is None:
        breakdown_col = "one"
        df[breakdown_col] = 1

    rec_counts = create_records(df, breakdown_col, "tax_unit_count")
    rec_amounts = create_records(df, breakdown_col, amount_name)

    # SPECIAL CASE: A59664 (EITC with 3+ children) is already in dollars, not thousands!
    # All other EITC amounts (A59661-A59663) are correctly in thousands.
    # This was verified by checking that A59660 (total EITC) equals the sum only when
    # A59664 is treated as already being in dollars.
    if amount_col == "A59664":
        # Check if IRS has fixed the data inconsistency
        # If values are < 10 million, they're likely already in thousands (fixed)
        max_value = rec_amounts["target_value"].max()
        if max_value < 10_000_000:
            print(
                f"WARNING: A59664 values appear to be in thousands (max={max_value:,.0f})"
            )
            print("The IRS may have fixed their data inconsistency.")
            print("Please verify and remove the special case handling if confirmed.")
            # Don't apply the fix - data appears to already be in thousands
        else:
            # Convert from dollars to thousands to match other columns
            rec_amounts["target_value"] /= 1_000

    rec_amounts["target_value"] *= multiplier  # Apply standard multiplier
    # Note: tax_unit_count is the correct variable - the stratum constraints
    # indicate what is being counted (e.g., eitc > 0 for EITC recipients)

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
    old_districts_df = old_districts_df.sort_values("ucgid_str").reset_index(drop=True)
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


def _year_prefix(year: int) -> str:
    return f"{year % 100:02d}"


def extract_soi_data(year: int) -> pd.DataFrame:
    """Download and save congressional district AGI totals.

    In the file below, ``in`` is individual returns and ``cd`` is
    congressional districts.
    """
    year_prefix = _year_prefix(year)
    cache_file = f"irs_soi_{year_prefix}incd.csv"
    if is_cached(cache_file):
        logger.info(f"Using cached {cache_file}")
        df = pd.read_csv(cache_path(cache_file))
    else:
        url = f"https://www.irs.gov/pub/irs-soi/{year_prefix}incd.csv"
        import requests

        response = requests.get(url)
        response.raise_for_status()
        save_bytes(cache_file, response.content)
        df = pd.read_csv(cache_path(cache_file))

    # Validate EITC data consistency (check if IRS fixed the A59664 issue)
    us_data = df[(df["STATE"] == "US") & (df["agi_stub"] == 0)]
    if not us_data.empty and all(
        col in us_data.columns
        for col in ["A59660", "A59661", "A59662", "A59663", "A59664"]
    ):
        total_eitc = us_data["A59660"].values[0]
        sum_as_thousands = (
            us_data["A59661"].values[0]
            + us_data["A59662"].values[0]
            + us_data["A59663"].values[0]
            + us_data["A59664"].values[0]
        )
        sum_mixed = (
            us_data["A59661"].values[0]
            + us_data["A59662"].values[0]
            + us_data["A59663"].values[0]
            + us_data["A59664"].values[0] / 1000
        )

        # Check which interpretation matches the total
        if abs(total_eitc - sum_as_thousands) < 100:  # Within 100K (thousands)
            print("=" * 60)
            print("ALERT: IRS may have fixed the A59664 data inconsistency!")
            print(f"Total EITC (A59660): {total_eitc:,.0f}")
            print(f"Sum treating A59664 as thousands: {sum_as_thousands:,.0f}")
            print("These now match! Please verify and update the code.")
            print("=" * 60)
        elif abs(total_eitc - sum_mixed) < 100:
            print(
                "Note: A59664 still has the units inconsistency (in dollars, not thousands)"
            )

    return df


def _upsert_target(
    session: Session,
    *,
    stratum_id: int,
    variable: str,
    period: int,
    value: float,
    source: str,
    notes: Optional[str] = None,
) -> None:
    existing_target = (
        session.query(Target)
        .filter(
            Target.stratum_id == stratum_id,
            Target.variable == variable,
            Target.period == period,
            Target.reform_id == 0,
        )
        .first()
    )
    if existing_target:
        existing_target.value = value
        existing_target.source = source
        if notes is not None:
            existing_target.notes = notes
        return

    session.add(
        Target(
            stratum_id=stratum_id,
            variable=variable,
            period=period,
            value=value,
            active=True,
            source=source,
            notes=notes,
        )
    )


def _get_or_create_national_domain_stratum(
    session: Session, national_filer_stratum_id: int, variable: str
) -> Stratum:
    note = f"National filers with {variable} > 0"
    stratum = (
        session.query(Stratum)
        .filter(
            Stratum.parent_stratum_id == national_filer_stratum_id,
            Stratum.notes == note,
        )
        .first()
    )
    if stratum:
        return stratum

    stratum = Stratum(
        parent_stratum_id=national_filer_stratum_id,
        notes=note,
    )
    stratum.constraints_rel.extend(
        [
            StratumConstraint(
                constraint_variable="tax_unit_is_filer",
                operation="==",
                value="1",
            ),
            StratumConstraint(
                constraint_variable=variable,
                operation=">",
                value="0",
            ),
        ]
    )
    session.add(stratum)
    session.flush()
    return stratum


def load_national_workbook_soi_targets(
    session: Session, national_filer_stratum_id: int, target_year: int
) -> None:
    agi_row = get_tracked_soi_row("adjusted_gross_income", target_year, count=False)
    agi_period = int(agi_row["Year"])
    _upsert_target(
        session,
        stratum_id=national_filer_stratum_id,
        variable="adjusted_gross_income",
        period=agi_period,
        value=float(agi_row["Value"]),
        source="IRS SOI",
        notes=f"Publication 1304 {agi_row['SOI table']} aggregate target",
    )

    for pe_variable, soi_variable in WORKBOOK_NATIONAL_DOMAIN_TARGETS.items():
        amount_row = get_tracked_soi_row(soi_variable, target_year, count=False)
        count_row = get_tracked_soi_row(soi_variable, target_year, count=True)
        period = int(amount_row["Year"])
        if period != int(count_row["Year"]):
            raise ValueError(
                f"Count and amount source years differ for {pe_variable}: "
                f"{count_row['Year']} vs {amount_row['Year']}"
            )

        stratum = _get_or_create_national_domain_stratum(
            session, national_filer_stratum_id, pe_variable
        )
        notes = f"Publication 1304 {amount_row['SOI table']} aggregate target"
        _upsert_target(
            session,
            stratum_id=stratum.stratum_id,
            variable="tax_unit_count",
            period=period,
            value=float(count_row["Value"]),
            source="IRS SOI",
            notes=notes,
        )
        _upsert_target(
            session,
            stratum_id=stratum.stratum_id,
            variable=pe_variable,
            period=period,
            value=float(amount_row["Value"]),
            source="IRS SOI",
            notes=notes,
        )


def extract_state_fine_agi_data(year: int) -> pd.DataFrame:
    """Download the state-level SOI file (in55cmcsv) with stubs 9 and 10."""
    year_prefix = _year_prefix(year)
    cache_file = f"irs_soi_{year_prefix}in55cmcsv.csv"
    if is_cached(cache_file):
        logger.info(f"Using cached {cache_file}")
        df = pd.read_csv(cache_path(cache_file), thousands=",")
    else:
        import requests

        url = f"https://www.irs.gov/pub/irs-soi/{year_prefix}in55cmcsv.csv"
        response = requests.get(url)
        response.raise_for_status()
        save_bytes(cache_file, response.content)
        df = pd.read_csv(cache_path(cache_file), thousands=",")

    df = df[df["AGI_STUB"].isin(STATE_FINE_AGI_STUBS.keys())]
    df = df[df["STATE"].isin(STATE_ABBR_TO_FIPS.keys())]
    return df


def load_state_fine_agi_targets(
    session: Session, filer_strata: dict, year: int
) -> None:
    """Create strata and targets for state-level fine AGI brackets (stubs 9/10)."""
    df = extract_state_fine_agi_data(year)

    for _, row in df.iterrows():
        state_abbr = row["STATE"]
        stub = int(row["AGI_STUB"])
        fips_str = STATE_ABBR_TO_FIPS[state_abbr]
        fips_int = int(fips_str)
        lower, upper = STATE_FINE_AGI_STUBS[stub]

        parent_stratum_id = filer_strata["state"][fips_int]
        note = f"State FIPS {fips_int} filers, AGI >= {lower}, AGI < {upper}"

        existing = (
            session.query(Stratum)
            .filter(
                Stratum.parent_stratum_id == parent_stratum_id,
                Stratum.notes == note,
            )
            .first()
        )

        if existing:
            stratum = existing
        else:
            stratum = Stratum(
                parent_stratum_id=parent_stratum_id,
                notes=note,
            )
            stratum.constraints_rel.extend(
                [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    ),
                    StratumConstraint(
                        constraint_variable="state_fips",
                        operation="==",
                        value=str(fips_int),
                    ),
                    StratumConstraint(
                        constraint_variable="adjusted_gross_income",
                        operation=">=",
                        value=str(lower),
                    ),
                    StratumConstraint(
                        constraint_variable="adjusted_gross_income",
                        operation="<",
                        value=str(upper),
                    ),
                ]
            )
            session.add(stratum)
            session.flush()

        person_count = float(row["N2"])
        agi_amount = float(row["A00100"]) * 1000

        _upsert_target(
            session,
            stratum_id=stratum.stratum_id,
            variable="person_count",
            period=year,
            value=person_count,
            source="IRS SOI",
            notes=f"State fine AGI stub {stub} from in55cmcsv",
        )
        _upsert_target(
            session,
            stratum_id=stratum.stratum_id,
            variable="adjusted_gross_income",
            period=year,
            value=agi_amount,
            source="IRS SOI",
            notes=f"State fine AGI stub {stub} from in55cmcsv",
        )


def load_national_fine_agi_targets(
    session: Session, national_filer_stratum_id: int, target_year: int
) -> None:
    """Create strata and targets for national fine AGI brackets from Table 1.4."""
    workbook = _load_workbook("Table 1.4", target_year)

    for excel_row, (lower, upper) in NATIONAL_FINE_AGI_BRACKETS.items():
        note = f"National filers, AGI >= {lower}, AGI < {upper}"

        existing = (
            session.query(Stratum)
            .filter(
                Stratum.parent_stratum_id == national_filer_stratum_id,
                Stratum.notes == note,
            )
            .first()
        )

        if existing:
            stratum = existing
        else:
            stratum = Stratum(
                parent_stratum_id=national_filer_stratum_id,
                notes=note,
            )
            stratum.constraints_rel.extend(
                [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    ),
                    StratumConstraint(
                        constraint_variable="adjusted_gross_income",
                        operation=">=",
                        value=str(lower),
                    ),
                    StratumConstraint(
                        constraint_variable="adjusted_gross_income",
                        operation="<",
                        value=str(upper),
                    ),
                ]
            )
            session.add(stratum)
            session.flush()

        count_value = _scaled_cell(workbook, excel_row, "B", is_count=True)
        agi_value = _scaled_cell(workbook, excel_row, "C", is_count=False)

        _upsert_target(
            session,
            stratum_id=stratum.stratum_id,
            variable="tax_unit_count",
            period=target_year,
            value=count_value,
            source="IRS SOI",
            notes=f"Table 1.4 row {excel_row} fine AGI bracket",
        )
        _upsert_target(
            session,
            stratum_id=stratum.stratum_id,
            variable="adjusted_gross_income",
            period=target_year,
            value=agi_value,
            source="IRS SOI",
            notes=f"Table 1.4 row {excel_row} fine AGI bracket",
        )


def transform_soi_data(raw_df):

    TARGETS = [
        dict(code="59661", name="eitc", breakdown=("eitc_child_count", 0)),
        dict(code="59662", name="eitc", breakdown=("eitc_child_count", 1)),
        dict(code="59663", name="eitc", breakdown=("eitc_child_count", 2)),
        dict(
            code="59664", name="eitc", breakdown=("eitc_child_count", "3+")
        ),  # Doc says "three" but data shows this is 3+
        dict(
            code="04475",
            name="qualified_business_income_deduction",
            breakdown=None,
        ),
        dict(code="00900", name="self_employment_income", breakdown=None),
        dict(
            code="01000", name="net_capital_gains", breakdown=None
        ),  # Not to be confused with the always positive net_capital_gain
        dict(code="18500", name="real_estate_taxes", breakdown=None),
        dict(code="25870", name="rental_income", breakdown=None),
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
        dict(code="05800", name="income_tax_before_credits", breakdown=None),
        dict(code="85530", name="aca_ptc", breakdown=None),
    ]

    # National ---------------
    national_df = raw_df.copy().loc[(raw_df.STATE == "US")]
    national_df["ucgid_str"] = "0100000US"

    # State -------------------
    # You've got agi_stub == 0 in here, which you want to use any time you don't want to
    # divide data by AGI classes (i.e., agi_stub)
    state_df = raw_df.copy().loc[(raw_df.STATE != "US") & (raw_df.CONG_DISTRICT == 0)]
    state_df["ucgid_str"] = "0400000US" + state_df["STATEFIPS"].astype(str).str.zfill(2)

    # District ------------------
    district_df = raw_df.copy().loc[(raw_df.CONG_DISTRICT > 0)]

    max_cong_district_by_state = raw_df.groupby("STATE")["CONG_DISTRICT"].transform(
        "max"
    )
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
        convert_district_data(r, mapping["mapping_matrix"], mapping["new_codes"])
        for r in records
    ]

    return converted


def load_soi_data(long_dfs, year, national_year: Optional[int] = None):
    """Load a list of databases into the db, critically dependent on order"""

    DATABASE_URL = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(DATABASE_URL)

    session = Session(engine)

    # Fetch existing geographic strata
    geo_strata = get_geographic_strata(session)

    # Create filer strata as intermediate layer between geographic and IRS-specific strata
    # All IRS data represents only tax filers, not the entire population
    filer_strata = {"national": None, "state": {}, "district": {}}

    # National filer stratum - check if it exists first
    national_filer_stratum = (
        session.query(Stratum)
        .filter(
            Stratum.parent_stratum_id == geo_strata["national"],
            Stratum.notes == "United States - Tax Filers",
        )
        .first()
    )

    if not national_filer_stratum:
        national_filer_stratum = Stratum(
            parent_stratum_id=geo_strata["national"],
            notes="United States - Tax Filers",
        )
        national_filer_stratum.constraints_rel = [
            StratumConstraint(
                constraint_variable="tax_unit_is_filer",
                operation="==",
                value="1",
            )
        ]
        session.add(national_filer_stratum)
        session.flush()

    filer_strata["national"] = national_filer_stratum.stratum_id

    # State filer strata
    for state_fips, state_geo_stratum_id in geo_strata["state"].items():
        # Check if state filer stratum exists
        state_filer_stratum = (
            session.query(Stratum)
            .filter(
                Stratum.parent_stratum_id == state_geo_stratum_id,
                Stratum.notes == f"State FIPS {state_fips} - Tax Filers",
            )
            .first()
        )

        if not state_filer_stratum:
            state_filer_stratum = Stratum(
                parent_stratum_id=state_geo_stratum_id,
                notes=f"State FIPS {state_fips} - Tax Filers",
            )
            state_filer_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="tax_unit_is_filer",
                    operation="==",
                    value="1",
                ),
                StratumConstraint(
                    constraint_variable="state_fips",
                    operation="==",
                    value=str(state_fips),
                ),
            ]
            session.add(state_filer_stratum)
            session.flush()

        filer_strata["state"][state_fips] = state_filer_stratum.stratum_id

    # District filer strata
    for district_geoid, district_geo_stratum_id in geo_strata["district"].items():
        # Check if district filer stratum exists
        district_filer_stratum = (
            session.query(Stratum)
            .filter(
                Stratum.parent_stratum_id == district_geo_stratum_id,
                Stratum.notes
                == f"Congressional District {district_geoid} - Tax Filers",
            )
            .first()
        )

        if not district_filer_stratum:
            district_filer_stratum = Stratum(
                parent_stratum_id=district_geo_stratum_id,
                notes=f"Congressional District {district_geoid} - Tax Filers",
            )
            district_filer_stratum.constraints_rel = [
                StratumConstraint(
                    constraint_variable="tax_unit_is_filer",
                    operation="==",
                    value="1",
                ),
                StratumConstraint(
                    constraint_variable="congressional_district_geoid",
                    operation="==",
                    value=str(district_geoid),
                ),
            ]
            session.add(district_filer_stratum)
            session.flush()

        filer_strata["district"][district_geoid] = district_filer_stratum.stratum_id

    if national_year is not None:
        load_national_workbook_soi_targets(
            session,
            filer_strata["national"],
            national_year,
        )
        load_national_fine_agi_targets(session, filer_strata["national"], national_year)

    load_state_fine_agi_targets(session, filer_strata, year)
    session.commit()

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

            # Determine parent stratum based on geographic level - use filer strata not geo strata
            if geo_info["type"] == "national":
                parent_stratum_id = filer_strata["national"]
                note = f"National EITC received with {n_children} children (filers)"
                constraints = [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    )
                ]
            elif geo_info["type"] == "state":
                parent_stratum_id = filer_strata["state"][geo_info["state_fips"]]
                note = f"State FIPS {geo_info['state_fips']} EITC received with {n_children} children (filers)"
                constraints = [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    ),
                    StratumConstraint(
                        constraint_variable="state_fips",
                        operation="==",
                        value=str(geo_info["state_fips"]),
                    ),
                ]
            elif geo_info["type"] == "district":
                parent_stratum_id = filer_strata["district"][
                    geo_info["congressional_district_geoid"]
                ]
                note = f"Congressional District {geo_info['congressional_district_geoid']} EITC received with {n_children} children (filers)"
                constraints = [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    ),
                    StratumConstraint(
                        constraint_variable="congressional_district_geoid",
                        operation="==",
                        value=str(geo_info["congressional_district_geoid"]),
                    ),
                ]

            # Check if stratum already exists
            existing_stratum = (
                session.query(Stratum)
                .filter(
                    Stratum.parent_stratum_id == parent_stratum_id,
                    Stratum.notes == note,
                )
                .first()
            )

            if existing_stratum:
                new_stratum = existing_stratum
            else:
                new_stratum = Stratum(
                    parent_stratum_id=parent_stratum_id,
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

                session.add(new_stratum)
                session.flush()

            # Get both count and amount values
            count_value = eitc_count_i.iloc[i][["target_value"]].values[0]
            amount_value = eitc_amount_i.iloc[i][["target_value"]].values[0]

            # Check if targets already exist and update or create them
            for variable, value in [
                ("tax_unit_count", count_value),
                ("eitc", amount_value),
            ]:
                existing_target = (
                    session.query(Target)
                    .filter(
                        Target.stratum_id == new_stratum.stratum_id,
                        Target.variable == variable,
                        Target.period == year,
                    )
                    .first()
                )

                if existing_target:
                    existing_target.value = value

                else:
                    new_stratum.targets_rel.append(
                        Target(
                            variable=variable,
                            period=year,
                            value=value,
                            active=True,
                            source="IRS SOI",
                        )
                    )

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
        if long_dfs[i][["target_variable"]].values[0] == "adjusted_gross_income"
        and long_dfs[i][["breakdown_variable"]].values[0] == "one"
    ][0]
    for j in range(8, first_agi_index, 2):
        count_j, amount_j = long_dfs[j], long_dfs[j + 1]
        count_variable_name = count_j.iloc[0][["target_variable"]].values[
            0
        ]  # Should be tax_unit_count
        amount_variable_name = amount_j.iloc[0][["target_variable"]].values[0]

        print(
            f"Loading count and amount data for IRS SOI data on {amount_variable_name}"
        )

        for i in range(count_j.shape[0]):
            ucgid_i = count_j[["ucgid_str"]].iloc[i].values[0]
            geo_info = parse_ucgid(ucgid_i)

            # Get parent filer stratum (not geographic stratum)
            if geo_info["type"] == "national":
                parent_stratum_id = filer_strata["national"]
                geo_description = "National"
            elif geo_info["type"] == "state":
                parent_stratum_id = filer_strata["state"][geo_info["state_fips"]]
                geo_description = f"State {geo_info['state_fips']}"
            elif geo_info["type"] == "district":
                parent_stratum_id = filer_strata["district"][
                    geo_info["congressional_district_geoid"]
                ]
                geo_description = f"CD {geo_info['congressional_district_geoid']}"

            # Create child stratum with constraint for this IRS variable
            # Note: This stratum will have the constraint that amount_variable > 0
            is_itemized = amount_variable_name in ITEMIZED_DEDUCTION_VARIABLES
            if is_itemized:
                note = f"{geo_description} itemizing filers with {amount_variable_name} > 0"
            else:
                note = f"{geo_description} filers with {amount_variable_name} > 0"

            # Check if child stratum already exists
            existing_stratum = (
                session.query(Stratum)
                .filter(
                    Stratum.parent_stratum_id == parent_stratum_id,
                    Stratum.notes == note,
                )
                .first()
            )

            if existing_stratum:
                child_stratum = existing_stratum
            else:
                # Create new child stratum with constraint
                child_stratum = Stratum(
                    parent_stratum_id=parent_stratum_id,
                    notes=note,
                )

                # Add constraints - filer status and this IRS variable must be positive
                child_stratum.constraints_rel.extend(
                    [
                        StratumConstraint(
                            constraint_variable="tax_unit_is_filer",
                            operation="==",
                            value="1",
                        ),
                        StratumConstraint(
                            constraint_variable=amount_variable_name,
                            operation=">",
                            value="0",
                        ),
                    ]
                )

                if is_itemized:
                    child_stratum.constraints_rel.append(
                        StratumConstraint(
                            constraint_variable="tax_unit_itemizes",
                            operation="==",
                            value="1",
                        )
                    )

                # Add geographic constraints if applicable
                if geo_info["type"] == "state":
                    child_stratum.constraints_rel.append(
                        StratumConstraint(
                            constraint_variable="state_fips",
                            operation="==",
                            value=str(geo_info["state_fips"]),
                        )
                    )
                elif geo_info["type"] == "district":
                    child_stratum.constraints_rel.append(
                        StratumConstraint(
                            constraint_variable="congressional_district_geoid",
                            operation="==",
                            value=str(geo_info["congressional_district_geoid"]),
                        )
                    )

                session.add(child_stratum)
                session.flush()

            count_value = count_j.iloc[i][["target_value"]].values[0]
            amount_value = amount_j.iloc[i][["target_value"]].values[0]

            # Check if targets already exist and update or create them
            for variable, value in [
                (count_variable_name, count_value),
                (amount_variable_name, amount_value),
            ]:
                existing_target = (
                    session.query(Target)
                    .filter(
                        Target.stratum_id == child_stratum.stratum_id,
                        Target.variable == variable,
                        Target.period == year,
                    )
                    .first()
                )

                if existing_target:
                    existing_target.value = value

                else:
                    child_stratum.targets_rel.append(
                        Target(
                            variable=variable,
                            period=year,
                            value=value,
                            active=True,
                            source="IRS SOI",
                        )
                    )

            session.add(child_stratum)
            session.flush()

    session.commit()

    # Adjusted Gross Income ------
    agi_values = long_dfs[first_agi_index]
    assert agi_values[["target_variable"]].values[0] == "adjusted_gross_income"

    for i in range(agi_values.shape[0]):
        ucgid_i = agi_values[["ucgid_str"]].iloc[i].values[0]
        geo_info = parse_ucgid(ucgid_i)

        # Add target to existing FILER stratum (not geographic stratum)
        if geo_info["type"] == "national":
            stratum = session.get(Stratum, filer_strata["national"])
        elif geo_info["type"] == "state":
            stratum = session.get(
                Stratum, filer_strata["state"][geo_info["state_fips"]]
            )
        elif geo_info["type"] == "district":
            stratum = session.get(
                Stratum,
                filer_strata["district"][geo_info["congressional_district_geoid"]],
            )

        # Check if target already exists
        existing_target = (
            session.query(Target)
            .filter(
                Target.stratum_id == stratum.stratum_id,
                Target.variable == "adjusted_gross_income",
                Target.period == year,
            )
            .first()
        )

        if existing_target:
            existing_target.value = agi_values.iloc[i][["target_value"]].values[0]
        else:
            stratum.targets_rel.append(
                Target(
                    variable="adjusted_gross_income",
                    period=year,
                    value=agi_values.iloc[i][["target_value"]].values[0],
                    active=True,
                    source="IRS SOI",
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
        note = f"National filers, AGI >= {agi_income_lower}, AGI < {agi_income_upper}"

        # Check if national AGI stratum already exists
        nat_stratum = (
            session.query(Stratum)
            .filter(
                Stratum.parent_stratum_id == filer_strata["national"],
                Stratum.notes == note,
            )
            .first()
        )

        if not nat_stratum:
            nat_stratum = Stratum(
                parent_stratum_id=filer_strata["national"],
                notes=note,
            )
            nat_stratum.constraints_rel.extend(
                [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    ),
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

            if _skip_coarse_state_agi_person_count_target(geo_info["type"], agi_stub):
                continue

            if geo_info["type"] == "state":
                parent_stratum_id = filer_strata["state"][geo_info["state_fips"]]
                note = f"State FIPS {geo_info['state_fips']} filers, AGI >= {agi_income_lower}, AGI < {agi_income_upper}"
                constraints = [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    ),
                    StratumConstraint(
                        constraint_variable="state_fips",
                        operation="==",
                        value=str(geo_info["state_fips"]),
                    ),
                ]
            elif geo_info["type"] == "district":
                parent_stratum_id = filer_strata["district"][
                    geo_info["congressional_district_geoid"]
                ]
                note = f"Congressional District {geo_info['congressional_district_geoid']} filers, AGI >= {agi_income_lower}, AGI < {agi_income_upper}"
                constraints = [
                    StratumConstraint(
                        constraint_variable="tax_unit_is_filer",
                        operation="==",
                        value="1",
                    ),
                    StratumConstraint(
                        constraint_variable="congressional_district_geoid",
                        operation="==",
                        value=str(geo_info["congressional_district_geoid"]),
                    ),
                ]
            else:
                continue  # Skip if not state or district (shouldn't happen, but defensive)

            # Check if stratum already exists
            existing_stratum = (
                session.query(Stratum)
                .filter(
                    Stratum.parent_stratum_id == parent_stratum_id,
                    Stratum.notes == note,
                )
                .first()
            )

            if existing_stratum:
                new_stratum = existing_stratum
            else:
                new_stratum = Stratum(
                    parent_stratum_id=parent_stratum_id,
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
                session.add(new_stratum)
                session.flush()

            # Check if target already exists and update or create it
            existing_target = (
                session.query(Target)
                .filter(
                    Target.stratum_id == new_stratum.stratum_id,
                    Target.variable == "person_count",
                    Target.period == year,
                )
                .first()
            )

            if existing_target:
                existing_target.value = person_count
            else:
                new_stratum.targets_rel.append(
                    Target(
                        variable="person_count",
                        period=year,
                        value=person_count,
                        active=True,
                        source="IRS SOI",
                    )
                )

            session.add(new_stratum)
            session.flush()

            if geo_info["type"] == "state":
                agi_stratum_lookup["state"][geo_info["state_fips"]] = (
                    new_stratum.stratum_id
                )
            elif geo_info["type"] == "district":
                agi_stratum_lookup["district"][
                    geo_info["congressional_district_geoid"]
                ] = new_stratum.stratum_id

    session.commit()


def main():
    def add_lag_arg(parser):
        parser.add_argument(
            "--lag",
            type=int,
            default=IRS_SOI_LAG_YEARS,
            help=(
                "Years to subtract from dataset year for IRS SOI data "
                "(default: %(default)s, since IRS data is ~2 years behind)"
            ),
        )

    args, dataset_year = etl_argparser(
        "ETL for IRS SOI calibration targets",
        extra_args_fn=add_lag_arg,
    )
    lagged_year = dataset_year - args.lag
    geography_year = min(lagged_year, LATEST_PUBLISHED_GEOGRAPHIC_SOI_YEAR)
    national_year = min(dataset_year, LATEST_PUBLISHED_NATIONAL_SOI_YEAR)
    print(f"IRS SOI geography year: {geography_year} (lag={args.lag})")
    print(f"IRS SOI national workbook year: {national_year}")
    if geography_year != lagged_year:
        print(
            "Clamped IRS SOI geography year to the latest published release: "
            f"{LATEST_PUBLISHED_GEOGRAPHIC_SOI_YEAR}"
        )

    # Extract -----------------------
    raw_df = extract_soi_data(geography_year)

    # Transform ---------------------
    long_dfs = transform_soi_data(raw_df)

    # Load ---------------------
    load_soi_data(long_dfs, geography_year, national_year=national_year)


if __name__ == "__main__":
    main()
