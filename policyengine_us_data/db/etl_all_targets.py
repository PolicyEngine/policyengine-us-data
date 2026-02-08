"""Comprehensive ETL that migrates ALL calibration targets from the
legacy ``loss.py`` system into the ``policy_data.db`` database.

This covers every target category that is NOT already handled by the
specialised ETLs (etl_national_targets, etl_irs_soi, etl_age,
etl_snap, etl_medicaid).

Categories handled here
-----------------------
1.  Census single-year age populations (86 bins)
2.  EITC by child count (returns + spending)
3.  SOI filer counts by AGI band (7 bands)
4.  Healthcare spending by age band (9 bands x 4 expense types)
5.  AGI by SPM threshold decile (10 deciles x 2 metrics)
6.  Negative household market income (total + count)
7.  Infant count
8.  Net worth
9.  State population (total + under-5)
10. Tax expenditure targets (SALT, medical, charitable, interest, QBI)
11. State real estate taxes (51 rows)
12. State ACA spending and enrollment (51 x 2)
13. State Medicaid enrollment (51 rows)
14. State 10-year age targets (50 states x 18 ranges = 900)
15. State AGI targets (918 rows)
16. SOI filing-status x AGI bin targets (1222 rows)
"""

import argparse
import logging

import numpy as np
import pandas as pd
from sqlmodel import Session, create_engine, select

from policyengine_us_data.storage import (
    STORAGE_FOLDER,
    CALIBRATION_FOLDER,
)
from policyengine_us_data.db.create_database_tables import (
    Source,
    SourceType,
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.utils.db_metadata import (
    get_or_create_source,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET = (
    "hf://policyengine/policyengine-us-data/"
    "calibration/stratified_extended_cps.h5"
)

# ------------------------------------------------------------------
# Hard-coded constants mirrored from loss.py
# ------------------------------------------------------------------

HARD_CODED_NATIONAL_TOTAL = 500e9  # real_estate_taxes national

SOI_FILER_COUNTS_2015 = {
    (-np.inf, 0): 2_072_066,
    (0, 5_000): 10_134_703,
    (5_000, 10_000): 11_398_595,
    (10_000, 25_000): 23_447_927,
    (25_000, 50_000): 23_727_745,
    (50_000, 100_000): 32_801_908,
    (100_000, np.inf): 25_120_985,
}

ITEMIZED_DEDUCTIONS = {
    "salt_deduction": 21.247e9,
    "medical_expense_deduction": 11.4e9,
    "charitable_deduction": 65.301e9,
    "interest_deduction": 24.8e9,
    "qualified_business_income_deduction": 63.1e9,
}

INFANTS_2023 = 3_491_679
INFANTS_2022 = 3_437_933

ACA_SPENDING_2024 = 9.8e10
NET_WORTH_2024 = 160e12


# ------------------------------------------------------------------
# Helper: format AGI bounds (mirrors loss.py ``fmt``)
# ------------------------------------------------------------------


def _fmt(x):
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    if x < 1e3:
        return f"{x:.0f}"
    if x < 1e6:
        return f"{x/1e3:.0f}k"
    if x < 1e9:
        return f"{x/1e6:.0f}m"
    return f"{x/1e9:.1f}bn"


# ------------------------------------------------------------------
# Extract helpers -- pure functions that read CSVs / constants and
# return plain Python data structures (no DB dependency).
# ------------------------------------------------------------------


def extract_census_age_populations(time_period: int):
    """Return list of 86 dicts with keys ``age`` and ``value``."""
    populations = pd.read_csv(CALIBRATION_FOLDER / "np2023_d5_mid.csv")
    populations = populations[
        (populations.SEX == 0) & (populations.RACE_HISP == 0)
    ]
    pop_cols = [f"POP_{i}" for i in range(86)]
    year_pops = (
        populations.groupby("YEAR").sum()[pop_cols].T[time_period].values
    )
    return [{"age": i, "value": float(year_pops[i])} for i in range(86)]


def extract_eitc_by_child_count():
    """Return list of 4 dicts (one per child bucket)."""
    df = pd.read_csv(CALIBRATION_FOLDER / "eitc.csv")
    return [
        {
            "count_children": int(row["count_children"]),
            "eitc_returns": float(row["eitc_returns"]),
            "eitc_total": float(row["eitc_total"]),
        }
        for _, row in df.iterrows()
    ]


def extract_soi_filer_counts():
    """Return list of 7 dicts (one per AGI band)."""
    return [
        {
            "agi_lower": lo,
            "agi_upper": hi,
            "filer_count_2015": count,
        }
        for (lo, hi), count in SOI_FILER_COUNTS_2015.items()
    ]


def extract_healthcare_by_age():
    """Return list of 9 dicts (one per 10-year age band)."""
    df = pd.read_csv(CALIBRATION_FOLDER / "healthcare_spending.csv")
    expense_cols = [
        "health_insurance_premiums_without_medicare_part_b",
        "over_the_counter_health_expenses",
        "other_medical_expenses",
        "medicare_part_b_premiums",
    ]
    records = []
    for _, row in df.iterrows():
        age_lower = int(row["age_10_year_lower_bound"])
        expenses = {c: float(row[c]) for c in expense_cols}
        records.append({"age_lower": age_lower, "expenses": expenses})
    return records


def extract_spm_threshold_agi():
    """Return list of 10 dicts (one per decile)."""
    df = pd.read_csv(CALIBRATION_FOLDER / "spm_threshold_agi.csv")
    return [
        {
            "decile": int(row["decile"]),
            "lower_spm_threshold": float(row["lower_spm_threshold"]),
            "upper_spm_threshold": float(row["upper_spm_threshold"]),
            "adjusted_gross_income": float(row["adjusted_gross_income"]),
            "count": float(row["count"]),
        }
        for _, row in df.iterrows()
    ]


def extract_negative_market_income():
    return {"total": -138e9, "count": 3e6}


def extract_infant_count():
    return INFANTS_2023 * (INFANTS_2023 / INFANTS_2022)


def extract_net_worth():
    return NET_WORTH_2024


def extract_state_population():
    df = pd.read_csv(CALIBRATION_FOLDER / "population_by_state.csv")
    return [
        {
            "state": row["state"],
            "population": float(row["population"]),
            "population_under_5": float(row["population_under_5"]),
        }
        for _, row in df.iterrows()
    ]


def extract_tax_expenditure_targets():
    return [
        {"variable": var, "value": val}
        for var, val in ITEMIZED_DEDUCTIONS.items()
    ]


def extract_state_real_estate_taxes():
    df = pd.read_csv(CALIBRATION_FOLDER / "real_estate_taxes_by_state_acs.csv")
    state_sum = df["real_estate_taxes_bn"].sum() * 1e9
    scale = HARD_CODED_NATIONAL_TOTAL / state_sum
    return [
        {
            "state_code": row["state_code"],
            "value": float(row["real_estate_taxes_bn"] * scale * 1e9),
        }
        for _, row in df.iterrows()
    ]


def extract_state_aca():
    df = pd.read_csv(
        CALIBRATION_FOLDER / "aca_spending_and_enrollment_2024.csv"
    )
    # Monthly to yearly, then scale to national target
    df["spending_annual"] = df["spending"] * 12
    spending_scale = ACA_SPENDING_2024 / df["spending_annual"].sum()
    df["spending_scaled"] = df["spending_annual"] * spending_scale
    return [
        {
            "state": row["state"],
            "spending": float(row["spending_scaled"]),
            "enrollment": float(row["enrollment"]),
        }
        for _, row in df.iterrows()
    ]


def extract_state_medicaid_enrollment():
    df = pd.read_csv(CALIBRATION_FOLDER / "medicaid_enrollment_2024.csv")
    return [
        {
            "state": row["state"],
            "enrollment": float(row["enrollment"]),
        }
        for _, row in df.iterrows()
    ]


def extract_state_10yr_age():
    df = pd.read_csv(CALIBRATION_FOLDER / "age_state.csv")
    records = []
    for _, row in df.iterrows():
        state = row["GEO_NAME"]
        for col in df.columns[2:]:  # skip GEO_ID, GEO_NAME
            records.append(
                {
                    "state": state,
                    "age_range": col,
                    "value": float(row[col]),
                }
            )
    return records


def extract_state_agi():
    df = pd.read_csv(CALIBRATION_FOLDER / "agi_state.csv")
    return [
        {
            "geo_name": row["GEO_NAME"],
            "agi_lower": float(row["AGI_LOWER_BOUND"]),
            "agi_upper": float(row["AGI_UPPER_BOUND"]),
            "value": float(row["VALUE"]),
            "is_count": bool(row["IS_COUNT"]),
            "variable": row["VARIABLE"],
        }
        for _, row in df.iterrows()
    ]


def _get_pe_variables():
    """Return the set of valid PolicyEngine US variable names."""
    try:
        from policyengine_us import CountryTaxBenefitSystem

        system = CountryTaxBenefitSystem()
        return set(system.variables.keys())
    except Exception:
        return None


# Map SOI variable names to PE variable names
_SOI_VAR_MAP = {
    "count": "tax_unit_count",
}


def extract_soi_filing_status_targets():
    df = pd.read_csv(CALIBRATION_FOLDER / "soi_targets.csv")
    filtered = df[
        (df["Taxable only"] == True)  # noqa: E712
        & (df["AGI upper bound"] > 10_000)
    ]

    pe_vars = _get_pe_variables()

    records = []
    for _, row in filtered.iterrows():
        var = row["Variable"]
        mapped = _SOI_VAR_MAP.get(var, var)
        if pe_vars is not None and mapped not in pe_vars:
            continue
        records.append(
            {
                "variable": mapped,
                "filing_status": row["Filing status"],
                "agi_lower": float(row["AGI lower bound"]),
                "agi_upper": float(row["AGI upper bound"]),
                "is_count": bool(row["Count"]),
                "taxable_only": bool(row["Taxable only"]),
                "value": float(row["Value"]),
            }
        )
    return records


# ------------------------------------------------------------------
# Load helper -- takes extracted data + engine and inserts into DB
# ------------------------------------------------------------------


def _get_or_create_stratum(
    session,
    parent_id,
    constraints,
    stratum_group_id,
    notes,
    category_tag=None,
):
    """Find an existing stratum by notes + parent, or create one.

    Parameters
    ----------
    category_tag : str, optional
        If given, an extra ``target_category == <tag>`` constraint
        is appended so the definition hash is unique across categories
        that otherwise share the same constraints.
    """
    existing = session.exec(
        select(Stratum).where(
            Stratum.parent_stratum_id == parent_id,
            Stratum.notes == notes,
        )
    ).first()
    if existing:
        return existing

    all_constraints = list(constraints)
    if category_tag:
        all_constraints.append(
            {
                "constraint_variable": "target_category",
                "operation": "==",
                "value": category_tag,
            }
        )

    stratum = Stratum(
        parent_stratum_id=parent_id,
        stratum_group_id=stratum_group_id,
        notes=notes,
    )
    stratum.constraints_rel = [StratumConstraint(**c) for c in all_constraints]
    session.add(stratum)
    session.flush()
    return stratum


def _upsert_target(
    session,
    stratum_id,
    variable,
    period,
    value,
    source_id,
    notes,
    reform_id=0,
):
    """Insert or update a target row."""
    existing = session.exec(
        select(Target).where(
            Target.stratum_id == stratum_id,
            Target.variable == variable,
            Target.period == period,
            Target.reform_id == reform_id,
        )
    ).first()
    if existing:
        existing.value = value
        existing.notes = notes
        existing.source_id = source_id
    else:
        t = Target(
            stratum_id=stratum_id,
            variable=variable,
            period=period,
            value=value,
            source_id=source_id,
            reform_id=reform_id,
            active=True,
            notes=notes,
        )
        session.add(t)


# Filing status mapping from SOI names to PE enum values
_FILING_STATUS_MAP = {
    "Single": "SINGLE",
    "Married Filing Jointly/Surviving Spouse": "JOINT",
    "Head of Household": "HEAD_OF_HOUSEHOLD",
    "Married Filing Separately": "SEPARATE",
    "All": None,
}


def load_all_targets(
    engine,
    time_period: int,
    root_stratum_id: int,
):
    """Load every target category into the database.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        Database engine (can be in-memory for tests).
    time_period : int
        Year for the targets (e.g. 2024).
    root_stratum_id : int
        ID of the national root stratum.
    """
    with Session(engine) as session:
        source = get_or_create_source(
            session,
            name="Legacy loss.py calibration targets",
            source_type=SourceType.HARDCODED,
            vintage=str(time_period),
            description=(
                "Comprehensive calibration targets migrated from "
                "the legacy build_loss_matrix() in loss.py"
            ),
        )
        sid = source.source_id

        # -- 1. Census single-year age populations (86 bins) --------
        age_pops = extract_census_age_populations(time_period)
        age_strata = {}  # age -> Stratum, reused for infant target
        for rec in age_pops:
            age = rec["age"]
            notes_str = f"Census age bin {age}"
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=[
                    {
                        "constraint_variable": "age",
                        "operation": ">=",
                        "value": str(age),
                    },
                    {
                        "constraint_variable": "age",
                        "operation": "<",
                        "value": str(age + 1),
                    },
                ],
                stratum_group_id=10,
                notes=notes_str,
            )
            age_strata[age] = stratum
            _upsert_target(
                session,
                stratum.stratum_id,
                "person_count",
                time_period,
                rec["value"],
                sid,
                notes=notes_str,
            )

        # -- 2. EITC by child count --------------------------------
        eitc_records = extract_eitc_by_child_count()
        for rec in eitc_records:
            cc = rec["count_children"]
            if cc < 2:
                op, val = "==", str(cc)
            else:
                op, val = ">=", str(cc)

            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=[
                    {
                        "constraint_variable": "eitc_child_count",
                        "operation": op,
                        "value": val,
                    },
                    {
                        "constraint_variable": "eitc",
                        "operation": ">",
                        "value": "0",
                    },
                ],
                stratum_group_id=11,
                notes=f"EITC {cc} children",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "tax_unit_count",
                time_period,
                rec["eitc_returns"],
                sid,
                notes=f"EITC returns, {cc} children",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "eitc",
                time_period,
                rec["eitc_total"],
                sid,
                notes=f"EITC spending, {cc} children",
            )

        # -- 3. SOI filer counts by AGI band -----------------------
        filer_counts = extract_soi_filer_counts()
        for rec in filer_counts:
            lo, hi = rec["agi_lower"], rec["agi_upper"]
            label = f"{_fmt(lo)}_{_fmt(hi)}"
            constraints = [
                {
                    "constraint_variable": "tax_unit_is_filer",
                    "operation": "==",
                    "value": "1",
                },
                {
                    "constraint_variable": "adjusted_gross_income",
                    "operation": ">=",
                    "value": str(lo),
                },
                {
                    "constraint_variable": "adjusted_gross_income",
                    "operation": "<",
                    "value": str(hi),
                },
            ]
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=constraints,
                stratum_group_id=12,
                notes=f"SOI filer count AGI {label}",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "tax_unit_count",
                time_period,
                rec["filer_count_2015"],
                sid,
                notes=f"SOI filer count AGI {label}",
            )

        # -- 4. Healthcare spending by age band --------------------
        hc_records = extract_healthcare_by_age()
        for rec in hc_records:
            age_lo = rec["age_lower"]
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=[
                    {
                        "constraint_variable": "age",
                        "operation": ">=",
                        "value": str(age_lo),
                    },
                    {
                        "constraint_variable": "age",
                        "operation": "<",
                        "value": str(age_lo + 10),
                    },
                ],
                stratum_group_id=13,
                notes=(f"Healthcare age {age_lo}-{age_lo + 9}"),
                category_tag="healthcare",
            )
            for var_name, amount in rec["expenses"].items():
                _upsert_target(
                    session,
                    stratum.stratum_id,
                    var_name,
                    time_period,
                    amount,
                    sid,
                    notes=(
                        f"Healthcare {var_name} " f"age {age_lo}-{age_lo + 9}"
                    ),
                )

        # -- 5. AGI by SPM threshold decile ------------------------
        spm_records = extract_spm_threshold_agi()
        for rec in spm_records:
            d = rec["decile"]
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=[
                    {
                        "constraint_variable": ("spm_unit_spm_threshold"),
                        "operation": ">=",
                        "value": str(rec["lower_spm_threshold"]),
                    },
                    {
                        "constraint_variable": ("spm_unit_spm_threshold"),
                        "operation": "<",
                        "value": str(rec["upper_spm_threshold"]),
                    },
                ],
                stratum_group_id=14,
                notes=f"SPM threshold decile {d}",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "adjusted_gross_income",
                time_period,
                rec["adjusted_gross_income"],
                sid,
                notes=f"SPM threshold decile {d} AGI",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "spm_unit_count",
                time_period,
                rec["count"],
                sid,
                notes=f"SPM threshold decile {d} count",
            )

        # -- 6. Negative household market income -------------------
        nmi = extract_negative_market_income()
        nmi_stratum = _get_or_create_stratum(
            session,
            parent_id=root_stratum_id,
            constraints=[
                {
                    "constraint_variable": ("household_market_income"),
                    "operation": "<",
                    "value": "0",
                },
            ],
            stratum_group_id=15,
            notes="Negative household market income",
        )
        _upsert_target(
            session,
            nmi_stratum.stratum_id,
            "household_market_income",
            time_period,
            nmi["total"],
            sid,
            notes="Negative household market income total",
        )
        _upsert_target(
            session,
            nmi_stratum.stratum_id,
            "household_count",
            time_period,
            nmi["count"],
            sid,
            notes="Negative household market income count",
        )

        # -- 7. Infant count ---------------------------------------
        # Reuse the age-0 stratum from census age populations
        # (same constraints: age >= 0, age < 1)
        # The infant target goes on the same stratum as census
        # age bin 0, but we use it to validate the estimate.
        # Since person_count for age 0 is already set from Census,
        # we update it with the projected infant count value.
        infant_count = extract_infant_count()
        infant_stratum = age_strata[0]
        _upsert_target(
            session,
            infant_stratum.stratum_id,
            "person_count",
            time_period,
            infant_count,
            sid,
            notes="Census age bin 0",
        )

        # -- 8. Net worth -----------------------------------------
        nw_value = extract_net_worth()
        _upsert_target(
            session,
            root_stratum_id,
            "net_worth",
            time_period,
            nw_value,
            sid,
            notes="Total household net worth (Fed Reserve)",
        )

        # -- 9. State population (total + under-5) -----------------
        state_pops = extract_state_population()
        for rec in state_pops:
            st = rec["state"]
            st_stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=[
                    {
                        "constraint_variable": "state_code",
                        "operation": "==",
                        "value": st,
                    },
                ],
                stratum_group_id=17,
                notes=f"State {st} population",
                category_tag="state_population",
            )
            _upsert_target(
                session,
                st_stratum.stratum_id,
                "person_count",
                time_period,
                rec["population"],
                sid,
                notes=f"State {st} total population",
            )

            under5_stratum = _get_or_create_stratum(
                session,
                parent_id=st_stratum.stratum_id,
                constraints=[
                    {
                        "constraint_variable": "state_code",
                        "operation": "==",
                        "value": st,
                    },
                    {
                        "constraint_variable": "age",
                        "operation": "<",
                        "value": "5",
                    },
                ],
                stratum_group_id=17,
                notes=f"State {st} under 5",
            )
            _upsert_target(
                session,
                under5_stratum.stratum_id,
                "person_count",
                time_period,
                rec["population_under_5"],
                sid,
                notes=f"State {st} population under 5",
            )

        # -- 10. Tax expenditure targets ---------------------------
        te_records = extract_tax_expenditure_targets()
        te_stratum = _get_or_create_stratum(
            session,
            parent_id=root_stratum_id,
            constraints=[],
            stratum_group_id=18,
            notes="Tax expenditure targets (counterfactual)",
            category_tag="tax_expenditure",
        )
        for rec in te_records:
            _upsert_target(
                session,
                te_stratum.stratum_id,
                rec["variable"],
                time_period,
                rec["value"],
                sid,
                notes=(
                    f"Tax expenditure: {rec['variable']} "
                    "(JCT 2024, requires counterfactual sim)"
                ),
                reform_id=1,
            )

        # -- 11. State real estate taxes ---------------------------
        ret_records = extract_state_real_estate_taxes()
        for rec in ret_records:
            sc = rec["state_code"]
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=[
                    {
                        "constraint_variable": "state_code",
                        "operation": "==",
                        "value": sc,
                    },
                ],
                stratum_group_id=19,
                notes=f"State real estate taxes {sc}",
                category_tag="real_estate_tax",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "real_estate_taxes",
                time_period,
                rec["value"],
                sid,
                notes=f"State real estate taxes {sc}",
            )

        # -- 12. State ACA spending + enrollment -------------------
        aca_records = extract_state_aca()
        for rec in aca_records:
            st = rec["state"]
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=[
                    {
                        "constraint_variable": "state_code",
                        "operation": "==",
                        "value": st,
                    },
                ],
                stratum_group_id=20,
                notes=f"State ACA {st}",
                category_tag="aca",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "aca_ptc",
                time_period,
                rec["spending"],
                sid,
                notes=f"ACA spending {st}",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "person_count",
                time_period,
                rec["enrollment"],
                sid,
                notes=f"ACA enrollment {st}",
            )

        # -- 13. State Medicaid enrollment -------------------------
        med_records = extract_state_medicaid_enrollment()
        for rec in med_records:
            st = rec["state"]
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=[
                    {
                        "constraint_variable": "state_code",
                        "operation": "==",
                        "value": st,
                    },
                    {
                        "constraint_variable": "medicaid_enrolled",
                        "operation": "==",
                        "value": "True",
                    },
                ],
                stratum_group_id=21,
                notes=f"State Medicaid {st}",
                category_tag="medicaid",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "person_count",
                time_period,
                rec["enrollment"],
                sid,
                notes=f"State Medicaid enrollment {st}",
            )

        # -- 14. State 10-year age targets -------------------------
        age_records = extract_state_10yr_age()
        for rec in age_records:
            st = rec["state"]
            ar = rec["age_range"]
            if "+" in ar:
                age_lo = int(ar.replace("+", ""))
                age_hi_str = "inf"
                constraints = [
                    {
                        "constraint_variable": "state_code",
                        "operation": "==",
                        "value": st,
                    },
                    {
                        "constraint_variable": "age",
                        "operation": ">=",
                        "value": str(age_lo),
                    },
                ]
            else:
                parts = ar.split("-")
                age_lo = int(parts[0])
                age_hi = int(parts[1])
                age_hi_str = str(age_hi)
                constraints = [
                    {
                        "constraint_variable": "state_code",
                        "operation": "==",
                        "value": st,
                    },
                    {
                        "constraint_variable": "age",
                        "operation": ">=",
                        "value": str(age_lo),
                    },
                    {
                        "constraint_variable": "age",
                        "operation": "<=",
                        "value": str(age_hi),
                    },
                ]
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=constraints,
                stratum_group_id=22,
                notes=f"State 10yr age {st} {ar}",
                category_tag="state_age",
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                "person_count",
                time_period,
                rec["value"],
                sid,
                notes=f"State 10yr age {st} {ar}",
            )

        # -- 15. State AGI targets ---------------------------------
        # Group count + amount rows onto the same stratum
        agi_records = extract_state_agi()
        for rec in agi_records:
            gn = rec["geo_name"]
            lo = rec["agi_lower"]
            hi = rec["agi_upper"]
            stratum_notes = f"State AGI {gn} {lo}-{hi}"

            constraints = [
                {
                    "constraint_variable": "state_code",
                    "operation": "==",
                    "value": gn,
                },
                {
                    "constraint_variable": "adjusted_gross_income",
                    "operation": ">",
                    "value": str(lo),
                },
                {
                    "constraint_variable": "adjusted_gross_income",
                    "operation": "<=",
                    "value": str(hi),
                },
            ]
            stratum = _get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=constraints,
                stratum_group_id=23,
                notes=stratum_notes,
                category_tag="state_agi",
            )
            variable = (
                "tax_unit_count"
                if rec["is_count"]
                else "adjusted_gross_income"
            )
            _upsert_target(
                session,
                stratum.stratum_id,
                variable,
                time_period,
                rec["value"],
                sid,
                notes=(f"State AGI {gn} " f"{lo}-{hi} {rec['variable']}"),
            )

        # -- 16. SOI filing-status x AGI bin targets ---------------
        # Multiple variables share the same stratum (fs + AGI band).
        soi_records = extract_soi_filing_status_targets()
        soi_strata_cache = {}  # (fs, lo, hi) -> Stratum
        for rec in soi_records:
            lo = rec["agi_lower"]
            hi = rec["agi_upper"]
            fs = rec["filing_status"]
            var = rec["variable"]
            is_count = rec["is_count"]

            cache_key = (fs, lo, hi)
            if cache_key not in soi_strata_cache:
                constraints = [
                    {
                        "constraint_variable": "tax_unit_is_filer",
                        "operation": "==",
                        "value": "1",
                    },
                    {
                        "constraint_variable": ("adjusted_gross_income"),
                        "operation": ">=",
                        "value": str(lo),
                    },
                    {
                        "constraint_variable": ("adjusted_gross_income"),
                        "operation": "<",
                        "value": str(hi),
                    },
                    {
                        "constraint_variable": "total_income_tax",
                        "operation": ">",
                        "value": "0",
                    },
                ]

                pe_fs = _FILING_STATUS_MAP.get(fs)
                if pe_fs is not None:
                    constraints.append(
                        {
                            "constraint_variable": "filing_status",
                            "operation": "==",
                            "value": pe_fs,
                        }
                    )

                stratum_notes = (
                    f"SOI filing-status {fs} " f"AGI {_fmt(lo)}-{_fmt(hi)}"
                )
                stratum = _get_or_create_stratum(
                    session,
                    parent_id=root_stratum_id,
                    constraints=constraints,
                    stratum_group_id=24,
                    notes=stratum_notes,
                )
                soi_strata_cache[cache_key] = stratum
            else:
                stratum = soi_strata_cache[cache_key]

            target_variable = var
            if is_count and var != "count":
                target_variable = "tax_unit_count"
            elif var == "count":
                target_variable = "tax_unit_count"

            count_label = "count" if is_count else "total"
            _upsert_target(
                session,
                stratum.stratum_id,
                target_variable,
                time_period,
                rec["value"],
                sid,
                notes=(
                    f"SOI filing-status {fs} "
                    f"{var} {count_label} "
                    f"AGI {_fmt(lo)}-{_fmt(hi)}"
                ),
            )

        session.commit()
        logger.info("All legacy targets loaded successfully.")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ETL: migrate ALL calibration targets "
            "from legacy loss.py into the database"
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Source dataset. Default: %(default)s",
    )
    args = parser.parse_args()

    from policyengine_us import Microsimulation

    print(f"Loading dataset: {args.dataset}")
    sim = Microsimulation(dataset=args.dataset)
    time_period = int(sim.default_calculation_period)
    print(f"Derived time_period={time_period}")

    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    engine = create_engine(f"sqlite:///{db_path}")
    from sqlmodel import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as sess:
        root = sess.exec(
            select(Stratum).where(
                Stratum.parent_stratum_id == None  # noqa: E711
            )
        ).first()
        if not root:
            root = Stratum(
                definition_hash="root_national",
                parent_stratum_id=None,
                stratum_group_id=1,
                notes="United States",
            )
            sess.add(root)
            sess.commit()
            sess.refresh(root)
        root_id = root.stratum_id

    load_all_targets(
        engine=engine,
        time_period=time_period,
        root_stratum_id=root_id,
    )
    print("Done.")


if __name__ == "__main__":
    main()
