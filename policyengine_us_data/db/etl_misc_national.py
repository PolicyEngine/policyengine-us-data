"""ETL: Miscellaneous national calibration targets.

Combines several categories from the legacy ``etl_all_targets.py``
that are national-level and not covered by other focused ETLs:
  1.  Census single-year age populations (86 bins)
  2.  EITC by child count (returns + spending)
  3.  SOI filer counts by AGI band (7 bands)
  6.  Negative household market income (total + count)
  7.  Infant count
  8.  Net worth
 13.  State Medicaid enrollment (51 rows)
 16.  SOI filing-status x AGI bin targets
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
    SourceType,
    Stratum,
)
from policyengine_us_data.utils.db_metadata import get_or_create_source
from policyengine_us_data.db.etl_helpers import (
    fmt,
    get_or_create_stratum,
    upsert_target,
    FILING_STATUS_MAP,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

SOI_FILER_COUNTS_2015 = {
    (-np.inf, 0): 2_072_066,
    (0, 5_000): 10_134_703,
    (5_000, 10_000): 11_398_595,
    (10_000, 25_000): 23_447_927,
    (25_000, 50_000): 23_727_745,
    (50_000, 100_000): 32_801_908,
    (100_000, np.inf): 25_120_985,
}

INFANTS_2023 = 3_491_679
INFANTS_2022 = 3_437_933
NET_WORTH_2024 = 160e12

# Map SOI variable names to PE variable names
_SOI_VAR_MAP = {
    "count": "tax_unit_count",
}


# ------------------------------------------------------------------
# Extract helpers
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


def extract_negative_market_income():
    """Return dict with total and count."""
    return {"total": -138e9, "count": 3e6}


def extract_infant_count():
    """Return projected infant count."""
    return INFANTS_2023 * (INFANTS_2023 / INFANTS_2022)


def extract_net_worth():
    """Return total household net worth."""
    return NET_WORTH_2024


def extract_state_medicaid_enrollment():
    """Return list of 51 dicts with state and enrollment."""
    df = pd.read_csv(CALIBRATION_FOLDER / "medicaid_enrollment_2024.csv")
    return [
        {
            "state": row["state"],
            "enrollment": float(row["enrollment"]),
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


def extract_soi_filing_status_targets():
    """Return filtered list of SOI filing-status x AGI bin records."""
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
# Load
# ------------------------------------------------------------------


def load_misc_national(engine, time_period, root_stratum_id):
    """Load all miscellaneous national targets into the database."""
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

        # -- 1. Census single-year age populations ------
        age_pops = extract_census_age_populations(time_period)
        age_strata = {}
        for rec in age_pops:
            age = rec["age"]
            notes_str = f"Census age bin {age}"
            stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                stratum.stratum_id,
                "person_count",
                time_period,
                rec["value"],
                sid,
                notes=notes_str,
            )

        # -- 2. EITC by child count --------------------
        eitc_records = extract_eitc_by_child_count()
        for rec in eitc_records:
            cc = rec["count_children"]
            if cc < 2:
                op, val = "==", str(cc)
            else:
                op, val = ">=", str(cc)

            stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                stratum.stratum_id,
                "tax_unit_count",
                time_period,
                rec["eitc_returns"],
                sid,
                notes=f"EITC returns, {cc} children",
            )
            upsert_target(
                session,
                stratum.stratum_id,
                "eitc",
                time_period,
                rec["eitc_total"],
                sid,
                notes=f"EITC spending, {cc} children",
            )

        # -- 3. SOI filer counts by AGI band ------------
        filer_counts = extract_soi_filer_counts()
        for rec in filer_counts:
            lo, hi = rec["agi_lower"], rec["agi_upper"]
            label = f"{fmt(lo)}_{fmt(hi)}"
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
            stratum = get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=constraints,
                stratum_group_id=12,
                notes=f"SOI filer count AGI {label}",
            )
            upsert_target(
                session,
                stratum.stratum_id,
                "tax_unit_count",
                time_period,
                rec["filer_count_2015"],
                sid,
                notes=f"SOI filer count AGI {label}",
            )

        # -- 6. Negative household market income --------
        nmi = extract_negative_market_income()
        nmi_stratum = get_or_create_stratum(
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
        upsert_target(
            session,
            nmi_stratum.stratum_id,
            "household_market_income",
            time_period,
            nmi["total"],
            sid,
            notes="Negative household market income total",
        )
        upsert_target(
            session,
            nmi_stratum.stratum_id,
            "household_count",
            time_period,
            nmi["count"],
            sid,
            notes="Negative household market income count",
        )

        # -- 7. Infant count ----------------------------
        infant_count = extract_infant_count()
        infant_stratum = age_strata[0]
        upsert_target(
            session,
            infant_stratum.stratum_id,
            "person_count",
            time_period,
            infant_count,
            sid,
            notes="Census age bin 0",
        )

        # -- 8. Net worth -------------------------------
        nw_value = extract_net_worth()
        upsert_target(
            session,
            root_stratum_id,
            "net_worth",
            time_period,
            nw_value,
            sid,
            notes="Total household net worth (Fed Reserve)",
        )

        # -- 13. State Medicaid enrollment --------------
        med_records = extract_state_medicaid_enrollment()
        for rec in med_records:
            st = rec["state"]
            stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                stratum.stratum_id,
                "person_count",
                time_period,
                rec["enrollment"],
                sid,
                notes=f"State Medicaid enrollment {st}",
            )

        # -- 16. SOI filing-status x AGI bin targets ----
        soi_records = extract_soi_filing_status_targets()
        soi_strata_cache = {}
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
                        "constraint_variable": "income_tax",
                        "operation": ">",
                        "value": "0",
                    },
                ]

                pe_fs = FILING_STATUS_MAP.get(fs)
                if pe_fs is not None:
                    constraints.append(
                        {
                            "constraint_variable": "filing_status",
                            "operation": "==",
                            "value": pe_fs,
                        }
                    )

                stratum_notes = (
                    f"SOI filing-status {fs} " f"AGI {fmt(lo)}-{fmt(hi)}"
                )
                stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                stratum.stratum_id,
                target_variable,
                time_period,
                rec["value"],
                sid,
                notes=(
                    f"SOI filing-status {fs} "
                    f"{var} {count_label} "
                    f"AGI {fmt(lo)}-{fmt(hi)}"
                ),
            )

        session.commit()
        logger.info("Misc national targets loaded successfully.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=("ETL: miscellaneous national calibration targets")
    )
    parser.add_argument(
        "--time-period",
        type=int,
        default=2024,
        help="Target year (default: %(default)s)",
    )
    args = parser.parse_args()

    from sqlmodel import SQLModel

    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as sess:
        root = sess.exec(
            select(Stratum).where(
                Stratum.parent_stratum_id == None  # noqa: E711
            )
        ).first()
        if not root:
            raise RuntimeError("Root stratum not found.")
        root_id = root.stratum_id

    load_misc_national(engine, args.time_period, root_id)
    print("Done.")


if __name__ == "__main__":
    main()
