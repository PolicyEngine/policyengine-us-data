"""ETL: State-level calibration targets.

Combines several categories from the legacy ``etl_all_targets.py``:
  9.  State population (total + under-5)
 11.  State real estate taxes (51 rows)
 12.  State ACA spending and enrollment (51 x 2)
 14.  State 10-year age targets (50 states x 18 ranges)
 15.  State AGI targets (918 rows)
"""

import argparse
import logging

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
    get_or_create_stratum,
    upsert_target,
)

logger = logging.getLogger(__name__)

HARD_CODED_NATIONAL_TOTAL = 500e9  # real_estate_taxes national
ACA_SPENDING_2024 = 9.8e10


# ------------------------------------------------------------------
# Extract helpers
# ------------------------------------------------------------------


def extract_state_population():
    """Return list of dicts with state, population, population_under_5."""
    df = pd.read_csv(CALIBRATION_FOLDER / "population_by_state.csv")
    return [
        {
            "state": row["state"],
            "population": float(row["population"]),
            "population_under_5": float(row["population_under_5"]),
        }
        for _, row in df.iterrows()
    ]


def extract_state_real_estate_taxes():
    """Return list of 51 dicts with state_code and scaled value."""
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
    """Return list of dicts with state, spending (scaled), enrollment."""
    df = pd.read_csv(
        CALIBRATION_FOLDER / "aca_spending_and_enrollment_2024.csv"
    )
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


def extract_state_10yr_age():
    """Return list of dicts with state, age_range, value."""
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
    """Return list of 918 dicts with state AGI data."""
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


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------


def load_state_targets(engine, time_period, root_stratum_id):
    """Load all state-level targets into the database."""
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

        # -- 9. State population (total + under-5) ------
        state_pops = extract_state_population()
        for rec in state_pops:
            st = rec["state"]
            st_stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                st_stratum.stratum_id,
                "person_count",
                time_period,
                rec["population"],
                sid,
                notes=f"State {st} total population",
            )

            under5_stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                under5_stratum.stratum_id,
                "person_count",
                time_period,
                rec["population_under_5"],
                sid,
                notes=f"State {st} population under 5",
            )

        # -- 11. State real estate taxes -----------------
        ret_records = extract_state_real_estate_taxes()
        for rec in ret_records:
            sc = rec["state_code"]
            stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                stratum.stratum_id,
                "real_estate_taxes",
                time_period,
                rec["value"],
                sid,
                notes=f"State real estate taxes {sc}",
            )

        # -- 12. State ACA spending + enrollment ---------
        aca_records = extract_state_aca()
        for rec in aca_records:
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
                ],
                stratum_group_id=20,
                notes=f"State ACA {st}",
                category_tag="aca",
            )
            upsert_target(
                session,
                stratum.stratum_id,
                "aca_ptc",
                time_period,
                rec["spending"],
                sid,
                notes=f"ACA spending {st}",
            )
            upsert_target(
                session,
                stratum.stratum_id,
                "person_count",
                time_period,
                rec["enrollment"],
                sid,
                notes=f"ACA enrollment {st}",
            )

        # -- 14. State 10-year age targets ---------------
        age_records = extract_state_10yr_age()
        for rec in age_records:
            st = rec["state"]
            ar = rec["age_range"]
            if "+" in ar:
                age_lo = int(ar.replace("+", ""))
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
            stratum = get_or_create_stratum(
                session,
                parent_id=root_stratum_id,
                constraints=constraints,
                stratum_group_id=22,
                notes=f"State 10yr age {st} {ar}",
                category_tag="state_age",
            )
            upsert_target(
                session,
                stratum.stratum_id,
                "person_count",
                time_period,
                rec["value"],
                sid,
                notes=f"State 10yr age {st} {ar}",
            )

        # -- 15. State AGI targets -----------------------
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
            stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                stratum.stratum_id,
                variable,
                time_period,
                rec["value"],
                sid,
                notes=(f"State AGI {gn} " f"{lo}-{hi} {rec['variable']}"),
            )

        session.commit()
        logger.info("State targets loaded successfully.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ETL: state-level calibration targets"
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

    load_state_targets(engine, args.time_period, root_id)
    print("Done.")


if __name__ == "__main__":
    main()
