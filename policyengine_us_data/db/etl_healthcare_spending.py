"""ETL: Healthcare spending by age band (9 bands x 4 expense types).

Migrated from category 4 of the legacy ``etl_all_targets.py``.
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


# ------------------------------------------------------------------
# Extract
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------


def load_healthcare_spending(engine, time_period, root_stratum_id):
    """Load healthcare spending targets into the database."""
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

        hc_records = extract_healthcare_by_age()
        for rec in hc_records:
            age_lo = rec["age_lower"]
            stratum = get_or_create_stratum(
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
                notes=f"Healthcare age {age_lo}-{age_lo + 9}",
                category_tag="healthcare",
            )
            for var_name, amount in rec["expenses"].items():
                upsert_target(
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

        session.commit()
        logger.info("Healthcare spending targets loaded.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ETL: healthcare spending by age band"
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

    load_healthcare_spending(engine, args.time_period, root_id)
    print("Done.")


if __name__ == "__main__":
    main()
