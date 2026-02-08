"""ETL: Tax expenditure targets (SALT, medical, charitable,
interest, QBI).

Migrated from category 10 of the legacy ``etl_all_targets.py``.
"""

import argparse
import logging

from sqlmodel import Session, create_engine, select

from policyengine_us_data.storage import STORAGE_FOLDER
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

ITEMIZED_DEDUCTIONS = {
    "salt_deduction": 21.247e9,
    "medical_expense_deduction": 11.4e9,
    "charitable_deduction": 65.301e9,
    "interest_deduction": 24.8e9,
    "qualified_business_income_deduction": 63.1e9,
}


# ------------------------------------------------------------------
# Extract
# ------------------------------------------------------------------


def extract_tax_expenditure_targets():
    """Return list of 5 dicts (one per deduction type)."""
    return [
        {"variable": var, "value": val}
        for var, val in ITEMIZED_DEDUCTIONS.items()
    ]


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------


def load_tax_expenditure(engine, time_period, root_stratum_id):
    """Load tax-expenditure targets into the database."""
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

        te_records = extract_tax_expenditure_targets()
        te_stratum = get_or_create_stratum(
            session,
            parent_id=root_stratum_id,
            constraints=[],
            stratum_group_id=18,
            notes="Tax expenditure targets (counterfactual)",
            category_tag="tax_expenditure",
        )
        for rec in te_records:
            upsert_target(
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

        session.commit()
        logger.info("Tax expenditure targets loaded.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ETL: tax expenditure targets"
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

    load_tax_expenditure(engine, args.time_period, root_id)
    print("Done.")


if __name__ == "__main__":
    main()
