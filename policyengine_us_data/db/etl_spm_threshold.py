"""ETL: AGI by SPM threshold decile (10 deciles x 2 metrics).

Migrated from category 5 of the legacy ``etl_all_targets.py``.
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


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------


def load_spm_threshold(engine, time_period, root_stratum_id):
    """Load SPM-threshold-decile targets into the database."""
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

        spm_records = extract_spm_threshold_agi()
        for rec in spm_records:
            d = rec["decile"]
            stratum = get_or_create_stratum(
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
            upsert_target(
                session,
                stratum.stratum_id,
                "adjusted_gross_income",
                time_period,
                rec["adjusted_gross_income"],
                sid,
                notes=f"SPM threshold decile {d} AGI",
            )
            upsert_target(
                session,
                stratum.stratum_id,
                "spm_unit_count",
                time_period,
                rec["count"],
                sid,
                notes=f"SPM threshold decile {d} count",
            )

        session.commit()
        logger.info("SPM threshold targets loaded.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ETL: AGI by SPM threshold decile"
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

    load_spm_threshold(engine, args.time_period, root_id)
    print("Done.")


if __name__ == "__main__":
    main()
