"""Orchestrator that loads ALL calibration targets from the legacy
``loss.py`` system into the ``policy_data.db`` database.

This thin wrapper delegates to focused ETL modules:
  - etl_misc_national  (census age, EITC, SOI filers, neg. market
                         income, infant, net worth, Medicaid,
                         SOI filing-status)
  - etl_healthcare_spending (healthcare by age band)
  - etl_spm_threshold      (AGI by SPM threshold decile)
  - etl_tax_expenditure     (SALT, medical, charitable, interest, QBI)
  - etl_state_targets       (state pop, real estate taxes, ACA,
                              10-yr age, state AGI)

Individual extract functions are re-exported here so existing
callers (tests, notebooks) continue to work.
"""

import argparse
import logging

from sqlmodel import Session, create_engine, select

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import Stratum

# -- Focused ETL loaders -------------------------------------------
from policyengine_us_data.db.etl_misc_national import (
    load_misc_national,
)
from policyengine_us_data.db.etl_healthcare_spending import (
    load_healthcare_spending,
)
from policyengine_us_data.db.etl_spm_threshold import (
    load_spm_threshold,
)
from policyengine_us_data.db.etl_tax_expenditure import (
    load_tax_expenditure,
)
from policyengine_us_data.db.etl_state_targets import (
    load_state_targets,
)

# -- Re-export every extract function for backward compatibility ---
from policyengine_us_data.db.etl_misc_national import (  # noqa: F401
    extract_census_age_populations,
    extract_eitc_by_child_count,
    extract_soi_filer_counts,
    extract_negative_market_income,
    extract_infant_count,
    extract_net_worth,
    extract_state_medicaid_enrollment,
    extract_soi_filing_status_targets,
)
from policyengine_us_data.db.etl_healthcare_spending import (  # noqa: F401
    extract_healthcare_by_age,
)
from policyengine_us_data.db.etl_spm_threshold import (  # noqa: F401
    extract_spm_threshold_agi,
)
from policyengine_us_data.db.etl_tax_expenditure import (  # noqa: F401
    extract_tax_expenditure_targets,
)
from policyengine_us_data.db.etl_state_targets import (  # noqa: F401
    extract_state_population,
    extract_state_real_estate_taxes,
    extract_state_aca,
    extract_state_10yr_age,
    extract_state_agi,
)

# Re-export shared helpers under their old private names
from policyengine_us_data.db.etl_helpers import (  # noqa: F401
    fmt as _fmt,
    get_or_create_stratum as _get_or_create_stratum,
    upsert_target as _upsert_target,
    FILING_STATUS_MAP as _FILING_STATUS_MAP,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET = (
    "hf://policyengine/policyengine-us-data/"
    "calibration/stratified_extended_cps.h5"
)


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------


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
    load_misc_national(engine, time_period, root_stratum_id)
    load_healthcare_spending(engine, time_period, root_stratum_id)
    load_spm_threshold(engine, time_period, root_stratum_id)
    load_tax_expenditure(engine, time_period, root_stratum_id)
    load_state_targets(engine, time_period, root_stratum_id)
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
