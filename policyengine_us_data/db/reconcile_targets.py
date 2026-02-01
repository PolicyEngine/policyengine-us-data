"""Reconcile stale database targets with policyengine-us parameters.

After the ETL scripts populate policy_data.db with historical
administrative data (IRS SOI 2022, USDA SNAP FY2023), the aggregate
totals drift from the projections that loss.py uses for national
calibration.

This script scales affected targets so that their national aggregates
match the policyengine-us parameter values for the simulation year,
applying the same proportional adjustment at every geographic level
(national, state, congressional district).

See: https://github.com/PolicyEngine/policyengine-us-data/issues/503
"""

import logging
from typing import Dict, Optional, Tuple

from sqlalchemy import text
from sqlmodel import Session, create_engine, select

from policyengine_us.system import system
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import (
    Target,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_YEAR = 2024


def _get_authoritative_targets(year: int) -> Dict[str, float]:
    """Read national targets from policyengine-us parameters.

    Covers every database variable for which a 2024 parameter exists
    in the policyengine-us calibration tree.  The dict keys are the
    *database* variable names (as written by the ETL scripts); the
    values are the authoritative national totals for *year*.

    Args:
        year: The simulation year.

    Returns:
        Mapping of database variable name to authoritative value.
    """
    p = system.parameters(year)

    cbo = p.calibration.gov.cbo
    cbo_inc = cbo.income_by_source
    soi = p.calibration.gov.irs.soi

    targets: Dict[str, float] = {
        # ---- CBO budget projections ----
        "income_tax": cbo._children["income_tax"],
        "snap": cbo._children["snap"],
        "unemployment_compensation": cbo._children[
            "unemployment_compensation"
        ],
        # ---- Treasury ----
        "eitc": (
            system.parameters.calibration.gov.treasury.tax_expenditures.eitc(
                year
            )
        ),
        # ---- CBO income-by-source (DB vars from IRS SOI ETL) ----
        "adjusted_gross_income": cbo_inc._children["adjusted_gross_income"],
        "taxable_social_security": cbo_inc._children[
            "taxable_social_security"
        ],
        "taxable_pension_income": cbo_inc._children["taxable_pension_income"],
        "net_capital_gain": cbo_inc._children["net_capital_gain"],
        # ---- IRS SOI calibration parameters ----
        "qualified_dividend_income": soi._children[
            "qualified_dividend_income"
        ],
        "taxable_interest_income": soi._children["taxable_interest_income"],
        "tax_exempt_interest_income": soi._children[
            "tax_exempt_interest_income"
        ],
        # DB name differs from param name
        "tax_unit_partnership_s_corp_income": soi._children[
            "partnership_s_corp_income"
        ],
        # ordinary dividends = qualified + non-qualified
        "dividend_income": (
            soi._children["qualified_dividend_income"]
            + soi._children["non_qualified_dividend_income"]
        ),
    }
    return targets


def _compute_state_aggregate(
    session: Session, variable: str
) -> Tuple[float, int]:
    """Sum state-level target values for a variable.

    State strata are identified by a ``ucgid_str`` constraint whose
    value starts with ``0400000US`` (two-digit state FIPS).

    A raw SQL join is used instead of the ORM because the
    ``constraint_variable`` column stores values (like ``ucgid_str``)
    that fall outside the ``USVariable`` enum, causing SQLAlchemy
    deserialization errors if read through the model.

    Args:
        session: Active database session.
        variable: Target variable name.

    Returns:
        Tuple of (sum of state-level targets, count of rows).
    """
    result = session.execute(
        text("""
            SELECT COALESCE(SUM(t.value), 0) AS total,
                   COUNT(*)                  AS cnt
            FROM   targets t
            JOIN   stratum_constraints sc
                   ON sc.stratum_id = t.stratum_id
            WHERE  t.variable = :variable
               AND t.active   = 1
               AND sc.constraint_variable = 'ucgid_str'
               AND sc.value LIKE '0400000US%'
            """),
        {"variable": variable},
    )
    row = result.one()
    return float(row.total), int(row.cnt)


def _scale_targets(
    session: Session,
    variable: str,
    scale_factor: float,
    target_year: int,
) -> int:
    """Multiply every target for *variable* by *scale_factor*.

    Also updates the ``period`` column to *target_year* so the
    database reflects the simulation year rather than the original
    source year.

    Args:
        session: Active database session.
        variable: Target variable name.
        scale_factor: Multiplicative adjustment factor.
        target_year: New period value for updated rows.

    Returns:
        Number of target rows updated.
    """
    stmt = select(Target).where(Target.variable == variable)
    all_targets = session.exec(stmt).all()

    updated = 0
    for t in all_targets:
        if t.value is not None:
            t.value *= scale_factor
        t.period = target_year
        session.add(t)
        updated += 1

    return updated


def reconcile_targets(
    session: Session,
    target_year: int = TARGET_YEAR,
) -> Dict[str, float]:
    """Scale database targets to match policyengine-us parameters.

    For each reconcilable variable the script:

    1. Sums current state-level DB targets to obtain the aggregate.
    2. Looks up the authoritative value from policyengine-us.
    3. Scales **all** geographic levels proportionally.

    Args:
        session: Active database session.
        target_year: Simulation year for the parameter lookup.

    Returns:
        Mapping of variable name to the scale factor applied.
    """
    authoritative = _get_authoritative_targets(target_year)
    scale_factors: Dict[str, float] = {}

    for variable, auth_value in authoritative.items():
        state_sum, state_count = _compute_state_aggregate(session, variable)

        if state_sum == 0:
            logger.warning(
                "Skipping '%s': no state-level targets in database",
                variable,
            )
            continue

        scale = auth_value / state_sum
        pct = (scale - 1) * 100

        logger.info(
            "Reconciling '%s': "
            "%d state rows summing to $%.1fB -> $%.1fB "
            "(x%.4f, %+.1f%%)",
            variable,
            state_count,
            state_sum / 1e9,
            auth_value / 1e9,
            scale,
            pct,
        )

        n = _scale_targets(session, variable, scale, target_year)
        logger.info("  Updated %d target rows for '%s'", n, variable)
        scale_factors[variable] = scale

    session.commit()
    logger.info("Reconciliation complete.")
    return scale_factors


def main() -> None:
    db_url = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(db_url)

    with Session(engine) as session:
        reconcile_targets(session)


if __name__ == "__main__":
    main()
