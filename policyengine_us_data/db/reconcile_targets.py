"""Reconcile stale database targets with policyengine-us parameters.

After the ETL scripts populate policy_data.db with historical
administrative data (IRS SOI 2022, USDA SNAP FY2023), the aggregate
totals drift from the projections that loss.py uses for national
calibration.

This script scales affected targets so that their national aggregates
match the policyengine-us parameter values for the simulation year,
applying the same proportional adjustment at every geographic level
(national, state, congressional district).

Each target is identified by both its variable name AND its source_id
to disambiguate cases like person_count, which appears in the age ETL
(source_id=1), medicaid ETL (source_id=2), and IRS SOI ETL
(source_id=5) with different meanings.

See: https://github.com/PolicyEngine/policyengine-us-data/issues/503
"""

import logging
from typing import Dict, Tuple

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

# Type alias: (variable_name, source_id)
TargetKey = Tuple[str, int]


def _get_authoritative_targets(
    year: int,
) -> Dict[TargetKey, float]:
    """Read national targets from policyengine-us parameters.

    Each target is keyed by ``(variable_name, source_id)`` to handle
    variables like ``person_count`` that appear in multiple ETL
    sources with different meanings.

    Source IDs follow the ETL convention:
        1 = Census age (etl_age)
        2 = Medicaid enrollment (etl_medicaid)
        3 = SNAP admin (etl_snap)
        5 = IRS SOI (etl_irs_soi)

    Args:
        year: The simulation year.

    Returns:
        Mapping of (variable, source_id) to authoritative value.
    """
    p = system.parameters(year)

    cal = p.calibration.gov
    cbo = cal.cbo
    cbo_inc = cbo.income_by_source
    soi = cal.irs.soi

    # Medicaid enrollment: sum state-level values
    medicaid_enrollment = sum(
        v
        for v in cal.hhs.medicaid.totals.enrollment._children.values()
        if isinstance(v, (int, float))
    )

    # IRS SOI total returns: sum by filing status
    soi_total_returns = sum(
        v
        for v in soi.returns_by_filing_status._children.values()
        if isinstance(v, (int, float))
    )

    targets: Dict[TargetKey, float] = {
        # ---- IRS SOI ETL (source_id=5) ----
        # CBO budget projections
        ("income_tax", 5): cbo._children["income_tax"],
        ("unemployment_compensation", 5): cbo._children[
            "unemployment_compensation"
        ],
        # Treasury
        ("eitc", 5): (
            system.parameters.calibration.gov.treasury.tax_expenditures.eitc(
                year
            )
        ),
        # CBO income-by-source
        ("adjusted_gross_income", 5): cbo_inc._children[
            "adjusted_gross_income"
        ],
        ("taxable_social_security", 5): cbo_inc._children[
            "taxable_social_security"
        ],
        ("taxable_pension_income", 5): cbo_inc._children[
            "taxable_pension_income"
        ],
        ("net_capital_gain", 5): cbo_inc._children["net_capital_gain"],
        # IRS SOI calibration parameters
        ("qualified_dividend_income", 5): soi._children[
            "qualified_dividend_income"
        ],
        ("taxable_interest_income", 5): soi._children[
            "taxable_interest_income"
        ],
        ("tax_exempt_interest_income", 5): soi._children[
            "tax_exempt_interest_income"
        ],
        # DB name differs from param name
        ("tax_unit_partnership_s_corp_income", 5): soi._children[
            "partnership_s_corp_income"
        ],
        # ordinary dividends = qualified + non-qualified
        ("dividend_income", 5): (
            soi._children["qualified_dividend_income"]
            + soi._children["non_qualified_dividend_income"]
        ),
        # IRS SOI person_count (total returns)
        ("person_count", 5): soi_total_returns,
        # ---- SNAP ETL (source_id=3) ----
        ("snap", 3): cbo._children["snap"],
        # ---- Census Age ETL (source_id=1) ----
        ("person_count", 1): cal.census.populations._children["total"],
        # ---- Medicaid ETL (source_id=2) ----
        ("person_count", 2): medicaid_enrollment,
    }
    return targets


def _compute_state_aggregate(
    session: Session, variable: str, source_id: int
) -> Tuple[float, int]:
    """Sum state-level target values for a variable and source.

    State strata are identified by a ``ucgid_str`` constraint whose
    value starts with ``0400000US`` (two-digit state FIPS).

    A raw SQL join is used instead of the ORM because the
    ``constraint_variable`` column stores values (like ``ucgid_str``)
    that fall outside the ``USVariable`` enum, causing SQLAlchemy
    deserialization errors if read through the model.

    Args:
        session: Active database session.
        variable: Target variable name.
        source_id: ETL source identifier to disambiguate targets.

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
               AND t.source_id = :source_id
               AND t.active   = 1
               AND sc.constraint_variable = 'ucgid_str'
               AND sc.value LIKE '0400000US%'
            """),
        {"variable": variable, "source_id": source_id},
    )
    row = result.one()
    return float(row.total), int(row.cnt)


def _scale_targets(
    session: Session,
    variable: str,
    source_id: int,
    scale_factor: float,
    target_year: int,
) -> int:
    """Multiply every target for *variable* / *source_id* by *scale*.

    Also updates the ``period`` column to *target_year* so the
    database reflects the simulation year rather than the original
    source year.

    Args:
        session: Active database session.
        variable: Target variable name.
        source_id: ETL source identifier to disambiguate targets.
        scale_factor: Multiplicative adjustment factor.
        target_year: New period value for updated rows.

    Returns:
        Number of target rows updated.
    """
    stmt = (
        select(Target)
        .where(Target.variable == variable)
        .where(Target.source_id == source_id)
    )
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
) -> Dict[TargetKey, float]:
    """Scale database targets to match policyengine-us parameters.

    For each reconcilable (variable, source_id) pair the script:

    1. Sums current state-level DB targets to obtain the aggregate.
    2. Looks up the authoritative value from policyengine-us.
    3. Scales **all** geographic levels proportionally.

    Args:
        session: Active database session.
        target_year: Simulation year for the parameter lookup.

    Returns:
        Mapping of (variable, source_id) to the scale factor applied.
    """
    authoritative = _get_authoritative_targets(target_year)
    scale_factors: Dict[TargetKey, float] = {}

    for (variable, source_id), auth_value in authoritative.items():
        state_sum, state_count = _compute_state_aggregate(
            session, variable, source_id
        )

        if state_sum == 0:
            logger.warning(
                "Skipping '%s' (source %d): "
                "no state-level targets in database",
                variable,
                source_id,
            )
            continue

        scale = auth_value / state_sum
        pct = (scale - 1) * 100

        logger.info(
            "Reconciling '%s' (source %d): "
            "%d state rows summing to %.3g -> %.3g "
            "(x%.4f, %+.1f%%)",
            variable,
            source_id,
            state_count,
            state_sum,
            auth_value,
            scale,
            pct,
        )

        n = _scale_targets(session, variable, source_id, scale, target_year)
        logger.info("  Updated %d target rows for '%s'", n, variable)
        scale_factors[(variable, source_id)] = scale

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
