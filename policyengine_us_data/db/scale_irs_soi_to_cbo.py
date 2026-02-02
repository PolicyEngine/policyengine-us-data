"""Scale IRS SOI 2022 database targets to 2024 using CBO projections.

The IRS SOI congressional-district data (22incd.csv) is the most recent
available; 2023/2024 CD data has not been published yet.  To align the
DB targets with the 2024 simulation year we scale each variable's
aggregate to match the corresponding CBO / Treasury projection, using
the same parameter values that the enhanced CPS calibration uses in
loss.py.

Only targets with source_id=5 (IRS SOI ETL) are affected.  Census,
Medicaid, and SNAP targets already pull 2024 data directly from their
administrative sources.

See: https://github.com/PolicyEngine/policyengine-us-data/issues/503
"""

import logging
from typing import Dict, Tuple

from sqlalchemy import text
from sqlmodel import Session, create_engine, select

from policyengine_us.system import system
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.db.create_database_tables import Target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SOI_SOURCE_ID = 5
SOI_YEAR = 2022
TARGET_YEAR = 2024


def _get_cbo_targets(year: int) -> Dict[str, float]:
    """CBO / Treasury national totals for IRS SOI variables.

    Mirrors the targets used in loss.py for enhanced CPS calibration.

    Args:
        year: Simulation year.

    Returns:
        Mapping of DB variable name to national total.
    """
    p = system.parameters(year)
    cbo = p.calibration.gov.cbo
    cbo_inc = cbo.income_by_source
    soi = p.calibration.gov.irs.soi

    # IRS SOI total returns (sum by filing status)
    soi_total_returns = sum(
        v
        for v in soi.returns_by_filing_status._children.values()
        if isinstance(v, (int, float))
    )

    return {
        # CBO budget projections
        "income_tax": cbo._children["income_tax"],
        "unemployment_compensation": cbo._children[
            "unemployment_compensation"
        ],
        # Treasury
        "eitc": (
            system.parameters.calibration.gov.treasury.tax_expenditures.eitc(
                year
            )
        ),
        # CBO income-by-source
        "adjusted_gross_income": cbo_inc._children["adjusted_gross_income"],
        "taxable_social_security": cbo_inc._children[
            "taxable_social_security"
        ],
        "taxable_pension_income": cbo_inc._children["taxable_pension_income"],
        "net_capital_gain": cbo_inc._children["net_capital_gain"],
        # IRS SOI projections
        "qualified_dividend_income": soi._children[
            "qualified_dividend_income"
        ],
        "taxable_interest_income": soi._children["taxable_interest_income"],
        "tax_exempt_interest_income": soi._children[
            "tax_exempt_interest_income"
        ],
        "tax_unit_partnership_s_corp_income": soi._children[
            "partnership_s_corp_income"
        ],
        "dividend_income": (
            soi._children["qualified_dividend_income"]
            + soi._children["non_qualified_dividend_income"]
        ),
        # Return counts
        "person_count": soi_total_returns,
    }


def _compute_state_aggregate(
    session: Session, variable: str
) -> Tuple[float, int]:
    """Sum state-level IRS SOI targets for *variable*.

    Uses raw SQL to avoid the USVariable enum deserialization issue.

    Args:
        session: Active database session.
        variable: Target variable name.

    Returns:
        (sum of state-level values, row count).
    """
    result = session.execute(
        text("""
            SELECT COALESCE(SUM(t.value), 0) AS total,
                   COUNT(*)                  AS cnt
            FROM   targets t
            JOIN   stratum_constraints sc
                   ON sc.stratum_id = t.stratum_id
            WHERE  t.variable   = :variable
               AND t.source_id  = :source_id
               AND t.active     = 1
               AND sc.constraint_variable = 'ucgid_str'
               AND sc.value LIKE '0400000US%'
            """),
        {"variable": variable, "source_id": SOI_SOURCE_ID},
    )
    row = result.one()
    return float(row.total), int(row.cnt)


def _scale_targets(
    session: Session,
    variable: str,
    scale_factor: float,
    target_year: int,
) -> int:
    """Scale all IRS SOI targets for *variable* and update period.

    Args:
        session: Active database session.
        variable: Target variable name.
        scale_factor: Multiplicative adjustment.
        target_year: New period value.

    Returns:
        Number of rows updated.
    """
    stmt = (
        select(Target)
        .where(Target.variable == variable)
        .where(Target.source_id == SOI_SOURCE_ID)
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


def scale_soi_to_cbo(
    session: Session,
    target_year: int = TARGET_YEAR,
) -> Dict[str, float]:
    """Scale IRS SOI DB targets to CBO 2024 projections.

    For each variable with a CBO/Treasury projection:
    1. Sum current state-level DB targets.
    2. Compute scale factor = CBO target / DB aggregate.
    3. Apply proportionally to all geographic levels.

    Args:
        session: Active database session.
        target_year: Simulation year.

    Returns:
        Mapping of variable name to scale factor applied.
    """
    cbo_targets = _get_cbo_targets(target_year)
    scale_factors: Dict[str, float] = {}

    for variable, cbo_value in cbo_targets.items():
        state_sum, state_count = _compute_state_aggregate(session, variable)

        if state_sum == 0:
            logger.warning(
                "Skipping '%s': no state-level SOI targets",
                variable,
            )
            continue

        scale = cbo_value / state_sum
        pct = (scale - 1) * 100

        logger.info(
            "%-35s  %4d states  %.3g -> %.3g  " "(x%.4f, %+.1f%%)",
            variable,
            state_count,
            state_sum,
            cbo_value,
            scale,
            pct,
        )

        n = _scale_targets(session, variable, scale, target_year)
        logger.info("  Updated %d rows", n)
        scale_factors[variable] = scale

    session.commit()
    logger.info("IRS SOI -> CBO scaling complete.")
    return scale_factors


def main() -> None:
    db_url = f"sqlite:///{STORAGE_FOLDER / 'policy_data.db'}"
    engine = create_engine(db_url)

    with Session(engine) as session:
        scale_soi_to_cbo(session)


if __name__ == "__main__":
    main()
