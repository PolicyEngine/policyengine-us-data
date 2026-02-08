"""Shared helpers for legacy-target ETL modules.

Functions extracted from the original monolithic ``etl_all_targets.py``
so that every focused ETL module can reuse them without duplication.
"""

import numpy as np
from sqlmodel import select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)


# ------------------------------------------------------------------
# Format AGI bounds (mirrors loss.py ``fmt``)
# ------------------------------------------------------------------


def fmt(x):
    """Human-readable label for an AGI bound value."""
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
# Stratum upsert
# ------------------------------------------------------------------


def get_or_create_stratum(
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
        is appended so the definition hash is unique across
        categories that otherwise share the same constraints.
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


# ------------------------------------------------------------------
# Target upsert
# ------------------------------------------------------------------


def upsert_target(
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


# ------------------------------------------------------------------
# Filing-status mapping (SOI names -> PE enum values)
# ------------------------------------------------------------------

FILING_STATUS_MAP = {
    "Single": "SINGLE",
    "Married Filing Jointly/Surviving Spouse": "JOINT",
    "Head of Household": "HEAD_OF_HOUSEHOLD",
    "Married Filing Separately": "SEPARATE",
    "All": None,
}
