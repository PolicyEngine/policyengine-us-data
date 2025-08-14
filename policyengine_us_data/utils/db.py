from typing import List, Optional

from sqlmodel import Session, select
import sqlalchemy as sa

from policyengine_us_data.db.create_database_tables import Stratum, StratumConstraint


def get_stratum_by_id(session: Session, stratum_id: int) -> Optional[Stratum]:
    """Retrieves a single Stratum by its primary key"""
    return session.get(Stratum, stratum_id)


def get_simple_stratum_by_ucgid(session: Session, ucgid: str) -> Optional[Stratum]:
    """
    Finds a stratum defined *only* by a single ucgid_str constraint.
    """
    constraint_count_subquery = (
        select(
            StratumConstraint.stratum_id,
            sa.func.count(StratumConstraint.stratum_id).label("constraint_count")
        )
        .group_by(StratumConstraint.stratum_id)
        .subquery()
    )

    statement = (
        select(Stratum)
        .join(StratumConstraint)
        .join(
            constraint_count_subquery,
            Stratum.stratum_id == constraint_count_subquery.c.stratum_id
        )
        .where(StratumConstraint.constraint_variable == "ucgid_str")
        .where(StratumConstraint.value == ucgid)
        .where(constraint_count_subquery.c.constraint_count == 1)
    )

    return session.exec(statement).first()


def get_root_strata(session: Session) -> List[Stratum]:
    """Finds all strata that do not have a parent"""
    statement = select(Stratum).where(Stratum.parent_stratum_id == None)
    return session.exec(statement).all()


def get_stratum_children(session: Session, stratum_id: int) -> List[Stratum]:
    """Retrieves all direct children for a given stratum"""
    parent_stratum = get_stratum_by_id(session, stratum_id)
    if parent_stratum:
        return parent_stratum.children_rel
    return []


def get_stratum_parent(session: Session, stratum_id: int) -> Optional[Stratum]:
    """Retrieves the direct parent for a given stratum."""
    child_stratum = get_stratum_by_id(session, stratum_id)
    if child_stratum:
        return child_stratum.parent_rel
    return None
