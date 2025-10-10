from typing import List, Optional, Dict

from sqlmodel import Session, select
import sqlalchemy as sa

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
)


def get_stratum_by_id(session: Session, stratum_id: int) -> Optional[Stratum]:
    """Retrieves a single Stratum by its primary key"""
    return session.get(Stratum, stratum_id)


def get_simple_stratum_by_ucgid(
    session: Session, ucgid: str
) -> Optional[Stratum]:
    """
    Finds a stratum defined *only* by a single ucgid_str constraint.
    """
    constraint_count_subquery = (
        select(
            StratumConstraint.stratum_id,
            sa.func.count(StratumConstraint.stratum_id).label(
                "constraint_count"
            ),
        )
        .group_by(StratumConstraint.stratum_id)
        .subquery()
    )

    statement = (
        select(Stratum)
        .join(StratumConstraint)
        .join(
            constraint_count_subquery,
            Stratum.stratum_id == constraint_count_subquery.c.stratum_id,
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


def parse_ucgid(ucgid_str: str) -> Dict:
    """Parse UCGID string to extract geographic information.

    UCGID (Universal Census Geographic ID) is a Census Bureau format
    for identifying geographic areas.

    Returns:
        dict with keys: 'type' ('national', 'state', 'district'),
                       'state_fips' (if applicable),
                       'district_number' (if applicable),
                       'congressional_district_geoid' (if applicable)
    """
    if ucgid_str == "0100000US":
        return {"type": "national"}
    elif ucgid_str.startswith("0400000US"):
        state_fips = int(ucgid_str[9:])
        return {"type": "state", "state_fips": state_fips}
    elif ucgid_str.startswith("5001800US"):
        # Format: 5001800USSSDD where SS is state FIPS, DD is district
        state_and_district = ucgid_str[9:]
        state_fips = int(state_and_district[:2])
        district_number = int(state_and_district[2:])
        # Convert district 00 to 01 for at-large districts (matches create_initial_strata.py)
        # Also convert DC's delegate district 98 to 01
        if district_number == 0 or (
            state_fips == 11 and district_number == 98
        ):
            district_number = 1
        cd_geoid = state_fips * 100 + district_number
        return {
            "type": "district",
            "state_fips": state_fips,
            "district_number": district_number,
            "congressional_district_geoid": cd_geoid,
        }
    else:
        raise ValueError(f"Unknown UCGID format: {ucgid_str}")


def get_geographic_strata(session: Session) -> Dict:
    """Fetch existing geographic strata from database.

    Returns dict mapping:
        - 'national': stratum_id for US
        - 'state': {state_fips: stratum_id}
        - 'district': {congressional_district_geoid: stratum_id}
    """
    strata_map = {
        "national": None,
        "state": {},
        "district": {},
    }

    # Get all strata with stratum_group_id = 1 (geographic strata)
    stmt = select(Stratum).where(Stratum.stratum_group_id == 1)
    geographic_strata = session.exec(stmt).unique().all()

    for stratum in geographic_strata:
        # Get constraints for this stratum
        constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == stratum.stratum_id
            )
        ).all()

        if not constraints:
            # No constraints = national level
            strata_map["national"] = stratum.stratum_id
        else:
            # Check constraint types
            constraint_vars = {
                c.constraint_variable: c.value for c in constraints
            }

            if "congressional_district_geoid" in constraint_vars:
                cd_geoid = int(constraint_vars["congressional_district_geoid"])
                strata_map["district"][cd_geoid] = stratum.stratum_id
            elif "state_fips" in constraint_vars:
                state_fips = int(constraint_vars["state_fips"])
                strata_map["state"][state_fips] = stratum.stratum_id

    return strata_map
