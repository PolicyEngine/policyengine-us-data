import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlmodel import Session, select
import sqlalchemy as sa

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
)
from policyengine_us_data.storage import STORAGE_FOLDER

DEFAULT_DATASET = str(STORAGE_FOLDER / "stratified_extended_cps_2024.h5")


def etl_argparser(
    description: str,
    extra_args_fn=None,
) -> Tuple[argparse.Namespace, int]:
    """Shared argument parsing and dataset-year derivation for ETL scripts.

    Args:
        description: Description for the argparse help text.
        extra_args_fn: Optional callable that receives the parser to add
            extra arguments before parsing.

    Returns:
        (args, year) where *year* is derived from the dataset's
        ``default_calculation_period``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "Source dataset (local path or HuggingFace URL). "
            "The year is derived from the dataset's "
            "default_calculation_period. Default: %(default)s"
        ),
    )
    if extra_args_fn is not None:
        extra_args_fn(parser)

    args = parser.parse_args()

    if (
        not args.dataset.startswith("hf://")
        and not Path(args.dataset).exists()
    ):
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}\n"
            f"Either build it locally (`make data`) or pass a "
            f"HuggingFace URL via --dataset hf://policyengine/..."
        )

    from policyengine_us import Microsimulation

    print(f"Loading dataset: {args.dataset}")
    sim = Microsimulation(dataset=args.dataset)
    year = int(sim.default_calculation_period)
    print(f"Derived year from dataset: {year}")

    return args, year


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
    elif ucgid_str.startswith("5001800US") or ucgid_str.startswith(
        "5001900US"
    ):
        # 5001800US = 118th Congress, 5001900US = 119th Congress
        state_and_district = ucgid_str[9:]
        state_fips = int(state_and_district[:2])
        district_number = int(state_and_district[2:])
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

    Returns:
        dict mapping:
        - 'national': stratum_id for US
        - 'state': {state_fips: stratum_id}
        - 'district': {congressional_district_geoid: stratum_id}
    """
    strata_map = {
        "national": None,
        "state": {},
        "district": {},
    }

    stmt = select(Stratum).where(Stratum.stratum_group_id == 1)
    geographic_strata = session.exec(stmt).unique().all()

    for stratum in geographic_strata:
        constraints = session.exec(
            select(StratumConstraint).where(
                StratumConstraint.stratum_id == stratum.stratum_id
            )
        ).all()

        if not constraints:
            strata_map["national"] = stratum.stratum_id
        else:
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
