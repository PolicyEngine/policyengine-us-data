"""Two-pass proportional rescaling of geographic targets.

Ensures that child-level targets (state, congressional district) sum
to their parent-level target for each (variable, period, reform_id)
group.  Original source values are preserved in ``raw_value``.

Pass 1: scale state targets so they sum to the national target.
Pass 2: scale CD targets so they sum to their (corrected) state target.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sqlmodel import Session, create_engine, select

from policyengine_us_data.db.create_database_tables import (
    Stratum,
    StratumConstraint,
    Target,
)
from policyengine_us_data.storage import STORAGE_FOLDER

logger = logging.getLogger(__name__)

# Type alias for the grouping key
GroupKey = Tuple[str, int, int]  # (variable, period, reform_id)


def _resolve_geo_ancestor(
    stratum: Stratum,
    stratum_cache: Dict[int, Stratum],
) -> Optional[int]:
    """Walk up the parent chain to find the nearest geo ancestor.

    A geographic stratum has ``stratum_group_id == 1``.

    Returns:
        The ``stratum_id`` of the geographic ancestor, or ``None``
        if the stratum itself is geographic (group 1) or has no
        geographic ancestor.
    """
    if stratum.stratum_group_id == 1:
        return stratum.stratum_id

    current = stratum
    while current.parent_stratum_id is not None:
        parent = stratum_cache.get(current.parent_stratum_id)
        if parent is None:
            break
        if parent.stratum_group_id == 1:
            return parent.stratum_id
        current = parent
    return None


def _classify_geo_level(
    stratum: Stratum,
    stratum_cache: Dict[int, Stratum],
) -> Optional[str]:
    """Classify a geographic stratum as national/state/district.

    Works by inspecting the constraint variables on the stratum.
    """
    if stratum.stratum_group_id != 1:
        return None

    constraint_vars = {
        c.constraint_variable for c in (stratum.constraints_rel or [])
    }

    if "congressional_district_geoid" in constraint_vars:
        return "district"
    elif "state_fips" in constraint_vars:
        return "state"
    elif stratum.parent_stratum_id is None:
        return "national"

    return None


def _get_state_fips(stratum: Stratum) -> Optional[int]:
    """Extract state_fips from a stratum's constraints."""
    for c in stratum.constraints_rel or []:
        if c.constraint_variable == "state_fips":
            return int(c.value)
    return None


def _get_cd_geoid(stratum: Stratum) -> Optional[int]:
    """Extract congressional_district_geoid from constraints."""
    for c in stratum.constraints_rel or []:
        if c.constraint_variable == "congressional_district_geoid":
            return int(c.value)
    return None


def reconcile_targets(session: Session) -> Dict[str, int]:
    """Run two-pass proportional rescaling on all active targets.

    Args:
        session: An open SQLModel session.

    Returns:
        A dict with counts: ``scaled_state``, ``scaled_cd``,
        ``skipped_zero_sum``.
    """
    stats = {
        "scaled_state": 0,
        "scaled_cd": 0,
        "skipped_zero_sum": 0,
    }

    # Load all strata into a cache
    all_strata = session.exec(select(Stratum)).unique().all()
    stratum_cache: Dict[int, Stratum] = {s.stratum_id: s for s in all_strata}

    # Build geo-stratum lookup: geo_stratum_id -> (level, state_fips)
    geo_info: Dict[int, dict] = {}
    for s in all_strata:
        level = _classify_geo_level(s, stratum_cache)
        if level is not None:
            info = {"level": level}
            if level == "state":
                info["state_fips"] = _get_state_fips(s)
            elif level == "district":
                info["cd_geoid"] = _get_cd_geoid(s)
                # Derive state_fips from CD geoid (first 1-2 digits)
                cd = _get_cd_geoid(s)
                if cd is not None:
                    info["state_fips"] = cd // 100
            geo_info[s.stratum_id] = info

    # Load all active targets
    stmt = select(Target).where(Target.active == True)  # noqa: E712
    all_targets = session.exec(stmt).all()

    # Group targets by (variable, period, reform_id)
    groups: Dict[GroupKey, List[Target]] = defaultdict(list)
    for t in all_targets:
        key = (t.variable, t.period, t.reform_id)
        groups[key].append(t)

    for key, targets in groups.items():
        variable, period, reform_id = key

        # Resolve each target to its geographic ancestor
        national_targets = []
        state_targets: Dict[int, List[Target]] = defaultdict(list)
        district_targets: Dict[int, List[Target]] = defaultdict(list)

        for t in targets:
            stratum = stratum_cache.get(t.stratum_id)
            if stratum is None:
                continue

            geo_ancestor_id = _resolve_geo_ancestor(stratum, stratum_cache)
            if geo_ancestor_id is None:
                continue

            info = geo_info.get(geo_ancestor_id)
            if info is None:
                continue

            level = info["level"]
            if level == "national":
                national_targets.append(t)
            elif level == "state":
                fips = info.get("state_fips")
                if fips is not None:
                    state_targets[fips].append(t)
            elif level == "district":
                fips = info.get("state_fips")
                if fips is not None:
                    district_targets[fips].append(t)

        # Pass 1: scale states -> national
        if national_targets and state_targets:
            national_value = sum(
                (t.raw_value if t.raw_value is not None else t.value)
                for t in national_targets
                if t.value is not None
            )

            all_state_targets = []
            for fips_targets in state_targets.values():
                all_state_targets.extend(fips_targets)

            state_sum = sum(
                (t.raw_value if t.raw_value is not None else t.value)
                for t in all_state_targets
                if t.value is not None
            )

            if state_sum == 0:
                logger.warning(
                    "State sum is zero for %s/%s/%s; skipping",
                    variable,
                    period,
                    reform_id,
                )
                stats["skipped_zero_sum"] += 1
            elif national_value != 0:
                scale = national_value / state_sum
                if abs(scale - 1.0) > 1e-9:
                    logger.info(
                        "Scaling %d state targets for %s/%s: "
                        "factor=%.6f (state_sum=%.2f, "
                        "national=%.2f)",
                        len(all_state_targets),
                        variable,
                        period,
                        scale,
                        state_sum,
                        national_value,
                    )
                    for t in all_state_targets:
                        if t.value is None:
                            continue
                        base = (
                            t.raw_value if t.raw_value is not None else t.value
                        )
                        t.raw_value = base
                        t.value = base * scale
                        stats["scaled_state"] += 1

        # Pass 2: scale CDs -> state
        # After pass 1, state target .value is already corrected.
        for fips, cd_targets_for_state in district_targets.items():
            state_targets_for_fips = state_targets.get(fips, [])
            if not state_targets_for_fips:
                continue

            state_value = sum(
                t.value for t in state_targets_for_fips if t.value is not None
            )

            cd_sum = sum(
                (t.raw_value if t.raw_value is not None else t.value)
                for t in cd_targets_for_state
                if t.value is not None
            )

            if cd_sum == 0:
                logger.warning(
                    "CD sum is zero for state %s, %s/%s/%s; " "skipping",
                    fips,
                    variable,
                    period,
                    reform_id,
                )
                stats["skipped_zero_sum"] += 1
            elif state_value != 0:
                scale = state_value / cd_sum
                if abs(scale - 1.0) > 1e-9:
                    logger.info(
                        "Scaling %d CD targets for state %s, "
                        "%s/%s: factor=%.6f",
                        len(cd_targets_for_state),
                        fips,
                        variable,
                        period,
                        scale,
                    )
                    for t in cd_targets_for_state:
                        if t.value is None:
                            continue
                        base = (
                            t.raw_value if t.raw_value is not None else t.value
                        )
                        t.raw_value = base
                        t.value = base * scale
                        stats["scaled_cd"] += 1

    session.commit()

    logger.info(
        "Reconciliation complete: %d state targets scaled, "
        "%d CD targets scaled, %d groups skipped (zero sum)",
        stats["scaled_state"],
        stats["scaled_cd"],
        stats["skipped_zero_sum"],
    )

    return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    db_uri = f"sqlite:///{STORAGE_FOLDER / 'calibration' / 'policy_data.db'}"
    engine = create_engine(db_uri)

    with Session(engine) as session:
        stats = reconcile_targets(session)

    logger.info("Final stats: %s", stats)


if __name__ == "__main__":
    main()
