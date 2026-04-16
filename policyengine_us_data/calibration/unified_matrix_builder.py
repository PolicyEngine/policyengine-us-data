"""
Unified sparse matrix builder for clone-based calibration.

Builds a sparse calibration matrix for cloned+geography-assigned CPS
records. Processes clone-by-clone: for each clone, sets each
record's state_fips to its assigned value, simulates, and extracts
variable values.

Matrix shape: (n_targets, n_records * n_clones)
Column ordering: index i = clone_idx * n_records + record_idx
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sqlalchemy import create_engine, text

from policyengine_us_data.db.create_database_tables import create_or_replace_views
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.census import STATE_ABBREV_TO_FIPS, STATE_NAME_TO_FIPS
from policyengine_us_data.calibration.signatures import (
    build_chunk_lineage_signature,
    signature_mismatches,
)
from policyengine_us_data.calibration.calibration_utils import (
    get_calculated_variables,
    apply_op,
    get_geo_level,
)

logger = logging.getLogger(__name__)

_GEO_VARS = {
    "state_fips",
    "state_code",
    "congressional_district_geoid",
}

COUNTY_DEPENDENT_VARS = {
    "aca_ptc",
}


def _validate_chunked_geography(geography, n_total: int) -> None:
    """Validate geography arrays before chunked matrix slicing."""
    required_arrays = (
        "block_geoid",
        "cd_geoid",
        "county_fips",
        "state_fips",
    )
    for field in required_arrays:
        if not hasattr(geography, field):
            raise ValueError(f"geography is missing required field '{field}'")
        actual = len(getattr(geography, field))
        if actual != n_total:
            raise ValueError(
                f"geography.{field} has length {actual}, expected {n_total} "
                "(n_records * n_clones)"
            )


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds for progress logs."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _current_rss_mb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        return None


def _load_chunk_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chunk_manifest(path: Path, signature: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"signature": signature}, f, indent=2, sort_keys=True)
        f.write("\n")


def _validate_chunk_manifest(path: Path, expected_signature: dict) -> None:
    if not path.exists():
        raise ValueError(
            f"Cannot resume chunk cache at {path.parent}: missing chunk manifest"
        )
    stored = _load_chunk_manifest(path)
    stored_signature = stored.get("signature")
    if stored_signature is None:
        raise ValueError(f"Chunk manifest at {path} is missing its signature")
    fatal, _ = signature_mismatches(stored_signature, expected_signature)
    if fatal:
        joined = "; ".join(fatal)
        raise ValueError(f"Chunk cache lineage mismatch for {path.parent}: {joined}")


def _has_existing_chunk_cache(coo_dir: Path) -> bool:
    return any(coo_dir.glob("chunk_*.npz"))


def _make_neutralize_variable_reform(variable_name: str):
    from policyengine_core.reforms import Reform

    class NeutralizeVariable(Reform):
        def apply(self):
            self.neutralize_variable(variable_name)

    NeutralizeVariable.__name__ = f"Neutralize_{variable_name}"
    return NeutralizeVariable


def _compute_reform_household_values(
    dataset_path: str,
    time_period: int,
    state: int,
    n_hh: int,
    reform_vars: list,
    baseline_income_tax: np.ndarray,
) -> dict:
    """Compute repeal-based household income tax deltas for target vars."""
    from policyengine_us import Microsimulation

    reform_hh = {}
    if not reform_vars:
        return reform_hh

    state_input = np.full(n_hh, state, dtype=np.int32)
    for var in reform_vars:
        try:
            reform_sim = Microsimulation(
                dataset=dataset_path,
                reform=_make_neutralize_variable_reform(var),
            )
            reform_sim.set_input("state_fips", time_period, state_input)
            for calc_var in get_calculated_variables(reform_sim):
                reform_sim.delete_arrays(calc_var)
            reform_income_tax = reform_sim.calculate(
                "income_tax",
                time_period,
                map_to="household",
            ).values.astype(np.float32)
            reform_hh[var] = reform_income_tax - baseline_income_tax
        except Exception as exc:
            logger.warning(
                "Cannot calculate tax expenditure '%s' for state %d: %s",
                var,
                state,
                exc,
            )

    return reform_hh


def _merged_person_constraint_dtype(
    state_values: dict,
    states: np.ndarray,
    variable: str,
):
    """Pick a merge dtype that preserves string-valued constraints."""
    dtypes = []
    for state in states:
        person = state_values[int(state)]["person"]
        if variable in person:
            dtypes.append(np.asarray(person[variable]).dtype)
    if not dtypes:
        return None
    merged_dtype = np.result_type(*dtypes)
    if np.issubdtype(merged_dtype, np.number) or np.issubdtype(
        merged_dtype,
        np.bool_,
    ):
        return np.float32
    return merged_dtype


def _compute_single_state(
    dataset_path: str,
    time_period: int,
    state: int,
    n_hh: int,
    target_vars: list,
    constraint_vars: list,
    reform_vars: list,
    rerandomize_takeup: bool,
    affected_targets: dict,
):
    """Compute household/person/entity values for one state.

    Top-level function (not a method) so it is picklable for
    ``ProcessPoolExecutor``.

    Args:
        dataset_path: Path to the base CPS h5 file.
        time_period: Tax year for simulation.
        state: State FIPS code.
        n_hh: Number of household records.
        target_vars: Target variable names (list for determinism).
        constraint_vars: Constraint variable names (list).
        rerandomize_takeup: Force takeup=True if True.
        affected_targets: Takeup-affected target info dict.

    Returns:
        (state_fips, {"hh": {...}, "person": {...}, "entity": {...}})
    """
    from policyengine_us import Microsimulation
    from policyengine_us_data.utils.takeup import SIMPLE_TAKEUP_VARS
    from policyengine_us_data.calibration.calibration_utils import (
        get_calculated_variables,
    )

    state_sim = Microsimulation(dataset=dataset_path)

    state_sim.set_input(
        "state_fips",
        time_period,
        np.full(n_hh, state, dtype=np.int32),
    )
    for var in get_calculated_variables(state_sim):
        state_sim.delete_arrays(var)

    hh = {}
    for var in target_vars:
        if var.endswith("_count"):
            continue
        try:
            hh[var] = state_sim.calculate(
                var,
                time_period,
                map_to="household",
            ).values.astype(np.float32)
        except Exception as exc:
            logger.warning(
                "Cannot calculate '%s' for state %d: %s",
                var,
                state,
                exc,
            )

    person = {}
    for var in constraint_vars:
        try:
            raw = state_sim.calculate(
                var,
                time_period,
                map_to="person",
            ).values
            try:
                person[var] = raw.astype(np.float32)
            except (ValueError, TypeError):
                person[var] = raw
        except Exception as exc:
            logger.warning(
                "Cannot calculate constraint '%s' for state %d: %s",
                var,
                state,
                exc,
            )

    baseline_income_tax = None
    reform_hh = {}
    if reform_vars:
        baseline_income_tax = state_sim.calculate(
            "income_tax",
            time_period,
            map_to="household",
        ).values.astype(np.float32)
        reform_hh = _compute_reform_household_values(
            dataset_path,
            time_period,
            state,
            n_hh,
            reform_vars,
            baseline_income_tax,
        )

    if rerandomize_takeup:
        for spec in SIMPLE_TAKEUP_VARS:
            entity = spec["entity"]
            n_ent = len(state_sim.calculate(f"{entity}_id", map_to=entity).values)
            state_sim.set_input(
                spec["variable"],
                time_period,
                np.ones(n_ent, dtype=bool),
            )
        for var in get_calculated_variables(state_sim):
            state_sim.delete_arrays(var)

    entity_vals = {}
    if rerandomize_takeup:
        for tvar, info in affected_targets.items():
            entity_level = info["entity"]
            try:
                entity_vals[tvar] = state_sim.calculate(
                    tvar,
                    time_period,
                    map_to=entity_level,
                ).values.astype(np.float32)
            except Exception as exc:
                logger.warning(
                    "Cannot calculate entity-level '%s' (map_to=%s) for state %d: %s",
                    tvar,
                    entity_level,
                    state,
                    exc,
                )

    entity_wf_false = {}
    if rerandomize_takeup:
        has_tu_target = any(
            info["entity"] == "tax_unit" for info in affected_targets.values()
        )
        if has_tu_target:
            n_tu = len(state_sim.calculate("tax_unit_id", map_to="tax_unit").values)
            state_sim.set_input(
                "would_file_taxes_voluntarily",
                time_period,
                np.zeros(n_tu, dtype=bool),
            )
            for var in get_calculated_variables(state_sim):
                state_sim.delete_arrays(var)
            for tvar, info in affected_targets.items():
                if info["entity"] != "tax_unit":
                    continue
                entity_wf_false[tvar] = state_sim.calculate(
                    tvar,
                    time_period,
                    map_to="tax_unit",
                ).values.astype(np.float32)

    return (
        state,
        {
            "hh": hh,
            "person": person,
            "reform_hh": reform_hh,
            "entity": entity_vals,
            "entity_wf_false": entity_wf_false,
        },
    )


def _compute_single_state_group_counties(
    dataset_path: str,
    time_period: int,
    state_fips: int,
    counties: list,
    n_hh: int,
    county_dep_targets: list,
    rerandomize_takeup: bool,
    affected_targets: dict,
):
    """Compute county-dependent values for all counties in one state.

    Top-level function (not a method) so it is picklable for
    ``ProcessPoolExecutor``. Creates one ``Microsimulation`` per
    state and reuses it across counties within that state.

    Args:
        dataset_path: Path to the base CPS h5 file.
        time_period: Tax year for simulation.
        state_fips: State FIPS code for this group.
        counties: List of county FIPS strings in this state.
        n_hh: Number of household records.
        county_dep_targets: County-dependent target var names.
        rerandomize_takeup: Force takeup=True if True.
        affected_targets: Takeup-affected target info dict.

    Returns:
        list of (county_fips_str, {"hh": {...}, "entity": {...}})
    """
    from policyengine_us import Microsimulation
    from policyengine_us_data.utils.takeup import SIMPLE_TAKEUP_VARS
    from policyengine_us_data.calibration.calibration_utils import (
        get_calculated_variables,
    )
    from policyengine_us_data.calibration.block_assignment import (
        get_county_enum_index_from_fips,
    )

    state_sim = Microsimulation(dataset=dataset_path)

    state_sim.set_input(
        "state_fips",
        time_period,
        np.full(n_hh, state_fips, dtype=np.int32),
    )

    original_takeup = {}
    if rerandomize_takeup:
        for spec in SIMPLE_TAKEUP_VARS:
            entity = spec["entity"]
            original_takeup[spec["variable"]] = (
                entity,
                state_sim.calculate(
                    spec["variable"],
                    time_period,
                    map_to=entity,
                ).values.copy(),
            )

    results = []
    for county_fips in counties:
        county_idx = get_county_enum_index_from_fips(county_fips)
        state_sim.set_input(
            "county",
            time_period,
            np.full(n_hh, county_idx, dtype=np.int32),
        )
        if county_fips == "06037":
            state_sim.set_input(
                "zip_code",
                time_period,
                np.full(n_hh, "90001"),
            )
        if rerandomize_takeup:
            for vname, (ent, orig) in original_takeup.items():
                state_sim.set_input(vname, time_period, orig)
        for var in get_calculated_variables(state_sim):
            if var not in ("county", "zip_code"):
                state_sim.delete_arrays(var)

        hh = {}
        for var in county_dep_targets:
            if var.endswith("_count"):
                continue
            try:
                hh[var] = state_sim.calculate(
                    var,
                    time_period,
                    map_to="household",
                ).values.astype(np.float32)
            except Exception as exc:
                logger.warning(
                    "Cannot calculate '%s' for county %s: %s",
                    var,
                    county_fips,
                    exc,
                )

        if rerandomize_takeup:
            for spec in SIMPLE_TAKEUP_VARS:
                entity = spec["entity"]
                n_ent = len(state_sim.calculate(f"{entity}_id", map_to=entity).values)
                state_sim.set_input(
                    spec["variable"],
                    time_period,
                    np.ones(n_ent, dtype=bool),
                )
            for var in get_calculated_variables(state_sim):
                if var not in ("county", "zip_code"):
                    state_sim.delete_arrays(var)

        entity_vals = {}
        if rerandomize_takeup:
            for tvar, info in affected_targets.items():
                entity_level = info["entity"]
                try:
                    entity_vals[tvar] = state_sim.calculate(
                        tvar,
                        time_period,
                        map_to=entity_level,
                    ).values.astype(np.float32)
                except Exception as exc:
                    logger.warning(
                        "Cannot calculate entity-level '%s' for county %s: %s",
                        tvar,
                        county_fips,
                        exc,
                    )

        results.append(
            (
                county_fips,
                {
                    "hh": hh,
                    "entity": entity_vals,
                },
            )
        )

    return results


# ---------------------------------------------------------------
# Clone-loop parallelisation helpers (module-level for pickling)
# ---------------------------------------------------------------

_CLONE_SHARED: dict = {}


def _init_clone_worker(shared_data: dict) -> None:
    """Initialise worker process with shared read-only data.

    Called once per worker at ``ProcessPoolExecutor`` startup so the
    ~50-200 MB payload is pickled *per worker* (not per clone).
    """
    _CLONE_SHARED.update(shared_data)


def _assemble_clone_values_standalone(
    state_values: dict,
    clone_states: np.ndarray,
    person_hh_indices: np.ndarray,
    target_vars: set,
    constraint_vars: set,
    reform_vars: set = None,
    county_values: dict = None,
    clone_counties: np.ndarray = None,
    county_dependent_vars: set = None,
) -> tuple:
    """Standalone clone-value assembly (no ``self``).

    Identical logic to
    ``UnifiedMatrixBuilder._assemble_clone_values`` but usable
    from a worker process.
    """
    n_records = len(clone_states)
    n_persons = len(person_hh_indices)
    person_states = clone_states[person_hh_indices]
    unique_clone_states = np.unique(clone_states)
    cdv = county_dependent_vars or set()

    state_masks = {int(s): clone_states == s for s in unique_clone_states}
    unique_person_states = np.unique(person_states)
    person_state_masks = {int(s): person_states == s for s in unique_person_states}
    county_masks = {}
    unique_counties = None
    if clone_counties is not None and county_values:
        unique_counties = np.unique(clone_counties)
        county_masks = {c: clone_counties == c for c in unique_counties}

    hh_vars: dict = {}
    for var in target_vars:
        if var.endswith("_count"):
            continue
        if var in cdv and county_values and clone_counties is not None:
            first_county = unique_counties[0]
            if var not in county_values.get(first_county, {}).get("hh", {}):
                continue
            arr = np.empty(n_records, dtype=np.float32)
            for county in unique_counties:
                mask = county_masks[county]
                county_hh = county_values.get(county, {}).get("hh", {})
                if var in county_hh:
                    arr[mask] = county_hh[var][mask]
                else:
                    st = int(county[:2])
                    arr[mask] = state_values[st]["hh"][var][mask]
            hh_vars[var] = arr
        else:
            if var not in state_values[unique_clone_states[0]]["hh"]:
                continue
            arr = np.empty(n_records, dtype=np.float32)
            for state in unique_clone_states:
                mask = state_masks[int(state)]
                arr[mask] = state_values[int(state)]["hh"][var][mask]
            hh_vars[var] = arr

    person_vars: dict = {}
    for var in constraint_vars:
        dtype = _merged_person_constraint_dtype(
            state_values,
            unique_person_states,
            var,
        )
        if dtype is None:
            continue
        arr = np.empty(n_persons, dtype=dtype)
        for state in unique_person_states:
            mask = person_state_masks[int(state)]
            arr[mask] = state_values[int(state)]["person"][var][mask]
        person_vars[var] = arr

    reform_hh_vars: dict = {}
    for var in reform_vars or set():
        if not any(
            var in state_values[int(state)].get("reform_hh", {})
            for state in unique_clone_states
        ):
            continue
        arr = np.zeros(n_records, dtype=np.float32)
        for state in unique_clone_states:
            mask = state_masks[int(state)]
            reform_data = state_values[int(state)].get("reform_hh", {})
            if var in reform_data:
                arr[mask] = reform_data[var][mask]
        reform_hh_vars[var] = arr

    return hh_vars, person_vars, reform_hh_vars


def _evaluate_constraints_standalone(
    constraints,
    person_vars: dict,
    entity_rel: pd.DataFrame,
    household_ids: np.ndarray,
    n_households: int,
) -> np.ndarray:
    """Standalone constraint evaluation (no class instance).

    Evaluates person-level constraints and aggregates to
    household level via .any().
    """
    if not constraints:
        return np.ones(n_households, dtype=bool)

    n_persons = len(entity_rel)
    person_mask = np.ones(n_persons, dtype=bool)

    for c in constraints:
        var = c["variable"]
        if var not in person_vars:
            logger.warning(
                "Constraint var '%s' not in precomputed person_vars",
                var,
            )
            return np.zeros(n_households, dtype=bool)
        vals = person_vars[var]
        person_mask &= apply_op(vals, c["operation"], c["value"])

    df = entity_rel.copy()
    df["satisfies"] = person_mask
    hh_mask = df.groupby("household_id")["satisfies"].any()
    return np.array([hh_mask.get(hid, False) for hid in household_ids])


def _calculate_target_values_standalone(
    target_variable: str,
    non_geo_constraints: list,
    n_households: int,
    hh_vars: dict,
    reform_hh_vars: dict,
    person_vars: dict,
    entity_rel: pd.DataFrame,
    household_ids: np.ndarray,
    variable_entity_map: dict,
    reform_id: int = 0,
) -> np.ndarray:
    """Standalone target-value calculation (no class instance).

    Uses ``variable_entity_map`` dict for entity resolution
    (picklable, unlike ``tax_benefit_system``).
    """
    is_count = target_variable.endswith("_count")

    if not is_count:
        mask = _evaluate_constraints_standalone(
            non_geo_constraints,
            person_vars,
            entity_rel,
            household_ids,
            n_households,
        )
        source_vars = reform_hh_vars if reform_id > 0 else hh_vars
        vals = source_vars.get(target_variable)
        if vals is None:
            return np.zeros(n_households, dtype=np.float32)
        return (vals * mask).astype(np.float32)

    # Count target: entity-aware counting
    n_persons = len(entity_rel)
    person_mask = np.ones(n_persons, dtype=bool)

    for c in non_geo_constraints:
        var = c["variable"]
        if var not in person_vars:
            return np.zeros(n_households, dtype=np.float32)
        cv = person_vars[var]
        person_mask &= apply_op(cv, c["operation"], c["value"])

    target_entity = variable_entity_map.get(target_variable)
    if target_entity is None:
        return np.zeros(n_households, dtype=np.float32)

    if target_entity == "household":
        if non_geo_constraints:
            mask = _evaluate_constraints_standalone(
                non_geo_constraints,
                person_vars,
                entity_rel,
                household_ids,
                n_households,
            )
            return mask.astype(np.float32)
        return np.ones(n_households, dtype=np.float32)

    if target_entity == "person":
        er = entity_rel.copy()
        er["satisfies"] = person_mask
        filtered = er[er["satisfies"]]
        counts = filtered.groupby("household_id")["person_id"].nunique()
    else:
        eid_col = f"{target_entity}_id"
        er = entity_rel.copy()
        er["satisfies"] = person_mask
        entity_ok = er.groupby(eid_col)["satisfies"].any()
        unique = er[["household_id", eid_col]].drop_duplicates()
        unique["entity_ok"] = unique[eid_col].map(entity_ok)
        filtered = unique[unique["entity_ok"]]
        counts = filtered.groupby("household_id")[eid_col].nunique()

    return np.array(
        [counts.get(hid, 0) for hid in household_ids],
        dtype=np.float32,
    )


def _process_single_clone(
    clone_idx: int,
    col_start: int,
    col_end: int,
    cache_path: str,
) -> tuple:
    """Process one clone in a worker process.

    Reads shared read-only data from ``_CLONE_SHARED``
    (populated by ``_init_clone_worker``).  Writes COO
    entries as a compressed ``.npz`` file to *cache_path*.

    Args:
        clone_idx: Zero-based clone index.
        col_start: First column index for this clone.
        col_end: One-past-last column index.
        cache_path: File path for output ``.npz``.

    Returns:
        (clone_idx, n_nonzero) tuple.
    """
    sd = _CLONE_SHARED

    # Unpack shared data
    geo_states = sd["geography_state_fips"]
    geo_counties = sd["geography_county_fips"]
    geo_blocks = sd["geography_block_geoid"]
    state_values = sd["state_values"]
    county_values = sd["county_values"]
    person_hh_indices = sd["person_hh_indices"]
    unique_variables = sd["unique_variables"]
    unique_constraint_vars = sd["unique_constraint_vars"]
    county_dep_targets = sd["county_dep_targets"]
    target_variables = sd["target_variables"]
    target_reform_ids = sd["target_reform_ids"]
    target_geo_info = sd["target_geo_info"]
    non_geo_constraints_list = sd["non_geo_constraints_list"]
    reform_vars = sd["reform_vars"]
    n_records = sd["n_records"]
    n_total = sd["n_total"]
    n_targets = sd["n_targets"]
    state_to_cols = sd["state_to_cols"]
    cd_to_cols = sd["cd_to_cols"]
    entity_rel = sd["entity_rel"]
    household_ids = sd["household_ids"]
    variable_entity_map = sd["variable_entity_map"]
    do_takeup = sd["rerandomize_takeup"]
    affected_target_info = sd["affected_target_info"]
    entity_hh_idx_map = sd.get("entity_hh_idx_map", {})
    entity_to_person_idx = sd.get("entity_to_person_idx", {})
    precomputed_rates = sd.get("precomputed_rates", {})
    reported_takeup_anchors = sd.get("reported_takeup_anchors", {})

    # Slice geography for this clone
    clone_states = geo_states[col_start:col_end]
    clone_counties = geo_counties[col_start:col_end]

    # Assemble hh/person values from precomputed state/county
    hh_vars, person_vars, reform_hh_vars = _assemble_clone_values_standalone(
        state_values,
        clone_states,
        person_hh_indices,
        unique_variables,
        unique_constraint_vars,
        reform_vars=reform_vars,
        county_values=county_values,
        clone_counties=clone_counties,
        county_dependent_vars=county_dep_targets,
    )

    # Takeup re-randomisation
    if do_takeup and affected_target_info:
        from policyengine_us_data.utils.takeup import (
            SIMPLE_TAKEUP_VARS,
            compute_block_takeup_for_entities,
        )

        clone_blocks = geo_blocks[col_start:col_end]

        # Phase 1: compute non-target draws (would_file) FIRST
        wf_draws = {}
        for spec in SIMPLE_TAKEUP_VARS:
            if spec.get("target") is not None:
                continue
            var_name = spec["variable"]
            entity = spec["entity"]
            rate_key = spec["rate_key"]
            if rate_key not in precomputed_rates:
                continue
            ent_hh = entity_hh_idx_map[entity]
            ent_blocks = clone_blocks[ent_hh]
            ent_hh_ids = household_ids[ent_hh]
            ent_ci = np.full(len(ent_hh), clone_idx, dtype=np.int64)
            draws = compute_block_takeup_for_entities(
                var_name,
                precomputed_rates[rate_key],
                ent_blocks,
                ent_hh_ids,
                ent_ci,
            )
            wf_draws[entity] = draws
            if var_name in person_vars:
                pidx = entity_to_person_idx[entity]
                person_vars[var_name] = draws[pidx].astype(np.float32)

        # Phase 2: target loop with would_file blending
        for tvar, info in affected_target_info.items():
            if tvar.endswith("_count"):
                continue
            entity_level = info["entity"]
            takeup_var = info["takeup_var"]
            ent_hh = entity_hh_idx_map[entity_level]
            n_ent = len(ent_hh)
            ent_states = clone_states[ent_hh]

            ent_eligible = np.zeros(n_ent, dtype=np.float32)
            if tvar in county_dep_targets and county_values:
                ent_counties = clone_counties[ent_hh]
                for cfips in np.unique(ent_counties):
                    m = ent_counties == cfips
                    cv = county_values.get(cfips, {}).get("entity", {})
                    if tvar in cv:
                        ent_eligible[m] = cv[tvar][m]
                    else:
                        st = int(cfips[:2])
                        sv = state_values[st]["entity"]
                        if tvar in sv:
                            ent_eligible[m] = sv[tvar][m]
            else:
                for st in np.unique(ent_states):
                    m = ent_states == st
                    sv = state_values[int(st)]["entity"]
                    if tvar in sv:
                        ent_eligible[m] = sv[tvar][m]

            # Blend: for tax_unit targets, select between
            # all-takeup-true and would_file=false values
            if entity_level == "tax_unit" and "tax_unit" in wf_draws:
                ent_wf_false = np.zeros(n_ent, dtype=np.float32)
                if tvar in county_dep_targets and county_values:
                    ent_counties = clone_counties[ent_hh]
                    for cfips in np.unique(ent_counties):
                        m = ent_counties == cfips
                        cv = county_values.get(cfips, {}).get("entity_wf_false", {})
                        if tvar in cv:
                            ent_wf_false[m] = cv[tvar][m]
                        else:
                            st = int(cfips[:2])
                            sv = state_values[st].get("entity_wf_false", {})
                            if tvar in sv:
                                ent_wf_false[m] = sv[tvar][m]
                else:
                    for st in np.unique(ent_states):
                        m = ent_states == st
                        sv = state_values[int(st)].get("entity_wf_false", {})
                        if tvar in sv:
                            ent_wf_false[m] = sv[tvar][m]
                ent_eligible = np.where(
                    wf_draws["tax_unit"],
                    ent_eligible,
                    ent_wf_false,
                )

            ent_blocks = clone_blocks[ent_hh]
            ent_hh_ids = household_ids[ent_hh]
            ent_ci = np.full(n_ent, clone_idx, dtype=np.int64)

            ent_takeup = compute_block_takeup_for_entities(
                takeup_var,
                precomputed_rates[info["rate_key"]],
                ent_blocks,
                ent_hh_ids,
                ent_ci,
                reported_mask=reported_takeup_anchors.get(takeup_var),
            )

            ent_values = (ent_eligible * ent_takeup).astype(np.float32)

            hh_result = np.zeros(n_records, dtype=np.float32)
            np.add.at(hh_result, ent_hh, ent_values)
            hh_vars[tvar] = hh_result

            if tvar in person_vars:
                pidx = entity_to_person_idx[entity_level]
                person_vars[tvar] = ent_values[pidx]

    # Build COO entries for every target row
    mask_cache: dict = {}
    count_cache: dict = {}
    rows_list: list = []
    cols_list: list = []
    vals_list: list = []

    for row_idx in range(n_targets):
        variable = target_variables[row_idx]
        reform_id = target_reform_ids[row_idx]
        geo_level, geo_id = target_geo_info[row_idx]
        non_geo = non_geo_constraints_list[row_idx]

        if geo_level == "district":
            all_geo_cols = cd_to_cols.get(
                str(geo_id),
                np.array([], dtype=np.int64),
            )
        elif geo_level == "state":
            all_geo_cols = state_to_cols.get(
                int(geo_id),
                np.array([], dtype=np.int64),
            )
        else:
            all_geo_cols = np.arange(n_total)

        clone_cols = all_geo_cols[
            (all_geo_cols >= col_start) & (all_geo_cols < col_end)
        ]
        if len(clone_cols) == 0:
            continue

        rec_indices = clone_cols - col_start

        constraint_key = tuple(
            sorted(
                (
                    c["variable"],
                    c["operation"],
                    c["value"],
                )
                for c in non_geo
            )
        )

        if variable.endswith("_count"):
            vkey = (variable, constraint_key, reform_id)
            if vkey not in count_cache:
                count_cache[vkey] = _calculate_target_values_standalone(
                    variable,
                    non_geo,
                    n_records,
                    hh_vars,
                    reform_hh_vars,
                    person_vars,
                    entity_rel,
                    household_ids,
                    variable_entity_map,
                    reform_id=reform_id,
                )
            values = count_cache[vkey]
        else:
            source_vars = reform_hh_vars if reform_id > 0 else hh_vars
            if variable not in source_vars:
                continue
            if constraint_key not in mask_cache:
                mask_cache[constraint_key] = _evaluate_constraints_standalone(
                    non_geo,
                    person_vars,
                    entity_rel,
                    household_ids,
                    n_records,
                )
            mask = mask_cache[constraint_key]
            values = source_vars[variable] * mask

        vals = values[rec_indices]
        nonzero = vals != 0
        if nonzero.any():
            rows_list.append(
                np.full(
                    nonzero.sum(),
                    row_idx,
                    dtype=np.int32,
                )
            )
            cols_list.append(clone_cols[nonzero].astype(np.int32))
            vals_list.append(vals[nonzero])

    # Write COO
    if rows_list:
        cr = np.concatenate(rows_list)
        cc = np.concatenate(cols_list)
        cv = np.concatenate(vals_list)
    else:
        cr = np.array([], dtype=np.int32)
        cc = np.array([], dtype=np.int32)
        cv = np.array([], dtype=np.float32)

    np.savez_compressed(cache_path, rows=cr, cols=cc, vals=cv)
    return clone_idx, len(cv)


class UnifiedMatrixBuilder:
    """Build sparse calibration matrix for cloned CPS records.

    Processes clone-by-clone: each clone's records get their
    assigned geography, are simulated, and the results fill
    the corresponding columns.

    Args:
        db_uri: SQLAlchemy database URI.
        time_period: Tax year for calibration (e.g. 2024).
        dataset_path: Path to the base extended CPS h5 file.
    """

    def __init__(
        self,
        db_uri: str,
        time_period: int,
        dataset_path: Optional[str] = None,
    ):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        # Existing SQLite checkpoints may carry an older target_overview view.
        create_or_replace_views(self.engine)
        self.time_period = time_period
        self.dataset_path = dataset_path
        self._entity_rel_cache = None
        self._target_overview_columns = None

    # ---------------------------------------------------------------
    # Entity relationships
    # ---------------------------------------------------------------

    def _build_entity_relationship(self, sim) -> pd.DataFrame:
        if self._entity_rel_cache is not None:
            return self._entity_rel_cache

        self._entity_rel_cache = pd.DataFrame(
            {
                "person_id": sim.calculate("person_id", map_to="person").values,
                "household_id": sim.calculate("household_id", map_to="person").values,
                "tax_unit_id": sim.calculate("tax_unit_id", map_to="person").values,
                "spm_unit_id": sim.calculate("spm_unit_id", map_to="person").values,
            }
        )
        return self._entity_rel_cache

    # ---------------------------------------------------------------
    # Per-state precomputation
    # ---------------------------------------------------------------

    def _build_state_values(
        self,
        sim,
        target_vars: set,
        constraint_vars: set,
        reform_vars: set = None,
        geography=None,
        rerandomize_takeup: bool = True,
        workers: int = 1,
    ) -> dict:
        """Precompute household/person/entity values per state.

        Creates a fresh Microsimulation per state to prevent
        cross-state cache pollution (stale intermediate values
        from one state leaking into another's calculations).

        County-dependent variables (e.g. aca_ptc) are computed
        here as a state-level fallback; county-level overrides
        are applied later via ``_build_county_values``.

        Args:
            sim: Microsimulation instance (unused; kept for API
                compatibility).
            target_vars: Set of target variable names.
            constraint_vars: Set of constraint variable names.
            geography: GeographyAssignment with state_fips.
            rerandomize_takeup: If True, force takeup=True and
                also store entity-level eligible amounts for
                takeup-affected targets.
            workers: Number of parallel worker processes.
                When >1, uses ProcessPoolExecutor.

        Returns:
            {state_fips: {
                'hh': {var: array},
                'person': {var: array},
                'entity': {var: array}  # only if rerandomize
            }}
        """
        from policyengine_us_data.utils.takeup import (
            TAKEUP_AFFECTED_TARGETS,
        )

        if geography is None:
            raise ValueError("geography is required")

        unique_states = sorted(set(int(s) for s in geography.state_fips))
        n_hh = geography.n_records

        logger.info(
            "Per-state precomputation: %d states, "
            "%d hh vars, %d constraint vars "
            "(fresh sim per state, workers=%d)",
            len(unique_states),
            len([v for v in target_vars if not v.endswith("_count")]),
            len(constraint_vars),
            workers,
        )

        # Identify takeup-affected targets before the state loop
        affected_targets = {}
        if rerandomize_takeup:
            for tvar in target_vars:
                for key, info in TAKEUP_AFFECTED_TARGETS.items():
                    if tvar == key or tvar.startswith(key):
                        affected_targets[tvar] = info
                        break

        # Convert sets to sorted lists for deterministic iteration
        target_vars_list = sorted(target_vars)
        constraint_vars_list = sorted(constraint_vars)
        reform_vars_list = sorted(reform_vars or set())

        state_values = {}

        if workers > 1:
            from concurrent.futures import (
                ProcessPoolExecutor,
                as_completed,
            )

            logger.info(
                "Parallel state precomputation with %d workers",
                workers,
            )
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _compute_single_state,
                        self.dataset_path,
                        self.time_period,
                        st,
                        n_hh,
                        target_vars_list,
                        constraint_vars_list,
                        reform_vars_list,
                        rerandomize_takeup,
                        affected_targets,
                    ): st
                    for st in unique_states
                }
                completed = 0
                for future in as_completed(futures):
                    st = futures[future]
                    try:
                        sf, vals = future.result()
                        state_values[sf] = vals
                        completed += 1
                        if completed % 10 == 0 or completed == 1:
                            logger.info(
                                "State %d/%d complete",
                                completed,
                                len(unique_states),
                            )
                    except Exception as exc:
                        for f in futures:
                            f.cancel()
                        raise RuntimeError(f"State {st} failed: {exc}") from exc
        else:
            from policyengine_us import Microsimulation
            from policyengine_us_data.utils.takeup import (
                SIMPLE_TAKEUP_VARS,
            )

            for i, state in enumerate(unique_states):
                state_sim = Microsimulation(dataset=self.dataset_path)

                state_sim.set_input(
                    "state_fips",
                    self.time_period,
                    np.full(n_hh, state, dtype=np.int32),
                )
                for var in get_calculated_variables(state_sim):
                    state_sim.delete_arrays(var)

                hh = {}
                for var in target_vars:
                    if var.endswith("_count"):
                        continue
                    try:
                        hh[var] = state_sim.calculate(
                            var,
                            self.time_period,
                            map_to="household",
                        ).values.astype(np.float32)
                    except Exception as exc:
                        logger.warning(
                            "Cannot calculate '%s' for state %d: %s",
                            var,
                            state,
                            exc,
                        )

                person = {}
                for var in constraint_vars:
                    try:
                        raw = state_sim.calculate(
                            var,
                            self.time_period,
                            map_to="person",
                        ).values
                        try:
                            person[var] = raw.astype(np.float32)
                        except (ValueError, TypeError):
                            person[var] = raw
                    except Exception as exc:
                        logger.warning(
                            "Cannot calculate constraint '%s' for state %d: %s",
                            var,
                            state,
                            exc,
                        )

                reform_hh = {}
                if reform_vars_list:
                    baseline_income_tax = state_sim.calculate(
                        "income_tax",
                        self.time_period,
                        map_to="household",
                    ).values.astype(np.float32)
                    reform_hh = _compute_reform_household_values(
                        self.dataset_path,
                        self.time_period,
                        state,
                        n_hh,
                        reform_vars_list,
                        baseline_income_tax,
                    )

                if rerandomize_takeup:
                    for spec in SIMPLE_TAKEUP_VARS:
                        entity = spec["entity"]
                        n_ent = len(
                            state_sim.calculate(f"{entity}_id", map_to=entity).values
                        )
                        state_sim.set_input(
                            spec["variable"],
                            self.time_period,
                            np.ones(n_ent, dtype=bool),
                        )
                    for var in get_calculated_variables(state_sim):
                        state_sim.delete_arrays(var)

                entity_vals = {}
                if rerandomize_takeup:
                    for tvar, info in affected_targets.items():
                        entity_level = info["entity"]
                        try:
                            entity_vals[tvar] = state_sim.calculate(
                                tvar,
                                self.time_period,
                                map_to=entity_level,
                            ).values.astype(np.float32)
                        except Exception as exc:
                            logger.warning(
                                "Cannot calculate entity-level "
                                "'%s' (map_to=%s) for "
                                "state %d: %s",
                                tvar,
                                entity_level,
                                state,
                                exc,
                            )

                entity_wf_false = {}
                if rerandomize_takeup:
                    has_tu_target = any(
                        info["entity"] == "tax_unit"
                        for info in affected_targets.values()
                    )
                    if has_tu_target:
                        n_tu = len(
                            state_sim.calculate(
                                "tax_unit_id",
                                map_to="tax_unit",
                            ).values
                        )
                        state_sim.set_input(
                            "would_file_taxes_voluntarily",
                            self.time_period,
                            np.zeros(n_tu, dtype=bool),
                        )
                        for var in get_calculated_variables(state_sim):
                            state_sim.delete_arrays(var)
                        for (
                            tvar,
                            info,
                        ) in affected_targets.items():
                            if info["entity"] != "tax_unit":
                                continue
                            entity_wf_false[tvar] = state_sim.calculate(
                                tvar,
                                self.time_period,
                                map_to="tax_unit",
                            ).values.astype(np.float32)

                state_values[state] = {
                    "hh": hh,
                    "person": person,
                    "reform_hh": reform_hh,
                    "entity": entity_vals,
                    "entity_wf_false": entity_wf_false,
                }
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(
                        "State %d/%d complete",
                        i + 1,
                        len(unique_states),
                    )

        logger.info(
            "Per-state precomputation done: %d states",
            len(state_values),
        )
        return state_values

    def _build_county_values(
        self,
        sim,
        county_dep_targets: set,
        geography,
        rerandomize_takeup: bool = True,
        county_level: bool = True,
        workers: int = 1,
    ) -> dict:
        """Precompute county-dependent variable values per county.

        Only iterates over COUNTY_DEPENDENT_VARS that actually
        benefit from per-county computation. All other target
        variables use state-level values from _build_state_values.

        Creates a fresh Microsimulation per state group to prevent
        cross-state cache pollution. Counties within the same state
        share a simulation since within-state recalculation is clean
        (only cross-state switches cause pollution).

        When county_level=False, returns an empty dict immediately
        (all values come from state-level precomputation).

        Args:
            sim: Microsimulation instance (unused; kept for API
                compatibility).
            county_dep_targets: Subset of target vars that depend
                on county (intersection of targets with
                COUNTY_DEPENDENT_VARS).
            geography: GeographyAssignment with county_fips.
            rerandomize_takeup: If True, force takeup=True and
                also store entity-level eligible amounts for
                takeup-affected targets.
            county_level: If True, iterate counties within each
                state. If False, return empty dict (skip county
                computation entirely).
            workers: Number of parallel worker processes.
                When >1, uses ProcessPoolExecutor.

        Returns:
            {county_fips_str: {
                'hh': {var: array},
                'entity': {var: array}
            }}
        """
        if not county_level or not county_dep_targets:
            logger.info(
                "Skipping county precomputation (county_level=%s, targets=%d)",
                county_level,
                len(county_dep_targets),
            )
            return {}

        from policyengine_us_data.utils.takeup import (
            TAKEUP_AFFECTED_TARGETS,
        )

        unique_counties = sorted(set(geography.county_fips))
        n_hh = geography.n_records

        state_to_counties = defaultdict(list)
        for county in unique_counties:
            state_to_counties[int(county[:2])].append(county)

        logger.info(
            "Per-county precomputation: %d counties in %d "
            "states, %d county-dependent vars "
            "(fresh sim per state, workers=%d)",
            len(unique_counties),
            len(state_to_counties),
            len(county_dep_targets),
            workers,
        )

        affected_targets = {}
        if rerandomize_takeup:
            for tvar in county_dep_targets:
                for key, info in TAKEUP_AFFECTED_TARGETS.items():
                    if tvar == key or tvar.startswith(key):
                        affected_targets[tvar] = info
                        break

        # Convert to sorted list for deterministic iteration
        county_dep_targets_list = sorted(county_dep_targets)

        county_values = {}

        if workers > 1:
            from concurrent.futures import (
                ProcessPoolExecutor,
                as_completed,
            )

            logger.info(
                "Parallel county precomputation with %d workers (%d state groups)",
                workers,
                len(state_to_counties),
            )
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _compute_single_state_group_counties,
                        self.dataset_path,
                        self.time_period,
                        sf,
                        counties,
                        n_hh,
                        county_dep_targets_list,
                        rerandomize_takeup,
                        affected_targets,
                    ): sf
                    for sf, counties in sorted(state_to_counties.items())
                }
                completed = 0
                county_count = 0
                for future in as_completed(futures):
                    sf = futures[future]
                    try:
                        results = future.result()
                        for cfips, vals in results:
                            county_values[cfips] = vals
                            county_count += 1
                        completed += 1
                        if county_count % 500 == 0 or completed == 1:
                            logger.info(
                                "County %d/%d complete (%d/%d state groups)",
                                county_count,
                                len(unique_counties),
                                completed,
                                len(state_to_counties),
                            )
                    except Exception as exc:
                        for f in futures:
                            f.cancel()
                        raise RuntimeError(f"State group {sf} failed: {exc}") from exc
        else:
            county_count = 0
            for sf, counties in sorted(state_to_counties.items()):
                results = _compute_single_state_group_counties(
                    self.dataset_path,
                    self.time_period,
                    sf,
                    counties,
                    n_hh,
                    county_dep_targets_list,
                    rerandomize_takeup,
                    affected_targets,
                )
                for county_fips, vals in results:
                    county_values[county_fips] = vals
                    county_count += 1
                    if county_count % 500 == 0 or county_count == 1:
                        logger.info(
                            "County %d/%d complete",
                            county_count,
                            len(unique_counties),
                        )

        logger.info(
            "Per-county precomputation done: %d counties",
            len(county_values),
        )
        return county_values

    def _assemble_clone_values(
        self,
        state_values: dict,
        clone_states: np.ndarray,
        person_hh_indices: np.ndarray,
        target_vars: set,
        constraint_vars: set,
        reform_vars: set = None,
        county_values: dict = None,
        clone_counties: np.ndarray = None,
        county_dependent_vars: set = None,
    ) -> tuple:
        """Assemble per-clone values from state/county precomputation.

        For each target variable, selects values from either
        county_values (if the var is county-dependent) or
        state_values (otherwise) using numpy fancy indexing.

        Args:
            state_values: Output of _build_state_values.
            clone_states: State FIPS per record for this clone.
            person_hh_indices: Maps person index to household
                index (0..n_records-1).
            target_vars: Set of target variable names.
            constraint_vars: Set of constraint variable names.
            county_values: Output of _build_county_values.
            clone_counties: County FIPS per record for this
                clone (str array).
            county_dependent_vars: Set of var names that should
                be looked up by county instead of state.

        Returns:
            (hh_vars, person_vars, reform_hh_vars) where hh_vars maps
            baseline variables to household-level float32 arrays,
            person_vars maps constraint variables to person-level arrays,
            and reform_hh_vars maps repeal-based expenditure targets to
            household-level arrays.
        """
        n_records = len(clone_states)
        n_persons = len(person_hh_indices)
        person_states = clone_states[person_hh_indices]
        unique_clone_states = np.unique(clone_states)
        cdv = county_dependent_vars or set()

        # Pre-compute masks to avoid recomputing per variable
        state_masks = {int(s): clone_states == s for s in unique_clone_states}
        unique_person_states = np.unique(person_states)
        person_state_masks = {int(s): person_states == s for s in unique_person_states}
        county_masks = {}
        unique_counties = None
        if clone_counties is not None and county_values:
            unique_counties = np.unique(clone_counties)
            county_masks = {c: clone_counties == c for c in unique_counties}

        hh_vars = {}
        for var in target_vars:
            if var.endswith("_count"):
                continue
            if var in cdv and county_values and clone_counties is not None:
                first_county = unique_counties[0]
                if var not in county_values.get(first_county, {}).get("hh", {}):
                    continue
                arr = np.empty(n_records, dtype=np.float32)
                for county in unique_counties:
                    mask = county_masks[county]
                    county_hh = county_values.get(county, {}).get("hh", {})
                    if var in county_hh:
                        arr[mask] = county_hh[var][mask]
                    else:
                        st = int(county[:2])
                        arr[mask] = state_values[st]["hh"][var][mask]
                hh_vars[var] = arr
            else:
                if var not in state_values[unique_clone_states[0]]["hh"]:
                    continue
                arr = np.empty(n_records, dtype=np.float32)
                for state in unique_clone_states:
                    mask = state_masks[int(state)]
                    arr[mask] = state_values[int(state)]["hh"][var][mask]
                hh_vars[var] = arr

        person_vars = {}
        for var in constraint_vars:
            dtype = _merged_person_constraint_dtype(
                state_values,
                unique_person_states,
                var,
            )
            if dtype is None:
                continue
            arr = np.empty(n_persons, dtype=dtype)
            for state in unique_person_states:
                mask = person_state_masks[int(state)]
                arr[mask] = state_values[int(state)]["person"][var][mask]
            person_vars[var] = arr

        reform_hh_vars = {}
        for var in reform_vars or set():
            if not any(
                var in state_values[int(state)].get("reform_hh", {})
                for state in unique_clone_states
            ):
                continue
            arr = np.zeros(n_records, dtype=np.float32)
            for state in unique_clone_states:
                mask = state_masks[int(state)]
                reform_data = state_values[int(state)].get("reform_hh", {})
                if var in reform_data:
                    arr[mask] = reform_data[var][mask]
            reform_hh_vars[var] = arr

        return hh_vars, person_vars, reform_hh_vars

    # ---------------------------------------------------------------
    # Database queries
    # ---------------------------------------------------------------

    def _get_stratum_constraints(self, stratum_id: int) -> List[dict]:
        query = """
        SELECT constraint_variable AS variable, operation, value
        FROM stratum_constraints
        WHERE stratum_id = :stratum_id
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"stratum_id": int(stratum_id)},
            )
        return df.to_dict("records")

    def _get_target_overview_columns(self) -> set:
        if self._target_overview_columns is None:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text("PRAGMA table_info(target_overview)")
                ).fetchall()
            self._target_overview_columns = {row[1] for row in rows}
        return self._target_overview_columns

    def _query_targets(self, target_filter: dict) -> pd.DataFrame:
        """Query targets via target_overview view with
        best-period selection."""
        and_conditions = []

        if "domain_variables" in target_filter:
            dvs = target_filter["domain_variables"]
            ph = ",".join(f"'{dv}'" for dv in dvs)
            and_conditions.append(f"tv.domain_variable IN ({ph})")

        if "variables" in target_filter:
            vs = ",".join(f"'{v}'" for v in target_filter["variables"])
            and_conditions.append(f"tv.variable IN ({vs})")

        if "target_ids" in target_filter:
            ids = ",".join(map(str, target_filter["target_ids"]))
            and_conditions.append(f"tv.target_id IN ({ids})")

        if "stratum_ids" in target_filter:
            ids = ",".join(map(str, target_filter["stratum_ids"]))
            and_conditions.append(f"tv.stratum_id IN ({ids})")

        if not and_conditions:
            where_clause = "1=1"
        else:
            where_clause = " AND ".join(f"({c})" for c in and_conditions)

        if "reform_id" in self._get_target_overview_columns():
            query = f"""
            WITH filtered_targets AS (
                SELECT tv.target_id, tv.stratum_id, tv.variable, tv.reform_id,
                       tv.value, tv.period, tv.geo_level,
                       tv.geographic_id, tv.domain_variable
                FROM target_overview tv
                WHERE tv.active = 1
                  AND ({where_clause})
            ),
            best_periods AS (
                SELECT stratum_id, variable, reform_id,
                    CASE
                        WHEN MAX(CASE WHEN period <= :time_period
                                 THEN period END) IS NOT NULL
                        THEN MAX(CASE WHEN period <= :time_period
                                 THEN period END)
                        ELSE MIN(period)
                    END as best_period
                FROM filtered_targets
                GROUP BY stratum_id, variable, reform_id
            )
            SELECT ft.*
            FROM filtered_targets ft
            JOIN best_periods bp
                ON ft.stratum_id = bp.stratum_id
                AND ft.variable = bp.variable
                AND ft.reform_id = bp.reform_id
                AND ft.period = bp.best_period
            ORDER BY ft.target_id
            """
        else:
            query = f"""
            WITH filtered_targets AS (
                SELECT tv.target_id, tv.stratum_id, tv.variable,
                       0 AS reform_id, tv.value, tv.period, tv.geo_level,
                       tv.geographic_id, tv.domain_variable
                FROM target_overview tv
                WHERE tv.active = 1
                  AND ({where_clause})
            ),
            best_periods AS (
                SELECT stratum_id, variable,
                    CASE
                        WHEN MAX(CASE WHEN period <= :time_period
                                 THEN period END) IS NOT NULL
                        THEN MAX(CASE WHEN period <= :time_period
                                 THEN period END)
                        ELSE MIN(period)
                    END as best_period
                FROM filtered_targets
                GROUP BY stratum_id, variable
            )
            SELECT ft.*
            FROM filtered_targets ft
            JOIN best_periods bp
                ON ft.stratum_id = bp.stratum_id
                AND ft.variable = bp.variable
                AND ft.period = bp.best_period
            ORDER BY ft.target_id
            """

        with self.engine.connect() as conn:
            return pd.read_sql(
                query,
                conn,
                params={"time_period": self.time_period},
            )

    def get_district_agi_targets(self) -> Dict[str, float]:
        """Return current-law district AGI targets for geography assignment."""
        targets_df = self._query_targets({"variables": ["adjusted_gross_income"]})
        district_rows = targets_df[
            (targets_df["geo_level"] == "district")
            & (targets_df["reform_id"] == 0)
            & (targets_df["domain_variable"].fillna("") == "")
        ]
        return {
            str(row["geographic_id"]): float(row["value"])
            for _, row in district_rows.iterrows()
        }

    # ---------------------------------------------------------------
    # Uprating
    # ---------------------------------------------------------------

    def _calculate_uprating_factors(self, params) -> dict:
        factors = {}
        query = (
            "SELECT DISTINCT period FROM targets "
            "WHERE period IS NOT NULL ORDER BY period"
        )
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            years_needed = [row[0] for row in result]

        for from_year in years_needed:
            if from_year == self.time_period:
                factors[(from_year, "cpi")] = 1.0
                factors[(from_year, "pop")] = 1.0
                continue

            try:
                cpi_from = params.gov.bls.cpi.cpi_u(from_year)
                cpi_to = params.gov.bls.cpi.cpi_u(self.time_period)
                factors[(from_year, "cpi")] = float(cpi_to / cpi_from)
            except Exception:
                factors[(from_year, "cpi")] = 1.0

            try:
                pop_from = params.calibration.gov.census.populations.total(from_year)
                pop_to = params.calibration.gov.census.populations.total(
                    self.time_period
                )
                factors[(from_year, "pop")] = float(pop_to / pop_from)
            except Exception:
                factors[(from_year, "pop")] = 1.0

        return factors

    def _get_uprating_info(
        self,
        variable: str,
        period: int,
        factors: dict,
    ) -> Tuple[float, str]:
        if period == self.time_period:
            return 1.0, "none"

        is_count = variable.endswith("_count")
        uprating_type = "pop" if is_count else "cpi"
        factor = factors.get((period, uprating_type), 1.0)
        return factor, uprating_type

    def _load_aca_ptc_factors(
        self,
    ) -> Dict[int, Dict[str, float]]:
        csv_candidates = {}
        for path in STORAGE_FOLDER.glob("aca_ptc_multipliers_2022_*.csv"):
            suffix = path.stem.removeprefix("aca_ptc_multipliers_2022_")
            if suffix.isdigit():
                csv_candidates[int(suffix)] = path

        eligible_years = [year for year in csv_candidates if year <= self.time_period]
        if eligible_years:
            csv_path = csv_candidates[max(eligible_years)]
        else:
            csv_path = csv_candidates[min(csv_candidates)]
        df = pd.read_csv(csv_path)
        result = {}
        for _, row in df.iterrows():
            fips_str = STATE_NAME_TO_FIPS.get(row["state"])
            if fips_str is None:
                fips_str = STATE_ABBREV_TO_FIPS.get(row["state"])
            if fips_str is None:
                continue
            fips_int = int(fips_str)
            result[fips_int] = {
                "tax_unit_count": row["vol_mult"],
                "aca_ptc": row["vol_mult"] * row["val_mult"],
            }
        return result

    def _get_state_uprating_factors(
        self,
        domain: str,
        targets_df: pd.DataFrame,
        national_factors: dict,
    ) -> Dict[int, Dict[str, float]]:
        state_rows = targets_df[
            (targets_df["domain_variable"] == domain)
            & (targets_df["geo_level"] == "state")
        ]
        state_fips_list = state_rows["geographic_id"].unique()
        variables = state_rows["variable"].unique()

        if domain == "aca_ptc":
            csv_factors = self._load_aca_ptc_factors()
        else:
            csv_factors = None

        result = {}
        for sf in state_fips_list:
            state_int = int(sf)
            var_factors = {}

            if csv_factors and state_int in csv_factors:
                for var in variables:
                    var_factors[var] = csv_factors[state_int].get(var, 1.0)
            else:
                for var in variables:
                    row = state_rows[
                        (state_rows["geographic_id"] == sf)
                        & (state_rows["variable"] == var)
                    ]
                    if row.empty:
                        var_factors[var] = 1.0
                        continue
                    period = row.iloc[0]["period"]
                    factor, _ = self._get_uprating_info(var, period, national_factors)
                    var_factors[var] = factor

            result[state_int] = var_factors

        return result

    def _apply_hierarchical_uprating(
        self,
        targets_df: pd.DataFrame,
        hierarchical_domains: List[str],
        national_factors: dict,
    ) -> pd.DataFrame:
        """Apply state-level uprating and reconcile CDs.

        Two factors per CD row:
        - hif: state_original / sum(cd_originals)
        - uprating_factor: state-specific scaling

        Final CD value = original * hif * uprating_factor.
        """
        df = targets_df.copy()
        df["hif"] = np.nan
        df["state_uprating_factor"] = np.nan
        rows_to_drop = []

        for domain in hierarchical_domains:
            domain_mask = df["domain_variable"] == domain
            state_factors = self._get_state_uprating_factors(
                domain, df, national_factors
            )
            state_mask = domain_mask & (df["geo_level"] == "state")
            district_mask = domain_mask & (df["geo_level"] == "district")

            for sf, var_factors in state_factors.items():
                for var, uf in var_factors.items():
                    state_row = df[
                        state_mask
                        & (df["geographic_id"] == str(sf))
                        & (df["variable"] == var)
                    ]
                    if state_row.empty:
                        continue
                    state_original = state_row.iloc[0]["original_value"]

                    def _cd_in_state(g, s=sf):
                        try:
                            return int(g) // 100 == s
                        except (ValueError, TypeError):
                            return False

                    cd_mask = (
                        district_mask
                        & (df["variable"] == var)
                        & df["geographic_id"].apply(_cd_in_state)
                    )
                    cd_rows = df[cd_mask]
                    if cd_rows.empty:
                        continue

                    cd_original_sum = cd_rows["original_value"].sum()
                    if cd_original_sum == 0:
                        continue

                    hif = state_original / cd_original_sum
                    for cd_idx in cd_rows.index:
                        df.at[cd_idx, "hif"] = hif
                        df.at[cd_idx, "state_uprating_factor"] = uf
                        df.at[cd_idx, "value"] = (
                            df.at[cd_idx, "original_value"] * hif * uf
                        )

            # Drop national/state rows used for reconciliation
            national_mask = domain_mask & (df["geo_level"] == "national")
            for idx in df[national_mask | state_mask].index:
                row = df.loc[idx]
                if row["period"] != self.time_period:
                    rows_to_drop.append(idx)

        if rows_to_drop:
            df = df.drop(index=rows_to_drop).reset_index(drop=True)

        df["target_period"] = self.time_period
        return df

    def print_uprating_summary(self, targets_df: pd.DataFrame) -> None:
        has_state_uf = "state_uprating_factor" in targets_df.columns
        if has_state_uf:
            eff = targets_df["state_uprating_factor"].fillna(
                targets_df["uprating_factor"]
            )
        else:
            eff = targets_df["uprating_factor"]

        uprated = targets_df[eff != 1.0]
        if len(uprated) == 0:
            print("No targets were uprated.")
            return

        print("\n" + "=" * 60)
        print("UPRATING SUMMARY")
        print("=" * 60)
        print(f"Uprated {len(uprated)} of {len(targets_df)} targets")
        period_counts = uprated["period"].value_counts().sort_index()
        for period, count in period_counts.items():
            print(f"  Period {period}: {count} targets")
        factors = eff[eff != 1.0]
        print(f"  Factor range: [{factors.min():.4f}, {factors.max():.4f}]")

    # ---------------------------------------------------------------
    # Target naming
    # ---------------------------------------------------------------

    @staticmethod
    def _make_target_name(
        variable: str,
        constraints: List[dict],
        reform_id: int = 0,
    ) -> str:
        geo_parts: List[str] = []
        for c in constraints:
            if c["variable"] == "state_fips":
                geo_parts.append(f"state_{c['value']}")
            elif c["variable"] == "congressional_district_geoid":
                geo_parts.append(f"cd_{c['value']}")

        parts: List[str] = []
        parts.append("/".join(geo_parts) if geo_parts else "national")
        if reform_id > 0:
            parts.append(f"{variable}_expenditure")
        else:
            parts.append(variable)

        non_geo = [c for c in constraints if c["variable"] not in _GEO_VARS]
        if non_geo:
            strs = [f"{c['variable']}{c['operation']}{c['value']}" for c in non_geo]
            parts.append("[" + ",".join(strs) + "]")

        return "/".join(parts)

    # ---------------------------------------------------------------
    # Clone simulation
    # ---------------------------------------------------------------

    def _simulate_clone(
        self,
        clone_state_fips: np.ndarray,
        n_records: int,
        variables: set,
        sim_modifier=None,
        clone_idx: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], object]:
        """Simulate one clone with assigned geography.

        Args:
            clone_state_fips: State FIPS per record, shape
                (n_records,).
            n_records: Number of base records.
            variables: Target variable names to compute.
            sim_modifier: Optional callback(sim, clone_idx)
                called after state_fips is set but before
                cache clearing. Used for takeup
                re-randomization.
            clone_idx: Clone index passed to sim_modifier.

        Returns:
            (var_values, sim) where var_values maps variable
            name to household-level float32 array.
        """
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.dataset_path)
        sim.set_input(
            "state_fips",
            self.time_period,
            clone_state_fips.astype(np.int32),
        )
        if sim_modifier is not None:
            sim_modifier(sim, clone_idx)
        for var in get_calculated_variables(sim):
            sim.delete_arrays(var)

        var_values: Dict[str, np.ndarray] = {}
        for var in variables:
            if var.endswith("_count"):
                continue
            try:
                var_values[var] = sim.calculate(
                    var,
                    self.time_period,
                    map_to="household",
                ).values.astype(np.float32)
            except Exception as exc:
                logger.warning("Cannot calculate '%s': %s", var, exc)

        return var_values, sim

    # ---------------------------------------------------------------
    # Main build method
    # ---------------------------------------------------------------

    def build_matrix(
        self,
        geography,
        sim,
        target_filter: Optional[dict] = None,
        hierarchical_domains: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        sim_modifier=None,
        rerandomize_takeup: bool = True,
        county_level: bool = True,
        workers: int = 1,
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]:
        """Build sparse calibration matrix.

        Two-phase build: (1) simulate each clone and save
        COO entries to disk, (2) assemble CSR from caches.

        Args:
            geography: GeographyAssignment with state_fips,
                cd_geoid, block_geoid arrays and n_records,
                n_clones attributes.
            sim: Microsimulation for parameters and entity
                relationships.
            target_filter: Dict for target_overview filtering.
            hierarchical_domains: Domain names for
                hierarchical uprating + CD reconciliation.
            cache_dir: Directory for per-clone COO caches.
                If None, COO data held in memory.
            sim_modifier: Optional callback(sim, clone_idx)
                called per clone after state_fips is set but
                before cache clearing. Use for takeup
                re-randomization.
            rerandomize_takeup: If True, use geo-salted
                entity-level takeup draws instead of base h5
                takeup values for takeup-affected targets.
            county_level: If True (default), iterate counties
                within each state during precomputation. If
                False, compute once per state and alias to all
                counties (faster for county-invariant vars).

        Returns:
            (targets_df, X_sparse, target_names)
        """
        n_records = geography.n_records
        n_clones = geography.n_clones
        n_total = n_records * n_clones
        self._coo_parts = ([], [], [])

        # 1. Query and uprate targets
        targets_df = self._query_targets(target_filter or {})
        if len(targets_df) == 0:
            raise ValueError("No targets found matching filter")

        params = sim.tax_benefit_system.parameters
        uprating_factors = self._calculate_uprating_factors(params)
        targets_df["original_value"] = targets_df["value"].copy()
        targets_df["uprating_factor"] = targets_df.apply(
            lambda row: self._get_uprating_info(
                row["variable"],
                row["period"],
                uprating_factors,
            )[0],
            axis=1,
        )
        targets_df["value"] = (
            targets_df["original_value"] * targets_df["uprating_factor"]
        )

        if hierarchical_domains:
            targets_df = self._apply_hierarchical_uprating(
                targets_df,
                hierarchical_domains,
                uprating_factors,
            )

        n_targets = len(targets_df)

        # 2. Sort targets by geographic level
        targets_df["_geo_level"] = targets_df["geographic_id"].apply(get_geo_level)
        targets_df = targets_df.sort_values(
            ["_geo_level", "variable", "reform_id", "geographic_id"]
        )
        targets_df = targets_df.drop(columns=["_geo_level"]).reset_index(drop=True)

        # 3. Build column index structures from geography
        state_col_lists: Dict[int, list] = defaultdict(list)
        cd_col_lists: Dict[str, list] = defaultdict(list)
        for col in range(n_total):
            state_col_lists[int(geography.state_fips[col])].append(col)
            cd_col_lists[str(geography.cd_geoid[col])].append(col)
        state_to_cols = {s: np.array(c) for s, c in state_col_lists.items()}
        cd_to_cols = {cd: np.array(c) for cd, c in cd_col_lists.items()}

        # 4. Pre-process targets: resolve constraints
        constraint_cache: Dict[int, List[dict]] = {}
        target_geo_info: List[Tuple[str, str]] = []
        target_names: List[str] = []
        non_geo_constraints_list: List[List[dict]] = []
        target_reform_ids: List[int] = []

        for _, row in targets_df.iterrows():
            sid = int(row["stratum_id"])
            if sid not in constraint_cache:
                constraint_cache[sid] = self._get_stratum_constraints(sid)
            constraints = constraint_cache[sid]

            geo_level = row["geo_level"]
            geo_id = row["geographic_id"]
            target_geo_info.append((geo_level, geo_id))

            non_geo = [c for c in constraints if c["variable"] not in _GEO_VARS]
            non_geo_constraints_list.append(non_geo)
            reform_id = int(row.get("reform_id", 0))
            target_reform_ids.append(reform_id)

            target_names.append(
                self._make_target_name(
                    str(row["variable"]),
                    constraints,
                    reform_id=reform_id,
                )
            )

        unique_variables = set(targets_df["variable"].values)
        reform_variables = {
            str(row["variable"])
            for _, row in targets_df.iterrows()
            if int(row.get("reform_id", 0)) > 0
        }

        # 5a. Collect unique constraint variables
        unique_constraint_vars = set()
        for constraints in non_geo_constraints_list:
            for c in constraints:
                unique_constraint_vars.add(c["variable"])

        # 5b. Per-state precomputation (51 sims on one object)
        self._entity_rel_cache = None
        state_values = self._build_state_values(
            sim,
            unique_variables,
            unique_constraint_vars,
            reform_variables,
            geography,
            rerandomize_takeup=rerandomize_takeup,
            workers=workers,
        )

        # 5b-county. Per-county precomputation for county-dependent vars
        county_dep_targets = unique_variables & COUNTY_DEPENDENT_VARS
        county_values = self._build_county_values(
            sim,
            county_dep_targets,
            geography,
            rerandomize_takeup=rerandomize_takeup,
            county_level=county_level,
            workers=workers,
        )

        # 5c. State-independent structures (computed once)
        entity_rel = self._build_entity_relationship(sim)
        household_ids = sim.calculate("household_id", map_to="household").values
        person_hh_ids = sim.calculate("household_id", map_to="person").values
        hh_id_to_idx = {int(hid): idx for idx, hid in enumerate(household_ids)}
        person_hh_indices = np.array([hh_id_to_idx[int(hid)] for hid in person_hh_ids])
        tax_benefit_system = sim.tax_benefit_system

        # Pre-extract entity keys so workers don't need
        # the unpicklable TaxBenefitSystem object.
        variable_entity_map: Dict[str, str] = {}
        for var in unique_variables:
            if var.endswith("_count") and var in tax_benefit_system.variables:
                variable_entity_map[var] = tax_benefit_system.variables[var].entity.key

        # 5c-extra: Entity-to-household index maps for takeup
        affected_target_info = {}
        if rerandomize_takeup:
            from policyengine_us_data.utils.takeup import (
                TAKEUP_AFFECTED_TARGETS,
                compute_block_takeup_for_entities,
                reported_subsidized_marketplace_by_tax_unit,
            )
            from policyengine_us_data.parameters import (
                load_take_up_rate,
            )

            # Build entity-to-household index arrays
            spm_to_hh_id = (
                entity_rel.groupby("spm_unit_id")["household_id"].first().to_dict()
            )
            spm_ids = sim.calculate("spm_unit_id", map_to="spm_unit").values
            spm_hh_idx = np.array(
                [hh_id_to_idx[int(spm_to_hh_id[int(sid)])] for sid in spm_ids]
            )

            tu_to_hh_id = (
                entity_rel.groupby("tax_unit_id")["household_id"].first().to_dict()
            )
            tu_ids = sim.calculate("tax_unit_id", map_to="tax_unit").values
            tu_hh_idx = np.array(
                [hh_id_to_idx[int(tu_to_hh_id[int(tid)])] for tid in tu_ids]
            )

            entity_hh_idx_map = {
                "spm_unit": spm_hh_idx,
                "tax_unit": tu_hh_idx,
                "person": person_hh_indices,
            }

            reported_takeup_anchors = {}
            with h5py.File(self.dataset_path, "r") as f:
                period_key = str(self.time_period)
                if (
                    "reported_has_subsidized_marketplace_health_coverage_at_interview"
                    in f
                    and period_key
                    in f[
                        "reported_has_subsidized_marketplace_health_coverage_at_interview"
                    ]
                ):
                    person_marketplace = f[
                        "reported_has_subsidized_marketplace_health_coverage_at_interview"
                    ][period_key][...].astype(bool)
                    person_tax_unit_ids = f["person_tax_unit_id"][period_key][...]
                    tax_unit_ids = f["tax_unit_id"][period_key][...]
                    reported_takeup_anchors["takes_up_aca_if_eligible"] = (
                        reported_subsidized_marketplace_by_tax_unit(
                            person_tax_unit_ids,
                            tax_unit_ids,
                            person_marketplace,
                        )
                    )
                if (
                    "has_medicaid_health_coverage_at_interview" in f
                    and period_key in f["has_medicaid_health_coverage_at_interview"]
                ):
                    reported_takeup_anchors["takes_up_medicaid_if_eligible"] = f[
                        "has_medicaid_health_coverage_at_interview"
                    ][period_key][...].astype(bool)

            entity_to_person_idx = {}
            for entity_level in ("spm_unit", "tax_unit"):
                ent_ids = sim.calculate(
                    f"{entity_level}_id",
                    map_to=entity_level,
                ).values
                ent_id_to_idx = {int(eid): idx for idx, eid in enumerate(ent_ids)}
                person_ent_ids = entity_rel[f"{entity_level}_id"].values
                entity_to_person_idx[entity_level] = np.array(
                    [ent_id_to_idx[int(eid)] for eid in person_ent_ids]
                )
            entity_to_person_idx["person"] = np.arange(len(entity_rel))

            for tvar in unique_variables:
                for key, info in TAKEUP_AFFECTED_TARGETS.items():
                    if tvar == key:
                        affected_target_info[tvar] = info
                        break

            logger.info(
                "Block-level takeup enabled, %d affected target vars",
                len(affected_target_info),
            )

            # Pre-compute takeup rates for ALL takeup vars
            from policyengine_us_data.utils.takeup import (
                SIMPLE_TAKEUP_VARS as _ALL_TAKEUP,
            )

            precomputed_rates = {}
            for spec in _ALL_TAKEUP:
                rk = spec["rate_key"]
                if rk not in precomputed_rates:
                    precomputed_rates[rk] = load_take_up_rate(rk, self.time_period)

            # Store for post-optimization stacked takeup
            self.entity_hh_idx_map = entity_hh_idx_map
            self.household_ids = household_ids
            self.precomputed_rates = precomputed_rates
            self.affected_target_info = affected_target_info
            self.reported_takeup_anchors = reported_takeup_anchors

        # 5d. Clone loop
        from pathlib import Path

        if workers > 1:
            # ---- Parallel clone processing ----
            import concurrent.futures
            import tempfile

            if cache_dir:
                clone_dir = Path(cache_dir)
            else:
                clone_dir = Path(tempfile.mkdtemp(prefix="clone_coo_"))
            clone_dir.mkdir(parents=True, exist_ok=True)

            target_variables = [
                str(targets_df.iloc[i]["variable"]) for i in range(n_targets)
            ]

            shared_data = {
                "geography_state_fips": geography.state_fips,
                "geography_county_fips": geography.county_fips,
                "geography_block_geoid": geography.block_geoid,
                "state_values": state_values,
                "county_values": county_values,
                "person_hh_indices": person_hh_indices,
                "unique_variables": unique_variables,
                "unique_constraint_vars": unique_constraint_vars,
                "reform_vars": reform_variables,
                "county_dep_targets": county_dep_targets,
                "target_variables": target_variables,
                "target_reform_ids": target_reform_ids,
                "target_geo_info": target_geo_info,
                "non_geo_constraints_list": (non_geo_constraints_list),
                "n_records": n_records,
                "n_total": n_total,
                "n_targets": n_targets,
                "state_to_cols": state_to_cols,
                "cd_to_cols": cd_to_cols,
                "entity_rel": entity_rel,
                "household_ids": household_ids,
                "variable_entity_map": variable_entity_map,
                "rerandomize_takeup": rerandomize_takeup,
                "affected_target_info": affected_target_info,
            }
            if rerandomize_takeup and affected_target_info:
                shared_data["entity_hh_idx_map"] = entity_hh_idx_map
                shared_data["entity_to_person_idx"] = entity_to_person_idx
                shared_data["precomputed_rates"] = precomputed_rates
                shared_data["reported_takeup_anchors"] = reported_takeup_anchors

            logger.info(
                "Starting parallel clone processing: %d clones, %d workers",
                n_clones,
                workers,
            )

            futures: dict = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_clone_worker,
                initargs=(shared_data,),
            ) as pool:
                for ci in range(n_clones):
                    coo_path = str(clone_dir / f"clone_{ci:04d}.npz")
                    if Path(coo_path).exists():
                        logger.info(
                            "Clone %d/%d cached.",
                            ci + 1,
                            n_clones,
                        )
                        continue
                    cs = ci * n_records
                    ce = cs + n_records
                    fut = pool.submit(
                        _process_single_clone,
                        ci,
                        cs,
                        ce,
                        coo_path,
                    )
                    futures[fut] = ci

                for fut in concurrent.futures.as_completed(futures):
                    ci = futures[fut]
                    try:
                        _, nnz = fut.result()
                        if (ci + 1) % 50 == 0:
                            logger.info(
                                "Clone %d/%d done (%d nnz).",
                                ci + 1,
                                n_clones,
                                nnz,
                            )
                    except Exception as exc:
                        for f in futures:
                            f.cancel()
                        raise RuntimeError(f"Clone {ci} failed: {exc}") from exc

        else:
            # ---- Sequential clone processing ----
            clone_dir = Path(cache_dir) if cache_dir else None
            if clone_dir:
                clone_dir.mkdir(parents=True, exist_ok=True)

            for clone_idx in range(n_clones):
                if clone_dir:
                    coo_path = clone_dir / f"clone_{clone_idx:04d}.npz"
                    if coo_path.exists():
                        logger.info(
                            "Clone %d/%d cached, skipping.",
                            clone_idx + 1,
                            n_clones,
                        )
                        continue

                col_start = clone_idx * n_records
                col_end = col_start + n_records
                clone_states = geography.state_fips[col_start:col_end]
                clone_counties = geography.county_fips[col_start:col_end]

                if (clone_idx + 1) % 50 == 0 or clone_idx == 0:
                    logger.info(
                        "Assembling clone %d/%d (cols %d-%d, %d unique states)...",
                        clone_idx + 1,
                        n_clones,
                        col_start,
                        col_end - 1,
                        len(np.unique(clone_states)),
                    )

                hh_vars, person_vars, reform_hh_vars = self._assemble_clone_values(
                    state_values,
                    clone_states,
                    person_hh_indices,
                    unique_variables,
                    unique_constraint_vars,
                    reform_vars=reform_variables,
                    county_values=county_values,
                    clone_counties=clone_counties,
                    county_dependent_vars=(county_dep_targets),
                )

                # Apply geo-specific entity-level takeup
                # for affected target variables
                if rerandomize_takeup and affected_target_info:
                    clone_blocks = geography.block_geoid[col_start:col_end]

                    from policyengine_us_data.utils.takeup import (
                        SIMPLE_TAKEUP_VARS as _SEQ_TAKEUP,
                    )

                    # Phase 1: non-target draws (would_file) FIRST
                    wf_draws = {}
                    for spec in _SEQ_TAKEUP:
                        if spec.get("target") is not None:
                            continue
                        var_name = spec["variable"]
                        entity = spec["entity"]
                        rate_key = spec["rate_key"]
                        if rate_key not in precomputed_rates:
                            continue
                        ent_hh = entity_hh_idx_map[entity]
                        ent_blocks = clone_blocks[ent_hh]
                        ent_hh_ids = household_ids[ent_hh]
                        ent_ci = np.full(len(ent_hh), clone_idx, dtype=np.int64)
                        draws = compute_block_takeup_for_entities(
                            var_name,
                            precomputed_rates[rate_key],
                            ent_blocks,
                            ent_hh_ids,
                            ent_ci,
                        )
                        wf_draws[entity] = draws
                        if var_name in person_vars:
                            pidx = entity_to_person_idx[entity]
                            person_vars[var_name] = draws[pidx].astype(np.float32)

                    # Phase 2: target loop with would_file blending
                    for (
                        tvar,
                        info,
                    ) in affected_target_info.items():
                        if tvar.endswith("_count"):
                            continue
                        entity_level = info["entity"]
                        takeup_var = info["takeup_var"]
                        ent_hh = entity_hh_idx_map[entity_level]
                        n_ent = len(ent_hh)

                        ent_states = clone_states[ent_hh]

                        ent_eligible = np.zeros(n_ent, dtype=np.float32)
                        if tvar in county_dep_targets and county_values:
                            ent_counties = clone_counties[ent_hh]
                            for cfips in np.unique(ent_counties):
                                m = ent_counties == cfips
                                cv = county_values.get(cfips, {}).get("entity", {})
                                if tvar in cv:
                                    ent_eligible[m] = cv[tvar][m]
                                else:
                                    st = int(cfips[:2])
                                    sv = state_values[st]["entity"]
                                    if tvar in sv:
                                        ent_eligible[m] = sv[tvar][m]
                        else:
                            for st in np.unique(ent_states):
                                m = ent_states == st
                                sv = state_values[int(st)]["entity"]
                                if tvar in sv:
                                    ent_eligible[m] = sv[tvar][m]

                        # Blend for tax_unit targets
                        if entity_level == "tax_unit" and "tax_unit" in wf_draws:
                            ent_wf_false = np.zeros(n_ent, dtype=np.float32)
                            if tvar in county_dep_targets and county_values:
                                ent_counties = clone_counties[ent_hh]
                                for cfips in np.unique(ent_counties):
                                    m = ent_counties == cfips
                                    cv = county_values.get(cfips, {}).get(
                                        "entity_wf_false", {}
                                    )
                                    if tvar in cv:
                                        ent_wf_false[m] = cv[tvar][m]
                                    else:
                                        st = int(cfips[:2])
                                        sv = state_values[st].get("entity_wf_false", {})
                                        if tvar in sv:
                                            ent_wf_false[m] = sv[tvar][m]
                            else:
                                for st in np.unique(ent_states):
                                    m = ent_states == st
                                    sv = state_values[int(st)].get(
                                        "entity_wf_false", {}
                                    )
                                    if tvar in sv:
                                        ent_wf_false[m] = sv[tvar][m]
                            ent_eligible = np.where(
                                wf_draws["tax_unit"],
                                ent_eligible,
                                ent_wf_false,
                            )

                        ent_blocks = clone_blocks[ent_hh]
                        ent_hh_ids = household_ids[ent_hh]
                        ent_ci = np.full(n_ent, clone_idx, dtype=np.int64)

                        ent_takeup = compute_block_takeup_for_entities(
                            takeup_var,
                            precomputed_rates[info["rate_key"]],
                            ent_blocks,
                            ent_hh_ids,
                            ent_ci,
                            reported_mask=reported_takeup_anchors.get(takeup_var),
                        )

                        ent_values = (ent_eligible * ent_takeup).astype(np.float32)

                        hh_result = np.zeros(n_records, dtype=np.float32)
                        np.add.at(hh_result, ent_hh, ent_values)
                        hh_vars[tvar] = hh_result

                        if tvar in person_vars:
                            pidx = entity_to_person_idx[entity_level]
                            person_vars[tvar] = ent_values[pidx]

                mask_cache: Dict[tuple, np.ndarray] = {}
                count_cache: Dict[tuple, np.ndarray] = {}

                rows_list: list = []
                cols_list: list = []
                vals_list: list = []

                for row_idx in range(n_targets):
                    variable = str(targets_df.iloc[row_idx]["variable"])
                    reform_id = int(targets_df.iloc[row_idx].get("reform_id", 0))
                    geo_level, geo_id = target_geo_info[row_idx]
                    non_geo = non_geo_constraints_list[row_idx]

                    if geo_level == "district":
                        all_geo_cols = cd_to_cols.get(
                            str(geo_id),
                            np.array([], dtype=np.int64),
                        )
                    elif geo_level == "state":
                        all_geo_cols = state_to_cols.get(
                            int(geo_id),
                            np.array([], dtype=np.int64),
                        )
                    else:
                        all_geo_cols = np.arange(n_total)

                    clone_cols = all_geo_cols[
                        (all_geo_cols >= col_start) & (all_geo_cols < col_end)
                    ]
                    if len(clone_cols) == 0:
                        continue

                    rec_indices = clone_cols - col_start

                    constraint_key = tuple(
                        sorted(
                            (
                                c["variable"],
                                c["operation"],
                                c["value"],
                            )
                            for c in non_geo
                        )
                    )

                    if variable.endswith("_count"):
                        vkey = (
                            variable,
                            constraint_key,
                            reform_id,
                        )
                        if vkey not in count_cache:
                            count_cache[vkey] = _calculate_target_values_standalone(
                                target_variable=variable,
                                non_geo_constraints=non_geo,
                                n_households=n_records,
                                hh_vars=hh_vars,
                                reform_hh_vars=reform_hh_vars,
                                person_vars=person_vars,
                                entity_rel=entity_rel,
                                household_ids=household_ids,
                                variable_entity_map=variable_entity_map,
                                reform_id=reform_id,
                            )
                        values = count_cache[vkey]
                    else:
                        source_vars = reform_hh_vars if reform_id > 0 else hh_vars
                        if variable not in source_vars:
                            continue
                        if constraint_key not in mask_cache:
                            mask_cache[constraint_key] = (
                                _evaluate_constraints_standalone(
                                    non_geo,
                                    person_vars,
                                    entity_rel,
                                    household_ids,
                                    n_records,
                                )
                            )
                        mask = mask_cache[constraint_key]
                        values = source_vars[variable] * mask

                    vals = values[rec_indices]
                    nonzero = vals != 0
                    if nonzero.any():
                        rows_list.append(
                            np.full(
                                nonzero.sum(),
                                row_idx,
                                dtype=np.int32,
                            )
                        )
                        cols_list.append(clone_cols[nonzero].astype(np.int32))
                        vals_list.append(vals[nonzero])

                # Save COO entries
                if rows_list:
                    cr = np.concatenate(rows_list)
                    cc = np.concatenate(cols_list)
                    cv = np.concatenate(vals_list)
                else:
                    cr = np.array([], dtype=np.int32)
                    cc = np.array([], dtype=np.int32)
                    cv = np.array([], dtype=np.float32)

                if clone_dir:
                    np.savez_compressed(
                        str(coo_path),
                        rows=cr,
                        cols=cc,
                        vals=cv,
                    )
                    if (clone_idx + 1) % 50 == 0:
                        logger.info(
                            "Clone %d: %d nonzero entries saved.",
                            clone_idx + 1,
                            len(cv),
                        )
                    del hh_vars, person_vars
                else:
                    self._coo_parts[0].append(cr)
                    self._coo_parts[1].append(cc)
                    self._coo_parts[2].append(cv)

        # 6. Assemble sparse matrix from COO data
        logger.info("Assembling matrix from %d clones...", n_clones)
        if clone_dir:
            all_r, all_c, all_v = [], [], []
            for ci in range(n_clones):
                p = clone_dir / f"clone_{ci:04d}.npz"
                data = np.load(str(p))
                all_r.append(data["rows"])
                all_c.append(data["cols"])
                all_v.append(data["vals"])
            rows = np.concatenate(all_r)
            cols = np.concatenate(all_c)
            vals = np.concatenate(all_v)
        else:
            rows = np.concatenate(self._coo_parts[0])
            cols = np.concatenate(self._coo_parts[1])
            vals = np.concatenate(self._coo_parts[2])
            del self._coo_parts

        X_csr = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_targets, n_total),
            dtype=np.float32,
        )

        logger.info(
            "Matrix: %d targets x %d cols, %d nnz",
            X_csr.shape[0],
            X_csr.shape[1],
            X_csr.nnz,
        )

        return targets_df, X_csr, target_names

    def build_matrix_chunked(
        self,
        geography,
        sim,
        target_filter: Optional[dict] = None,
        hierarchical_domains: Optional[List[str]] = None,
        chunk_size: int = 25_000,
        chunk_dir: Optional[str] = None,
        keep_chunks: bool = False,
        resume_chunks: bool = False,
        rerandomize_takeup: bool = True,
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]:
        """Build a sparse matrix by materializing mixed-geography chunks.

        This path chunks over global clone-household columns.  Each chunk H5
        contains only the selected base households and their dependent entities,
        with geography inputs assigned row-by-row from the clone assignment.
        """
        import shutil
        import tempfile
        import time

        from policyengine_us import Microsimulation
        from policyengine_us_data.calibration.entity_clone import (
            build_household_entity_maps,
            materialize_clone_household_chunk,
        )

        if self.dataset_path is None:
            raise ValueError("dataset_path is required for chunked matrix building")
        if chunk_size <= 0:
            raise ValueError("--chunk-size must be positive")

        n_records = geography.n_records
        n_clones = geography.n_clones
        n_total = n_records * n_clones
        _validate_chunked_geography(geography, n_total)

        targets_df = self._query_targets(target_filter or {})
        if len(targets_df) == 0:
            raise ValueError("No targets found matching filter")

        params = sim.tax_benefit_system.parameters
        uprating_factors = self._calculate_uprating_factors(params)
        targets_df["original_value"] = targets_df["value"].copy()
        targets_df["uprating_factor"] = targets_df.apply(
            lambda row: self._get_uprating_info(
                row["variable"],
                row["period"],
                uprating_factors,
            )[0],
            axis=1,
        )
        targets_df["value"] = (
            targets_df["original_value"] * targets_df["uprating_factor"]
        )

        if hierarchical_domains:
            targets_df = self._apply_hierarchical_uprating(
                targets_df,
                hierarchical_domains,
                uprating_factors,
            )

        targets_df["_geo_level"] = targets_df["geographic_id"].apply(get_geo_level)
        targets_df = targets_df.sort_values(
            ["_geo_level", "variable", "reform_id", "geographic_id"]
        )
        targets_df = targets_df.drop(columns=["_geo_level"]).reset_index(drop=True)
        n_targets = len(targets_df)

        constraint_cache: Dict[int, List[dict]] = {}
        target_geo_info: List[Tuple[str, str]] = []
        target_names: List[str] = []
        non_geo_constraints_list: List[List[dict]] = []
        target_reform_ids: List[int] = []

        for _, row in targets_df.iterrows():
            sid = int(row["stratum_id"])
            if sid not in constraint_cache:
                constraint_cache[sid] = self._get_stratum_constraints(sid)
            constraints = constraint_cache[sid]
            target_geo_info.append((row["geo_level"], row["geographic_id"]))
            non_geo = [c for c in constraints if c["variable"] not in _GEO_VARS]
            non_geo_constraints_list.append(non_geo)
            reform_id = int(row.get("reform_id", 0))
            target_reform_ids.append(reform_id)
            target_names.append(
                self._make_target_name(
                    str(row["variable"]),
                    constraints,
                    reform_id=reform_id,
                )
            )

        unique_variables = set(targets_df["variable"].values)
        reform_variables = {
            str(row["variable"])
            for _, row in targets_df.iterrows()
            if int(row.get("reform_id", 0)) > 0
        }
        unique_constraint_vars = set()
        for constraints in non_geo_constraints_list:
            for constraint in constraints:
                unique_constraint_vars.add(constraint["variable"])

        base_entity_maps = build_household_entity_maps(sim)

        if chunk_dir is None:
            chunk_root = Path(tempfile.mkdtemp(prefix="matrix_chunks_"))
            remove_chunk_root = not keep_chunks
        else:
            chunk_root = Path(chunk_dir)
            remove_chunk_root = False
        coo_dir = chunk_root / "coo"
        h5_dir = chunk_root / "h5"
        coo_dir.mkdir(parents=True, exist_ok=True)
        h5_dir.mkdir(parents=True, exist_ok=True)

        n_chunks = (n_total + chunk_size - 1) // chunk_size
        logger.info(
            "Chunked matrix build: %d targets x %d columns, "
            "%d chunks of up to %d columns",
            n_targets,
            n_total,
            n_chunks,
            chunk_size,
        )

        target_variables = [
            str(targets_df.iloc[i]["variable"]) for i in range(n_targets)
        ]
        chunk_manifest_path = chunk_root / "chunk_manifest.json"
        chunk_signature = build_chunk_lineage_signature(
            dataset_path=self.dataset_path,
            db_uri=self.db_uri,
            time_period=self.time_period,
            geography=geography,
            targets_df=targets_df,
            target_names=target_names,
            chunk_size=chunk_size,
            rerandomize_takeup=rerandomize_takeup,
        )
        if resume_chunks:
            if chunk_manifest_path.exists():
                _validate_chunk_manifest(chunk_manifest_path, chunk_signature)
            elif _has_existing_chunk_cache(coo_dir):
                raise ValueError(
                    f"Cannot resume chunk cache at {chunk_root}: missing chunk manifest"
                )
            else:
                _save_chunk_manifest(chunk_manifest_path, chunk_signature)
        else:
            _save_chunk_manifest(chunk_manifest_path, chunk_signature)

        t_build = time.time()
        processed_chunk_times: list[float] = []
        cached_chunks = 0

        for chunk_id, col_start in enumerate(range(0, n_total, chunk_size)):
            t_chunk = time.time()
            col_end = min(col_start + chunk_size, n_total)
            coo_path = coo_dir / f"chunk_{chunk_id:06d}.npz"
            h5_path = h5_dir / f"chunk_{chunk_id:06d}.h5"

            if resume_chunks and coo_path.exists():
                with np.load(str(coo_path)) as cached_chunk:
                    if "col_start" not in cached_chunk or "col_end" not in cached_chunk:
                        raise ValueError(
                            f"Cached chunk {coo_path} is missing col_start/col_end metadata"
                        )
                    cached_col_start = int(np.asarray(cached_chunk["col_start"]).item())
                    cached_col_end = int(np.asarray(cached_chunk["col_end"]).item())
                if cached_col_start != col_start or cached_col_end != col_end:
                    raise ValueError(
                        f"Cached chunk {coo_path} covers cols {cached_col_start}-{cached_col_end - 1}, "
                        f"expected {col_start}-{col_end - 1}"
                    )
                cached_chunks += 1
                logger.info(
                    "Chunk %d/%d cached: cols %d-%d, cached=%d",
                    chunk_id + 1,
                    n_chunks,
                    col_start,
                    col_end - 1,
                    cached_chunks,
                )
                continue

            global_cols = np.arange(col_start, col_end, dtype=np.int64)
            active_hh = global_cols % n_records
            active_clone_indices = global_cols // n_records
            active_blocks = np.asarray(geography.block_geoid)[global_cols]
            active_cd_geoids = np.asarray(geography.cd_geoid, dtype=str)[global_cols]
            active_states = np.asarray(geography.state_fips)[global_cols]
            active_counties = np.asarray(geography.county_fips, dtype=str)[global_cols]

            summary = materialize_clone_household_chunk(
                sim=sim,
                entity_maps=base_entity_maps,
                active_hh=active_hh,
                active_blocks=active_blocks,
                active_cd_geoids=active_cd_geoids,
                active_clone_indices=active_clone_indices,
                output_path=h5_path,
                apply_takeup=rerandomize_takeup,
            )

            chunk_sim = Microsimulation(dataset=str(h5_path))
            chunk_n = len(global_cols)
            self._entity_rel_cache = None
            entity_rel = self._build_entity_relationship(chunk_sim)
            household_ids = chunk_sim.calculate(
                "household_id",
                map_to="household",
            ).values

            hh_vars = {}
            for variable in sorted(unique_variables):
                if variable.endswith("_count"):
                    continue
                try:
                    hh_vars[variable] = chunk_sim.calculate(
                        variable,
                        self.time_period,
                        map_to="household",
                    ).values.astype(np.float32)
                except Exception as exc:
                    logger.warning(
                        "Chunk %d cannot calculate target '%s': %s",
                        chunk_id,
                        variable,
                        exc,
                    )

            person_vars = {}
            for variable in sorted(unique_constraint_vars):
                try:
                    raw = chunk_sim.calculate(
                        variable,
                        self.time_period,
                        map_to="person",
                    ).values
                    try:
                        person_vars[variable] = raw.astype(np.float32)
                    except (ValueError, TypeError):
                        person_vars[variable] = raw
                except Exception as exc:
                    logger.warning(
                        "Chunk %d cannot calculate constraint '%s': %s",
                        chunk_id,
                        variable,
                        exc,
                    )

            reform_hh_vars = {}
            if reform_variables:
                baseline_income_tax = chunk_sim.calculate(
                    "income_tax",
                    self.time_period,
                    map_to="household",
                ).values.astype(np.float32)
                for variable in sorted(reform_variables):
                    try:
                        reform_sim = Microsimulation(
                            dataset=str(h5_path),
                            reform=_make_neutralize_variable_reform(variable),
                        )
                        reform_income_tax = reform_sim.calculate(
                            "income_tax",
                            self.time_period,
                            map_to="household",
                        ).values.astype(np.float32)
                        reform_hh_vars[variable] = (
                            reform_income_tax - baseline_income_tax
                        )
                    except Exception as exc:
                        logger.warning(
                            "Chunk %d cannot calculate reform target '%s': %s",
                            chunk_id,
                            variable,
                            exc,
                        )

            variable_entity_map: Dict[str, str] = {}
            for variable in unique_variables:
                if (
                    variable.endswith("_count")
                    and variable in chunk_sim.tax_benefit_system.variables
                ):
                    variable_entity_map[variable] = (
                        chunk_sim.tax_benefit_system.variables[variable].entity.key
                    )

            mask_cache: Dict[tuple, np.ndarray] = {}
            count_cache: Dict[tuple, np.ndarray] = {}
            rows_list: list = []
            cols_list: list = []
            vals_list: list = []

            for row_idx in range(n_targets):
                variable = target_variables[row_idx]
                reform_id = target_reform_ids[row_idx]
                geo_level, geo_id = target_geo_info[row_idx]
                non_geo = non_geo_constraints_list[row_idx]

                if geo_level == "district":
                    geo_mask = active_cd_geoids == str(geo_id)
                elif geo_level == "state":
                    geo_mask = active_states.astype(np.int64) == int(geo_id)
                elif geo_level == "county":
                    geo_mask = active_counties == str(geo_id).zfill(5)
                else:
                    geo_mask = np.ones(chunk_n, dtype=bool)

                if not geo_mask.any():
                    continue

                constraint_key = tuple(
                    sorted(
                        (
                            c["variable"],
                            c["operation"],
                            c["value"],
                        )
                        for c in non_geo
                    )
                )

                if variable.endswith("_count"):
                    value_key = (variable, constraint_key, reform_id)
                    if value_key not in count_cache:
                        count_cache[value_key] = _calculate_target_values_standalone(
                            target_variable=variable,
                            non_geo_constraints=non_geo,
                            n_households=chunk_n,
                            hh_vars=hh_vars,
                            reform_hh_vars=reform_hh_vars,
                            person_vars=person_vars,
                            entity_rel=entity_rel,
                            household_ids=household_ids,
                            variable_entity_map=variable_entity_map,
                            reform_id=reform_id,
                        )
                    values = count_cache[value_key]
                else:
                    source_vars = reform_hh_vars if reform_id > 0 else hh_vars
                    if variable not in source_vars:
                        continue
                    if constraint_key not in mask_cache:
                        mask_cache[constraint_key] = _evaluate_constraints_standalone(
                            non_geo,
                            person_vars,
                            entity_rel,
                            household_ids,
                            chunk_n,
                        )
                    values = source_vars[variable] * mask_cache[constraint_key]

                vals = values[geo_mask]
                nonzero = vals != 0
                if nonzero.any():
                    rows_list.append(np.full(nonzero.sum(), row_idx, dtype=np.int32))
                    cols_list.append(global_cols[geo_mask][nonzero].astype(np.int32))
                    vals_list.append(vals[nonzero].astype(np.float32, copy=False))

            if rows_list:
                rows = np.concatenate(rows_list)
                cols = np.concatenate(cols_list)
                vals = np.concatenate(vals_list)
            else:
                rows = np.array([], dtype=np.int32)
                cols = np.array([], dtype=np.int32)
                vals = np.array([], dtype=np.float32)

            np.savez_compressed(
                str(coo_path),
                rows=rows,
                cols=cols,
                vals=vals,
                col_start=np.array([col_start], dtype=np.int64),
                col_end=np.array([col_end], dtype=np.int64),
            )

            if not keep_chunks and h5_path.exists():
                h5_path.unlink()

            chunk_seconds = time.time() - t_chunk
            processed_chunk_times.append(chunk_seconds)
            remaining_chunks = n_chunks - (chunk_id + 1)
            avg_seconds = float(np.mean(processed_chunk_times))
            eta_seconds = avg_seconds * remaining_chunks
            elapsed_seconds = time.time() - t_build
            rss = _current_rss_mb()
            rss_part = f", rss={rss:,.0f} MB" if rss is not None else ""
            logger.info(
                "Chunk %d/%d: cols %d-%d, hh=%d, persons=%d, "
                "states=%d, counties=%d, cds=%d, nnz=%d, "
                "chunk=%s, avg=%s, elapsed=%s, eta=%s%s",
                chunk_id + 1,
                n_chunks,
                col_start,
                col_end - 1,
                summary.n_households,
                summary.n_persons,
                summary.unique_states,
                summary.unique_counties,
                summary.unique_cds,
                len(vals),
                _format_duration(chunk_seconds),
                _format_duration(avg_seconds),
                _format_duration(elapsed_seconds),
                _format_duration(eta_seconds),
                rss_part,
            )

            del chunk_sim, hh_vars, person_vars, reform_hh_vars

        logger.info("Assembling matrix from %d chunk files...", n_chunks)
        all_rows, all_cols, all_vals = [], [], []
        for chunk_id in range(n_chunks):
            path = coo_dir / f"chunk_{chunk_id:06d}.npz"
            data = np.load(str(path))
            all_rows.append(data["rows"])
            all_cols.append(data["cols"])
            all_vals.append(data["vals"])

        rows = np.concatenate(all_rows) if all_rows else np.array([], dtype=np.int32)
        cols = np.concatenate(all_cols) if all_cols else np.array([], dtype=np.int32)
        vals = np.concatenate(all_vals) if all_vals else np.array([], dtype=np.float32)

        X_csr = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_targets, n_total),
            dtype=np.float32,
        )
        logger.info(
            "Chunked matrix: %d targets x %d cols, %d nnz",
            X_csr.shape[0],
            X_csr.shape[1],
            X_csr.nnz,
        )

        if remove_chunk_root and chunk_root.exists():
            shutil.rmtree(chunk_root)
        elif not keep_chunks and h5_dir.exists():
            shutil.rmtree(h5_dir)

        return targets_df, X_csr, target_names
