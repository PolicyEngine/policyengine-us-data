"""
Shared takeup draw logic for calibration and stacked dataset building.

Both the matrix builder and the stacked dataset builder need to produce
identical takeup draws for each geographic unit so that calibration
targets match stacked-h5 aggregations.  The geo_id salt (today a CD
GEOID, tomorrow an SLD/tract/etc.) ensures:
  - Same (variable, geo_id, n_entities) → same draws
  - Different geo_ids → different draws

Entity-level draws respect the native entity of each takeup variable
(spm_unit for SNAP/TANF, tax_unit for ACA/DC-PTC, person for SSI/
Medicaid/Head Start).
"""

import numpy as np
from typing import Dict, List

from policyengine_us_data.utils.randomness import seeded_rng
from policyengine_us_data.parameters import load_take_up_rate

SIMPLE_TAKEUP_VARS = [
    {
        "variable": "takes_up_snap_if_eligible",
        "entity": "spm_unit",
        "rate_key": "snap",
    },
    {
        "variable": "takes_up_aca_if_eligible",
        "entity": "tax_unit",
        "rate_key": "aca",
    },
    {
        "variable": "takes_up_dc_ptc",
        "entity": "tax_unit",
        "rate_key": "dc_ptc",
    },
    {
        "variable": "takes_up_head_start_if_eligible",
        "entity": "person",
        "rate_key": "head_start",
    },
    {
        "variable": "takes_up_early_head_start_if_eligible",
        "entity": "person",
        "rate_key": "early_head_start",
    },
    {
        "variable": "takes_up_ssi_if_eligible",
        "entity": "person",
        "rate_key": "ssi",
    },
    {
        "variable": "would_file_taxes_voluntarily",
        "entity": "tax_unit",
        "rate_key": "voluntary_filing",
    },
    {
        "variable": "takes_up_medicaid_if_eligible",
        "entity": "person",
        "rate_key": "medicaid",
    },
    {
        "variable": "takes_up_tanf_if_eligible",
        "entity": "spm_unit",
        "rate_key": "tanf",
    },
]

TAKEUP_AFFECTED_TARGETS: Dict[str, dict] = {
    "snap": {
        "takeup_var": "takes_up_snap_if_eligible",
        "entity": "spm_unit",
        "rate_key": "snap",
    },
    "tanf": {
        "takeup_var": "takes_up_tanf_if_eligible",
        "entity": "spm_unit",
        "rate_key": "tanf",
    },
    "aca_ptc": {
        "takeup_var": "takes_up_aca_if_eligible",
        "entity": "tax_unit",
        "rate_key": "aca",
    },
    "ssi": {
        "takeup_var": "takes_up_ssi_if_eligible",
        "entity": "person",
        "rate_key": "ssi",
    },
    "medicaid": {
        "takeup_var": "takes_up_medicaid_if_eligible",
        "entity": "person",
        "rate_key": "medicaid",
    },
    "head_start": {
        "takeup_var": "takes_up_head_start_if_eligible",
        "entity": "person",
        "rate_key": "head_start",
    },
    "early_head_start": {
        "takeup_var": "takes_up_early_head_start_if_eligible",
        "entity": "person",
        "rate_key": "early_head_start",
    },
    "dc_property_tax_credit": {
        "takeup_var": "takes_up_dc_ptc",
        "entity": "tax_unit",
        "rate_key": "dc_ptc",
    },
}

# FIPS -> 2-letter state code for Medicaid rate lookup
_FIPS_TO_STATE_CODE = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}


def _resolve_rate(
    rate_or_dict,
    state_fips: int,
) -> float:
    """Resolve a scalar or state-keyed rate to a single float."""
    if isinstance(rate_or_dict, dict):
        code = _FIPS_TO_STATE_CODE.get(state_fips, "")
        return rate_or_dict.get(
            code,
            rate_or_dict.get(str(state_fips), 0.8),
        )
    return float(rate_or_dict)


def draw_takeup_for_geo(
    var_name: str,
    geo_id: str,
    n_entities: int,
) -> np.ndarray:
    """Draw uniform [0, 1) values for a takeup variable in a geo unit.

    Args:
        var_name: Takeup variable name.
        geo_id: Geographic unit identifier (e.g. CD GEOID "3701").
        n_entities: Number of entities at the native level.

    Returns:
        float64 array of shape (n_entities,).
    """
    rng = seeded_rng(var_name, salt=f"geo:{geo_id}")
    return rng.random(n_entities)


def compute_entity_takeup_for_geo(
    geo_id: str,
    n_entities_by_level: Dict[str, int],
    state_fips: int,
    time_period: int,
) -> Dict[str, np.ndarray]:
    """Compute boolean takeup arrays for all SIMPLE_TAKEUP_VARS.

    Args:
        geo_id: Geographic unit identifier.
        n_entities_by_level: {"person": n, "tax_unit": n, "spm_unit": n}.
        state_fips: State FIPS for state-specific rates.
        time_period: Tax year.

    Returns:
        {takeup_var_name: bool array at native entity level}
    """
    result = {}
    for spec in SIMPLE_TAKEUP_VARS:
        var_name = spec["variable"]
        entity = spec["entity"]
        rate_key = spec["rate_key"]

        n_entities = n_entities_by_level[entity]
        draws = draw_takeup_for_geo(var_name, geo_id, n_entities)

        rate_or_dict = load_take_up_rate(rate_key, time_period)
        rate = _resolve_rate(rate_or_dict, state_fips)

        result[var_name] = draws < rate
    return result


def apply_takeup_draws_to_sim(
    sim,
    geo_id: str,
    time_period: int,
) -> None:
    """Set all takeup inputs on a sim using CD-level geo-salted draws.

    Deprecated: use apply_block_takeup_draws_to_sim for block-level
    seeding that works for any aggregation level.

    Args:
        sim: Microsimulation instance (state_fips already set).
        geo_id: Geographic unit identifier (CD GEOID).
        time_period: Tax year.
    """
    state_fips_arr = sim.calculate(
        "state_fips", time_period, map_to="household"
    ).values
    state_fips = int(state_fips_arr[0])

    n_entities_by_level = {}
    for entity in ("person", "tax_unit", "spm_unit"):
        ids = sim.calculate(f"{entity}_id", map_to=entity).values
        n_entities_by_level[entity] = len(ids)

    takeup = compute_entity_takeup_for_geo(
        geo_id, n_entities_by_level, state_fips, time_period
    )
    for var_name, bools in takeup.items():
        entity = next(
            s["entity"]
            for s in SIMPLE_TAKEUP_VARS
            if s["variable"] == var_name
        )
        sim.set_input(var_name, time_period, bools)


def compute_block_takeup_for_entities(
    var_name: str,
    rate_or_dict,
    entity_blocks: np.ndarray,
    entity_state_fips: np.ndarray,
    entity_hh_ids: np.ndarray = None,
) -> np.ndarray:
    """Compute boolean takeup via block-level seeded draws.

    Each unique (block, household) pair gets its own seeded RNG,
    producing reproducible draws regardless of how many households
    share the same block across clones.

    Args:
        var_name: Takeup variable name.
        rate_or_dict: Scalar rate or {state_code: rate} dict.
        entity_blocks: Block GEOID per entity (str array).
        entity_state_fips: State FIPS per entity (int array).
        entity_hh_ids: Household ID per entity (int array).
            When provided, seeds per (block, household) for
            clone-independent draws.

    Returns:
        Boolean array of shape (n_entities,).
    """
    n = len(entity_blocks)
    draws = np.zeros(n, dtype=np.float64)
    rates = np.ones(n, dtype=np.float64)

    for block in np.unique(entity_blocks):
        if block == "":
            continue
        blk_mask = entity_blocks == block
        sf = int(str(block)[:2])
        rate = _resolve_rate(rate_or_dict, sf)
        rates[blk_mask] = rate

        if entity_hh_ids is not None:
            for hh_id in np.unique(entity_hh_ids[blk_mask]):
                hh_mask = blk_mask & (entity_hh_ids == hh_id)
                rng = seeded_rng(var_name, salt=f"{block}:{int(hh_id)}")
                draws[hh_mask] = rng.random(int(hh_mask.sum()))
        else:
            rng = seeded_rng(var_name, salt=str(block))
            draws[blk_mask] = rng.random(int(blk_mask.sum()))

    return draws < rates


def _build_entity_to_hh_index(sim) -> Dict[str, np.ndarray]:
    """Map each entity instance to its household index.

    Uses person-level bridge IDs (person_household_id,
    person_tax_unit_id, etc.) which are reliable across
    all dataset formats.

    Returns:
        {"person": arr, "tax_unit": arr, "spm_unit": arr}
        where each arr[i] is the household index for entity i.
    """
    hh_ids = sim.calculate("household_id", map_to="household").values
    hh_id_to_idx = {int(h): i for i, h in enumerate(hh_ids)}

    p_hh_ids = sim.calculate("person_household_id", map_to="person").values
    person_hh_idx = np.array([hh_id_to_idx[int(h)] for h in p_hh_ids])

    result = {"person": person_hh_idx}

    for entity, id_var in (
        ("tax_unit", "person_tax_unit_id"),
        ("spm_unit", "person_spm_unit_id"),
    ):
        p_ent_ids = sim.calculate(id_var, map_to="person").values
        ent_ids = sim.calculate(f"{entity}_id", map_to=entity).values

        ent_id_to_hh_idx = {}
        for p_idx in range(len(p_ent_ids)):
            eid = int(p_ent_ids[p_idx])
            if eid not in ent_id_to_hh_idx:
                ent_id_to_hh_idx[eid] = person_hh_idx[p_idx]

        result[entity] = np.array(
            [ent_id_to_hh_idx[int(eid)] for eid in ent_ids]
        )

    return result


def apply_block_takeup_to_arrays(
    hh_blocks: np.ndarray,
    hh_state_fips: np.ndarray,
    hh_ids: np.ndarray,
    entity_hh_indices: Dict[str, np.ndarray],
    entity_counts: Dict[str, int],
    time_period: int,
    takeup_filter: List[str] = None,
) -> Dict[str, np.ndarray]:
    """Compute block-level takeup draws from raw arrays.

    Works without a Microsimulation instance. For each takeup
    variable, maps entity-level arrays from household-level block/
    state/id arrays using entity->household index mappings, then
    calls compute_block_takeup_for_entities.

    Args:
        hh_blocks: Block GEOID per cloned household (str array).
        hh_state_fips: State FIPS per cloned household (int array).
        hh_ids: Household ID per cloned household (int array).
        entity_hh_indices: {entity_key: array} mapping each entity
            instance to its household index. Keys: "person",
            "tax_unit", "spm_unit".
        entity_counts: {entity_key: count} number of entities per
            type.
        time_period: Tax year.
        takeup_filter: Optional list of takeup variable names to
            re-randomize. If None, all SIMPLE_TAKEUP_VARS are
            processed. Non-filtered vars are set to True.

    Returns:
        {variable_name: bool_array} for each takeup variable.
    """
    filter_set = set(takeup_filter) if takeup_filter is not None else None
    result = {}

    for spec in SIMPLE_TAKEUP_VARS:
        var_name = spec["variable"]
        entity = spec["entity"]
        rate_key = spec["rate_key"]
        n_ent = entity_counts[entity]

        if filter_set is not None and var_name not in filter_set:
            result[var_name] = np.ones(n_ent, dtype=bool)
            continue

        ent_hh_idx = entity_hh_indices[entity]
        ent_blocks = hh_blocks[ent_hh_idx].astype(str)
        ent_states = hh_state_fips[ent_hh_idx]
        ent_hh_ids = hh_ids[ent_hh_idx]

        rate_or_dict = load_take_up_rate(rate_key, time_period)
        bools = compute_block_takeup_for_entities(
            var_name,
            rate_or_dict,
            ent_blocks,
            ent_states,
            ent_hh_ids,
        )
        result[var_name] = bools

    return result


def apply_block_takeup_draws_to_sim(
    sim,
    hh_blocks: np.ndarray,
    time_period: int,
    takeup_filter: List[str] = None,
) -> None:
    """Set all takeup inputs on a sim using block-level draws.

    Groups entities by their household's block GEOID and uses
    block-level seeded draws. This produces draws that are
    consistent regardless of the aggregation level.

    Args:
        sim: Microsimulation instance (state_fips already set).
        hh_blocks: Block GEOID per household (str array).
        time_period: Tax year.
        takeup_filter: Optional list of takeup variable names
            to re-randomize. If None, all SIMPLE_TAKEUP_VARS
            are processed. Use this to match the matrix builder's
            set of re-randomized variables.
    """
    state_fips_arr = sim.calculate(
        "state_fips", time_period, map_to="household"
    ).values
    hh_ids = sim.calculate("household_id", map_to="household").values

    entity_hh_idx = _build_entity_to_hh_index(sim)

    filter_set = set(takeup_filter) if takeup_filter is not None else None

    for spec in SIMPLE_TAKEUP_VARS:
        var_name = spec["variable"]
        entity = spec["entity"]
        rate_key = spec["rate_key"]

        n_ent = len(sim.calculate(f"{entity}_id", map_to=entity).values)

        if filter_set is not None and var_name not in filter_set:
            # Force non-filtered vars to True to match
            # the matrix builder's precomputation assumption
            sim.set_input(
                var_name,
                time_period,
                np.ones(n_ent, dtype=bool),
            )
            continue

        ent_hh_idx = entity_hh_idx[entity]
        ent_blocks = np.array([str(hh_blocks[h]) for h in ent_hh_idx])
        ent_states = state_fips_arr[ent_hh_idx]
        ent_hh_ids = hh_ids[ent_hh_idx]

        rate_or_dict = load_take_up_rate(rate_key, time_period)
        bools = compute_block_takeup_for_entities(
            var_name,
            rate_or_dict,
            ent_blocks,
            ent_states,
            ent_hh_ids,
        )
        sim.set_input(var_name, time_period, bools)
