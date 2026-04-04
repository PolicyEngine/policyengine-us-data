"""
Shared takeup draw logic for calibration and local-area H5 building.

Block-level seeded draws ensure that calibration targets match
local-area H5 aggregations.  The (block, household) salt ensures:
  - Same (variable, block, household) → same draws
  - Different blocks/households → different draws

Entity-level draws respect the native entity of each takeup variable
(spm_unit for SNAP/TANF, tax_unit for ACA/DC-PTC, person for SSI/
Medicaid/Head Start).
"""

import numpy as np
from typing import Any, Dict, List, Optional

from policyengine_us_data.utils.randomness import seeded_rng
from policyengine_us_data.parameters import load_take_up_rate

SIMPLE_TAKEUP_VARS = [
    {
        "variable": "takes_up_snap_if_eligible",
        "entity": "spm_unit",
        "rate_key": "snap",
        "target": "snap",
    },
    {
        "variable": "takes_up_aca_if_eligible",
        "entity": "tax_unit",
        "rate_key": "aca",
        "target": "aca_ptc",
    },
    {
        "variable": "takes_up_dc_ptc",
        "entity": "tax_unit",
        "rate_key": "dc_ptc",
        "target": "dc_property_tax_credit",
    },
    {
        "variable": "takes_up_head_start_if_eligible",
        "entity": "person",
        "rate_key": "head_start",
        "target": "head_start",
    },
    {
        "variable": "takes_up_early_head_start_if_eligible",
        "entity": "person",
        "rate_key": "early_head_start",
        "target": "early_head_start",
    },
    {
        "variable": "takes_up_ssi_if_eligible",
        "entity": "person",
        "rate_key": "ssi",
        "target": "ssi",
    },
    {
        "variable": "would_file_taxes_voluntarily",
        "entity": "tax_unit",
        "rate_key": "voluntary_filing",
        "target": None,
    },
    {
        "variable": "takes_up_medicaid_if_eligible",
        "entity": "person",
        "rate_key": "medicaid",
        "target": "medicaid",
    },
    {
        "variable": "takes_up_tanf_if_eligible",
        "entity": "spm_unit",
        "rate_key": "tanf",
        "target": "tanf",
    },
]

TAKEUP_AFFECTED_TARGETS: Dict[str, dict] = {
    spec["target"]: {
        "takeup_var": spec["variable"],
        "entity": spec["entity"],
        "rate_key": spec["rate_key"],
    }
    for spec in SIMPLE_TAKEUP_VARS
    if spec.get("target") is not None
}

# CMS 2025 Marketplace OEP State-Level Public Use File, Total / All row.
# This is the number of consumers receiving APTC in plan year 2025.
ACA_POST_CALIBRATION_PERSON_TARGETS = {
    2025: 22_380_137,
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


def compute_block_takeup_draws_for_entities(
    var_name: str,
    entity_blocks: np.ndarray,
    entity_hh_ids: np.ndarray,
    entity_clone_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute deterministic uniform draws for entity-level takeup.

    Each unique (household id, clone index) pair gets its own seeded RNG,
    producing reproducible draws that are stable for a given donor household
    and independent across clones. Rates are applied separately by the caller
    after resolving state FIPS from the block GEOID.

    Args:
        var_name: Takeup variable name.
        entity_blocks: Block GEOID per entity (str array).
        entity_hh_ids: Original household ID per entity.
        entity_clone_indices: Clone index per entity.

    Returns:
        Float array of shape (n_entities,) in [0, 1).
    """
    n = len(entity_blocks)
    draws = np.zeros(n, dtype=np.float64)
    if entity_clone_indices is None:
        entity_clone_indices = np.zeros(n, dtype=np.int64)

    # Iterate block groups first so draws stay stable within geography slices.
    for block in np.unique(entity_blocks):
        if block == "":
            continue
        blk_mask = entity_blocks == block

    # Draw per (hh_id, clone_idx) pair
    for hh_id in np.unique(entity_hh_ids):
        hh_mask = entity_hh_ids == hh_id
        for ci in np.unique(entity_clone_indices[hh_mask]):
            ci_mask = hh_mask & (entity_clone_indices == ci)
            n_ent = int(ci_mask.sum())
            rng = seeded_rng(var_name, salt=f"{int(hh_id)}:{int(ci)}")
            draws[ci_mask] = rng.random(n_ent)

    return draws


def compute_block_takeup_for_entities(
    var_name: str,
    rate_or_dict,
    entity_blocks: np.ndarray,
    entity_hh_ids: np.ndarray = None,
    entity_clone_ids: np.ndarray = None,
) -> np.ndarray:
    """Compute boolean takeup via block-level seeded draws."""
    draws = compute_block_takeup_draws_for_entities(
        var_name,
        entity_blocks,
        entity_hh_ids,
        entity_clone_ids,
    )
    rates = np.ones(len(entity_blocks), dtype=np.float64)

    for block in np.unique(entity_blocks):
        if block == "":
            continue
        blk_mask = entity_blocks == block
        rates[blk_mask] = _resolve_rate(rate_or_dict, int(str(block)[:2]))

    return draws < rates


def extend_aca_takeup_to_match_target(
    base_takeup: np.ndarray,
    entity_draws: np.ndarray,
    enrolled_person_weights: np.ndarray,
    target_people: float,
) -> np.ndarray:
    """Turn on extra tax units until weighted ACA enrollment hits target.

    ``enrolled_person_weights`` should be the weighted number of
    people in each tax unit who would receive ACA PTC if that tax
    unit takes up coverage in the target year.
    """
    result = base_takeup.copy()
    current_people = enrolled_person_weights[result].sum()
    if current_people >= target_people:
        return result

    available_mask = (~result) & (enrolled_person_weights > 0)
    if not available_mask.any():
        return result

    available_idx = np.flatnonzero(available_mask)
    ordered_idx = available_idx[np.argsort(entity_draws[available_idx], kind="stable")]
    cumulative_people = current_people + np.cumsum(enrolled_person_weights[ordered_idx])
    n_to_add = (
        np.searchsorted(
            cumulative_people,
            target_people,
            side="left",
        )
        + 1
    )
    result[ordered_idx[:n_to_add]] = True
    return result


def apply_block_takeup_to_arrays(
    hh_blocks: np.ndarray,
    hh_state_fips: np.ndarray,
    hh_ids: np.ndarray,
    hh_clone_indices: np.ndarray,
    entity_hh_indices: Dict[str, np.ndarray],
    entity_counts: Dict[str, int],
    time_period: int,
    takeup_filter: List[str] = None,
    precomputed_rates: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Compute takeup draws from raw arrays.

    Works without a Microsimulation instance. For each takeup
    variable, maps entity-level arrays from household-level block/
    state/id arrays using entity->household index mappings, then
    calls compute_block_takeup_for_entities.

    Args:
        hh_blocks: Block GEOID per cloned household (str array).
        hh_state_fips: State FIPS per cloned household (int array).
        hh_ids: Original household ID per cloned household.
        hh_clone_indices: Clone index per cloned household.
        entity_hh_indices: {entity_key: array} mapping each entity
            instance to its household index. Keys: "person",
            "tax_unit", "spm_unit".
        entity_counts: {entity_key: count} number of entities per
            type.
        time_period: Tax year.
        takeup_filter: Optional list of takeup variable names to
            re-randomize. If None, all SIMPLE_TAKEUP_VARS are
            processed. Non-filtered vars are set to True.
        precomputed_rates: Optional {rate_key: rate_or_dict} cache.
            When provided, skips ``load_take_up_rate`` calls and
            uses cached values instead.

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
        ent_hh_ids = hh_ids[ent_hh_idx]
        ent_clone_indices = hh_clone_indices[ent_hh_idx]

        if precomputed_rates is not None and rate_key in precomputed_rates:
            rate_or_dict = precomputed_rates[rate_key]
        else:
            rate_or_dict = load_take_up_rate(rate_key, time_period)
        bools = compute_block_takeup_for_entities(
            var_name,
            rate_or_dict,
            ent_blocks,
            ent_hh_ids,
            ent_clone_indices,
        )
        result[var_name] = bools

    return result
