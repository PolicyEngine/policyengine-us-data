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


def compute_block_takeup_for_entities(
    var_name: str,
    rate_or_dict,
    entity_blocks: np.ndarray,
    entity_hh_ids: np.ndarray = None,
) -> np.ndarray:
    """Compute boolean takeup via block-level seeded draws.

    Each unique (block, household) pair gets its own seeded RNG,
    producing reproducible draws regardless of how many households
    share the same block across clones.

    State FIPS for rate resolution is derived from the first two
    characters of each block GEOID.

    Args:
        var_name: Takeup variable name.
        rate_or_dict: Scalar rate or {state_code: rate} dict.
        entity_blocks: Block GEOID per entity (str array).
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
        ent_hh_ids = hh_ids[ent_hh_idx]

        rate_or_dict = load_take_up_rate(rate_key, time_period)
        bools = compute_block_takeup_for_entities(
            var_name,
            rate_or_dict,
            ent_blocks,
            ent_hh_ids,
        )
        result[var_name] = bools

    return result
