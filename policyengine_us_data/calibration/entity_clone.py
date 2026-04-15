"""Helpers for materializing cloned household subsets.

The calibration matrix and local-area publishing paths both need to clone
selected household rows while preserving person and sub-entity joins.  This
module keeps that logic in one place so chunked matrix builds can materialize
small mixed-geography H5 files instead of precomputing whole-dataset geography
cartesian products.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from policyengine_us_data.calibration.block_assignment import (
    derive_geography_from_blocks,
)
from policyengine_us_data.calibration.calibration_utils import (
    calculate_spm_thresholds_vectorized,
    load_cd_geoadj_values,
)
from policyengine_us_data.utils.takeup import (
    apply_block_takeup_to_arrays,
    reported_subsidized_marketplace_by_tax_unit,
)


SUB_ENTITIES = [
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
]


@dataclass
class EntityMaps:
    """Base household to person/sub-entity membership maps."""

    time_period: int
    household_ids: np.ndarray
    person_hh_ids: np.ndarray
    hh_to_persons: dict
    hh_to_entity: dict
    entity_id_arrays: dict
    person_entity_id_arrays: dict


@dataclass
class MaterializedCloneSummary:
    """Diagnostics for one materialized cloned-H5 chunk."""

    output_path: Path
    n_households: int
    n_persons: int
    entity_counts: dict
    unique_states: int
    unique_counties: int
    unique_cds: int


def build_household_entity_maps(sim) -> EntityMaps:
    """Build reusable maps from base households to dependent entities."""

    time_period = int(sim.default_calculation_period)
    household_ids = sim.calculate("household_id", map_to="household").values
    person_hh_ids = sim.calculate("household_id", map_to="person").values
    hh_id_to_idx = {int(hid): i for i, hid in enumerate(household_ids)}

    hh_to_persons = defaultdict(list)
    for p_idx, p_hh_id in enumerate(person_hh_ids):
        hh_to_persons[hh_id_to_idx[int(p_hh_id)]].append(p_idx)

    hh_to_entity = {}
    entity_id_arrays = {}
    person_entity_id_arrays = {}

    for entity_key in SUB_ENTITIES:
        entity_ids = sim.calculate(f"{entity_key}_id", map_to=entity_key).values
        person_entity_ids = sim.calculate(
            f"person_{entity_key}_id",
            map_to="person",
        ).values
        entity_id_arrays[entity_key] = entity_ids
        person_entity_id_arrays[entity_key] = person_entity_ids
        entity_id_to_idx = {int(eid): i for i, eid in enumerate(entity_ids)}

        mapping = defaultdict(list)
        seen = defaultdict(set)
        for p_idx in range(len(person_hh_ids)):
            hh_idx = hh_id_to_idx[int(person_hh_ids[p_idx])]
            entity_idx = entity_id_to_idx[int(person_entity_ids[p_idx])]
            if entity_idx not in seen[hh_idx]:
                seen[hh_idx].add(entity_idx)
                mapping[hh_idx].append(entity_idx)
        for hh_idx in mapping:
            mapping[hh_idx].sort()
        hh_to_entity[entity_key] = mapping

    return EntityMaps(
        time_period=time_period,
        household_ids=household_ids,
        person_hh_ids=person_hh_ids,
        hh_to_persons=hh_to_persons,
        hh_to_entity=hh_to_entity,
        entity_id_arrays=entity_id_arrays,
        person_entity_id_arrays=person_entity_id_arrays,
    )


def _build_reported_takeup_anchors(data: dict, time_period: int) -> dict:
    reported_anchors = {}
    if (
        "reported_has_subsidized_marketplace_health_coverage_at_interview" in data
        and time_period
        in data["reported_has_subsidized_marketplace_health_coverage_at_interview"]
    ):
        reported_anchors["takes_up_aca_if_eligible"] = (
            reported_subsidized_marketplace_by_tax_unit(
                data["person_tax_unit_id"][time_period],
                data["tax_unit_id"][time_period],
                data[
                    "reported_has_subsidized_marketplace_health_coverage_at_interview"
                ][time_period],
            )
        )
    if (
        "has_medicaid_health_coverage_at_interview" in data
        and time_period in data["has_medicaid_health_coverage_at_interview"]
    ):
        reported_anchors["takes_up_medicaid_if_eligible"] = data[
            "has_medicaid_health_coverage_at_interview"
        ][time_period].astype(bool)
    return reported_anchors


def materialize_clone_household_chunk(
    sim,
    entity_maps: EntityMaps,
    active_hh: np.ndarray,
    active_blocks: np.ndarray,
    active_cd_geoids: np.ndarray,
    active_clone_indices: np.ndarray,
    output_path: Path,
    household_weights: Optional[np.ndarray] = None,
    apply_takeup: bool = True,
    takeup_filter: Optional[list[str]] = None,
) -> MaterializedCloneSummary:
    """Write a cloned household subset H5 for one chunk.

    Args:
        sim: Base ``Microsimulation``.
        entity_maps: Output of :func:`build_household_entity_maps`.
        active_hh: Base household row index for each cloned household.
        active_blocks: Assigned block GEOID for each cloned household.
        active_cd_geoids: Assigned congressional district GEOID per row.
        active_clone_indices: Global clone index per row, used for salted takeup.
        output_path: H5 path to write.
        household_weights: Optional household weights. Defaults to ones because
            matrix entries are per-column feature values, not weighted totals.
        apply_takeup: Whether to write block-salted takeup input arrays.
        takeup_filter: Optional takeup variable filter.
    """
    from policyengine_core.enums import Enum

    active_hh = np.asarray(active_hh, dtype=np.int64)
    active_blocks = np.asarray(active_blocks, dtype=str)
    active_cd_geoids = np.asarray(active_cd_geoids, dtype=str)
    active_clone_indices = np.asarray(active_clone_indices, dtype=np.int64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(active_hh) == 0:
        raise ValueError("Cannot materialize an empty clone-household chunk")
    if not (
        len(active_hh)
        == len(active_blocks)
        == len(active_cd_geoids)
        == len(active_clone_indices)
    ):
        raise ValueError("Chunk arrays must all have the same length")
    if np.any(active_blocks == ""):
        raise ValueError("Active clone-households include empty block GEOIDs")

    time_period = entity_maps.time_period
    n_clones = len(active_hh)

    persons_per_clone = np.array(
        [len(entity_maps.hh_to_persons.get(h, [])) for h in active_hh],
        dtype=np.int64,
    )
    person_parts = [
        np.array(entity_maps.hh_to_persons.get(h, []), dtype=np.int64)
        for h in active_hh
    ]
    person_clone_idx = (
        np.concatenate(person_parts) if person_parts else np.array([], dtype=np.int64)
    )

    entity_clone_idx = {}
    entities_per_clone = {}
    for entity_key in SUB_ENTITIES:
        per_clone = np.array(
            [len(entity_maps.hh_to_entity[entity_key].get(h, [])) for h in active_hh],
            dtype=np.int64,
        )
        entities_per_clone[entity_key] = per_clone
        parts = [
            np.array(
                entity_maps.hh_to_entity[entity_key].get(h, []),
                dtype=np.int64,
            )
            for h in active_hh
        ]
        entity_clone_idx[entity_key] = (
            np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        )

    n_persons = len(person_clone_idx)
    new_hh_ids = np.arange(n_clones, dtype=np.int32)
    new_person_ids = np.arange(n_persons, dtype=np.int32)
    new_person_hh_ids = np.repeat(new_hh_ids, persons_per_clone)

    new_entity_ids = {}
    new_person_entity_ids = {}
    clone_ids_for_persons = np.repeat(
        np.arange(n_clones, dtype=np.int64),
        persons_per_clone,
    )

    for entity_key in SUB_ENTITIES:
        n_entities = len(entity_clone_idx[entity_key])
        new_entity_ids[entity_key] = np.arange(n_entities, dtype=np.int32)

        old_entity_ids = entity_maps.entity_id_arrays[entity_key][
            entity_clone_idx[entity_key]
        ].astype(np.int64)
        clone_ids_for_entities = np.repeat(
            np.arange(n_clones, dtype=np.int64),
            entities_per_clone[entity_key],
        )

        offset = int(old_entity_ids.max()) + 1 if len(old_entity_ids) > 0 else 1
        entity_keys = clone_ids_for_entities * offset + old_entity_ids
        sorted_order = np.argsort(entity_keys)
        sorted_keys = entity_keys[sorted_order]
        sorted_new = new_entity_ids[entity_key][sorted_order]

        person_old_entity_ids = entity_maps.person_entity_id_arrays[entity_key][
            person_clone_idx
        ].astype(np.int64)
        person_keys = clone_ids_for_persons * offset + person_old_entity_ids
        positions = np.searchsorted(sorted_keys, person_keys)
        positions = np.clip(positions, 0, len(sorted_keys) - 1)
        if len(sorted_keys) and not np.array_equal(sorted_keys[positions], person_keys):
            raise ValueError(
                f"Failed to rebuild person_{entity_key}_id references for chunk"
            )
        new_person_entity_ids[entity_key] = sorted_new[positions]

    unique_blocks, block_inv = np.unique(active_blocks, return_inverse=True)
    unique_geo = derive_geography_from_blocks(unique_blocks)
    clone_geo = {k: v[block_inv] for k, v in unique_geo.items()}

    vars_to_save = set(sim.input_variables)
    vars_to_save.add("county")
    vars_to_save.add("spm_unit_spm_threshold")
    vars_to_save.add("congressional_district_geoid")
    for geo_var in [
        "block_geoid",
        "tract_geoid",
        "cbsa_code",
        "sldu",
        "sldl",
        "place_fips",
        "vtd",
        "puma",
        "zcta",
    ]:
        vars_to_save.add(geo_var)

    clone_idx_map = {
        "household": active_hh,
        "person": person_clone_idx,
    }
    for entity_key in SUB_ENTITIES:
        clone_idx_map[entity_key] = entity_clone_idx[entity_key]

    data = {}
    for variable in sim.tax_benefit_system.variables:
        if variable not in vars_to_save:
            continue

        holder = sim.get_holder(variable)
        periods = holder.get_known_periods()
        if not periods:
            continue

        var_def = sim.tax_benefit_system.variables.get(variable)
        entity_key = var_def.entity.key
        if entity_key not in clone_idx_map:
            continue

        cloned_indices = clone_idx_map[entity_key]
        var_data = {}
        for period in periods:
            values = holder.get_array(period)
            if hasattr(values, "_pa_array") or hasattr(values, "_ndarray"):
                values = np.asarray(values)

            if var_def.value_type in (Enum, str) and variable != "county_fips":
                if hasattr(values, "decode_to_str"):
                    values = values.decode_to_str().astype("S")
                else:
                    values = np.asarray(values).astype("S")
            elif variable == "county_fips":
                values = np.asarray(values).astype("int32")
            else:
                values = np.asarray(values)

            var_data[period] = values[cloned_indices]

        if var_data:
            data[variable] = var_data

    data["household_id"] = {time_period: new_hh_ids}
    data["person_id"] = {time_period: new_person_ids}
    data["person_household_id"] = {time_period: new_person_hh_ids}
    for entity_key in SUB_ENTITIES:
        data[f"{entity_key}_id"] = {time_period: new_entity_ids[entity_key]}
        data[f"person_{entity_key}_id"] = {
            time_period: new_person_entity_ids[entity_key],
        }

    if household_weights is None:
        household_weights = np.ones(n_clones, dtype=np.float32)
    data["household_weight"] = {
        time_period: np.asarray(household_weights, dtype=np.float32),
    }

    data["state_fips"] = {
        time_period: clone_geo["state_fips"].astype(np.int32),
    }
    data["county"] = {
        time_period: clone_geo["county_index"].astype(np.int32),
    }
    data["county_fips"] = {
        time_period: clone_geo["county_fips"].astype(np.int32),
    }
    for geo_var in [
        "block_geoid",
        "tract_geoid",
        "cbsa_code",
        "sldu",
        "sldl",
        "place_fips",
        "vtd",
        "puma",
        "zcta",
    ]:
        if geo_var in clone_geo:
            data[geo_var] = {time_period: clone_geo[geo_var].astype("S")}

    la_mask = clone_geo["county_fips"].astype(str) == "06037"
    if la_mask.any():
        zip_codes = np.full(len(la_mask), "00000")
        zip_codes[la_mask] = "90001"
        data["zip_code"] = {time_period: zip_codes.astype("S")}

    data["congressional_district_geoid"] = {
        time_period: np.array([int(cd) for cd in active_cd_geoids], dtype=np.int32),
    }

    unique_cds_list = sorted(set(active_cd_geoids))
    cd_geoadj_values = load_cd_geoadj_values(unique_cds_list)
    spm_clone_ids = np.repeat(
        np.arange(n_clones, dtype=np.int64),
        entities_per_clone["spm_unit"],
    )
    spm_unit_geoadj = np.array(
        [cd_geoadj_values[active_cd_geoids[c]] for c in spm_clone_ids],
        dtype=np.float64,
    )

    person_ages = sim.calculate("age", map_to="person").values[person_clone_idx]
    spm_tenure_holder = sim.get_holder("spm_unit_tenure_type")
    spm_tenure_periods = spm_tenure_holder.get_known_periods()
    if spm_tenure_periods:
        raw_tenure = spm_tenure_holder.get_array(spm_tenure_periods[0])
        if hasattr(raw_tenure, "decode_to_str"):
            raw_tenure = raw_tenure.decode_to_str().astype("S")
        else:
            raw_tenure = np.array(raw_tenure).astype("S")
        spm_tenure_cloned = raw_tenure[entity_clone_idx["spm_unit"]]
    else:
        spm_tenure_cloned = np.full(
            len(entity_clone_idx["spm_unit"]),
            b"RENTER",
            dtype="S30",
        )

    data["spm_unit_spm_threshold"] = {
        time_period: calculate_spm_thresholds_vectorized(
            person_ages=person_ages,
            person_spm_unit_ids=new_person_entity_ids["spm_unit"],
            spm_unit_tenure_types=spm_tenure_cloned,
            spm_unit_geoadj=spm_unit_geoadj,
            year=time_period,
        ),
    }

    if apply_takeup:
        entity_hh_indices = {
            "person": np.repeat(
                np.arange(n_clones, dtype=np.int64),
                persons_per_clone,
            ).astype(np.int64),
            "tax_unit": np.repeat(
                np.arange(n_clones, dtype=np.int64),
                entities_per_clone["tax_unit"],
            ).astype(np.int64),
            "spm_unit": np.repeat(
                np.arange(n_clones, dtype=np.int64),
                entities_per_clone["spm_unit"],
            ).astype(np.int64),
        }
        entity_counts = {
            "person": n_persons,
            "tax_unit": len(entity_clone_idx["tax_unit"]),
            "spm_unit": len(entity_clone_idx["spm_unit"]),
        }
        reported_anchors = _build_reported_takeup_anchors(data, time_period)
        takeup_results = apply_block_takeup_to_arrays(
            hh_blocks=active_blocks,
            hh_state_fips=clone_geo["state_fips"].astype(np.int32),
            hh_ids=entity_maps.household_ids[active_hh].astype(np.int64),
            hh_clone_indices=active_clone_indices.astype(np.int64),
            entity_hh_indices=entity_hh_indices,
            entity_counts=entity_counts,
            time_period=time_period,
            takeup_filter=takeup_filter,
            reported_anchors=reported_anchors,
        )
        for variable, values in takeup_results.items():
            data[variable] = {time_period: values}

    with h5py.File(str(output_path), "w") as f:
        for variable, periods in data.items():
            group = f.create_group(variable)
            for period, values in periods.items():
                group.create_dataset(str(period), data=values)

    return MaterializedCloneSummary(
        output_path=output_path,
        n_households=n_clones,
        n_persons=n_persons,
        entity_counts={
            entity_key: len(entity_clone_idx[entity_key]) for entity_key in SUB_ENTITIES
        },
        unique_states=len(np.unique(clone_geo["state_fips"])),
        unique_counties=len(np.unique(clone_geo["county_fips"])),
        unique_cds=len(np.unique(active_cd_geoids)),
    )
