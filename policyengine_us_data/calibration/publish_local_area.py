"""
Publish local area H5 files to GCP and Hugging Face.

Downloads calibration inputs from HF, generates state/district H5s
with checkpointing, and uploads to both destinations.

Usage:
    python publish_local_area.py [--skip-download] [--states-only] [--districts-only]
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Set

from policyengine_us import Microsimulation
from policyengine_us_data.utils.huggingface import download_calibration_inputs
from policyengine_us_data.utils.data_upload import (
    upload_local_area_file,
    upload_local_area_batch_to_hf,
)
from policyengine_us_data.calibration.stacked_dataset_builder import (
    create_sparse_cd_stacked_dataset,
    NYC_COUNTIES,
    NYC_CDS,
)
from policyengine_us_data.calibration.calibration_utils import (
    get_all_cds_from_database,
    STATE_CODES,
)
from policyengine_us_data.utils.takeup import TAKEUP_AFFECTED_TARGETS

CHECKPOINT_FILE = Path("completed_states.txt")
CHECKPOINT_FILE_DISTRICTS = Path("completed_districts.txt")
CHECKPOINT_FILE_CITIES = Path("completed_cities.txt")
WORK_DIR = Path("local_area_build")


def load_completed_states() -> set:
    if CHECKPOINT_FILE.exists():
        content = CHECKPOINT_FILE.read_text().strip()
        if content:
            return set(content.split("\n"))
    return set()


def record_completed_state(state_code: str):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{state_code}\n")


def load_completed_districts() -> set:
    if CHECKPOINT_FILE_DISTRICTS.exists():
        content = CHECKPOINT_FILE_DISTRICTS.read_text().strip()
        if content:
            return set(content.split("\n"))
    return set()


def record_completed_district(district_name: str):
    with open(CHECKPOINT_FILE_DISTRICTS, "a") as f:
        f.write(f"{district_name}\n")


def load_completed_cities() -> set:
    if CHECKPOINT_FILE_CITIES.exists():
        content = CHECKPOINT_FILE_CITIES.read_text().strip()
        if content:
            return set(content.split("\n"))
    return set()


def record_completed_city(city_name: str):
    with open(CHECKPOINT_FILE_CITIES, "a") as f:
        f.write(f"{city_name}\n")


def build_state_h5(
    state_code: str,
    weights: np.ndarray,
    cds_to_calibrate: List[str],
    dataset_path: Path,
    output_dir: Path,
    rerandomize_takeup: bool = False,
    calibration_blocks: np.ndarray = None,
    takeup_filter: List[str] = None,
) -> Optional[Path]:
    """
    Build a single state H5 file (build only, no upload).

    Args:
        state_code: Two-letter state code (e.g., "AL", "CA")
        weights: Calibrated weight vector
        cds_to_calibrate: Full list of CD GEOIDs from calibration
        dataset_path: Path to base dataset H5 file
        output_dir: Output directory for H5 file
        rerandomize_takeup: Re-draw takeup using block-level seeds
        calibration_blocks: Stacked block GEOID array from calibration
        takeup_filter: List of takeup vars to re-randomize

    Returns:
        Path to output H5 file if successful, None if no CDs found
    """
    state_fips = None
    for fips, code in STATE_CODES.items():
        if code == state_code:
            state_fips = fips
            break

    if state_fips is None:
        print(f"Unknown state code: {state_code}")
        return None

    cd_subset = [cd for cd in cds_to_calibrate if int(cd) // 100 == state_fips]
    if not cd_subset:
        print(f"No CDs found for {state_code}, skipping")
        return None

    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    output_path = states_dir / f"{state_code}.h5"

    print(f"\n{'='*60}")
    print(f"Building {state_code} ({len(cd_subset)} CDs)")
    print(f"{'='*60}")

    create_sparse_cd_stacked_dataset(
        weights,
        cds_to_calibrate,
        cd_subset=cd_subset,
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        rerandomize_takeup=rerandomize_takeup,
        calibration_blocks=calibration_blocks,
        takeup_filter=takeup_filter,
    )

    return output_path


def build_district_h5(
    cd_geoid: str,
    weights: np.ndarray,
    cds_to_calibrate: List[str],
    dataset_path: Path,
    output_dir: Path,
    rerandomize_takeup: bool = False,
    calibration_blocks: np.ndarray = None,
    takeup_filter: List[str] = None,
) -> Path:
    """
    Build a single district H5 file (build only, no upload).

    Args:
        cd_geoid: Congressional district GEOID (e.g., "0101" for AL-01)
        weights: Calibrated weight vector
        cds_to_calibrate: Full list of CD GEOIDs from calibration
        dataset_path: Path to base dataset H5 file
        output_dir: Output directory for H5 file
        rerandomize_takeup: Re-draw takeup using block-level seeds
        calibration_blocks: Stacked block GEOID array from calibration
        takeup_filter: List of takeup vars to re-randomize

    Returns:
        Path to output H5 file
    """
    cd_int = int(cd_geoid)
    state_fips = cd_int // 100
    district_num = cd_int % 100
    if district_num in AT_LARGE_DISTRICTS:
        district_num = 1
    state_code = STATE_CODES.get(state_fips, str(state_fips))
    friendly_name = f"{state_code}-{district_num:02d}"

    districts_dir = output_dir / "districts"
    districts_dir.mkdir(parents=True, exist_ok=True)
    output_path = districts_dir / f"{friendly_name}.h5"

    print(f"\n{'='*60}")
    print(f"Building {friendly_name}")
    print(f"{'='*60}")

    create_sparse_cd_stacked_dataset(
        weights,
        cds_to_calibrate,
        cd_subset=[cd_geoid],
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        rerandomize_takeup=rerandomize_takeup,
        calibration_blocks=calibration_blocks,
        takeup_filter=takeup_filter,
    )

    return output_path


def build_city_h5(
    city_name: str,
    weights: np.ndarray,
    cds_to_calibrate: List[str],
    dataset_path: Path,
    output_dir: Path,
    rerandomize_takeup: bool = False,
    calibration_blocks: np.ndarray = None,
    takeup_filter: List[str] = None,
) -> Optional[Path]:
    """
    Build a city H5 file (build only, no upload).

    Currently supports NYC only.

    Args:
        city_name: City name (currently only "NYC" supported)
        weights: Calibrated weight vector
        cds_to_calibrate: Full list of CD GEOIDs from calibration
        dataset_path: Path to base dataset H5 file
        output_dir: Output directory for H5 file
        rerandomize_takeup: Re-draw takeup using block-level seeds
        calibration_blocks: Stacked block GEOID array from calibration
        takeup_filter: List of takeup vars to re-randomize

    Returns:
        Path to output H5 file if successful, None otherwise
    """
    if city_name != "NYC":
        print(f"Unsupported city: {city_name}")
        return None

    cd_subset = [cd for cd in cds_to_calibrate if cd in NYC_CDS]
    if not cd_subset:
        print("No NYC-related CDs found, skipping")
        return None

    cities_dir = output_dir / "cities"
    cities_dir.mkdir(parents=True, exist_ok=True)
    output_path = cities_dir / "NYC.h5"

    print(f"\n{'='*60}")
    print(f"Building NYC ({len(cd_subset)} CDs)")
    print(f"{'='*60}")

    create_sparse_cd_stacked_dataset(
        weights,
        cds_to_calibrate,
        cd_subset=cd_subset,
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        county_filter=NYC_COUNTIES,
        rerandomize_takeup=rerandomize_takeup,
        calibration_blocks=calibration_blocks,
        takeup_filter=takeup_filter,
    )

    return output_path


def build_national_h5(
    weights: np.ndarray,
    blocks: np.ndarray,
    dataset_path: Path,
    output_dir: Path,
) -> Path:
    """Build national US.h5 by cloning records for each nonzero weight.

    Each nonzero entry in the (n_geo, n_hh) weight matrix represents a
    distinct household clone placed at a specific census block. This
    function clones entity arrays via fancy indexing, derives geography
    from the blocks array, reindexes all entity IDs, and writes the H5.
    """
    import h5py
    from collections import defaultdict
    from policyengine_core.enums import Enum
    from policyengine_us_data.calibration.block_assignment import (
        derive_geography_from_blocks,
    )
    from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
        County,
    )

    national_dir = output_dir / "national"
    national_dir.mkdir(parents=True, exist_ok=True)
    output_path = national_dir / "US.h5"

    # === Load base simulation ===
    sim = Microsimulation(dataset=str(dataset_path))
    time_period = int(sim.default_calculation_period)
    household_ids = sim.calculate("household_id", map_to="household").values
    n_hh = len(household_ids)

    if weights.shape[0] % n_hh != 0:
        raise ValueError(
            f"Weight vector length {weights.shape[0]} is not "
            f"divisible by n_hh={n_hh}"
        )
    if len(blocks) != len(weights):
        raise ValueError(
            f"Blocks length {len(blocks)} != " f"weights length {len(weights)}"
        )
    n_geo = weights.shape[0] // n_hh

    print(f"\n{'='*60}")
    print(
        f"Building national US.h5 " f"({n_geo} geo units, {n_hh} households)"
    )
    print(f"{'='*60}")

    # === Identify active clones ===
    W = weights.reshape(n_geo, n_hh)
    active_geo, active_hh = np.where(W > 0)
    n_clones = len(active_geo)
    clone_weights = W[active_geo, active_hh]
    active_blocks = blocks[active_geo * n_hh + active_hh]

    empty_count = np.sum(active_blocks == "")
    if empty_count > 0:
        raise ValueError(
            f"{empty_count} active clones have empty block GEOIDs"
        )

    print(f"Active clones: {n_clones:,}")
    print(f"Total weight: {clone_weights.sum():,.0f}")

    # === Build entity membership maps ===
    hh_id_to_idx = {int(hid): i for i, hid in enumerate(household_ids)}
    person_hh_ids = sim.calculate("household_id", map_to="person").values

    hh_to_persons = defaultdict(list)
    for p_idx, p_hh_id in enumerate(person_hh_ids):
        hh_to_persons[hh_id_to_idx[int(p_hh_id)]].append(p_idx)

    SUB_ENTITIES = [
        "tax_unit",
        "spm_unit",
        "family",
        "marital_unit",
    ]
    hh_to_entity = {}
    entity_id_arrays = {}
    person_entity_id_arrays = {}

    for ek in SUB_ENTITIES:
        eids = sim.calculate(f"{ek}_id", map_to=ek).values
        peids = sim.calculate(f"person_{ek}_id", map_to="person").values
        entity_id_arrays[ek] = eids
        person_entity_id_arrays[ek] = peids
        eid_to_idx = {int(eid): i for i, eid in enumerate(eids)}

        mapping = defaultdict(list)
        seen = defaultdict(set)
        for p_idx in range(len(person_hh_ids)):
            hh_idx = hh_id_to_idx[int(person_hh_ids[p_idx])]
            e_idx = eid_to_idx[int(peids[p_idx])]
            if e_idx not in seen[hh_idx]:
                seen[hh_idx].add(e_idx)
                mapping[hh_idx].append(e_idx)
        hh_to_entity[ek] = mapping

    # === Build clone index arrays ===
    hh_clone_idx = active_hh

    persons_per_clone = np.array(
        [len(hh_to_persons.get(h, [])) for h in active_hh]
    )
    person_parts = [
        np.array(hh_to_persons.get(h, []), dtype=np.int64) for h in active_hh
    ]
    person_clone_idx = (
        np.concatenate(person_parts)
        if person_parts
        else np.array([], dtype=np.int64)
    )

    entity_clone_idx = {}
    entities_per_clone = {}
    for ek in SUB_ENTITIES:
        epc = np.array([len(hh_to_entity[ek].get(h, [])) for h in active_hh])
        entities_per_clone[ek] = epc
        parts = [
            np.array(hh_to_entity[ek].get(h, []), dtype=np.int64)
            for h in active_hh
        ]
        entity_clone_idx[ek] = (
            np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        )

    n_persons = len(person_clone_idx)
    print(f"Cloned persons: {n_persons:,}")
    for ek in SUB_ENTITIES:
        print(f"Cloned {ek}s: {len(entity_clone_idx[ek]):,}")

    # === Build new entity IDs and cross-references ===
    new_hh_ids = np.arange(n_clones, dtype=np.int32)
    new_person_ids = np.arange(n_persons, dtype=np.int32)
    new_person_hh_ids = np.repeat(new_hh_ids, persons_per_clone)

    new_entity_ids = {}
    new_person_entity_ids = {}
    clone_ids_for_persons = np.repeat(
        np.arange(n_clones, dtype=np.int64), persons_per_clone
    )

    for ek in SUB_ENTITIES:
        n_ents = len(entity_clone_idx[ek])
        new_entity_ids[ek] = np.arange(n_ents, dtype=np.int32)

        old_eids = entity_id_arrays[ek][entity_clone_idx[ek]].astype(np.int64)
        clone_ids_e = np.repeat(
            np.arange(n_clones, dtype=np.int64),
            entities_per_clone[ek],
        )

        offset = int(old_eids.max()) + 1 if len(old_eids) > 0 else 1
        entity_keys = clone_ids_e * offset + old_eids

        sorted_order = np.argsort(entity_keys)
        sorted_keys = entity_keys[sorted_order]
        sorted_new = new_entity_ids[ek][sorted_order]

        p_old_eids = person_entity_id_arrays[ek][person_clone_idx].astype(
            np.int64
        )
        person_keys = clone_ids_for_persons * offset + p_old_eids

        positions = np.searchsorted(sorted_keys, person_keys)
        positions = np.clip(positions, 0, len(sorted_keys) - 1)
        new_person_entity_ids[ek] = sorted_new[positions]

    # === Derive geography from blocks (dedup optimization) ===
    print("Deriving geography from blocks...")
    unique_blocks, block_inv = np.unique(active_blocks, return_inverse=True)
    print(f"  {n_clones:,} blocks -> " f"{len(unique_blocks):,} unique")
    unique_geo = derive_geography_from_blocks(unique_blocks)
    geography = {k: v[block_inv] for k, v in unique_geo.items()}

    # === Calculate weights for all entity levels ===
    person_weights = np.repeat(clone_weights, persons_per_clone)
    per_person_wt = clone_weights / np.maximum(persons_per_clone, 1)

    entity_weights = {}
    for ek in SUB_ENTITIES:
        n_ents = len(entity_clone_idx[ek])
        ent_person_counts = np.zeros(n_ents, dtype=np.int32)
        np.add.at(
            ent_person_counts,
            new_person_entity_ids[ek],
            1,
        )
        clone_ids_e = np.repeat(np.arange(n_clones), entities_per_clone[ek])
        entity_weights[ek] = per_person_wt[clone_ids_e] * ent_person_counts

    # === Determine variables to save ===
    vars_to_save = set(sim.input_variables)
    vars_to_save.add("county")
    vars_to_save.add("spm_unit_spm_threshold")
    for gv in [
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
        vars_to_save.add(gv)

    # === Clone variable arrays ===
    clone_idx_map = {
        "household": hh_clone_idx,
        "person": person_clone_idx,
    }
    for ek in SUB_ENTITIES:
        clone_idx_map[ek] = entity_clone_idx[ek]

    data = {}
    variables_saved = 0

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

        cidx = clone_idx_map[entity_key]
        var_data = {}

        for period in periods:
            values = holder.get_array(period)

            if var_def.value_type in (Enum, str) and variable != "county_fips":
                if hasattr(values, "decode_to_str"):
                    values = values.decode_to_str().astype("S")
                else:
                    values = values.astype("S")
            elif variable == "county_fips":
                values = values.astype("int32")
            else:
                values = np.array(values)

            var_data[period] = values[cidx]
            variables_saved += 1

        if var_data:
            data[variable] = var_data

    print(f"Variables cloned: {variables_saved}")

    # === Override entity IDs ===
    data["household_id"] = {time_period: new_hh_ids}
    data["person_id"] = {time_period: new_person_ids}
    data["person_household_id"] = {
        time_period: new_person_hh_ids,
    }
    for ek in SUB_ENTITIES:
        data[f"{ek}_id"] = {
            time_period: new_entity_ids[ek],
        }
        data[f"person_{ek}_id"] = {
            time_period: new_person_entity_ids[ek],
        }

    # === Override weights ===
    data["household_weight"] = {
        time_period: clone_weights.astype(np.float32),
    }
    data["person_weight"] = {
        time_period: person_weights.astype(np.float32),
    }
    for ek in SUB_ENTITIES:
        data[f"{ek}_weight"] = {
            time_period: entity_weights[ek].astype(np.float32),
        }

    # === Override geography ===
    data["state_fips"] = {
        time_period: geography["state_fips"].astype(np.int32),
    }
    county_names = np.array(
        [County._member_names_[i] for i in geography["county_index"]]
    ).astype("S")
    data["county"] = {time_period: county_names}
    data["county_fips"] = {
        time_period: geography["county_fips"].astype(np.int32),
    }
    for gv in [
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
        if gv in geography:
            data[gv] = {
                time_period: geography[gv].astype("S"),
            }

    # === Write H5 ===
    with h5py.File(str(output_path), "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    print(f"\nNational H5 saved to {output_path}")

    with h5py.File(str(output_path), "r") as f:
        tp = str(time_period)
        if "household_id" in f and tp in f["household_id"]:
            n = len(f["household_id"][tp][:])
            print(f"Verified: {n:,} households in output")
        if "person_id" in f and tp in f["person_id"]:
            n = len(f["person_id"][tp][:])
            print(f"Verified: {n:,} persons in output")
        if "household_weight" in f and tp in f["household_weight"]:
            hw = f["household_weight"][tp][:]
            print(f"Total population (HH weights): " f"{hw.sum():,.0f}")
        if "person_weight" in f and tp in f["person_weight"]:
            pw = f["person_weight"][tp][:]
            print(f"Total population (person weights): " f"{pw.sum():,.0f}")

    return output_path


AT_LARGE_DISTRICTS = {0, 98}


def get_district_friendly_name(cd_geoid: str) -> str:
    """Convert GEOID to friendly name (e.g., '0101' -> 'AL-01')."""
    cd_int = int(cd_geoid)
    state_fips = cd_int // 100
    district_num = cd_int % 100
    if district_num in AT_LARGE_DISTRICTS:
        district_num = 1
    state_code = STATE_CODES.get(state_fips, str(state_fips))
    return f"{state_code}-{district_num:02d}"


def build_and_upload_states(
    weights_path: Path,
    dataset_path: Path,
    db_path: Path,
    output_dir: Path,
    completed_states: set,
    hf_batch_size: int = 10,
    rerandomize_takeup: bool = False,
    calibration_blocks: np.ndarray = None,
    takeup_filter: List[str] = None,
):
    """Build and upload state H5 files with checkpointing."""
    db_uri = f"sqlite:///{db_path}"
    cds_to_calibrate = get_all_cds_from_database(db_uri)
    w = np.load(weights_path)

    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    hf_queue = []  # Queue for batched HuggingFace uploads

    for state_fips, state_code in STATE_CODES.items():
        if state_code in completed_states:
            print(f"Skipping {state_code} (already completed)")
            continue

        cd_subset = [
            cd for cd in cds_to_calibrate if int(cd) // 100 == state_fips
        ]
        if not cd_subset:
            print(f"No CDs found for {state_code}, skipping")
            continue

        output_path = states_dir / f"{state_code}.h5"
        print(f"\n{'='*60}")
        print(f"Building {state_code} ({len(cd_subset)} CDs)")
        print(f"{'='*60}")

        try:
            create_sparse_cd_stacked_dataset(
                w,
                cds_to_calibrate,
                cd_subset=cd_subset,
                dataset_path=str(dataset_path),
                output_path=str(output_path),
                rerandomize_takeup=rerandomize_takeup,
                calibration_blocks=calibration_blocks,
                takeup_filter=takeup_filter,
            )

            print(f"Uploading {state_code}.h5 to GCP...")
            upload_local_area_file(str(output_path), "states", skip_hf=True)

            # Queue for batched HuggingFace upload
            hf_queue.append((str(output_path), "states"))

            record_completed_state(state_code)
            print(f"Completed {state_code}")

            # Flush HF queue every batch_size files
            if len(hf_queue) >= hf_batch_size:
                print(
                    f"\nUploading batch of {len(hf_queue)} files to HuggingFace..."
                )
                upload_local_area_batch_to_hf(hf_queue)
                hf_queue = []

        except Exception as e:
            print(f"ERROR building {state_code}: {e}")
            raise

    # Flush remaining files to HuggingFace
    if hf_queue:
        print(
            f"\nUploading final batch of {len(hf_queue)} files to HuggingFace..."
        )
        upload_local_area_batch_to_hf(hf_queue)


def build_and_upload_districts(
    weights_path: Path,
    dataset_path: Path,
    db_path: Path,
    output_dir: Path,
    completed_districts: set,
    hf_batch_size: int = 10,
    rerandomize_takeup: bool = False,
    calibration_blocks: np.ndarray = None,
    takeup_filter: List[str] = None,
):
    """Build and upload district H5 files with checkpointing."""
    db_uri = f"sqlite:///{db_path}"
    cds_to_calibrate = get_all_cds_from_database(db_uri)
    w = np.load(weights_path)

    districts_dir = output_dir / "districts"
    districts_dir.mkdir(parents=True, exist_ok=True)

    hf_queue = []  # Queue for batched HuggingFace uploads

    for i, cd_geoid in enumerate(cds_to_calibrate):
        cd_int = int(cd_geoid)
        state_fips = cd_int // 100
        district_num = cd_int % 100
        if district_num in AT_LARGE_DISTRICTS:
            district_num = 1
        state_code = STATE_CODES.get(state_fips, str(state_fips))
        friendly_name = f"{state_code}-{district_num:02d}"

        if friendly_name in completed_districts:
            print(f"Skipping {friendly_name} (already completed)")
            continue

        output_path = districts_dir / f"{friendly_name}.h5"
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(cds_to_calibrate)}] Building {friendly_name}")
        print(f"{'='*60}")

        try:
            create_sparse_cd_stacked_dataset(
                w,
                cds_to_calibrate,
                cd_subset=[cd_geoid],
                dataset_path=str(dataset_path),
                output_path=str(output_path),
                rerandomize_takeup=rerandomize_takeup,
                calibration_blocks=calibration_blocks,
                takeup_filter=takeup_filter,
            )

            print(f"Uploading {friendly_name}.h5 to GCP...")
            upload_local_area_file(str(output_path), "districts", skip_hf=True)

            # Queue for batched HuggingFace upload
            hf_queue.append((str(output_path), "districts"))

            record_completed_district(friendly_name)
            print(f"Completed {friendly_name}")

            # Flush HF queue every batch_size files
            if len(hf_queue) >= hf_batch_size:
                print(
                    f"\nUploading batch of {len(hf_queue)} files to HuggingFace..."
                )
                upload_local_area_batch_to_hf(hf_queue)
                hf_queue = []

        except Exception as e:
            print(f"ERROR building {friendly_name}: {e}")
            raise

    # Flush remaining files to HuggingFace
    if hf_queue:
        print(
            f"\nUploading final batch of {len(hf_queue)} files to HuggingFace..."
        )
        upload_local_area_batch_to_hf(hf_queue)


def build_and_upload_cities(
    weights_path: Path,
    dataset_path: Path,
    db_path: Path,
    output_dir: Path,
    completed_cities: set,
    hf_batch_size: int = 10,
    rerandomize_takeup: bool = False,
    calibration_blocks: np.ndarray = None,
    takeup_filter: List[str] = None,
):
    """Build and upload city H5 files with checkpointing."""
    db_uri = f"sqlite:///{db_path}"
    cds_to_calibrate = get_all_cds_from_database(db_uri)
    w = np.load(weights_path)

    cities_dir = output_dir / "cities"
    cities_dir.mkdir(parents=True, exist_ok=True)

    hf_queue = []  # Queue for batched HuggingFace uploads

    # NYC
    if "NYC" in completed_cities:
        print("Skipping NYC (already completed)")
    else:
        cd_subset = [cd for cd in cds_to_calibrate if cd in NYC_CDS]
        if not cd_subset:
            print("No NYC-related CDs found, skipping")
        else:
            output_path = cities_dir / "NYC.h5"
            print(f"\n{'='*60}")
            print(f"Building NYC ({len(cd_subset)} CDs)")
            print(f"{'='*60}")

            try:
                create_sparse_cd_stacked_dataset(
                    w,
                    cds_to_calibrate,
                    cd_subset=cd_subset,
                    dataset_path=str(dataset_path),
                    output_path=str(output_path),
                    county_filter=NYC_COUNTIES,
                    rerandomize_takeup=rerandomize_takeup,
                    calibration_blocks=calibration_blocks,
                    takeup_filter=takeup_filter,
                )

                print("Uploading NYC.h5 to GCP...")
                upload_local_area_file(
                    str(output_path), "cities", skip_hf=True
                )

                # Queue for batched HuggingFace upload
                hf_queue.append((str(output_path), "cities"))

                record_completed_city("NYC")
                print("Completed NYC")

            except Exception as e:
                print(f"ERROR building NYC: {e}")
                raise

    # Flush remaining files to HuggingFace
    if hf_queue:
        print(
            f"\nUploading batch of {len(hf_queue)} city files to HuggingFace..."
        )
        upload_local_area_batch_to_hf(hf_queue)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and publish local area H5 files"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading inputs from HF (use existing files)",
    )
    parser.add_argument(
        "--states-only",
        action="store_true",
        help="Only build and upload state files",
    )
    parser.add_argument(
        "--districts-only",
        action="store_true",
        help="Only build and upload district files",
    )
    parser.add_argument(
        "--cities-only",
        action="store_true",
        help="Only build and upload city files (e.g., NYC)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        help="Override path to weights file (for local testing)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Override path to dataset file (for local testing)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Override path to database file (for local testing)",
    )
    parser.add_argument(
        "--rerandomize-takeup",
        action="store_true",
        help="Re-draw takeup using block-level seeds",
    )
    parser.add_argument(
        "--calibration-blocks",
        type=str,
        help="Path to stacked_blocks.npy from calibration",
    )
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.weights_path and args.dataset_path and args.db_path:
        inputs = {
            "weights": Path(args.weights_path),
            "dataset": Path(args.dataset_path),
            "database": Path(args.db_path),
        }
        print("Using provided paths:")
        for key, path in inputs.items():
            print(f"  {key}: {path}")
    elif args.skip_download:
        inputs = {
            "weights": WORK_DIR / "calibration_weights.npy",
            "dataset": WORK_DIR / "stratified_extended_cps.h5",
            "database": WORK_DIR / "policy_data.db",
        }
        source_imputed = WORK_DIR / "source_imputed_stratified_extended_cps.h5"
        if source_imputed.exists():
            inputs["source_imputed_dataset"] = source_imputed
        print("Using existing files in work directory:")
        for key, path in inputs.items():
            if not path.exists():
                raise FileNotFoundError(f"Expected file not found: {path}")
            print(f"  {key}: {path}")
    else:
        print("Downloading calibration inputs from Hugging Face...")
        inputs = download_calibration_inputs(str(WORK_DIR))
        for key, path in inputs.items():
            inputs[key] = Path(path)

    if "source_imputed_dataset" in inputs:
        inputs["dataset"] = inputs["source_imputed_dataset"]
        print("Using source-imputed dataset")
    else:
        print(
            "WARNING: Source-imputed dataset not found, " "using base dataset"
        )

    sim = Microsimulation(dataset=str(inputs["dataset"]))
    n_hh = sim.calculate("household_id", map_to="household").shape[0]
    print(f"\nBase dataset has {n_hh:,} households")

    rerandomize_takeup = args.rerandomize_takeup
    calibration_blocks = None
    takeup_filter = None

    if args.calibration_blocks:
        calibration_blocks = np.load(args.calibration_blocks)
        rerandomize_takeup = True
        print(f"Loaded calibration blocks: {len(calibration_blocks):,}")
    elif rerandomize_takeup:
        blocks_path = inputs.get("blocks")
        if blocks_path and Path(blocks_path).exists():
            calibration_blocks = np.load(str(blocks_path))
            print(
                f"Loaded calibration blocks: " f"{len(calibration_blocks):,}"
            )
        else:
            print(
                "WARNING: --rerandomize-takeup set but no " "blocks available"
            )

    if rerandomize_takeup:
        takeup_filter = [
            info["takeup_var"] for info in TAKEUP_AFFECTED_TARGETS.values()
        ]
        print(f"Takeup filter: {takeup_filter}")

    # Determine what to build based on flags
    build_states = not args.districts_only and not args.cities_only
    build_districts = not args.states_only and not args.cities_only
    build_cities = not args.states_only and not args.districts_only

    # If a specific *-only flag is set, only build that type
    if args.states_only:
        build_states = True
        build_districts = False
        build_cities = False
    elif args.districts_only:
        build_states = False
        build_districts = True
        build_cities = False
    elif args.cities_only:
        build_states = False
        build_districts = False
        build_cities = True

    if build_states:
        print("\n" + "=" * 60)
        print("BUILDING STATE FILES")
        print("=" * 60)
        completed_states = load_completed_states()
        print(f"Already completed: {len(completed_states)} states")
        build_and_upload_states(
            inputs["weights"],
            inputs["dataset"],
            inputs["database"],
            WORK_DIR,
            completed_states,
            rerandomize_takeup=rerandomize_takeup,
            calibration_blocks=calibration_blocks,
            takeup_filter=takeup_filter,
        )

    if build_districts:
        print("\n" + "=" * 60)
        print("BUILDING DISTRICT FILES")
        print("=" * 60)
        completed_districts = load_completed_districts()
        print(f"Already completed: {len(completed_districts)} districts")
        build_and_upload_districts(
            inputs["weights"],
            inputs["dataset"],
            inputs["database"],
            WORK_DIR,
            completed_districts,
            rerandomize_takeup=rerandomize_takeup,
            calibration_blocks=calibration_blocks,
            takeup_filter=takeup_filter,
        )

    if build_cities:
        print("\n" + "=" * 60)
        print("BUILDING CITY FILES")
        print("=" * 60)
        completed_cities = load_completed_cities()
        print(f"Already completed: {len(completed_cities)} cities")
        build_and_upload_cities(
            inputs["weights"],
            inputs["dataset"],
            inputs["database"],
            WORK_DIR,
            completed_cities,
            rerandomize_takeup=rerandomize_takeup,
            calibration_blocks=calibration_blocks,
            takeup_filter=takeup_filter,
        )

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
