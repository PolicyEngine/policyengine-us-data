"""
Build local area H5 files, optionally uploading to GCP and Hugging Face.

Downloads calibration inputs from HF, generates state/district H5s
with checkpointing. Uploads only occur when --upload is explicitly passed.

Usage:
    python publish_local_area.py [--skip-download] [--states-only] [--upload]
"""

import json
import shutil


import numpy as np
from pathlib import Path
from typing import List

from policyengine_us import Microsimulation
from policyengine_us_data.utils.huggingface import download_calibration_inputs
from policyengine_us_data.utils.data_upload import (
    upload_local_area_file,
    upload_local_area_batch_to_hf,
)
from policyengine_us_data.calibration.calibration_utils import (
    STATE_CODES,
)
from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
    assign_random_geography,
)
from policyengine_us_data.calibration.local_h5.contracts import AreaFilter
from policyengine_us_data.calibration.local_h5.reindexing import EntityReindexer
from policyengine_us_data.calibration.local_h5.selection import AreaSelector
from policyengine_us_data.calibration.local_h5.source_dataset import (
    PolicyEngineDatasetReader,
    SourceDatasetSnapshot,
)
from policyengine_us_data.calibration.local_h5.us_augmentations import (
    USAugmentationService,
    build_reported_takeup_anchors,
)
from policyengine_us_data.calibration.local_h5.variables import (
    VariableCloner,
    VariableExportPolicy,
)
from policyengine_us_data.calibration.local_h5.weights import (
    CloneWeightMatrix,
    infer_clone_count_from_weight_length,
)
from policyengine_us_data.utils.takeup import SIMPLE_TAKEUP_VARS

CHECKPOINT_FILE = Path("completed_states.txt")
CHECKPOINT_FILE_DISTRICTS = Path("completed_districts.txt")
CHECKPOINT_FILE_CITIES = Path("completed_cities.txt")
WORK_DIR = Path("local_area_build")

NYC_COUNTY_FIPS = {"36005", "36047", "36061", "36081", "36085"}


META_FILE = WORK_DIR / "checkpoint_meta.json"


def _build_selection_filters(
    *,
    cd_subset: List[str] | None = None,
    county_fips_filter: set[str] | None = None,
) -> tuple[AreaFilter, ...]:
    filters = []
    if cd_subset is not None:
        filters.append(
            AreaFilter(
                geography_field="cd_geoid",
                op="in",
                value=tuple(str(cd) for cd in cd_subset),
            )
        )
    if county_fips_filter is not None:
        filters.append(
            AreaFilter(
                geography_field="county_fips",
                op="in",
                value=tuple(str(fips) for fips in sorted(county_fips_filter)),
            )
        )
    return tuple(filters)


def compute_input_fingerprint(
    weights_path: Path,
    dataset_path: Path,
    n_clones: int,
    seed: int,
    calibration_package_path: Path | None = None,
) -> str:
    if calibration_package_path is None:
        import hashlib

        h = hashlib.sha256()
        for p in [weights_path, dataset_path]:
            with open(p, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
        h.update(f"{n_clones}:{seed}".encode())
        return h.hexdigest()[:16]

    from policyengine_us_data.calibration.local_h5.fingerprinting import (
        FingerprintService,
    )

    service = FingerprintService()
    record = service.create_publish_fingerprint(
        weights_path=weights_path,
        dataset_path=dataset_path,
        calibration_package_path=calibration_package_path,
        n_clones=n_clones,
        seed=seed,
    )
    return record.digest


def validate_or_clear_checkpoints(fingerprint):
    from policyengine_us_data.calibration.local_h5.fingerprinting import (
        FingerprintRecord,
        FingerprintService,
    )

    service = FingerprintService()
    if isinstance(fingerprint, FingerprintRecord):
        record = fingerprint
    else:
        record = service.legacy_record(str(fingerprint))

    if META_FILE.exists():
        stored = service.read_record(META_FILE)
        if service.matches(stored, record):
            print(f"Inputs unchanged ({record.digest}), resuming...")
            return
        print(
            f"Inputs changed "
            f"({stored.digest} -> {record.digest}), "
            f"clearing..."
        )
    else:
        print(f"No checkpoint metadata, starting fresh ({record.digest})")
    h5_count = sum(
        1
        for subdir in ["states", "districts", "cities"]
        if (WORK_DIR / subdir).exists()
        for _ in (WORK_DIR / subdir).rglob("*.h5")
    )
    if h5_count > 0:
        print(
            f"WARNING: {h5_count} H5 files exist. "
            f"Clearing only checkpoint files, preserving H5s."
        )
        for cp in [
            CHECKPOINT_FILE,
            CHECKPOINT_FILE_DISTRICTS,
            CHECKPOINT_FILE_CITIES,
        ]:
            if cp.exists():
                cp.unlink()
    else:
        for cp in [
            CHECKPOINT_FILE,
            CHECKPOINT_FILE_DISTRICTS,
            CHECKPOINT_FILE_CITIES,
        ]:
            if cp.exists():
                cp.unlink()
        for subdir in ["states", "districts", "cities"]:
            d = WORK_DIR / subdir
            if d.exists():
                shutil.rmtree(d)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    service.write_record(META_FILE, record)


SUB_ENTITIES = [
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
]


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


def _build_reported_takeup_anchors(
    data: dict, time_period: int
) -> dict[str, np.ndarray]:
    return build_reported_takeup_anchors(data, time_period)


def build_h5(
    weights: np.ndarray,
    geography,
    dataset_path: Path,
    output_path: Path,
    cd_subset: List[str] = None,
    county_fips_filter: set = None,
    takeup_filter: List[str] = None,
    source_snapshot: SourceDatasetSnapshot | None = None,
) -> Path:
    """Build an H5 file by cloning records for each nonzero weight.

    Args:
        weights: Clone-level weight vector, shape (n_clones_total * n_hh,).
        geography: GeographyAssignment from assign_random_geography.
        dataset_path: Path to base dataset H5 file.
        output_path: Where to write the output H5 file.
        cd_subset: If provided, only include clones for these CDs.
        county_fips_filter: If provided, zero out weights for clones
            whose county FIPS is not in this set.
        takeup_filter: List of takeup vars to apply.

    Returns:
        Path to the output H5 file.
    """
    import h5py

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if source_snapshot is None:
        source_snapshot = PolicyEngineDatasetReader(tuple(SUB_ENTITIES)).load(
            dataset_path
        )
    elif source_snapshot.dataset_path != Path(dataset_path):
        raise ValueError(
            "source_snapshot.dataset_path does not match dataset_path "
            f"({source_snapshot.dataset_path} != {Path(dataset_path)})"
        )

    time_period = source_snapshot.time_period
    n_hh = source_snapshot.n_households

    weight_matrix = CloneWeightMatrix.from_vector(weights, n_hh)
    selector = AreaSelector()
    selection = selector.select(
        weight_matrix,
        geography,
        filters=_build_selection_filters(
            cd_subset=cd_subset,
            county_fips_filter=county_fips_filter,
        ),
    )

    label = (
        f"CD subset {cd_subset}"
        if cd_subset is not None
        else f"{weight_matrix.n_clones} clone rows"
    )
    print(f"\n{'=' * 60}")
    print(f"Building {output_path.name} ({label}, {n_hh} households)")
    print(f"{'=' * 60}")

    # === Identify active clones ===
    n_clones = selection.n_household_clones
    if selection.is_empty:
        raise ValueError(
            f"No active clones after filtering. "
            f"cd_subset={cd_subset}, county_fips_filter={county_fips_filter}"
        )
    clone_weights = selection.active_weights
    active_blocks = selection.active_block_geoids
    empty_count = np.sum(active_blocks == "")
    if empty_count > 0:
        raise ValueError(f"{empty_count} active clones have empty block GEOIDs")

    print(f"Active clones: {n_clones:,}")
    print(f"Total weight: {clone_weights.sum():,.0f}")

    # === Build clone index arrays and output IDs ===
    reindexed = EntityReindexer().reindex(source_snapshot, selection)

    entity_clone_idx = reindexed.entity_source_indices

    n_persons = len(reindexed.person_source_indices)
    print(f"Cloned persons: {n_persons:,}")
    for ek in SUB_ENTITIES:
        print(f"Cloned {ek}s: {len(entity_clone_idx[ek]):,}")

    # === Clone variable arrays ===
    payload = VariableCloner().clone(
        source_snapshot,
        reindexed,
        VariableExportPolicy(include_input_variables=True),
    )
    data = {
        variable: dict(periods) for variable, periods in payload.variables.items()
    }
    print(f"Variables cloned: {payload.dataset_count}")

    # === Override entity IDs ===
    data["household_id"] = {time_period: reindexed.new_household_ids}
    data["person_id"] = {time_period: reindexed.new_person_ids}
    data["person_household_id"] = {
        time_period: reindexed.new_person_household_ids,
    }
    for ek in SUB_ENTITIES:
        data[f"{ek}_id"] = {
            time_period: reindexed.new_entity_ids[ek],
        }
        data[f"person_{ek}_id"] = {
            time_period: reindexed.new_person_entity_ids[ek],
        }

    # === Override weights ===
    # Only write household_weight; sub-entity weights (tax_unit_weight,
    # spm_unit_weight, person_weight, etc.) are formula variables in
    # policyengine-us that derive from household_weight at runtime.
    data["household_weight"] = {
        time_period: clone_weights.astype(np.float32),
    }

    print("Applying US-specific output augmentations...")
    USAugmentationService().apply_all(
        data,
        time_period=time_period,
        selection=selection,
        source=source_snapshot,
        reindexed=reindexed,
        takeup_filter=takeup_filter,
    )
    # === Write H5 ===
    with h5py.File(str(output_path), "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    print(f"\nH5 saved to {output_path}")

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
            print(f"Total population (HH weights): {hw.sum():,.0f}")
        if "person_weight" in f and tp in f["person_weight"]:
            pw = f["person_weight"][tp][:]
            print(f"Total population (person weights): {pw.sum():,.0f}")

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


def build_states(
    weights_path: Path,
    dataset_path: Path,
    geography,
    output_dir: Path,
    completed_states: set,
    hf_batch_size: int = 10,
    takeup_filter: List[str] = None,
    upload: bool = False,
    state_filter: str = None,
):
    """Build state H5 files with checkpointing, optionally uploading."""
    w = np.load(weights_path)

    all_cds = sorted(set(geography.cd_geoid.astype(str)))

    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    hf_queue = []

    for state_fips, state_code in STATE_CODES.items():
        if state_filter and state_code != state_filter:
            continue
        if state_code in completed_states:
            print(f"Skipping {state_code} (already completed)")
            continue

        cd_subset = [cd for cd in all_cds if int(cd) // 100 == state_fips]
        if not cd_subset:
            print(f"No CDs found for {state_code}, skipping")
            continue

        output_path = states_dir / f"{state_code}.h5"

        try:
            build_h5(
                weights=w,
                geography=geography,
                dataset_path=dataset_path,
                output_path=output_path,
                cd_subset=cd_subset,
                takeup_filter=takeup_filter,
            )

            if upload:
                print(f"Uploading {state_code}.h5 to GCP...")
                upload_local_area_file(str(output_path), "states", skip_hf=True)
                hf_queue.append((str(output_path), "states"))

            record_completed_state(state_code)
            print(f"Completed {state_code}")

            if upload and len(hf_queue) >= hf_batch_size:
                print(f"\nUploading batch of {len(hf_queue)} files to HuggingFace...")
                upload_local_area_batch_to_hf(hf_queue)
                hf_queue = []

        except Exception as e:
            print(f"ERROR building {state_code}: {e}")
            raise

    if upload and hf_queue:
        print(f"\nUploading final batch of {len(hf_queue)} files to HuggingFace...")
        upload_local_area_batch_to_hf(hf_queue)


def build_districts(
    weights_path: Path,
    dataset_path: Path,
    geography,
    output_dir: Path,
    completed_districts: set,
    hf_batch_size: int = 10,
    takeup_filter: List[str] = None,
    upload: bool = False,
):
    """Build district H5 files with checkpointing, optionally uploading."""
    w = np.load(weights_path)

    all_cds = sorted(set(geography.cd_geoid.astype(str)))

    districts_dir = output_dir / "districts"
    districts_dir.mkdir(parents=True, exist_ok=True)

    hf_queue = []

    for i, cd_geoid in enumerate(all_cds):
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
        print(f"\n[{i + 1}/{len(all_cds)}] Building {friendly_name}")

        try:
            build_h5(
                weights=w,
                geography=geography,
                dataset_path=dataset_path,
                output_path=output_path,
                cd_subset=[cd_geoid],
                takeup_filter=takeup_filter,
            )

            if upload:
                print(f"Uploading {friendly_name}.h5 to GCP...")
                upload_local_area_file(str(output_path), "districts", skip_hf=True)
                hf_queue.append((str(output_path), "districts"))

            record_completed_district(friendly_name)
            print(f"Completed {friendly_name}")

            if upload and len(hf_queue) >= hf_batch_size:
                print(f"\nUploading batch of {len(hf_queue)} files to HuggingFace...")
                upload_local_area_batch_to_hf(hf_queue)
                hf_queue = []

        except Exception as e:
            print(f"ERROR building {friendly_name}: {e}")
            raise

    if upload and hf_queue:
        print(f"\nUploading final batch of {len(hf_queue)} files to HuggingFace...")
        upload_local_area_batch_to_hf(hf_queue)


def build_cities(
    weights_path: Path,
    dataset_path: Path,
    geography,
    output_dir: Path,
    completed_cities: set,
    hf_batch_size: int = 10,
    takeup_filter: List[str] = None,
    upload: bool = False,
):
    """Build city H5 files with checkpointing, optionally uploading."""
    w = np.load(weights_path)

    cities_dir = output_dir / "cities"
    cities_dir.mkdir(parents=True, exist_ok=True)

    hf_queue = []

    # NYC
    if "NYC" in completed_cities:
        print("Skipping NYC (already completed)")
    else:
        output_path = cities_dir / "NYC.h5"

        try:
            build_h5(
                weights=w,
                geography=geography,
                dataset_path=dataset_path,
                output_path=output_path,
                county_fips_filter=NYC_COUNTY_FIPS,
                takeup_filter=takeup_filter,
            )

            if upload:
                print("Uploading NYC.h5 to GCP...")
                upload_local_area_file(str(output_path), "cities", skip_hf=True)
                hf_queue.append((str(output_path), "cities"))

            record_completed_city("NYC")
            print("Completed NYC")

        except Exception as e:
            print(f"ERROR building NYC: {e}")
            raise

    if upload and hf_queue:
        print(f"\nUploading batch of {len(hf_queue)} city files to HuggingFace...")
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
        "--state",
        type=str,
        help="Build only this state (e.g., SC, NY, CA)",
    )
    parser.add_argument(
        "--n-clones",
        type=int,
        required=True,
        help="Number of clones used in calibration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in calibration (default: 42)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to GCP and HuggingFace (default: build locally only)",
    )
    parser.add_argument(
        "--calibration-package-path",
        type=str,
        help="Optional calibration package path for exact geography reuse",
    )
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.weights_path and args.dataset_path:
        inputs = {
            "weights": Path(args.weights_path),
            "dataset": Path(args.dataset_path),
        }
        print("Using provided paths:")
        for key, path in inputs.items():
            print(f"  {key}: {path}")
    elif args.skip_download:
        inputs = {
            "weights": WORK_DIR / "calibration_weights.npy",
            "dataset": (WORK_DIR / "source_imputed_stratified_extended_cps.h5"),
        }
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

    print(f"Using dataset: {inputs['dataset']}")

    calibration_package_path = (
        Path(args.calibration_package_path)
        if args.calibration_package_path
        else None
    )

    print("Loading base simulation to get household count...")
    _sim = Microsimulation(dataset=str(inputs["dataset"]))
    n_hh = len(_sim.calculate("household_id", map_to="household").values)
    del _sim
    print(f"\nBase dataset has {n_hh:,} households")

    weights = np.load(inputs["weights"], mmap_mode="r")
    canonical_n_clones = infer_clone_count_from_weight_length(
        weights.shape[0],
        n_hh,
    )
    if canonical_n_clones != args.n_clones:
        print(
            f"WARNING: requested n_clones={args.n_clones} but "
            f"weights imply {canonical_n_clones}; using weights-derived value"
        )

    print("Computing input fingerprint...")
    if calibration_package_path is not None:
        from policyengine_us_data.calibration.local_h5.fingerprinting import (
            FingerprintService,
        )
        from policyengine_us_data.calibration.local_h5.package_geography import (
            require_calibration_package_path,
        )

        calibration_package_path = require_calibration_package_path(
            calibration_package_path
        )
        fingerprint_service = FingerprintService()
        fingerprint_record = fingerprint_service.create_publish_fingerprint(
            weights_path=inputs["weights"],
            dataset_path=inputs["dataset"],
            calibration_package_path=calibration_package_path,
            n_clones=canonical_n_clones,
            seed=args.seed,
        )
        fingerprint = fingerprint_record.digest
        validate_or_clear_checkpoints(fingerprint_record)
    else:
        fingerprint = compute_input_fingerprint(
            inputs["weights"],
            inputs["dataset"],
            canonical_n_clones,
            args.seed,
            calibration_package_path=calibration_package_path,
        )
        validate_or_clear_checkpoints(fingerprint)

    geo_cache = WORK_DIR / f"geography_{n_hh}x{canonical_n_clones}_s{args.seed}.npz"
    if calibration_package_path is not None and calibration_package_path.exists():
        from policyengine_us_data.calibration.local_h5.package_geography import (
            CalibrationPackageGeographyLoader,
        )

        loader = CalibrationPackageGeographyLoader()
        resolved = loader.resolve_for_weights(
            package_path=calibration_package_path,
            weights_length=weights.shape[0],
            n_records=n_hh,
            n_clones=canonical_n_clones,
            seed=args.seed,
        )
        geography = resolved.geography
        print(f"Loaded geography from {resolved.source}")
        for warning in resolved.warnings:
            print(f"WARNING: {warning}")
    elif geo_cache.exists():
        print(f"Loading cached geography from {geo_cache}")
        npz = np.load(geo_cache, allow_pickle=True)
        geography = GeographyAssignment(
            block_geoid=npz["block_geoid"],
            cd_geoid=npz["cd_geoid"],
            county_fips=npz["county_fips"],
            state_fips=npz["state_fips"],
            n_records=n_hh,
            n_clones=canonical_n_clones,
        )
    else:
        print(
            f"Generating geography: {n_hh} records x "
            f"{canonical_n_clones} clones, seed={args.seed}"
        )
        geography = assign_random_geography(
            n_records=n_hh,
            n_clones=canonical_n_clones,
            seed=args.seed,
        )
        np.savez_compressed(
            geo_cache,
            block_geoid=geography.block_geoid,
            cd_geoid=geography.cd_geoid,
            county_fips=geography.county_fips,
            state_fips=geography.state_fips,
        )
        print(f"Saved geography cache to {geo_cache}")
    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]
    print(f"Takeup filter: {takeup_filter}")

    # Determine what to build based on flags
    do_states = not args.districts_only and not args.cities_only
    do_districts = not args.states_only and not args.cities_only
    do_cities = not args.states_only and not args.districts_only

    # If a specific *-only flag is set, only build that type
    if args.states_only:
        do_states = True
        do_districts = False
        do_cities = False
    elif args.districts_only:
        do_states = False
        do_districts = True
        do_cities = False
    elif args.cities_only:
        do_states = False
        do_districts = False
        do_cities = True

    if do_states:
        print("\n" + "=" * 60)
        print("BUILDING STATE FILES")
        print("=" * 60)
        completed_states = load_completed_states()
        print(f"Already completed: {len(completed_states)} states")
        build_states(
            inputs["weights"],
            inputs["dataset"],
            geography,
            WORK_DIR,
            completed_states,
            takeup_filter=takeup_filter,
            upload=args.upload,
            state_filter=args.state,
        )

    if do_districts:
        print("\n" + "=" * 60)
        print("BUILDING DISTRICT FILES")
        print("=" * 60)
        completed_districts = load_completed_districts()
        print(f"Already completed: {len(completed_districts)} districts")
        build_districts(
            inputs["weights"],
            inputs["dataset"],
            geography,
            WORK_DIR,
            completed_districts,
            takeup_filter=takeup_filter,
            upload=args.upload,
        )

    if do_cities:
        print("\n" + "=" * 60)
        print("BUILDING CITY FILES")
        print("=" * 60)
        completed_cities = load_completed_cities()
        print(f"Already completed: {len(completed_cities)} cities")
        build_cities(
            inputs["weights"],
            inputs["dataset"],
            geography,
            WORK_DIR,
            completed_cities,
            takeup_filter=takeup_filter,
            upload=args.upload,
        )

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
