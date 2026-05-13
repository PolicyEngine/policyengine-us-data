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
from typing import List, Optional

from policyengine_us import Microsimulation
from policyengine_us_data.build_outputs.builder import LocalAreaDatasetBuilder
from policyengine_us_data.build_outputs.fingerprinting import (
    FingerprintingService,
    PublishingInputBundle,
)
from policyengine_us_data.build_outputs.geography_loader import (
    CalibrationGeographyLoader,
)
from policyengine_us_data.build_outputs.requests import AreaBuildRequest, AreaFilter
from policyengine_us_data.build_outputs.source_dataset import SourceDatasetSnapshot
from policyengine_us_data.build_outputs.us_augmentations import (
    USTakeupPostProcessor,
    default_us_postprocessors,
)
from policyengine_us_data.build_outputs.weights import CloneWeightMatrix
from policyengine_us_data.build_outputs.writer import H5Writer
from policyengine_us_data.utils.huggingface import download_calibration_inputs
from policyengine_us_data.utils.data_upload import (
    upload_local_area_file,
    upload_local_area_batch_to_hf,
)
from policyengine_us_data.calibration.calibration_utils import (
    STATE_CODES,
)
from policyengine_us_data.utils.takeup import (
    SIMPLE_TAKEUP_VARS,
)
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode

CHECKPOINT_FILE = Path("completed_states.txt")
CHECKPOINT_FILE_DISTRICTS = Path("completed_districts.txt")
CHECKPOINT_FILE_CITIES = Path("completed_cities.txt")
WORK_DIR = Path("local_area_build")

NYC_COUNTY_FIPS = {"36005", "36047", "36061", "36081", "36085"}


META_FILE = WORK_DIR / "checkpoint_meta.json"


@pipeline_node(
    PipelineNode(
        id="local_h5_input_fingerprint",
        label="Compute Local H5 Input Fingerprint",
        node_type="library",
        description="Compute a scope fingerprint for local H5 checkpoint and resume decisions.",
        source_file="policyengine_us_data/calibration/publish_local_area.py",
        status="legacy",
        stability="moving",
        pathways=["local_h5"],
        api_refs=[
            "policyengine_us_data.build_outputs.fingerprinting.FingerprintingService"
        ],
        validation_commands=[
            "uv run pytest tests/unit/build_outputs/test_fingerprinting.py"
        ],
    )
)
def compute_input_fingerprint(
    weights_path: Path,
    dataset_path: Path,
    n_clones: Optional[int] = None,
    seed: int = 42,
    geography_path: Optional[Path] = None,
    blocks_path: Optional[Path] = None,
    target_db_path: Optional[Path] = None,
    run_config_path: Optional[Path] = None,
    calibration_package_path: Optional[Path] = None,
    scope: str = "regional",
) -> str:
    service = FingerprintingService()
    inputs = PublishingInputBundle(
        weights_path=Path(weights_path),
        source_dataset_path=Path(dataset_path),
        target_db_path=Path(target_db_path) if target_db_path is not None else None,
        exact_geography_path=(
            Path(geography_path) if geography_path is not None else None
        ),
        calibration_package_path=(
            Path(calibration_package_path)
            if calibration_package_path is not None
            else None
        ),
        run_config_path=Path(run_config_path) if run_config_path is not None else None,
        run_id="",
        version="",
        n_clones=n_clones,
        seed=seed,
        legacy_blocks_path=Path(blocks_path) if blocks_path is not None else None,
    )
    traceability = service.build_traceability(inputs=inputs, scope=scope)
    return service.compute_scope_fingerprint(traceability)


@pipeline_node(
    PipelineNode(
        id="load_calibration_geography",
        label="Load Calibration Geography",
        node_type="library",
        description="Resolve exact geography from saved bundles, package metadata, or legacy block artifacts.",
        source_file="policyengine_us_data/calibration/publish_local_area.py",
        status="legacy",
        stability="moving",
        pathways=["local_h5"],
        api_refs=[
            "policyengine_us_data.build_outputs.geography_loader.CalibrationGeographyLoader"
        ],
        artifacts_in=[
            "calibration_weights.npy",
            "geography_assignment.npz",
            "stacked_blocks.npy",
        ],
        validation_commands=[
            "uv run pytest tests/unit/build_outputs/test_geography_loader.py"
        ],
    )
)
def load_calibration_geography(
    weights_path: Path,
    n_records: int,
    n_clones: Optional[int] = None,
    geography_path: Optional[Path] = None,
    blocks_path: Optional[Path] = None,
    calibration_package_path: Optional[Path] = None,
):
    loader = CalibrationGeographyLoader()
    resolved = loader.resolve_source(
        weights_path=Path(weights_path),
        geography_path=Path(geography_path) if geography_path is not None else None,
        blocks_path=Path(blocks_path) if blocks_path is not None else None,
        calibration_package_path=(
            Path(calibration_package_path)
            if calibration_package_path is not None
            else None
        ),
    )
    geography = loader.load(
        weights_path=Path(weights_path),
        n_records=n_records,
        n_clones=n_clones,
        geography_path=Path(geography_path) if geography_path is not None else None,
        blocks_path=Path(blocks_path) if blocks_path is not None else None,
        calibration_package_path=(
            Path(calibration_package_path)
            if calibration_package_path is not None
            else None
        ),
    )
    if resolved is not None:
        if resolved.kind == "saved_geography":
            print(f"Loaded calibration geography from {resolved.path}")
        elif resolved.kind == "calibration_package":
            print(f"Loaded calibration geography from package {resolved.path}")
        else:
            print(
                "Reconstructing geography from legacy stacked blocks at "
                f"{resolved.path}"
            )
    return geography


def validate_or_clear_checkpoints(fingerprint: str):
    if META_FILE.exists():
        stored = json.loads(META_FILE.read_text())
        if stored.get("fingerprint") == fingerprint:
            print(f"Inputs unchanged ({fingerprint}), resuming...")
            return
        print(
            f"Inputs changed "
            f"({stored.get('fingerprint')} -> {fingerprint}), "
            f"clearing..."
        )
    else:
        print(f"No checkpoint metadata, starting fresh ({fingerprint})")
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
    META_FILE.write_text(json.dumps({"fingerprint": fingerprint}))


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


def _build_legacy_area_filters(
    *,
    cd_subset: List[str] | None,
    county_fips_filter: set | None,
) -> tuple[AreaFilter, ...]:
    filters = []
    if cd_subset is not None:
        filters.append(
            AreaFilter(
                geography_field="cd_geoid",
                op="in",
                value=tuple(str(item) for item in cd_subset),
            )
        )
    if county_fips_filter is not None:
        filters.append(
            AreaFilter(
                geography_field="county_fips",
                op="in",
                value=tuple(sorted(str(item) for item in county_fips_filter)),
            )
        )
    return tuple(filters)


def _build_legacy_area_request(
    *,
    output_path: Path,
    filters: tuple[AreaFilter, ...],
) -> AreaBuildRequest:
    return AreaBuildRequest(
        area_type="custom",
        area_id=Path(output_path).stem,
        display_name=Path(output_path).name,
        output_relative_path=Path(output_path).name,
        filters=filters,
    )


@pipeline_node(
    PipelineNode(
        id="build_h5",
        label="Build Local Area H5",
        node_type="library",
        description="Expand calibrated clone weights into local-area H5 datasets with geography and takeup updates.",
        details="This is the main bundled H5 construction routine and remains a critical transitional waypoint.",
        source_file="policyengine_us_data/calibration/publish_local_area.py",
        status="transitional",
        stability="moving",
        pathways=["local_h5"],
        artifacts_in=[
            "calibration_weights.npy",
            "source_imputed_stratified_extended_cps*.h5",
        ],
        artifacts_out=["states/*.h5", "districts/*.h5", "cities/*.h5", "US.h5"],
        validation_commands=[
            "uv run pytest tests/unit/calibration/test_publish_local_area.py",
            "uv run pytest tests/integration/test_tiny_h5_pipeline.py",
        ],
    )
)
def build_h5(
    weights: np.ndarray,
    geography,
    dataset_path: Path,
    output_path: Path,
    cd_subset: List[str] = None,
    county_fips_filter: set = None,
    takeup_filter: List[str] = None,
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # === Load base simulation ===
    sim = Microsimulation(dataset=str(dataset_path))
    source = SourceDatasetSnapshot.from_simulation(Path(dataset_path), sim)
    n_hh = source.n_households
    weight_matrix = CloneWeightMatrix.from_vector(weights, n_records=n_hh)
    n_clones_total = weight_matrix.n_clones
    filters = _build_legacy_area_filters(
        cd_subset=cd_subset,
        county_fips_filter=county_fips_filter,
    )
    request = _build_legacy_area_request(output_path=output_path, filters=filters)

    label = (
        f"CD subset {cd_subset}"
        if cd_subset is not None
        else f"{n_clones_total} clone rows"
    )
    print(f"\n{'=' * 60}")
    print(f"Building {output_path.name} ({label}, {n_hh} households)")
    print(f"{'=' * 60}")

    result = LocalAreaDatasetBuilder(
        postprocessors=default_us_postprocessors(),
    ).build(
        source=source,
        simulation=sim,
        weights=weight_matrix,
        geography=geography,
        request=request,
        takeup_filter=tuple(takeup_filter) if takeup_filter is not None else None,
    )

    print(f"Active clones: {result.selection.n_selected_clones:,}")
    print(f"Total weight: {result.summary['total_weight']:,.0f}")
    print(f"Cloned persons: {len(result.reindexed.person_ids):,}")
    for entity_key, indices in result.reindexed.subentity_source_indices.items():
        print(f"Cloned {entity_key}s: {len(indices):,}")
    print(f"Variables cloned: {result.variables_saved}")

    unique_blocks = np.unique(result.selection.block_geoids)
    print("Derived geography from blocks.")
    print(
        f"  {result.selection.n_selected_clones:,} blocks -> "
        f"{len(unique_blocks):,} unique"
    )
    takeup = result.postprocessor_result(USTakeupPostProcessor)
    if takeup is not None and takeup.takeup_variables:
        print("Applied calibration takeup draws.")
        print(f"Takeup variables: {', '.join(takeup.takeup_variables)}")

    write_result = H5Writer().write(
        payload=result.payload,
        output_path=output_path,
    )

    print(f"\nH5 saved to {output_path}")
    if write_result.households is not None:
        print(f"Verified: {write_result.households:,} households in output")
    if write_result.persons is not None:
        print(f"Verified: {write_result.persons:,} persons in output")
    if write_result.household_weight_sum is not None:
        print(
            f"Total population (HH weights): {write_result.household_weight_sum:,.0f}"
        )
    if write_result.person_weight_sum is not None:
        print(
            f"Total population (person weights): {write_result.person_weight_sum:,.0f}"
        )

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


@pipeline_node(
    PipelineNode(
        id="build_states",
        label="Build State H5 Files",
        node_type="library",
        description="Build state-level H5 files from calibrated weights and exact geography.",
        source_file="policyengine_us_data/calibration/publish_local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        artifacts_out=["states/*.h5"],
    )
)
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
    if upload:
        raise RuntimeError(
            "Direct upload from publish_local_area.py is disabled. "
            "Use modal_app/local_area.py or promote_local_h5s.py so release "
            "manifests and tags are finalized atomically."
        )
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


@pipeline_node(
    PipelineNode(
        id="build_districts",
        label="Build District H5 Files",
        node_type="library",
        description="Build congressional-district H5 files from calibrated weights and exact geography.",
        source_file="policyengine_us_data/calibration/publish_local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        artifacts_out=["districts/*.h5"],
    )
)
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
    if upload:
        raise RuntimeError(
            "Direct upload from publish_local_area.py is disabled. "
            "Use modal_app/local_area.py or promote_local_h5s.py so release "
            "manifests and tags are finalized atomically."
        )
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


@pipeline_node(
    PipelineNode(
        id="build_cities",
        label="Build City H5 Files",
        node_type="library",
        description="Build supported city H5 files with county probability filtering.",
        source_file="policyengine_us_data/calibration/publish_local_area.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        artifacts_out=["cities/*.h5"],
    )
)
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
    if upload:
        raise RuntimeError(
            "Direct upload from publish_local_area.py is disabled. "
            "Use modal_app/local_area.py or promote_local_h5s.py so release "
            "manifests and tags are finalized atomically."
        )
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
        "--national-only",
        action="store_true",
        help="Only build the national US.h5 file",
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
        default=None,
        help="Clone count override for validating saved geography artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Legacy fallback seed used only if no saved geography is available",
    )
    parser.add_argument(
        "--geography-path",
        type=str,
        default=None,
        help="Override path to saved geography_assignment.npz",
    )
    parser.add_argument(
        "--blocks-path",
        type=str,
        default=None,
        help="Override path to legacy stacked_blocks.npy",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to GCP and HuggingFace (default: build locally only)",
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

    print("Computing input fingerprint...")
    fingerprint = compute_input_fingerprint(
        inputs["weights"],
        inputs["dataset"],
        args.n_clones,
        args.seed,
        geography_path=Path(args.geography_path) if args.geography_path else None,
        blocks_path=Path(args.blocks_path) if args.blocks_path else None,
    )
    validate_or_clear_checkpoints(fingerprint)

    print("Loading base simulation to get household count...")
    _sim = Microsimulation(dataset=str(inputs["dataset"]))
    n_hh = len(_sim.calculate("household_id", map_to="household").values)
    del _sim
    print(f"\nBase dataset has {n_hh:,} households")

    geography = load_calibration_geography(
        weights_path=inputs["weights"],
        n_records=n_hh,
        n_clones=args.n_clones,
        geography_path=Path(args.geography_path) if args.geography_path else None,
        blocks_path=Path(args.blocks_path) if args.blocks_path else None,
    )
    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]
    print(f"Takeup filter: {takeup_filter}")

    # Determine what to build based on flags
    do_national = args.national_only
    do_states = (
        not args.districts_only and not args.cities_only and not args.national_only
    )
    do_districts = (
        not args.states_only and not args.cities_only and not args.national_only
    )
    do_cities = (
        not args.states_only and not args.districts_only and not args.national_only
    )

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

    if do_national:
        print("\n" + "=" * 60)
        print("BUILDING NATIONAL US.h5")
        print("=" * 60)
        weights = np.load(inputs["weights"])
        national_dir = WORK_DIR / "national"
        national_dir.mkdir(parents=True, exist_ok=True)
        path = build_h5(
            weights=weights,
            geography=geography,
            dataset_path=inputs["dataset"],
            output_path=national_dir / "US.h5",
            takeup_filter=takeup_filter,
        )
        print(f"Built {path}")

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
