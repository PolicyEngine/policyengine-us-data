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
    cds_to_calibrate: List[str],
    dataset_path: Path,
    output_dir: Path,
    rerandomize_takeup: bool = False,
    calibration_blocks: np.ndarray = None,
    takeup_filter: List[str] = None,
) -> Path:
    """Build national US.h5 by collapsing CD weights to household level.

    Unlike state/district H5s which re-run simulations per CD with
    geographic reassignment, the national H5 keeps original geography
    and simply sums weights across CDs per household, filtering to
    nonzero-weight households.
    """
    import h5py
    from policyengine_core.enums import Enum

    national_dir = output_dir / "national"
    national_dir.mkdir(parents=True, exist_ok=True)
    output_path = national_dir / "US.h5"

    print(f"\n{'='*60}")
    print(f"Building national US.h5 ({len(cds_to_calibrate)} CDs)")
    print(f"{'='*60}")

    sim = Microsimulation(dataset=str(dataset_path))
    time_period = int(sim.default_calculation_period)

    household_ids = sim.calculate("household_id", map_to="household").values
    n_hh = len(household_ids)
    n_cds = len(cds_to_calibrate)

    W = weights.reshape(n_cds, n_hh)
    hh_weights = W.sum(axis=0)

    active_mask = hh_weights > 0
    n_active = active_mask.sum()
    print(f"Households: {n_hh:,} total, {n_active:,} active")
    print(f"Total weight: {hh_weights[active_mask].sum():,.0f}")

    sim.set_input("household_weight", time_period, hh_weights)

    person_hh_ids = sim.calculate("household_id", map_to="person").values
    active_hh_set = set(household_ids[active_mask])
    person_mask = np.isin(person_hh_ids, list(active_hh_set))

    print(
        f"Persons: {len(person_mask):,} total, "
        f"{person_mask.sum():,} active"
    )

    data = {}
    variables_saved = 0
    for variable in sim.tax_benefit_system.variables:
        holder = sim.get_holder(variable)
        periods = holder.get_known_periods()
        if not periods:
            continue

        var_data = {}
        for period in periods:
            values = holder.get_array(period)

            var_def = sim.tax_benefit_system.variables.get(variable)
            entity_key = var_def.entity.key

            if entity_key == "person":
                values = values[person_mask]
            elif entity_key == "household":
                values = values[active_mask]
            else:
                entity_id_var = f"{entity_key}_id"
                entity_ids = sim.calculate(
                    entity_id_var, map_to=entity_key
                ).values
                person_entity_ids = sim.calculate(
                    entity_id_var, map_to="person"
                ).values
                active_entity_ids = set(person_entity_ids[person_mask])
                entity_mask = np.isin(entity_ids, list(active_entity_ids))
                values = values[entity_mask]

            if var_def.value_type in (Enum, str) and variable != "county_fips":
                if hasattr(values, "decode_to_str"):
                    values = values.decode_to_str().astype("S")
                else:
                    values = values.astype("S")
            elif variable == "county_fips":
                values = values.astype("int32")
            else:
                values = np.array(values)

            var_data[period] = values
            variables_saved += 1

        if var_data:
            data[variable] = var_data

    print(f"Variables saved: {variables_saved}")

    with h5py.File(str(output_path), "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    print(f"National H5 saved to {output_path}")

    with h5py.File(str(output_path), "r") as f:
        if "household_id" in f and str(time_period) in f["household_id"]:
            n = len(f["household_id"][str(time_period)][:])
            print(f"Verified: {n:,} households in output")

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
