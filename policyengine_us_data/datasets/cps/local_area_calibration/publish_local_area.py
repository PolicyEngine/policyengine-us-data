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

from policyengine_us import Microsimulation
from policyengine_us_data.utils.huggingface import download_calibration_inputs
from policyengine_us_data.utils.data_upload import upload_local_area_file
from policyengine_us_data.datasets.cps.local_area_calibration.stacked_dataset_builder import (
    create_sparse_cd_stacked_dataset,
)
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_all_cds_from_database,
    STATE_CODES,
)

CHECKPOINT_FILE = Path("completed_states.txt")
CHECKPOINT_FILE_DISTRICTS = Path("completed_districts.txt")
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


def build_and_upload_states(
    weights_path: Path,
    dataset_path: Path,
    db_path: Path,
    output_dir: Path,
    completed_states: set,
):
    """Build and upload state H5 files with checkpointing."""
    db_uri = f"sqlite:///{db_path}"
    cds_to_calibrate = get_all_cds_from_database(db_uri)
    w = np.load(weights_path)

    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

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
            )

            print(f"Uploading {state_code}.h5...")
            upload_local_area_file(str(output_path), "states")

            record_completed_state(state_code)
            print(f"Completed {state_code}")

        except Exception as e:
            print(f"ERROR building {state_code}: {e}")
            raise


def build_and_upload_districts(
    weights_path: Path,
    dataset_path: Path,
    db_path: Path,
    output_dir: Path,
    completed_districts: set,
):
    """Build and upload district H5 files with checkpointing."""
    db_uri = f"sqlite:///{db_path}"
    cds_to_calibrate = get_all_cds_from_database(db_uri)
    w = np.load(weights_path)

    districts_dir = output_dir / "districts"
    districts_dir.mkdir(parents=True, exist_ok=True)

    for i, cd_geoid in enumerate(cds_to_calibrate):
        cd_int = int(cd_geoid)
        state_fips = cd_int // 100
        district_num = cd_int % 100
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
            )

            print(f"Uploading {friendly_name}.h5...")
            upload_local_area_file(str(output_path), "districts")

            record_completed_district(friendly_name)
            print(f"Completed {friendly_name}")

        except Exception as e:
            print(f"ERROR building {friendly_name}: {e}")
            raise


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
            "weights": WORK_DIR / "w_district_calibration.npy",
            "dataset": WORK_DIR / "stratified_extended_cps.h5",
            "database": WORK_DIR / "policy_data.db",
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

    sim = Microsimulation(dataset=str(inputs["dataset"]))
    n_hh = sim.calculate("household_id", map_to="household").shape[0]
    print(f"\nBase dataset has {n_hh:,} households")

    if not args.districts_only:
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
        )

    if not args.states_only:
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
        )

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
