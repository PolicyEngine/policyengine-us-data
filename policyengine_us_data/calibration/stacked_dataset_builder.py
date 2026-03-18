"""
CLI for creating CD-stacked datasets from calibration artifacts.

Thin wrapper around build_output_dataset/build_states/build_districts/build_cities
in publish_local_area.py. Loads a GeographyAssignment from geography.npz
and delegates all H5 building logic.
"""

import os
import numpy as np
from pathlib import Path

from policyengine_us_data.calibration.clone_and_assign import (
    load_geography,
)

if __name__ == "__main__":
    import argparse

    from policyengine_us import Microsimulation
    from policyengine_us_data.calibration.publish_local_area import (
        build_output_dataset,
        build_states,
        build_districts,
        build_cities,
    )
    from policyengine_us_data.utils.takeup import SIMPLE_TAKEUP_VARS

    parser = argparse.ArgumentParser(
        description="Create CD-stacked datasets from calibration artifacts"
    )
    parser.add_argument(
        "--weights-path",
        required=True,
        help="Path to w_cd.npy file",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to stratified dataset .h5 file",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to policy_data.db",
    )
    parser.add_argument(
        "--geography-path",
        required=True,
        help="Path to geography.npz from calibration",
    )
    parser.add_argument(
        "--output-dir",
        default="./temp",
        help="Output directory for files",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "national",
            "states",
            "cds",
            "single-cd",
            "single-state",
            "nyc",
        ],
        default="national",
        help="Output mode",
    )
    parser.add_argument(
        "--cd",
        type=str,
        help="Single CD GEOID (--mode single-cd)",
    )
    parser.add_argument(
        "--state",
        type=str,
        help="State code e.g. RI, CA (--mode single-state)",
    )

    args = parser.parse_args()
    weights_path = Path(args.weights_path)
    dataset_path = Path(args.dataset_path)
    db_path = Path(args.db_path).resolve()
    output_dir = Path(args.output_dir)
    mode = args.mode

    os.makedirs(output_dir, exist_ok=True)

    # === Load and validate ===
    w = np.load(str(weights_path))
    db_uri = f"sqlite:///{db_path}"

    # === Load geography (required) ===
    if not args.geography_path or not Path(args.geography_path).exists():
        raise ValueError(
            f"--geography-path is required and must exist. "
            f"Got: {args.geography_path}. "
            f"Re-run calibration to generate geography.npz."
        )
    geography = load_geography(args.geography_path)
    print(
        f"Loaded geography from {args.geography_path}: "
        f"{geography.n_clones} clones x "
        f"{geography.n_records} records"
    )

    print(f"Geography: {geography.n_clones} clones x {geography.n_records} records")

    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]

    # === Dispatch ===
    if mode == "national":
        output_base = output_dir / "US"
        print(f"\nCreating national dataset: {output_base}")
        build_output_dataset(
            weights=w,
            geography=geography,
            dataset_path=dataset_path,
            output_base=output_base,
            takeup_filter=takeup_filter,
        )

    elif mode == "states":
        build_states(
            weights_path=weights_path,
            dataset_path=dataset_path,
            geography=geography,
            output_dir=output_dir,
            completed_states=set(),
            takeup_filter=takeup_filter,
        )

    elif mode == "single-state":
        if not args.state:
            raise ValueError("--state required with --mode single-state")
        build_states(
            weights_path=weights_path,
            dataset_path=dataset_path,
            geography=geography,
            output_dir=output_dir,
            completed_states=set(),
            takeup_filter=takeup_filter,
            state_filter=args.state.upper(),
        )

    elif mode == "cds":
        build_districts(
            weights_path=weights_path,
            dataset_path=dataset_path,
            geography=geography,
            output_dir=output_dir,
            completed_districts=set(),
            takeup_filter=takeup_filter,
        )

    elif mode == "single-cd":
        if not args.cd:
            raise ValueError("--cd required with --mode single-cd")
        calibrated_cds = sorted(set(cd_geoid))
        if args.cd not in calibrated_cds:
            raise ValueError(f"CD {args.cd} not in calibrated CDs")
        output_base = output_dir / args.cd
        print(f"\nCreating single CD dataset: {output_base}")
        build_output_dataset(
            weights=w,
            geography=geography,
            dataset_path=dataset_path,
            output_base=output_base,
            cd_subset=[args.cd],
            takeup_filter=takeup_filter,
        )

    elif mode == "nyc":
        build_cities(
            weights_path=weights_path,
            dataset_path=dataset_path,
            geography=geography,
            output_dir=output_dir,
            completed_cities=set(),
            takeup_filter=takeup_filter,
        )

    print("\nDone!")
