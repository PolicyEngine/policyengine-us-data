"""
CLI for creating CD-stacked datasets from calibration artifacts.

Thin wrapper around build_h5/build_states/build_districts/build_cities
in publish_local_area.py. Constructs a GeographyAssignment from local
calibration outputs and delegates all H5 building logic.
"""

import os
import numpy as np
from pathlib import Path

from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
)

if __name__ == "__main__":
    import argparse

    from policyengine_us import Microsimulation
    from policyengine_us_data.calibration.publish_local_area import (
        build_h5,
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
        "--calibration-blocks",
        required=True,
        help="Path to stacked_blocks.npy",
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

    sim = Microsimulation(dataset=str(dataset_path))
    n_hh = sim.calculate("household_id", map_to="household").shape[0]
    del sim

    if len(w) % n_hh != 0:
        raise ValueError(
            f"Weight vector length ({len(w):,}) is not divisible "
            f"by n_hh ({n_hh:,})"
        )
    n_clones = len(w) // n_hh
    print(f"Detected {n_clones} clones from weights ({len(w):,} / {n_hh:,})")

    # === Construct geography from calibration artifacts ===
    cal_blocks = np.load(args.calibration_blocks, allow_pickle=True)
    print(f"Loaded calibration blocks: {len(cal_blocks):,}")

    if len(cal_blocks) != len(w):
        raise ValueError(
            f"Blocks length ({len(cal_blocks):,}) doesn't match "
            f"weights length ({len(w):,})"
        )

    # Derive CD GEOIDs from blocks (first 4 digits of block GEOID)
    cd_geoid = np.array([str(b)[:4] for b in cal_blocks], dtype=str)
    geography = GeographyAssignment(
        block_geoid=cal_blocks,
        cd_geoid=cd_geoid,
        county_fips=np.full(len(w), "", dtype="U5"),
        state_fips=np.array(
            [int(cd) // 100 for cd in cd_geoid], dtype=np.int32
        ),
        n_records=n_hh,
        n_clones=n_clones,
    )
    print(
        f"Geography: {geography.n_clones} clones x "
        f"{geography.n_records} records"
    )

    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]

    # === Dispatch ===
    if mode == "national":
        output_path = output_dir / "US.h5"
        print(f"\nCreating national dataset: {output_path}")
        build_h5(
            weights=w,
            geography=geography,
            dataset_path=dataset_path,
            output_path=output_path,
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
        output_path = output_dir / f"{args.cd}.h5"
        print(f"\nCreating single CD dataset: {output_path}")
        build_h5(
            weights=w,
            geography=geography,
            dataset_path=dataset_path,
            output_path=output_path,
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
