"""
Create a sparse congressional district-stacked dataset with non-zero weight
households.

DEPRECATED: This module is superseded by build_h5() in publish_local_area.py.
create_sparse_cd_stacked_dataset is now a thin wrapper that delegates to
build_h5, which uses a single simulation + fancy indexing instead of looping
over CDs.
"""

import os
import numpy as np
from pathlib import Path

from policyengine_us_data.calibration.calibration_utils import (
    get_all_cds_from_database,
    STATE_CODES,
)

NYC_COUNTIES = {
    "QUEENS_COUNTY_NY",
    "BRONX_COUNTY_NY",
    "RICHMOND_COUNTY_NY",
    "NEW_YORK_COUNTY_NY",
    "KINGS_COUNTY_NY",
}

NYC_CDS = [
    "3603",
    "3605",
    "3606",
    "3607",
    "3608",
    "3609",
    "3610",
    "3611",
    "3612",
    "3613",
    "3614",
    "3615",
    "3616",
]


def create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    cd_subset=None,
    output_path=None,
    dataset_path=None,
    county_filter=None,
    seed: int = 42,
    rerandomize_takeup: bool = False,
    calibration_blocks: np.ndarray = None,
    takeup_filter=None,
):
    """Thin wrapper around build_h5() for backward compatibility.

    DEPRECATED: Use build_h5() from publish_local_area.py directly.

    Args:
        w: Calibrated weight vector.
        cds_to_calibrate: Ordered list of CD GEOIDs.
        cd_subset: Optional list of CDs to include.
        output_path: Where to save the .h5 file.
        dataset_path: Path to base dataset .h5 file.
        county_filter: Optional county filter set.
        seed: Unused (kept for API compat).
        rerandomize_takeup: Re-draw takeup draws.
        calibration_blocks: Stacked block GEOID array.
        takeup_filter: List of takeup vars to re-randomize.

    Returns:
        output_path: Path to the saved .h5 file.
    """
    from policyengine_us_data.calibration.publish_local_area import (
        build_h5,
    )

    if output_path is None:
        raise ValueError("No output .h5 path given")

    return build_h5(
        weights=np.array(w),
        blocks=calibration_blocks,
        dataset_path=Path(dataset_path),
        output_path=Path(output_path),
        cds_to_calibrate=cds_to_calibrate,
        cd_subset=cd_subset,
        county_filter=county_filter,
        rerandomize_takeup=rerandomize_takeup,
        takeup_filter=takeup_filter,
    )


if __name__ == "__main__":
    import argparse

    from policyengine_us import Microsimulation

    parser = argparse.ArgumentParser(
        description="Create sparse CD-stacked datasets"
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
    parser.add_argument(
        "--rerandomize-takeup",
        action="store_true",
        help="Re-randomize takeup draws per CD",
    )
    parser.add_argument(
        "--calibration-blocks",
        default=None,
        help="Path to stacked_blocks.npy",
    )

    args = parser.parse_args()
    dataset_path_str = args.dataset_path
    weights_path_str = args.weights_path
    db_path = Path(args.db_path).resolve()
    output_dir = args.output_dir
    mode = args.mode

    os.makedirs(output_dir, exist_ok=True)

    w = np.load(weights_path_str)
    db_uri = f"sqlite:///{db_path}"

    cds_to_calibrate = get_all_cds_from_database(db_uri)
    print(f"Found {len(cds_to_calibrate)} congressional districts")

    assert_sim = Microsimulation(dataset=dataset_path_str)
    n_hh = assert_sim.calculate("household_id", map_to="household").shape[0]
    expected_length = len(cds_to_calibrate) * n_hh

    if len(w) != expected_length:
        raise ValueError(
            f"Weight vector length ({len(w):,}) doesn't match "
            f"expected ({expected_length:,})"
        )

    rerand = args.rerandomize_takeup
    cal_blocks = None
    if args.calibration_blocks:
        cal_blocks = np.load(args.calibration_blocks)
        print(f"Loaded calibration blocks: {len(cal_blocks):,}")

    if mode == "national":
        output_path = f"{output_dir}/national.h5"
        print(f"\nCreating national dataset: {output_path}")
        create_sparse_cd_stacked_dataset(
            w,
            cds_to_calibrate,
            dataset_path=dataset_path_str,
            output_path=output_path,
            rerandomize_takeup=rerand,
            calibration_blocks=cal_blocks,
        )

    elif mode == "states":
        for state_fips, state_code in STATE_CODES.items():
            cd_subset = [
                cd for cd in cds_to_calibrate if int(cd) // 100 == state_fips
            ]
            if not cd_subset:
                continue
            output_path = f"{output_dir}/{state_code}.h5"
            print(f"\nCreating {state_code}: {output_path}")
            create_sparse_cd_stacked_dataset(
                w,
                cds_to_calibrate,
                cd_subset=cd_subset,
                dataset_path=dataset_path_str,
                output_path=output_path,
                rerandomize_takeup=rerand,
                calibration_blocks=cal_blocks,
            )

    elif mode == "cds":
        for i, cd_geoid in enumerate(cds_to_calibrate):
            cd_int = int(cd_geoid)
            state_fips = cd_int // 100
            district_num = cd_int % 100
            if district_num in (0, 98):
                district_num = 1
            state_code = STATE_CODES.get(state_fips, str(state_fips))
            friendly_name = f"{state_code}-{district_num:02d}"

            output_path = f"{output_dir}/{friendly_name}.h5"
            print(
                f"\n[{i+1}/{len(cds_to_calibrate)}] "
                f"Creating {friendly_name}.h5"
            )
            create_sparse_cd_stacked_dataset(
                w,
                cds_to_calibrate,
                cd_subset=[cd_geoid],
                dataset_path=dataset_path_str,
                output_path=output_path,
                rerandomize_takeup=rerand,
                calibration_blocks=cal_blocks,
            )

    elif mode == "single-cd":
        if not args.cd:
            raise ValueError("--cd required with --mode single-cd")
        if args.cd not in cds_to_calibrate:
            raise ValueError(f"CD {args.cd} not in calibrated CDs list")
        output_path = f"{output_dir}/{args.cd}.h5"
        print(f"\nCreating single CD dataset: {output_path}")
        create_sparse_cd_stacked_dataset(
            w,
            cds_to_calibrate,
            cd_subset=[args.cd],
            dataset_path=dataset_path_str,
            output_path=output_path,
            rerandomize_takeup=rerand,
            calibration_blocks=cal_blocks,
        )

    elif mode == "single-state":
        if not args.state:
            raise ValueError("--state required with --mode single-state")
        state_code_upper = args.state.upper()
        state_fips = None
        for fips, code in STATE_CODES.items():
            if code == state_code_upper:
                state_fips = fips
                break
        if state_fips is None:
            raise ValueError(f"Unknown state code: {args.state}")

        cd_subset = [
            cd for cd in cds_to_calibrate if int(cd) // 100 == state_fips
        ]
        if not cd_subset:
            raise ValueError(f"No CDs found for state {state_code_upper}")

        output_path = f"{output_dir}/{state_code_upper}.h5"
        print(
            f"\nCreating {state_code_upper} with "
            f"{len(cd_subset)} CDs: {output_path}"
        )
        create_sparse_cd_stacked_dataset(
            w,
            cds_to_calibrate,
            cd_subset=cd_subset,
            dataset_path=dataset_path_str,
            output_path=output_path,
            rerandomize_takeup=rerand,
            calibration_blocks=cal_blocks,
        )

    elif mode == "nyc":
        cd_subset = [cd for cd in cds_to_calibrate if cd in NYC_CDS]
        if not cd_subset:
            raise ValueError("No NYC CDs found")

        output_path = f"{output_dir}/NYC.h5"
        print(f"\nCreating NYC with {len(cd_subset)} CDs: " f"{output_path}")
        create_sparse_cd_stacked_dataset(
            w,
            cds_to_calibrate,
            cd_subset=cd_subset,
            dataset_path=dataset_path_str,
            output_path=output_path,
            county_filter=NYC_COUNTIES,
            rerandomize_takeup=rerand,
            calibration_blocks=cal_blocks,
        )

    print("\nDone!")
