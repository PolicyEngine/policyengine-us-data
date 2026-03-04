#!/usr/bin/env python
"""
Worker script for building local area H5 files.

Called by Modal workers via subprocess to avoid import conflicts.
"""

import argparse
import json
import sys
import traceback
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-items", required=True, help="JSON work items")
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--calibration-blocks",
        type=str,
        default=None,
        help="Path to stacked_blocks.npy from calibration",
    )
    args = parser.parse_args()

    work_items = json.loads(args.work_items)
    weights_path = Path(args.weights_path)
    dataset_path = Path(args.dataset_path)
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    calibration_blocks = None
    if args.calibration_blocks:
        calibration_blocks = np.load(args.calibration_blocks)

    rerandomize_takeup = True
    from policyengine_us_data.utils.takeup import (
        TAKEUP_AFFECTED_TARGETS,
    )

    takeup_filter = [
        info["takeup_var"] for info in TAKEUP_AFFECTED_TARGETS.values()
    ]

    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    from policyengine_us_data.calibration.publish_local_area import (
        build_state_h5,
        build_district_h5,
        build_city_h5,
        build_national_h5,
    )
    from policyengine_us_data.calibration.calibration_utils import (
        get_all_cds_from_database,
        STATE_CODES,
    )

    db_uri = f"sqlite:///{db_path}"
    cds_to_calibrate = get_all_cds_from_database(db_uri)
    weights = np.load(weights_path)

    results = {
        "completed": [],
        "failed": [],
        "errors": [],
    }

    for item in work_items:
        item_type = item["type"]
        item_id = item["id"]

        try:
            if item_type == "state":
                path = build_state_h5(
                    state_code=item_id,
                    weights=weights,
                    cds_to_calibrate=cds_to_calibrate,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    rerandomize_takeup=rerandomize_takeup,
                    calibration_blocks=calibration_blocks,
                    takeup_filter=takeup_filter,
                )
            elif item_type == "district":
                state_code, dist_num = item_id.split("-")
                state_fips = None
                for fips, code in STATE_CODES.items():
                    if code == state_code:
                        state_fips = fips
                        break
                if state_fips is None:
                    raise ValueError(f"Unknown state in district: {item_id}")

                candidate = f"{state_fips}{int(dist_num):02d}"
                if candidate in cds_to_calibrate:
                    geoid = candidate
                else:
                    state_cds = [
                        cd
                        for cd in cds_to_calibrate
                        if int(cd) // 100 == state_fips
                    ]
                    if len(state_cds) == 1:
                        geoid = state_cds[0]
                    else:
                        raise ValueError(
                            f"CD {candidate} not found and "
                            f"state {state_code} has "
                            f"{len(state_cds)} CDs"
                        )

                path = build_district_h5(
                    cd_geoid=geoid,
                    weights=weights,
                    cds_to_calibrate=cds_to_calibrate,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    rerandomize_takeup=rerandomize_takeup,
                    calibration_blocks=calibration_blocks,
                    takeup_filter=takeup_filter,
                )
            elif item_type == "city":
                path = build_city_h5(
                    city_name=item_id,
                    weights=weights,
                    cds_to_calibrate=cds_to_calibrate,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    rerandomize_takeup=rerandomize_takeup,
                    calibration_blocks=calibration_blocks,
                    takeup_filter=takeup_filter,
                )
            elif item_type == "national":
                path = build_national_h5(
                    weights=weights,
                    blocks=calibration_blocks,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    cds_to_calibrate=cds_to_calibrate,
                )
            else:
                raise ValueError(f"Unknown item type: {item_type}")

            if path:
                results["completed"].append(f"{item_type}:{item_id}")
                print(f"Completed {item_type}:{item_id}", file=sys.stderr)

        except Exception as e:
            results["failed"].append(f"{item_type}:{item_id}")
            results["errors"].append(
                {
                    "item": f"{item_type}:{item_id}",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAILED {item_type}:{item_id}: {e}", file=sys.stderr)

    sys.stdout = original_stdout
    print(json.dumps(results))


if __name__ == "__main__":
    main()
