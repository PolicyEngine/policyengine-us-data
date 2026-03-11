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
        "--geography-path",
        required=True,
        help="Path to geography.npz from calibration",
    )
    args = parser.parse_args()

    work_items = json.loads(args.work_items)
    weights_path = Path(args.weights_path)
    dataset_path = Path(args.dataset_path)
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    from policyengine_us_data.utils.takeup import (
        SIMPLE_TAKEUP_VARS,
    )

    takeup_filter = [spec["variable"] for spec in SIMPLE_TAKEUP_VARS]

    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    from policyengine_us_data.calibration.publish_local_area import (
        build_h5,
        NYC_COUNTIES,
        NYC_CDS,
        AT_LARGE_DISTRICTS,
    )
    from policyengine_us_data.calibration.calibration_utils import (
        STATE_CODES,
    )
    from policyengine_us_data.calibration.clone_and_assign import (
        load_geography,
    )

    weights = np.load(weights_path)

    # Load geography from .npz (required)
    if not args.geography_path or not Path(args.geography_path).exists():
        raise RuntimeError(
            f"--geography-path is required and must exist. "
            f"Got: {args.geography_path}. "
            f"Re-run calibration to generate geography.npz."
        )
    geography = load_geography(args.geography_path)
    cds_to_calibrate = sorted(set(geography.cd_geoid.astype(str)))
    geo_labels = cds_to_calibrate
    print(
        f"Loaded geography from {args.geography_path}: "
        f"{geography.n_clones} clones x "
        f"{geography.n_records} records",
        file=sys.stderr,
    )

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
                state_fips = None
                for fips, code in STATE_CODES.items():
                    if code == item_id:
                        state_fips = fips
                        break
                if state_fips is None:
                    raise ValueError(f"Unknown state code: {item_id}")
                cd_subset = [
                    cd
                    for cd in cds_to_calibrate
                    if int(cd) // 100 == state_fips
                ]
                if not cd_subset:
                    print(
                        f"No CDs for {item_id}, skipping",
                        file=sys.stderr,
                    )
                    continue
                states_dir = output_dir / "states"
                states_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=states_dir / f"{item_id}.h5",
                    cd_subset=cd_subset,
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
                if candidate in geo_labels:
                    geoid = candidate
                else:
                    state_cds = [
                        cd for cd in geo_labels if int(cd) // 100 == state_fips
                    ]
                    if len(state_cds) == 1:
                        geoid = state_cds[0]
                    else:
                        raise ValueError(
                            f"CD {candidate} not found and "
                            f"state {state_code} has "
                            f"{len(state_cds)} CDs"
                        )

                cd_int = int(geoid)
                district_num = cd_int % 100
                if district_num in AT_LARGE_DISTRICTS:
                    district_num = 1
                friendly_name = f"{state_code}-{district_num:02d}"

                districts_dir = output_dir / "districts"
                districts_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=districts_dir / f"{friendly_name}.h5",
                    cd_subset=[geoid],
                    takeup_filter=takeup_filter,
                )

            elif item_type == "city":
                cd_subset = [cd for cd in cds_to_calibrate if cd in NYC_CDS]
                if not cd_subset:
                    print(
                        "No NYC CDs found, skipping",
                        file=sys.stderr,
                    )
                    continue
                cities_dir = output_dir / "cities"
                cities_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=cities_dir / "NYC.h5",
                    cd_subset=cd_subset,
                    county_filter=NYC_COUNTIES,
                    takeup_filter=takeup_filter,
                )

            elif item_type == "national":
                national_dir = output_dir / "national"
                national_dir.mkdir(parents=True, exist_ok=True)
                path = build_h5(
                    weights=weights,
                    geography=geography,
                    dataset_path=dataset_path,
                    output_path=national_dir / "US.h5",
                )
            else:
                raise ValueError(f"Unknown item type: {item_type}")

            if path:
                results["completed"].append(f"{item_type}:{item_id}")
                print(
                    f"Completed {item_type}:{item_id}",
                    file=sys.stderr,
                )

        except Exception as e:
            results["failed"].append(f"{item_type}:{item_id}")
            results["errors"].append(
                {
                    "item": f"{item_type}:{item_id}",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"FAILED {item_type}:{item_id}: {e}",
                file=sys.stderr,
            )

    sys.stdout = original_stdout
    print(json.dumps(results))


if __name__ == "__main__":
    main()
