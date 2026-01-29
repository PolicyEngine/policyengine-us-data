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
    args = parser.parse_args()

    work_items = json.loads(args.work_items)
    weights_path = Path(args.weights_path)
    dataset_path = Path(args.dataset_path)
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    from policyengine_us_data.datasets.cps.local_area_calibration.publish_local_area import (
        build_state_h5,
        build_district_h5,
        build_city_h5,
    )
    from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
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
                )
            elif item_type == "district":
                state_code, dist_num = item_id.split("-")
                geoid = None
                for fips, code in STATE_CODES.items():
                    if code == state_code:
                        geoid = f"{fips}{int(dist_num):02d}"
                        break
                if geoid is None:
                    raise ValueError(f"Unknown state in district: {item_id}")

                path = build_district_h5(
                    cd_geoid=geoid,
                    weights=weights,
                    cds_to_calibrate=cds_to_calibrate,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                )
            elif item_type == "city":
                path = build_city_h5(
                    city_name=item_id,
                    weights=weights,
                    cds_to_calibrate=cds_to_calibrate,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                )
            else:
                raise ValueError(f"Unknown item type: {item_type}")

            if path:
                results["completed"].append(f"{item_type}:{item_id}")
                print(f"Completed {item_type}:{item_id}", file=sys.stderr)

        except Exception as e:
            results["failed"].append(f"{item_type}:{item_id}")
            results["errors"].append({
                "item": f"{item_type}:{item_id}",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"FAILED {item_type}:{item_id}: {e}", file=sys.stderr)

    print(json.dumps(results))


if __name__ == "__main__":
    main()
