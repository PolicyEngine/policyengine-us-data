"""
Build NC stacked dataset from calibration weights and print
weighted sums of key variables.

Usage:
    python scripts/verify_nc_calibration.py
    python scripts/verify_nc_calibration.py --weights-path my_weights.npy
    python scripts/verify_nc_calibration.py --skip-build
"""

import argparse
import os
import subprocess
import sys

from policyengine_us import Microsimulation

DATASET_PATH = "policyengine_us_data/storage/stratified_extended_cps_2024.h5"
DB_PATH = "policyengine_us_data/storage/calibration/policy_data.db"
OUTPUT_DIR = "./temp"


def build_nc_dataset(weights_path: str) -> str:
    output_path = os.path.join(OUTPUT_DIR, "NC.h5")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable,
        "policyengine_us_data/datasets/cps/local_area_calibration"
        "/stacked_dataset_builder.py",
        "--weights-path",
        weights_path,
        "--dataset-path",
        DATASET_PATH,
        "--db-path",
        DB_PATH,
        "--output-dir",
        OUTPUT_DIR,
        "--mode",
        "single-state",
        "--state",
        "NC",
        "--rerandomize-takeup",
    ]
    print("Building NC stacked dataset...")
    subprocess.run(cmd, check=True)
    print(f"NC dataset saved to: {output_path}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-path",
        default="calibration_weights.npy",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use existing temp/NC.h5",
    )
    args = parser.parse_args()

    h5_path = os.path.join(OUTPUT_DIR, "NC.h5")
    if not args.skip_build:
        h5_path = build_nc_dataset(args.weights_path)

    sim = Microsimulation(dataset=h5_path)

    variables = [
        "snap",
        "aca_ptc",
        "eitc",
        "ssi",
        "social_security",
        "medicaid",
        "tanf",
        "refundable_ctc",
        "rent",
        "real_estate_taxes",
        "self_employment_income",
        "unemployment_compensation",
    ]

    hh_weight = sim.calculate(
        "household_weight", 2024, map_to="household"
    ).values
    hh_count = hh_weight.sum()
    print(f"{'household_count':<30s} {hh_count:>18,.0f}")
    print()
    print(f"{'Variable':<30s} {'Weighted Sum ($M)':>18s}")
    print("-" * 50)
    for var in variables:
        try:
            total = sim.calculate(var, period=2024).sum()
            print(f"{var:<30s} {total / 1e6:>18.2f}")
        except Exception as exc:
            print(f"{var:<30s}  ERROR: {exc}")


if __name__ == "__main__":
    main()
