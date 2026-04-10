from __future__ import annotations

import argparse
import json

from assess_publishable_horizon import assess_years
from support_augmentation import build_augmented_dataset


DEFAULT_BASE_DATASET_PATH = (
    "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
)


def parse_years(raw: str) -> list[int]:
    years = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not years:
        raise ValueError("At least one year must be provided.")
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assess publishability for an augmented late-tail dataset under the "
            "standard long-run contract."
        )
    )
    parser.add_argument(
        "--years",
        default="2075",
        help="Comma-separated years to assess.",
    )
    parser.add_argument(
        "--profile",
        default="ss-payroll-tob",
        help="Named calibration profile to assess.",
    )
    parser.add_argument(
        "--target-source",
        default="trustees_2025_current_law",
        help="Named long-run target source package.",
    )
    parser.add_argument(
        "--support-augmentation",
        required=True,
        help="Support augmentation profile name.",
    )
    parser.add_argument(
        "--base-dataset",
        default=DEFAULT_BASE_DATASET_PATH,
        help="Base microsimulation dataset path.",
    )
    args = parser.parse_args()

    augmented_dataset, augmentation_report = build_augmented_dataset(
        base_dataset=args.base_dataset,
        base_year=2024,
        profile=args.support_augmentation,
    )

    rows = assess_years(
        years=parse_years(args.years),
        profile_name=args.profile,
        target_source=args.target_source,
        base_dataset_path=augmented_dataset,
    )

    payload = {
        "years": parse_years(args.years),
        "profile": args.profile,
        "target_source": args.target_source,
        "support_augmentation_profile": args.support_augmentation,
        "augmentation_report": augmentation_report,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
