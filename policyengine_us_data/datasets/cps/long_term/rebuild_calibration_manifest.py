from __future__ import annotations

import argparse

from calibration_artifacts import rebuild_dataset_manifest_with_target_source
from ssa_data import describe_long_term_target_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild calibration_manifest.json from existing metadata sidecars.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory containing YYYY.h5 and YYYY.h5.metadata.json files.",
    )
    parser.add_argument(
        "--target-source",
        help="Optional target source name to stamp into each sidecar while rebuilding the manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_source = (
        describe_long_term_target_source(args.target_source)
        if args.target_source
        else None
    )
    manifest_path = rebuild_dataset_manifest_with_target_source(
        args.output_dir,
        target_source=target_source,
    )
    print(f"Rebuilt {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
