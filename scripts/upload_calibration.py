"""Upload calibration artifacts to HuggingFace.

Usage:
    python scripts/upload_calibration.py
    python scripts/upload_calibration.py --weights my_weights.npy
    python scripts/upload_calibration.py --weights w.npy --blocks b.npy --log-dir ./logs
"""

import argparse
import sys

from policyengine_us_data.utils.huggingface import (
    upload_calibration_artifacts,
)


def main():
    parser = argparse.ArgumentParser(
        description="Upload calibration artifacts to HuggingFace"
    )
    parser.add_argument(
        "--weights",
        default="calibration_weights.npy",
        help="Path to weights file (default: calibration_weights.npy)",
    )
    parser.add_argument(
        "--blocks",
        default="stacked_blocks.npy",
        help="Path to blocks file (default: stacked_blocks.npy)",
    )
    parser.add_argument(
        "--log-dir",
        default=".",
        help="Directory containing log files (default: .)",
    )
    args = parser.parse_args()

    import os

    if not os.path.exists(args.weights):
        print(f"ERROR: Weights file not found: {args.weights}")
        sys.exit(1)

    blocks = args.blocks if os.path.exists(args.blocks) else None

    uploaded = upload_calibration_artifacts(
        weights_path=args.weights,
        blocks_path=blocks,
        log_dir=args.log_dir,
    )
    if uploaded:
        print(f"Successfully uploaded {len(uploaded)} artifact(s)")
    else:
        print("Nothing was uploaded")
        sys.exit(1)


if __name__ == "__main__":
    main()
