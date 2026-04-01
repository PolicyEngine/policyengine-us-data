from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .prototype_synthetic_2100_support import summarize_realized_clone_translation
except ImportError:  # pragma: no cover - script execution fallback
    from prototype_synthetic_2100_support import summarize_realized_clone_translation


def _default_metadata_path(h5_path: Path) -> Path:
    return h5_path.with_suffix(h5_path.suffix + ".metadata.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare support-augmentation clone targets to the realized output H5."
        )
    )
    parser.add_argument("h5_path", type=Path, help="Path to the realized year H5 file.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to the year metadata sidecar. Defaults to <h5>.metadata.json.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Optional path to support_augmentation_report.json. Defaults to the "
            "report_file named in metadata, or <output_dir>/support_augmentation_report.json."
        ),
    )
    parser.add_argument("--year", type=int, required=True, help="Output year to inspect.")
    parser.add_argument(
        "--age-bucket-size",
        type=int,
        default=5,
        help="Age bucket size for the translation comparison summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write the diagnostic summary.",
    )
    args = parser.parse_args()

    metadata_path = args.metadata or _default_metadata_path(args.h5_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    augmentation_metadata = metadata.get("support_augmentation")
    if not augmentation_metadata:
        raise ValueError(f"No support_augmentation metadata found in {metadata_path}")

    report_path = args.report
    if report_path is None:
        report_file = augmentation_metadata.get("report_file")
        if report_file:
            report_path = metadata_path.parent / report_file
        else:
            report_path = metadata_path.parent / "support_augmentation_report.json"

    augmentation_report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = summarize_realized_clone_translation(
        str(args.h5_path),
        period=args.year,
        augmentation_report=augmentation_report,
        age_bucket_size=args.age_bucket_size,
    )
    payload = {
        "h5_path": str(args.h5_path),
        "metadata_path": str(metadata_path),
        "report_path": str(report_path),
        "year": int(args.year),
        "age_bucket_size": int(args.age_bucket_size),
        "summary": summary,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
