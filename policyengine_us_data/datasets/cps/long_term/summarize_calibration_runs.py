from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from calibration_artifacts import normalize_metadata
from profile_support_concentration import profile_support


SUPPORT_FIELDS = (
    "positive_household_count",
    "positive_household_pct",
    "effective_sample_size",
    "top_10_weight_share_pct",
    "top_100_weight_share_pct",
    "weighted_nonworking_share_pct",
    "weighted_nonworking_share_85_plus_pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize and compare calibration quality across projected dataset directories."
        )
    )
    parser.add_argument("left", type=Path, help="First projected dataset directory.")
    parser.add_argument(
        "right", type=Path, nargs="?", help="Optional second directory to compare."
    )
    parser.add_argument(
        "--years",
        help="Optional comma-separated list of years to include. Defaults to all years found.",
    )
    parser.add_argument(
        "--profile-support",
        action="store_true",
        help="Compute support-concentration metrics from the H5 files when metadata is missing them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def parse_years(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def metadata_for(directory: Path, year: int) -> dict[str, Any] | None:
    metadata_path = directory / f"{year}.h5.metadata.json"
    if not metadata_path.exists():
        return None
    return normalize_metadata(json.loads(metadata_path.read_text(encoding="utf-8")))


def support_metrics(
    directory: Path, year: int, metadata: dict[str, Any] | None, *, profile: bool
) -> dict[str, Any]:
    audit = (metadata or {}).get("calibration_audit", {})
    metrics = {field: audit.get(field) for field in SUPPORT_FIELDS}
    if not profile or all(value is not None for value in metrics.values()):
        return metrics

    dataset_path = directory / f"{year}.h5"
    if not dataset_path.exists():
        return metrics
    profiled = profile_support(dataset_path, year, top_n=10)
    return {
        "positive_household_count": profiled["positive_household_count"],
        "positive_household_pct": profiled["positive_household_pct"],
        "effective_sample_size": profiled["effective_sample_size"],
        "top_10_weight_share_pct": profiled["top_10_weight_share_pct"],
        "top_100_weight_share_pct": profiled["top_100_weight_share_pct"],
        "weighted_nonworking_share_pct": profiled["weighted_nonworking_share_pct"],
        "weighted_nonworking_share_85_plus_pct": profiled[
            "weighted_nonworking_share_85_plus_pct"
        ],
    }


def summarize_directory(
    directory: Path, years: list[int] | None, *, profile: bool
) -> dict[int, dict[str, Any]]:
    if years is None:
        years = sorted(
            int(path.name.split(".")[0])
            for path in directory.glob("*.h5.metadata.json")
        )

    rows: dict[int, dict[str, Any]] = {}
    for year in years:
        metadata = metadata_for(directory, year)
        if metadata is None:
            continue
        audit = metadata["calibration_audit"]
        row: dict[str, Any] = {
            "quality": audit.get("calibration_quality"),
            "method": audit.get("approximation_method") or audit.get("method_used"),
            "age_bucket_size": audit.get("age_bucket_size"),
            "max_constraint_pct_error": audit.get("max_constraint_pct_error"),
            "age_max_pct_error": audit.get("age_max_pct_error"),
            "negative_weight_pct": audit.get("negative_weight_pct"),
        }
        row.update(support_metrics(directory, year, metadata, profile=profile))
        rows[year] = row
    return rows


def build_rows(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    years = sorted(set(left) | set(right or {}))
    rows: list[dict[str, Any]] = []
    for year in years:
        row: dict[str, Any] = {"year": year}
        for prefix, source in [("left", left), ("right", right or {})]:
            values = source.get(year, {})
            for key, value in values.items():
                row[f"{prefix}_{key}"] = value
        if right is not None and year in left and year in right:
            for key in (
                "max_constraint_pct_error",
                "age_max_pct_error",
                "effective_sample_size",
                "top_10_weight_share_pct",
                "top_100_weight_share_pct",
                "weighted_nonworking_share_pct",
                "weighted_nonworking_share_85_plus_pct",
            ):
                left_value = left[year].get(key)
                right_value = right[year].get(key)
                if left_value is not None and right_value is not None:
                    row[f"delta_{key}"] = right_value - left_value
        rows.append(row)
    return rows


def write_rows(rows: list[dict[str, Any]], output: Path | None) -> None:
    if not rows:
        raise SystemExit("No rows to write.")

    fieldnames = sorted({key for row in rows for key in row})
    if output is None:
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    years = parse_years(args.years)
    left = summarize_directory(args.left, years, profile=args.profile_support)
    right = (
        summarize_directory(args.right, years, profile=args.profile_support)
        if args.right is not None
        else None
    )
    rows = build_rows(left, right)
    write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
