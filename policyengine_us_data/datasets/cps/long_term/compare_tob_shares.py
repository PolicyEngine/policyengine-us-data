from __future__ import annotations

import argparse
import json
from pathlib import Path


def _resolve_metadata_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            for candidate in sorted(path.glob("*.metadata.json")):
                paths.append(candidate)
            continue
        if path.is_file():
            paths.append(path)
            continue
        raise FileNotFoundError(f"Metadata path not found: {path}")
    if not paths:
        raise ValueError("No metadata files found.")
    return paths


def _load_record(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    audit = metadata["calibration_audit"]
    constraints = audit["constraints"]
    tob_section = audit.get("benchmarks") or audit.get("constraints")

    ss_actual = float(constraints["ss_total"]["achieved"])
    ss_target = float(constraints["ss_total"]["target"])
    oasdi_actual = float(tob_section["oasdi_tob"]["achieved"])
    oasdi_target = float(tob_section["oasdi_tob"]["target"])
    hi_actual = float(tob_section["hi_tob"]["achieved"])
    hi_target = float(tob_section["hi_tob"]["target"])

    combined_actual = oasdi_actual + hi_actual
    combined_target = oasdi_target + hi_target

    return {
        "year": int(metadata["year"]),
        "source_path": str(path),
        "oasdi_actual_share_pct": 100 * oasdi_actual / ss_actual,
        "oasdi_target_share_pct": 100 * oasdi_target / ss_target,
        "oasdi_gap_pct_pt": 100 * oasdi_actual / ss_actual
        - 100 * oasdi_target / ss_target,
        "combined_actual_share_pct": 100 * combined_actual / ss_actual,
        "combined_target_share_pct": 100 * combined_target / ss_target,
        "combined_gap_pct_pt": 100 * combined_actual / ss_actual
        - 100 * combined_target / ss_target,
    }


def _format_markdown(records: list[dict]) -> str:
    header = (
        "| Year | OASDI actual | OASDI target | OASDI gap | Combined actual | "
        "Combined target | Combined gap |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [
        (
            f"| {record['year']} | "
            f"{record['oasdi_actual_share_pct']:.2f}% | "
            f"{record['oasdi_target_share_pct']:.2f}% | "
            f"{record['oasdi_gap_pct_pt']:+.2f} pp | "
            f"{record['combined_actual_share_pct']:.2f}% | "
            f"{record['combined_target_share_pct']:.2f}% | "
            f"{record['combined_gap_pct_pt']:+.2f} pp |"
        )
        for record in records
    ]
    return "\n".join([header, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize OASDI-only and combined TOB shares from long-run "
            "metadata sidecars."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=("Metadata files or directories containing *.metadata.json sidecars."),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = _resolve_metadata_paths(args.paths)
    records = sorted((_load_record(path) for path in paths), key=lambda x: x["year"])
    if args.format == "json":
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        print(_format_markdown(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
