from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from calibration_artifacts import update_dataset_manifest


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "run_household_projection.py"


def parse_years(spec: str) -> list[int]:
    years: set[int] = set()
    for part in spec.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid year range: {chunk}")
            years.update(range(start, end + 1))
        else:
            years.add(int(chunk))
    if not years:
        raise ValueError("No years provided")
    return sorted(years)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run long-run household projections in parallel, one year per "
            "subprocess, then merge the resulting H5 artifacts into one output "
            "directory and rebuild the calibration manifest."
        )
    )
    parser.add_argument(
        "--years",
        required=True,
        help="Comma-separated years and ranges, e.g. 2026-2035,2045,2070.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Maximum number of year subprocesses to run concurrently.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Final output directory for merged YYYY.h5 artifacts.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep per-year temporary output directories after a successful merge.",
    )
    args, forwarded_args = parser.parse_known_args()
    return args, forwarded_args


def validate_forwarded_args(forwarded_args: list[str]) -> None:
    blocked = {"--output-dir", "--save-h5"}
    for arg in forwarded_args:
        if arg in blocked:
            raise ValueError(
                f"{arg} is controlled by run_household_projection_parallel.py; "
                "pass it to the wrapper instead."
            )


def year_output_dir(root: Path, year: int) -> Path:
    return root / ".parallel_tmp" / str(year)


def year_log_path(root: Path, year: int) -> Path:
    return root / ".parallel_logs" / f"{year}.log"


def run_year(
    *,
    year: int,
    output_root: Path,
    forwarded_args: list[str],
) -> tuple[int, Path]:
    output_dir = year_output_dir(output_root, year)
    log_path = year_log_path(output_root, year)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(RUNNER_PATH),
        str(year),
        str(year),
        "--output-dir",
        str(output_dir),
        "--save-h5",
        *forwarded_args,
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Year {year} failed with exit code {completed.returncode}. "
            f"See {log_path}."
        )

    expected_h5 = output_dir / f"{year}.h5"
    expected_metadata = output_dir / f"{year}.h5.metadata.json"
    if not expected_h5.exists() or not expected_metadata.exists():
        raise FileNotFoundError(
            f"Year {year} finished without expected artifacts in {output_dir}."
        )

    return year, output_dir


def copy_support_reports(temp_output_dir: Path, final_output_dir: Path) -> None:
    for report_path in sorted(temp_output_dir.glob("support_augmentation_report*.json")):
        target_path = final_output_dir / report_path.name
        if not target_path.exists():
            shutil.copy2(report_path, target_path)
            continue
        if report_path.read_bytes() != target_path.read_bytes():
            raise ValueError(
                f"Conflicting support augmentation report contents for {report_path.name}"
            )


def _json_clone(value):
    return json.loads(json.dumps(value))


def manifest_contract(manifest: dict) -> dict:
    return {
        "base_dataset_path": manifest["base_dataset_path"],
        "profile": _json_clone(manifest["profile"]),
        "target_source": _json_clone(manifest.get("target_source")),
        "tax_assumption": _json_clone(manifest.get("tax_assumption")),
        "support_augmentation": _json_clone(manifest.get("support_augmentation")),
    }


def merge_outputs(
    *,
    years: list[int],
    output_root: Path,
    keep_temp: bool,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_seed = None
    manifest_path = None

    for year in years:
        temp_output_dir = year_output_dir(output_root, year)
        temp_manifest_path = temp_output_dir / "calibration_manifest.json"
        if not temp_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing temp manifest for year {year}: {temp_manifest_path}"
            )

        temp_manifest = json.loads(temp_manifest_path.read_text(encoding="utf-8"))
        if manifest_seed is None:
            manifest_seed = manifest_contract(temp_manifest)
        else:
            for key, value in manifest_seed.items():
                if _json_clone(temp_manifest.get(key)) != value:
                    raise ValueError(
                        f"Temp manifest mismatch for {key} in year {year}: "
                        f"{temp_manifest.get(key)} != {value}"
                    )

        h5_name = f"{year}.h5"
        metadata_name = f"{year}.h5.metadata.json"
        shutil.copy2(temp_output_dir / h5_name, output_root / h5_name)
        shutil.copy2(temp_output_dir / metadata_name, output_root / metadata_name)
        copy_support_reports(temp_output_dir, output_root)

        metadata = json.loads(
            (temp_output_dir / metadata_name).read_text(encoding="utf-8")
        )
        manifest_path = update_dataset_manifest(
            output_root,
            year=year,
            h5_path=output_root / h5_name,
            metadata_path=output_root / metadata_name,
            base_dataset_path=manifest_seed["base_dataset_path"],
            profile=manifest_seed["profile"],
            calibration_audit=metadata["calibration_audit"],
            target_source=manifest_seed["target_source"],
            tax_assumption=manifest_seed["tax_assumption"],
            support_augmentation=manifest_seed["support_augmentation"],
        )

    if not keep_temp:
        shutil.rmtree(output_root / ".parallel_tmp", ignore_errors=True)

    return manifest_path


def main() -> int:
    args, forwarded_args = parse_args()
    validate_forwarded_args(forwarded_args)

    output_root = Path(args.output_dir).expanduser().resolve()
    years = parse_years(args.years)

    print(
        f"Running {len(years)} year jobs with concurrency {args.jobs} into {output_root}"
    )

    completed_years: list[int] = []
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as executor:
        future_map = {
            executor.submit(
                run_year,
                year=year,
                output_root=output_root,
                forwarded_args=forwarded_args,
            ): year
            for year in years
        }
        for future in as_completed(future_map):
            year = future_map[future]
            try:
                future.result()
            except Exception as error:
                print(f"Year {year} failed: {error}", file=sys.stderr)
                return 1
            completed_years.append(year)
            print(f"Completed year {year}")

    manifest_path = merge_outputs(
        years=years,
        output_root=output_root,
        keep_temp=args.keep_temp,
    )
    print(f"Merged {len(completed_years)} yearly artifacts into {output_root}")
    print(f"Rebuilt manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
