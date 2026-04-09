from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_export import export_bundle
from benchmark_manifest import load_manifest
from benchmark_metrics import (
    compute_common_metrics,
    load_targets_csv,
    write_method_summary,
)


ROOT = Path(__file__).resolve().parent
RUNNERS_DIR = ROOT / "runners"


def _run_subprocess(cmd, cwd=None):
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    elapsed = time.time() - started
    return proc, elapsed


def cmd_export(args):
    manifest = load_manifest(args.manifest)
    output_dir, info = export_bundle(manifest=manifest, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(output_dir), **info}, indent=2, sort_keys=True))
    return 0


def _run_l0(run_dir: Path):
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"

    from scipy.io import mmread
    from policyengine_us_data.calibration.unified_calibration import fit_l0_weights

    with open(inputs / "benchmark_manifest.json") as f:
        manifest = json.load(f)

    options = manifest.get("method_options", {}).get("l0", {})
    X_sparse = mmread(str(inputs / "X_targets_by_units.mtx")).tocsr()
    targets_df = pd.read_csv(inputs / "target_metadata.csv")
    initial_weights = np.load(inputs / "initial_weights.npy")

    weights = fit_l0_weights(
        X_sparse=X_sparse,
        targets=targets_df["value"].to_numpy(dtype=np.float64),
        lambda_l0=float(options.get("lambda_l0", 1e-8)),
        epochs=int(options.get("epochs", 1000)),
        device=str(options.get("device", "cpu")),
        beta=float(options.get("beta", 0.65)),
        lambda_l2=float(options.get("lambda_l2", 1e-12)),
        learning_rate=float(options.get("learning_rate", 0.15)),
        target_names=targets_df["target_name"].tolist(),
        initial_weights=initial_weights,
        targets_df=targets_df,
    )

    weights_path = outputs / "fitted_weights.npy"
    np.save(weights_path, weights.astype(np.float64))
    return weights_path


def _run_greg(run_dir: Path):
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    temp_csv = outputs / "_greg_weights.csv"

    with open(inputs / "benchmark_manifest.json") as f:
        manifest = json.load(f)
    options = manifest.get("method_options", {}).get("greg", {})

    cmd = [
        "Rscript",
        str(RUNNERS_DIR / "greg_runner.R"),
        str(inputs / "X_targets_by_units.mtx"),
        str(inputs / "target_metadata.csv"),
        str(inputs / "initial_weights.npy"),
        str(temp_csv),
        str(int(options.get("maxit", 50))),
        str(float(options.get("epsilon", 1e-7))),
    ]
    proc, elapsed = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"GREG runner failed with exit code {proc.returncode}")

    weights = pd.read_csv(temp_csv)["fitted_weight"].to_numpy(dtype=np.float64)
    weights_path = outputs / "fitted_weights.npy"
    np.save(weights_path, weights)
    temp_csv.unlink(missing_ok=True)
    return weights_path, elapsed


def _run_ipf(run_dir: Path):
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    temp_csv = outputs / "_ipf_weights.csv"

    with open(inputs / "benchmark_manifest.json") as f:
        manifest = json.load(f)
    options = manifest.get("method_options", {}).get("ipf", {})

    target_metadata_path = inputs / "ipf_target_metadata.csv"
    if not target_metadata_path.exists():
        raise FileNotFoundError(
            "IPF run requires inputs/ipf_target_metadata.csv. "
            "Provide external_inputs.ipf_target_metadata_csv in the manifest."
        )

    cmd = [
        "Rscript",
        str(RUNNERS_DIR / "ipf_runner.R"),
        str(inputs / "unit_metadata.csv"),
        str(target_metadata_path),
        str(inputs / "initial_weights.npy"),
        str(temp_csv),
        str(int(options.get("max_iter", 200))),
        str(float(options.get("bound", 4.0))),
        str(float(options.get("epsP", 1e-6))),
        str(float(options.get("epsH", 1e-2))),
        str(options.get("household_id_col", "household_id")),
        str(options.get("weight_col", "base_weight")),
    ]
    proc, elapsed = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"IPF runner failed with exit code {proc.returncode}")

    raw_weights = pd.read_csv(temp_csv)
    if "unit_index" not in raw_weights.columns:
        raise RuntimeError("IPF runner output must include a unit_index column")
    if raw_weights["unit_index"].isna().any():
        raise RuntimeError("IPF runner output contains missing unit_index values")

    raw_weights["unit_index"] = raw_weights["unit_index"].astype(np.int64)
    n_units = len(np.load(inputs / "initial_weights.npy"))
    if (raw_weights["unit_index"] < 0).any() or (
        raw_weights["unit_index"] >= n_units
    ).any():
        raise RuntimeError("IPF runner output contains out-of-range unit_index values")

    per_unit_spread = raw_weights.groupby("unit_index", sort=True)["fitted_weight"].agg(
        lambda series: float(series.max() - series.min())
    )
    inconsistent_units = per_unit_spread[per_unit_spread > 1e-9]
    if not inconsistent_units.empty:
        raise RuntimeError(
            "IPF runner produced inconsistent fitted weights within the same unit_index"
        )

    weights_by_unit = (
        raw_weights.groupby("unit_index", sort=True)["fitted_weight"]
        .first()
        .reindex(np.arange(n_units, dtype=np.int64))
    )
    if weights_by_unit.isna().any():
        raise RuntimeError(
            "Aggregated IPF weights do not cover the full benchmark unit range"
        )
    weights = weights_by_unit.to_numpy(dtype=np.float64)
    weights_path = outputs / "fitted_weights.npy"
    np.save(weights_path, weights)
    temp_csv.unlink(missing_ok=True)
    return weights_path, elapsed


def cmd_run(args):
    run_dir = Path(args.run_dir)
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    targets_df = load_targets_csv(inputs / "target_metadata.csv")

    started = time.time()
    if args.method == "l0":
        weights_path = _run_l0(run_dir)
    elif args.method == "greg":
        weights_path, _ = _run_greg(run_dir)
    elif args.method == "ipf":
        weights_path, _ = _run_ipf(run_dir)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    elapsed = time.time() - started

    weights = np.load(weights_path)
    summary = compute_common_metrics(
        weights=weights,
        targets_df=targets_df,
        matrix_path=inputs / "X_targets_by_units.mtx",
    )
    summary["method"] = args.method
    summary["run_dir"] = str(run_dir.resolve())
    summary["runtime_seconds"] = elapsed
    write_method_summary(summary, outputs / f"{args.method}_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark scaffold CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export a benchmark bundle")
    export_parser.add_argument(
        "--manifest", required=True, help="Path to benchmark manifest JSON"
    )
    export_parser.add_argument(
        "--output-dir", required=True, help="Output bundle directory"
    )
    export_parser.set_defaults(func=cmd_export)

    run_parser = subparsers.add_parser(
        "run", help="Run one method on an exported bundle"
    )
    run_parser.add_argument("--method", required=True, choices=["l0", "greg", "ipf"])
    run_parser.add_argument(
        "--run-dir", required=True, help="Exported benchmark bundle directory"
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
