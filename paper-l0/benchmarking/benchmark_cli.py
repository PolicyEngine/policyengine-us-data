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


def _select_training_inputs(run_dir: Path, train_on: str) -> tuple[Path, Path, str]:
    """Resolve the training-input pair for L0 / GREG.

    `shared_requested` returns the shared bundle's full target set. The
    `ipf_retained_authored` mode loads the IPF scoring subset so L0 and GREG
    can be fit on the same target set IPF was given — required for matched
    benchmark comparison. Fails fast if the exporter did not produce that
    subset.
    """
    inputs = run_dir / "inputs"
    if train_on == "ipf_retained_authored":
        targets_path = inputs / "ipf_scoring_target_metadata.csv"
        matrix_path = inputs / "ipf_scoring_X_targets_by_units.mtx"
        if not targets_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(
                "Requested --train-on ipf_retained_authored, but the exporter "
                "did not write inputs/ipf_scoring_*. Re-run export with the "
                "IPF method enabled and a calibration package that includes "
                "target_id."
            )
        return targets_path, matrix_path, "ipf_retained_authored"
    return (
        inputs / "target_metadata.csv",
        inputs / "X_targets_by_units.mtx",
        "shared_requested",
    )


def _run_l0(run_dir: Path, train_on: str = "shared_requested"):
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"

    from scipy.io import mmread
    from policyengine_us_data.calibration.unified_calibration import fit_l0_weights

    with open(inputs / "benchmark_manifest.json") as f:
        manifest = json.load(f)

    options = manifest.get("method_options", {}).get("l0", {})
    targets_path, matrix_path, _ = _select_training_inputs(run_dir, train_on)
    X_sparse = mmread(str(matrix_path)).tocsr()
    targets_df = pd.read_csv(targets_path)
    initial_weights = np.load(inputs / "initial_weights.npy")

    seed_value = options.get("seed")
    weights = fit_l0_weights(
        X_sparse=X_sparse,
        targets=targets_df["value"].to_numpy(dtype=np.float64),
        lambda_l0=float(options.get("lambda_l0", 1e-8)),
        epochs=int(options.get("epochs", 1000)),
        device=str(options.get("device", "cpu")),
        beta=float(options.get("beta", 0.65)),
        lambda_l2=float(options.get("lambda_l2", 1e-12)),
        learning_rate=float(options.get("learning_rate", 0.15)),
        seed=int(seed_value) if seed_value is not None else None,
        target_names=targets_df["target_name"].tolist(),
        initial_weights=initial_weights,
        targets_df=targets_df,
    )

    weights_path = outputs / "fitted_weights.npy"
    np.save(weights_path, weights.astype(np.float64))
    return weights_path


def _run_greg(run_dir: Path, train_on: str = "shared_requested"):
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    temp_csv = outputs / "_greg_weights.csv"

    with open(inputs / "benchmark_manifest.json") as f:
        manifest = json.load(f)
    options = manifest.get("method_options", {}).get("greg", {})

    targets_path, matrix_path, _ = _select_training_inputs(run_dir, train_on)
    cmd = [
        "Rscript",
        str(RUNNERS_DIR / "greg_runner.R"),
        str(matrix_path),
        str(targets_path),
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


def _collapse_ipf_rows_to_unit_weights(
    raw_weights: pd.DataFrame, n_units: int
) -> np.ndarray:
    """Validate a per-row IPF output and collapse it to a length-`n_units` vector.

    surveysd::ipf with `meanHH = TRUE` guarantees every row that shares a
    `unit_index` carries the same fitted weight; the spread check keeps that
    assumption honest.
    """
    if "unit_index" not in raw_weights.columns:
        raise RuntimeError("IPF runner output must include a unit_index column")
    if raw_weights["unit_index"].isna().any():
        raise RuntimeError("IPF runner output contains missing unit_index values")

    raw_weights = raw_weights.copy()
    raw_weights["unit_index"] = raw_weights["unit_index"].astype(np.int64)
    if (raw_weights["unit_index"] < 0).any() or (
        raw_weights["unit_index"] >= n_units
    ).any():
        raise RuntimeError("IPF runner output contains out-of-range unit_index values")

    per_unit_spread = raw_weights.groupby("unit_index", sort=True)["fitted_weight"].agg(
        lambda series: float(series.max() - series.min())
    )
    if (per_unit_spread > 1e-9).any():
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
    return weights_by_unit.to_numpy(dtype=np.float64)


def _run_ipf(run_dir: Path):
    """Run one coherent IPF problem in a single `surveysd::ipf` call."""
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"

    with open(inputs / "benchmark_manifest.json") as f:
        manifest = json.load(f)
    options = manifest.get("method_options", {}).get("ipf", {})

    target_metadata_path = inputs / "ipf_target_metadata.csv"
    if not target_metadata_path.exists():
        raise FileNotFoundError(
            "IPF run requires inputs/ipf_target_metadata.csv. "
            "Provide external_inputs.ipf_target_metadata_csv in the manifest."
        )
    unit_metadata_path = inputs / "unit_metadata.csv"
    if not unit_metadata_path.exists():
        raise FileNotFoundError("IPF run requires inputs/unit_metadata.csv.")

    full_targets = pd.read_csv(target_metadata_path)
    if full_targets.empty:
        raise RuntimeError("IPF target metadata is empty; nothing to run.")
    unit_metadata = pd.read_csv(unit_metadata_path)
    if "unit_index" not in unit_metadata.columns:
        raise RuntimeError("Unit metadata must include a unit_index column for IPF")

    weight_col = str(options.get("weight_col", "base_weight"))
    household_id_col = str(options.get("household_id_col", "household_id"))

    initial_weights = np.load(inputs / "initial_weights.npy").astype(np.float64)
    n_units = len(initial_weights)
    unit_indices = unit_metadata["unit_index"].astype(np.int64).to_numpy()
    if unit_indices.min() < 0 or unit_indices.max() >= n_units:
        raise RuntimeError(
            "Unit metadata unit_index values fall outside the initial weight vector"
        )
    temp_csv = outputs / "_ipf_weights.csv"
    unit_with_weights = unit_metadata.copy()
    unit_with_weights[weight_col] = initial_weights[unit_indices]
    temp_unit_csv = outputs / "_ipf_unit_metadata.csv"
    unit_with_weights.to_csv(temp_unit_csv, index=False)

    cmd = [
        "Rscript",
        str(RUNNERS_DIR / "ipf_runner.R"),
        str(temp_unit_csv),
        str(target_metadata_path),
        str(inputs / "initial_weights.npy"),
        str(temp_csv),
        str(int(options.get("max_iter", 200))),
        str(float(options.get("bound", 10.0))),
        str(float(options.get("epsP", 1e-6))),
        str(float(options.get("epsH", 1e-2))),
        household_id_col,
        weight_col,
    ]
    proc, elapsed = _run_subprocess(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"IPF runner failed with exit code {proc.returncode}")

    raw_weights = pd.read_csv(temp_csv)
    current_weights = _collapse_ipf_rows_to_unit_weights(raw_weights, n_units)
    weights_path = outputs / "fitted_weights.npy"
    np.save(weights_path, current_weights)
    temp_csv.unlink(missing_ok=True)
    temp_unit_csv.unlink(missing_ok=True)
    return weights_path, elapsed


def _select_scoring_inputs(
    run_dir: Path, method: str, score_on: str
) -> tuple[Path, Path, str]:
    inputs = run_dir / "inputs"
    ipf_targets = inputs / "ipf_scoring_target_metadata.csv"
    ipf_matrix = inputs / "ipf_scoring_X_targets_by_units.mtx"
    has_ipf_scoring = ipf_targets.exists() and ipf_matrix.exists()

    if score_on == "ipf_retained_authored":
        if not has_ipf_scoring:
            raise FileNotFoundError(
                "Requested score_on=ipf_retained_authored, but "
                "inputs/ipf_scoring_target_metadata.csv and "
                "inputs/ipf_scoring_X_targets_by_units.mtx are not both present."
            )
        return ipf_targets, ipf_matrix, "ipf_retained_authored"

    if score_on == "auto" and method == "ipf" and has_ipf_scoring:
        return ipf_targets, ipf_matrix, "ipf_retained_authored"
    return (
        inputs / "target_metadata.csv",
        inputs / "X_targets_by_units.mtx",
        "shared_requested",
    )


def _summary_filename(method: str, train_on: str) -> str:
    if train_on == "ipf_retained_authored":
        return f"{method}_matched_summary.json"
    return f"{method}_summary.json"


def cmd_run(args):
    run_dir = Path(args.run_dir)
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    train_on = getattr(args, "train_on", "shared_requested")
    targets_path, matrix_path, scoring_target_set = _select_scoring_inputs(
        run_dir,
        args.method,
        getattr(args, "score_on", "auto"),
    )
    targets_df = load_targets_csv(targets_path)

    started = time.time()
    if args.method == "l0":
        weights_path = _run_l0(run_dir, train_on=train_on)
        training_target_set = train_on
    elif args.method == "greg":
        weights_path, _ = _run_greg(run_dir, train_on=train_on)
        training_target_set = train_on
    elif args.method == "ipf":
        weights_path, _ = _run_ipf(run_dir)
        # IPF always trains on its own categorical-margin inputs.
        training_target_set = "ipf_categorical_margins"
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    elapsed = time.time() - started

    weights = np.load(weights_path)
    summary = compute_common_metrics(
        weights=weights,
        targets_df=targets_df,
        matrix_path=matrix_path,
    )
    summary["method"] = args.method
    summary["run_dir"] = str(run_dir.resolve())
    summary["runtime_seconds"] = elapsed
    summary["scoring_target_set"] = scoring_target_set
    summary["training_target_set"] = training_target_set
    summary_filename = (
        _summary_filename(args.method, train_on)
        if args.method != "ipf"
        else f"{args.method}_summary.json"
    )
    write_method_summary(summary, outputs / summary_filename)
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
    run_parser.add_argument(
        "--score-on",
        default="auto",
        choices=["auto", "shared_requested", "ipf_retained_authored"],
        help=(
            "Scoring target set. 'auto' uses IPF-retained-authored targets only "
            "for method=ipf when available; the other methods default to the "
            "shared requested target set unless explicitly overridden."
        ),
    )
    run_parser.add_argument(
        "--train-on",
        default="shared_requested",
        choices=["shared_requested", "ipf_retained_authored"],
        help=(
            "Target set used as L0 / GREG training inputs. Defaults to the "
            "shared requested set. 'ipf_retained_authored' loads the IPF "
            "scoring subset for matched-input comparison against IPF. Ignored "
            "for method=ipf, which always trains on its own categorical-margin "
            "inputs. When set to ipf_retained_authored the summary is written "
            "to {method}_matched_summary.json so a follow-up matched run does "
            "not overwrite the full-info run."
        ),
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
