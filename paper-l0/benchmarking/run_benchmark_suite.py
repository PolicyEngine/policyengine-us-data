"""End-to-end orchestration for the L0 / GREG / IPF benchmark suite.

Drives the workflow described in `paper-l0/BENCHMARK_PLAN.md`:

1. Export each manifest to its own bundle directory.
2. Run every method declared in the manifest on the shared exported bundle.
3. When IPF is in play, also run L0 and GREG (when present) on the matched
   IPF-retained-authored subset so the paper's matched-input rows are produced
   side-by-side with the full-information rows.
4. Aggregate per-method summaries into one `tier_summary.csv` per tier so the
   paper's tractable / scaling / production tables consume a single artifact.

Failures (export-time `IPFConversionError`, runtime errors, runner non-zero
exit codes) are recorded as rows with `status="failed"` rather than aborting
the suite — Tier 3 explicitly relies on this so a GREG out-of-memory or IPF
non-convergence is a reportable result, not a missing row.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Pin the in-tree repo root ahead of site-packages so child `python` invocations
# inherit the import path that finds this repo's `policyengine_us_data` rather
# than a sibling editable install. Without this, `fit_l0_weights` may resolve
# to an older copy that lacks the `seed` parameter the manifests pass through.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
DEFAULT_MANIFEST_DIR = ROOT / "manifests"
DEFAULT_RUNS_DIR = ROOT / "runs"


@dataclass
class RunRecord:
    """One row in a per-tier summary table."""

    tier: str
    manifest_name: str
    method: str
    training_target_set: str
    scoring_target_set: str
    status: str
    runtime_seconds: Optional[float]
    n_targets: Optional[int]
    n_units: Optional[int]
    mean_abs_rel_error: Optional[float]
    median_abs_rel_error: Optional[float]
    p95_abs_rel_error: Optional[float]
    max_abs_rel_error: Optional[float]
    ess: Optional[float]
    active_record_count: Optional[int]
    negative_weight_share: Optional[float]
    notes: str

    def as_row(self) -> Dict[str, object]:
        return {
            "tier": self.tier,
            "manifest_name": self.manifest_name,
            "method": self.method,
            "training_target_set": self.training_target_set,
            "scoring_target_set": self.scoring_target_set,
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
            "n_targets": self.n_targets,
            "n_units": self.n_units,
            "mean_abs_rel_error": self.mean_abs_rel_error,
            "median_abs_rel_error": self.median_abs_rel_error,
            "p95_abs_rel_error": self.p95_abs_rel_error,
            "max_abs_rel_error": self.max_abs_rel_error,
            "ess": self.ess,
            "active_record_count": self.active_record_count,
            "negative_weight_share": self.negative_weight_share,
            "notes": self.notes,
        }


def _tier_default_manifests(manifest_dir: Path, tier: str) -> List[Path]:
    """Manifest files that belong to a tier, ordered by target-count rung.

    Tier 2 is a scaling ladder: rungs must run from smallest to largest so the
    summary table reads top-to-bottom in increasing target count, and so a
    failure on a large rung does not cause smaller rungs to be skipped. The
    rung is taken from `target_filters.max_targets`; manifests with no cap
    (the largest pre-production rung and the production manifest) sort last.
    """
    prefix = {
        "tier_1": "tier1_",
        "tier_2": "tier2_",
        "tier_3": "tier3_",
    }[tier]
    candidates = list(manifest_dir.glob(f"{prefix}*.json"))

    def sort_key(path: Path) -> tuple:
        with open(path) as f:
            manifest = json.load(f)
        max_targets = manifest.get("target_filters", {}).get("max_targets")
        rung = float("inf") if max_targets is None else int(max_targets)
        return (rung, path.name)

    return sorted(candidates, key=sort_key)


def _read_manifest(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _export_bundle(manifest_path: Path, run_dir: Path) -> Dict:
    """Run `benchmark_cli.py export` and return the parsed JSON info."""
    cmd = [
        sys.executable,
        str(ROOT / "benchmark_cli.py"),
        "export",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(run_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"benchmark export failed for {manifest_path.name}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    last_brace = result.stdout.rfind("{")
    if last_brace == -1:
        return {}
    return json.loads(result.stdout[last_brace:])


def _run_method(
    run_dir: Path,
    method: str,
    *,
    train_on: str = "shared_requested",
    score_on: str = "auto",
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(ROOT / "benchmark_cli.py"),
        "run",
        "--method",
        method,
        "--run-dir",
        str(run_dir),
        "--score-on",
        score_on,
    ]
    if method != "ipf":
        cmd.extend(["--train-on", train_on])
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _summary_filename(method: str, train_on: str) -> str:
    if method != "ipf" and train_on == "ipf_retained_authored":
        return f"{method}_matched_summary.json"
    return f"{method}_summary.json"


def _summary_to_record(
    *,
    tier: str,
    manifest_name: str,
    method: str,
    train_on: str,
    summary: Dict,
    notes: str = "",
) -> RunRecord:
    return RunRecord(
        tier=tier,
        manifest_name=manifest_name,
        method=method,
        training_target_set=str(summary.get("training_target_set", train_on)),
        scoring_target_set=str(summary.get("scoring_target_set", "shared_requested")),
        status="completed",
        runtime_seconds=summary.get("runtime_seconds"),
        n_targets=summary.get("n_targets"),
        n_units=summary.get("n_units"),
        mean_abs_rel_error=summary.get("mean_abs_rel_error"),
        median_abs_rel_error=summary.get("median_abs_rel_error"),
        p95_abs_rel_error=summary.get("p95_abs_rel_error"),
        max_abs_rel_error=summary.get("max_abs_rel_error"),
        ess=summary.get("ess"),
        active_record_count=summary.get("active_record_count"),
        negative_weight_share=summary.get("negative_weight_share"),
        notes=notes,
    )


def _failure_record(
    *,
    tier: str,
    manifest_name: str,
    method: str,
    train_on: str,
    reason: str,
) -> RunRecord:
    return RunRecord(
        tier=tier,
        manifest_name=manifest_name,
        method=method,
        training_target_set=train_on,
        scoring_target_set="n/a",
        status="failed",
        runtime_seconds=None,
        n_targets=None,
        n_units=None,
        mean_abs_rel_error=None,
        median_abs_rel_error=None,
        p95_abs_rel_error=None,
        max_abs_rel_error=None,
        ess=None,
        active_record_count=None,
        negative_weight_share=None,
        notes=reason,
    )


def _run_method_and_record(
    *,
    tier: str,
    manifest_name: str,
    method: str,
    run_dir: Path,
    train_on: str,
    score_on: str,
) -> RunRecord:
    proc = _run_method(run_dir, method, train_on=train_on, score_on=score_on)
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or "unknown error").strip()
        return _failure_record(
            tier=tier,
            manifest_name=manifest_name,
            method=method,
            train_on=train_on,
            reason=f"runner_exit_{proc.returncode}: {reason[:500]}",
        )
    summary_path = run_dir / "outputs" / _summary_filename(method, train_on)
    if not summary_path.exists():
        return _failure_record(
            tier=tier,
            manifest_name=manifest_name,
            method=method,
            train_on=train_on,
            reason=f"missing_summary: {summary_path}",
        )
    with open(summary_path) as f:
        summary = json.load(f)
    return _summary_to_record(
        tier=tier,
        manifest_name=manifest_name,
        method=method,
        train_on=train_on,
        summary=summary,
    )


def _run_one_manifest(
    manifest_path: Path,
    runs_dir: Path,
) -> List[RunRecord]:
    """Export one manifest and run every method paired with it.

    Each manifest gets its own bundle directory under `runs_dir`. When IPF is
    declared, also run any other method in the manifest (L0 / GREG) on the
    matched IPF-retained-authored subset so the full-info and matched rows
    appear together in the tier summary.
    """
    manifest = _read_manifest(manifest_path)
    tier = str(manifest["tier"])
    name = str(manifest["name"])
    methods = list(manifest.get("methods", []))
    run_dir = runs_dir / name

    records: List[RunRecord] = []

    try:
        _export_bundle(manifest_path, run_dir)
    except Exception as exc:
        message = str(exc)
        for method in methods:
            records.append(
                _failure_record(
                    tier=tier,
                    manifest_name=name,
                    method=method,
                    train_on="shared_requested",
                    reason=f"export_failed: {message[:500]}",
                )
            )
        return records

    for method in methods:
        records.append(
            _run_method_and_record(
                tier=tier,
                manifest_name=name,
                method=method,
                run_dir=run_dir,
                train_on="shared_requested",
                score_on="auto",
            )
        )

    if "ipf" in methods:
        for method in methods:
            if method == "ipf":
                continue
            records.append(
                _run_method_and_record(
                    tier=tier,
                    manifest_name=name,
                    method=method,
                    run_dir=run_dir,
                    train_on="ipf_retained_authored",
                    score_on="ipf_retained_authored",
                )
            )

    return records


def run_suite(
    manifests: Iterable[Path],
    runs_dir: Path,
) -> pd.DataFrame:
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for path in manifests:
        manifest_started = time.time()
        try:
            records = _run_one_manifest(path, runs_dir)
        except Exception:
            tb = traceback.format_exc()
            manifest = _read_manifest(path)
            records = [
                _failure_record(
                    tier=str(manifest.get("tier", "unknown")),
                    manifest_name=str(manifest.get("name", path.stem)),
                    method=method,
                    train_on="shared_requested",
                    reason=f"unhandled_exception: {tb[-500:]}",
                )
                for method in manifest.get("methods", [])
            ]
        elapsed = time.time() - manifest_started
        for record in records:
            row = record.as_row()
            row["manifest_wall_seconds"] = elapsed
            rows.append(row)
    return pd.DataFrame(rows)


def write_tier_summaries(suite_df: pd.DataFrame, runs_dir: Path) -> Dict[str, Path]:
    """Split the orchestration DataFrame into one CSV per tier."""
    paths: Dict[str, Path] = {}
    for tier, group in suite_df.groupby("tier", sort=True):
        path = runs_dir / f"{tier}_summary.csv"
        group.to_csv(path, index=False)
        paths[str(tier)] = path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more benchmark tiers end-to-end. The orchestrator "
            "exports the manifest, runs every method it declares, "
            "schedules matched-input rows when IPF is in play, and writes a "
            "per-tier summary CSV ready to drop into the paper."
        )
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=["tier_1", "tier_2", "tier_3"],
        help=(
            "Tier(s) to run. Pass multiple times to run several tiers in one "
            "invocation. Defaults to all three tiers."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        help=(
            "Explicit manifest path(s). Overrides --tier. Repeatable. Useful "
            "for re-running a single tier rung after a CI failure."
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory holding tier manifests (default: paper-l0/benchmarking/manifests).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory under which each manifest gets its own bundle.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.manifest:
        manifests = [Path(p).resolve() for p in args.manifest]
    else:
        tiers = args.tier or ["tier_1", "tier_2", "tier_3"]
        manifests = [
            path
            for tier in tiers
            for path in _tier_default_manifests(args.manifest_dir, tier)
        ]
    if not manifests:
        raise SystemExit("No manifests selected.")

    suite_df = run_suite(manifests, args.runs_dir)
    suite_csv = args.runs_dir / "suite_summary.csv"
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    suite_df.to_csv(suite_csv, index=False)
    tier_paths = write_tier_summaries(suite_df, args.runs_dir)
    print(
        json.dumps(
            {
                "suite_summary_csv": str(suite_csv),
                "tier_summary_csvs": {tier: str(p) for tier, p in tier_paths.items()},
                "n_rows": int(len(suite_df)),
                "n_failures": int((suite_df["status"] == "failed").sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
