"""
Validate a calibration package before uploading to Modal.

Usage:
    python -m policyengine_us_data.calibration.validate_package [path]
        [--n-hardest N] [--strict [RATIO]]
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    n_targets: int
    n_columns: int
    nnz: int
    density: float
    metadata: dict
    n_achievable: int
    n_impossible: int
    impossible_targets: pd.DataFrame
    impossible_by_group: pd.DataFrame
    hardest_targets: pd.DataFrame
    group_summary: pd.DataFrame
    strict_ratio: Optional[float] = None
    strict_failures: int = 0


def validate_package(
    package: dict,
    n_hardest: int = 10,
    strict_ratio: float = None,
) -> ValidationResult:
    X_sparse = package["X_sparse"]
    targets_df = package["targets_df"]
    target_names = package["target_names"]
    metadata = package.get("metadata", {})

    n_targets, n_columns = X_sparse.shape
    nnz = X_sparse.nnz
    density = nnz / (n_targets * n_columns) if n_targets * n_columns else 0

    row_sums = np.array(X_sparse.sum(axis=1)).flatten()
    achievable_mask = row_sums > 0
    n_achievable = int(achievable_mask.sum())
    n_impossible = n_targets - n_achievable

    impossible_idx = np.where(~achievable_mask)[0]
    impossible_rows = targets_df.iloc[impossible_idx]
    impossible_targets = pd.DataFrame(
        {
            "target_name": [target_names[i] for i in impossible_idx],
            "domain_variable": impossible_rows["domain_variable"].values,
            "variable": impossible_rows["variable"].values,
            "geo_level": impossible_rows["geo_level"].values,
            "geographic_id": impossible_rows["geographic_id"].values,
            "target_value": impossible_rows["value"].values,
        }
    )
    impossible_by_group = (
        impossible_rows.groupby(["domain_variable", "variable", "geo_level"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    target_values = targets_df["value"].values
    achievable_idx = np.where(achievable_mask)[0]
    if len(achievable_idx) > 0:
        a_row_sums = row_sums[achievable_idx]
        a_target_vals = target_values[achievable_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                a_target_vals != 0,
                a_row_sums / a_target_vals,
                np.inf,
            )
        k = min(n_hardest, len(ratios))
        hardest_local_idx = np.argpartition(ratios, k)[:k]
        hardest_local_idx = hardest_local_idx[
            np.argsort(ratios[hardest_local_idx])
        ]
        hardest_global_idx = achievable_idx[hardest_local_idx]

        hardest_targets = pd.DataFrame(
            {
                "target_name": [target_names[i] for i in hardest_global_idx],
                "domain_variable": targets_df["domain_variable"]
                .iloc[hardest_global_idx]
                .values,
                "variable": targets_df["variable"]
                .iloc[hardest_global_idx]
                .values,
                "geographic_id": targets_df["geographic_id"]
                .iloc[hardest_global_idx]
                .values,
                "ratio": ratios[hardest_local_idx],
                "row_sum": a_row_sums[hardest_local_idx],
                "target_value": a_target_vals[hardest_local_idx],
            }
        )
    else:
        hardest_targets = pd.DataFrame(
            columns=[
                "target_name",
                "domain_variable",
                "variable",
                "geographic_id",
                "ratio",
                "row_sum",
                "target_value",
            ]
        )

    group_summary = (
        targets_df.assign(achievable=achievable_mask)
        .groupby(["domain_variable", "variable", "geo_level"])
        .agg(total=("value", "size"), ok=("achievable", "sum"))
        .reset_index()
    )
    group_summary["impossible"] = group_summary["total"] - group_summary["ok"]
    group_summary["ok"] = group_summary["ok"].astype(int)
    group_summary = group_summary.sort_values(
        ["domain_variable", "variable", "geo_level"]
    ).reset_index(drop=True)

    strict_failures = 0
    if strict_ratio is not None and len(achievable_idx) > 0:
        strict_failures = int((ratios < strict_ratio).sum())

    return ValidationResult(
        n_targets=n_targets,
        n_columns=n_columns,
        nnz=nnz,
        density=density,
        metadata=metadata,
        n_achievable=n_achievable,
        n_impossible=n_impossible,
        impossible_targets=impossible_targets,
        impossible_by_group=impossible_by_group,
        hardest_targets=hardest_targets,
        group_summary=group_summary,
        strict_ratio=strict_ratio,
        strict_failures=strict_failures,
    )


def format_report(result: ValidationResult, package_path: str = None) -> str:
    lines = ["", "=== Calibration Package Validation ===", ""]

    if package_path:
        lines.append(f"Package: {package_path}")
    meta = result.metadata
    if meta.get("created_at"):
        lines.append(f"Created: {meta['created_at']}")
    if meta.get("dataset_path"):
        lines.append(f"Dataset: {meta['dataset_path']}")
    lines.append("")

    lines.append(
        f"Matrix:  {result.n_targets:,} targets"
        f" x {result.n_columns:,} columns"
    )
    lines.append(f"Non-zero: {result.nnz:,} (density: {result.density:.6f})")
    if meta.get("n_clones"):
        parts = [f"Clones: {meta['n_clones']}"]
        if meta.get("n_records"):
            parts.append(f"Records: {meta['n_records']:,}")
        if meta.get("seed") is not None:
            parts.append(f"Seed: {meta['seed']}")
        lines.append(", ".join(parts))
    lines.append("")

    pct = (
        100 * result.n_achievable / result.n_targets if result.n_targets else 0
    )
    pct_imp = 100 - pct
    lines.append("--- Achievability ---")
    lines.append(
        f"Achievable: {result.n_achievable:>6,}"
        f" / {result.n_targets:,} ({pct:.1f}%)"
    )
    lines.append(
        f"Impossible: {result.n_impossible:>6,}"
        f" / {result.n_targets:,} ({pct_imp:.1f}%)"
    )
    lines.append("")

    if len(result.impossible_targets) > 0:
        lines.append("--- Impossible Targets ---")
        for _, row in result.impossible_targets.iterrows():
            lines.append(
                f"  {row['target_name']:<60s}"
                f" {row['target_value']:>14,.0f}"
            )
        lines.append("")

    if len(result.impossible_by_group) > 1:
        lines.append("--- Impossible Targets by Group ---")
        for _, row in result.impossible_by_group.iterrows():
            lines.append(
                f"  {row['domain_variable']:<20s}"
                f" {row['variable']:<25s}"
                f" {row['geo_level']:<12s}"
                f" {row['count']:>5d}"
            )
        lines.append("")

    if len(result.hardest_targets) > 0:
        n = len(result.hardest_targets)
        lines.append(
            f"--- Hardest Achievable Targets" f" ({n} lowest ratio) ---"
        )
        for _, row in result.hardest_targets.iterrows():
            lines.append(
                f"  {row['target_name']:<50s}"
                f" {row['ratio']:>10.4f}"
                f" {row['row_sum']:>14,.0f}"
                f" {row['target_value']:>14,.0f}"
            )
        lines.append("")

    if len(result.group_summary) > 0:
        lines.append("--- Group Summary ---")
        lines.append(
            f"  {'domain':<20s} {'variable':<25s}"
            f" {'geo_level':<12s}"
            f" {'total':>6s} {'ok':>6s} {'impossible':>10s}"
        )
        for _, row in result.group_summary.iterrows():
            lines.append(
                f"  {row['domain_variable']:<20s}"
                f" {row['variable']:<25s}"
                f" {row['geo_level']:<12s}"
                f" {row['total']:>6d}"
                f" {row['ok']:>6d}"
                f" {row['impossible']:>10d}"
            )
        lines.append("")

    if result.strict_ratio is not None:
        lines.append(
            f"Strict check (ratio < {result.strict_ratio}):"
            f" {result.strict_failures} failures"
        )
        lines.append("")

    if result.strict_ratio is not None and result.strict_failures > 0:
        lines.append(
            f"RESULT: FAIL ({result.strict_failures}"
            f" targets below ratio {result.strict_ratio})"
        )
    elif result.n_impossible > 0:
        lines.append(
            f"RESULT: FAIL ({result.n_impossible} impossible targets)"
        )
    else:
        lines.append("RESULT: PASS")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a calibration package"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to calibration_package.pkl",
    )
    parser.add_argument(
        "--n-hardest",
        type=int,
        default=10,
        help="Number of hardest achievable targets to show",
    )
    parser.add_argument(
        "--strict",
        nargs="?",
        const=0.01,
        type=float,
        default=None,
        metavar="RATIO",
        help="Fail if any achievable target has ratio below RATIO"
        " (default: 0.01)",
    )
    args = parser.parse_args()

    if args.path is None:
        from policyengine_us_data.storage import STORAGE_FOLDER

        path = STORAGE_FOLDER / "calibration" / "calibration_package.pkl"
    else:
        path = Path(args.path)

    if not path.exists():
        print(f"Error: package not found at {path}", file=sys.stderr)
        sys.exit(1)

    from policyengine_us_data.calibration.unified_calibration import (
        load_calibration_package,
    )

    package = load_calibration_package(str(path))
    result = validate_package(
        package,
        n_hardest=args.n_hardest,
        strict_ratio=args.strict,
    )
    print(format_report(result, package_path=str(path)))

    if args.strict is not None and result.strict_failures > 0:
        sys.exit(2)
    elif result.n_impossible > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
