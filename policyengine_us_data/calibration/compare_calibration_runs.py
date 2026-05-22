"""Compare a completed L0 pipeline run against legacy Enhanced CPS outputs.

This module is intentionally report-first. By default it writes diagnostics and
H5 aggregate comparisons without failing the process; pass
``--fail-on-threshold`` to turn the configured thresholds into a CI gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from policyengine_us_data.__version__ import __version__ as DATA_PACKAGE_VERSION
from policyengine_us_data.utils.run_context import staging_prefix

HF_REPO = "policyengine/policyengine-us-data"
HF_REPO_TYPE = "model"
DEFAULT_OUTPUT_DIR = Path("calibration_comparison_report")
DEFAULT_P95_THRESHOLD = 0.10
DEFAULT_P99_THRESHOLD = 0.50
DEFAULT_H5_DELTA_THRESHOLD = 0.10
DEFAULT_VARIABLES = [
    "adjusted_gross_income",
    "employment_income",
    "total_self_employment_income",
    "tax_unit_partnership_s_corp_income",
    "taxable_pension_income",
    "dividend_income",
    "net_capital_gains",
    "rental_income",
    "taxable_interest_income",
    "social_security",
    "snap",
    "ssi_federal_fiscal_year_outlays",
    "income_tax_before_credits",
    "ctc",
    "eitc",
    "non_refundable_ctc",
    "refundable_ctc",
    "real_estate_taxes",
    "rent",
    "is_pregnant",
    "ctc_qualifying_children",
    "person_count",
    "household_count",
]
COUNT_VARIABLES = {
    "person_count",
    "household_count",
    "is_pregnant",
    "ctc_qualifying_children",
}


@dataclass(frozen=True)
class RunComparisonPaths:
    """Default artifact paths for a run-scoped production pipeline attempt."""

    run_id: str
    version: str = DATA_PACKAGE_VERSION

    @property
    def regional_diagnostics(self) -> str:
        return (
            f"hf://{HF_REPO}/calibration/runs/{self.run_id}/diagnostics/"
            "unified_diagnostics.csv"
        )

    @property
    def national_diagnostics(self) -> str:
        return (
            f"hf://{HF_REPO}/calibration/runs/{self.run_id}/diagnostics/"
            "national_unified_diagnostics.csv"
        )

    @property
    def candidate_h5(self) -> str:
        prefix = staging_prefix(self.run_id, version=self.version)
        return f"hf://{HF_REPO}/{prefix}/national/US.h5"

    @property
    def legacy_h5(self) -> str:
        prefix = staging_prefix(self.run_id, version=self.version)
        return f"hf://{HF_REPO}/{prefix}/enhanced_cps_2024.h5"


def resolve_artifact_path(path: str) -> str:
    """Resolve local or ``hf://`` artifact paths to a local filesystem path."""
    if not path.startswith("hf://"):
        return path

    from huggingface_hub import hf_hub_download

    parts = path[5:].split("/", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected hf:// artifact path: {path}")
    return hf_hub_download(
        repo_id=f"{parts[0]}/{parts[1]}",
        filename=parts[2],
        repo_type=HF_REPO_TYPE,
        token=os.environ.get("HUGGING_FACE_TOKEN"),
    )


def parse_variables(raw: str | None) -> list[str]:
    """Return requested H5 variables, preserving input order."""
    if not raw:
        return list(DEFAULT_VARIABLES)
    return [value.strip() for value in raw.split(",") if value.strip()]


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _quantile(values: pd.Series, q: float) -> float | None:
    if values.empty:
        return None
    return float(values.quantile(q))


def summarize_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    label: str,
    p95_threshold: float = DEFAULT_P95_THRESHOLD,
    p99_threshold: float = DEFAULT_P99_THRESHOLD,
) -> dict[str, Any]:
    """Summarize per-target calibration diagnostics."""
    if "abs_rel_error" in diagnostics.columns:
        abs_errors = pd.to_numeric(diagnostics["abs_rel_error"], errors="coerce")
    elif "rel_error" in diagnostics.columns:
        abs_errors = pd.to_numeric(diagnostics["rel_error"], errors="coerce").abs()
    else:
        raise ValueError("Diagnostics must include abs_rel_error or rel_error.")

    achievable = (
        diagnostics["achievable"].astype(bool)
        if "achievable" in diagnostics.columns
        else pd.Series(True, index=diagnostics.index)
    )
    comparable = abs_errors[achievable & abs_errors.notna()]
    worst_idx = comparable.idxmax() if not comparable.empty else None
    p95 = _quantile(comparable, 0.95)
    p99 = _quantile(comparable, 0.99)
    status = "PASS"
    if p95 is not None and p95 > p95_threshold:
        status = "WARN"
    if p99 is not None and p99 > p99_threshold:
        status = "WARN"

    target_col = "target" if "target" in diagnostics.columns else None
    return {
        "label": label,
        "status": status,
        "target_count": int(len(diagnostics)),
        "achievable_target_count": int(achievable.sum()),
        "unachievable_target_count": int((~achievable).sum()),
        "mean_abs_rel_error": _safe_float(comparable.mean()),
        "median_abs_rel_error": _safe_float(comparable.median()),
        "p90_abs_rel_error": _quantile(comparable, 0.90),
        "p95_abs_rel_error": p95,
        "p99_abs_rel_error": p99,
        "max_abs_rel_error": _safe_float(comparable.max()),
        "share_abs_rel_error_le_1pct": _safe_float((comparable <= 0.01).mean()),
        "share_abs_rel_error_le_5pct": _safe_float((comparable <= 0.05).mean()),
        "share_abs_rel_error_le_10pct": _safe_float((comparable <= 0.10).mean()),
        "worst_target": (
            str(diagnostics.loc[worst_idx, target_col])
            if worst_idx is not None and target_col is not None
            else None
        ),
        "worst_abs_rel_error": _safe_float(comparable.loc[worst_idx])
        if worst_idx is not None
        else None,
    }


def load_diagnostic_summary(
    path: str,
    *,
    label: str,
    p95_threshold: float,
    p99_threshold: float,
) -> dict[str, Any]:
    """Load and summarize a diagnostics CSV from a local or HF path."""
    resolved = resolve_artifact_path(path)
    diagnostics = pd.read_csv(resolved)
    summary = summarize_diagnostics(
        diagnostics,
        label=label,
        p95_threshold=p95_threshold,
        p99_threshold=p99_threshold,
    )
    summary["path"] = path
    return summary


def calculate_h5_totals(dataset_path: str, variables: Iterable[str]) -> dict[str, Any]:
    """Calculate national aggregate values from a PolicyEngine H5 artifact."""
    from policyengine_us import Microsimulation

    resolved = resolve_artifact_path(dataset_path)
    sim = Microsimulation(dataset=resolved)
    totals: dict[str, Any] = {}
    for variable in variables:
        try:
            totals[variable] = float(sim.calculate(variable).sum())
        except Exception as exc:  # pragma: no cover - depends on runtime variables.
            totals[variable] = {"error": str(exc)}
    return totals


def load_reference_values() -> dict[str, tuple[float, str]]:
    """Load national reference totals used by the national H5 validator."""
    from policyengine_us_data.calibration.validate_national_h5 import (
        get_reference_values,
    )

    return get_reference_values()


def build_h5_comparison_rows(
    *,
    candidate_totals: dict[str, Any],
    legacy_totals: dict[str, Any],
    reference_values: dict[str, tuple[float, str]] | None = None,
    max_delta_vs_legacy: float = DEFAULT_H5_DELTA_THRESHOLD,
) -> list[dict[str, Any]]:
    """Build per-variable aggregate comparison rows."""
    reference_values = reference_values or {}
    rows: list[dict[str, Any]] = []
    variables = sorted(set(candidate_totals) | set(legacy_totals))
    for variable in variables:
        candidate = candidate_totals.get(variable)
        legacy = legacy_totals.get(variable)
        candidate_error = (
            candidate.get("error")
            if isinstance(candidate, dict) and "error" in candidate
            else None
        )
        legacy_error = (
            legacy.get("error")
            if isinstance(legacy, dict) and "error" in legacy
            else None
        )
        candidate_value = None if candidate_error else _safe_float(candidate)
        legacy_value = None if legacy_error else _safe_float(legacy)
        delta = (
            candidate_value - legacy_value
            if candidate_value is not None and legacy_value is not None
            else None
        )
        pct_delta = (
            delta / abs(legacy_value)
            if delta is not None and legacy_value not in (None, 0)
            else None
        )
        ref_value = reference_values.get(variable, (None, None))[0]
        ref_label = reference_values.get(variable, (None, None))[1]
        candidate_ref_error = (
            (candidate_value - ref_value) / abs(ref_value)
            if candidate_value is not None and ref_value not in (None, 0)
            else None
        )
        legacy_ref_error = (
            (legacy_value - ref_value) / abs(ref_value)
            if legacy_value is not None and ref_value not in (None, 0)
            else None
        )
        status = "PASS"
        if candidate_error or legacy_error:
            status = "ERROR"
        elif pct_delta is not None and abs(pct_delta) > max_delta_vs_legacy:
            status = "WARN"
        rows.append(
            {
                "variable": variable,
                "status": status,
                "candidate_value": candidate_value,
                "legacy_value": legacy_value,
                "delta_vs_legacy": delta,
                "pct_delta_vs_legacy": pct_delta,
                "reference_value": ref_value,
                "reference_label": ref_label,
                "candidate_pct_error_vs_reference": candidate_ref_error,
                "legacy_pct_error_vs_reference": legacy_ref_error,
                "candidate_error": candidate_error,
                "legacy_error": legacy_error,
                "display_type": "count" if variable in COUNT_VARIABLES else "amount",
            }
        )
    return rows


def summarize_h5_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize national H5 aggregate comparison rows."""
    comparable = [
        row
        for row in rows
        if row.get("pct_delta_vs_legacy") is not None and row.get("status") != "ERROR"
    ]
    abs_deltas = [abs(float(row["pct_delta_vs_legacy"])) for row in comparable]
    worst = max(
        comparable,
        key=lambda row: abs(float(row["pct_delta_vs_legacy"])),
        default=None,
    )
    status = "PASS"
    if any(row["status"] == "ERROR" for row in rows):
        status = "ERROR"
    elif any(row["status"] == "WARN" for row in rows):
        status = "WARN"
    return {
        "label": "national_h5",
        "status": status,
        "variable_count": len(rows),
        "comparable_variable_count": len(comparable),
        "max_abs_pct_delta_vs_legacy": max(abs_deltas) if abs_deltas else None,
        "mean_abs_pct_delta_vs_legacy": float(np.mean(abs_deltas))
        if abs_deltas
        else None,
        "worst_variable": worst["variable"] if worst else None,
        "worst_pct_delta_vs_legacy": worst["pct_delta_vs_legacy"] if worst else None,
    }


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _format_number(value: Any, display_type: str = "amount") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    prefix = "" if display_type == "count" else "$"
    return f"{prefix}{float(value):,.0f}"


def build_markdown_report(
    *,
    run_id: str,
    summaries: list[dict[str, Any]],
    h5_rows: list[dict[str, Any]],
    overall_status: str,
    fail_on_threshold: bool,
) -> str:
    """Render a compact Markdown report for GitHub summaries and artifacts."""
    lines = [
        f"# Calibration comparison report: `{run_id}`",
        "",
        f"Overall status: **{overall_status}**",
        f"Threshold mode: `{'fail' if fail_on_threshold else 'report-only'}`",
        "",
        "## Calibration diagnostics",
        "",
    ]
    if summaries:
        lines.extend(
            [
                "| Scope | Status | Targets | p95 abs err | p99 abs err | Worst target | Worst abs err |",
                "|---|---:|---:|---:|---:|---|---:|",
            ]
        )
        for summary in summaries:
            lines.append(
                "| {label} | {status} | {targets:,} | {p95} | {p99} | {worst} | {max_err} |".format(
                    label=summary["label"],
                    status=summary["status"],
                    targets=summary["achievable_target_count"],
                    p95=_format_pct(summary.get("p95_abs_rel_error")),
                    p99=_format_pct(summary.get("p99_abs_rel_error")),
                    worst=summary.get("worst_target") or "n/a",
                    max_err=_format_pct(summary.get("worst_abs_rel_error")),
                )
            )
    else:
        lines.append("No diagnostics CSVs were included.")

    lines.extend(["", "## National H5 vs legacy Enhanced CPS", ""])
    if h5_rows:
        h5_summary = summarize_h5_rows(h5_rows)
        lines.extend(
            [
                f"Status: **{h5_summary['status']}**",
                (
                    "Worst delta vs legacy: "
                    f"`{h5_summary.get('worst_variable') or 'n/a'}` "
                    f"({_format_pct(h5_summary.get('worst_pct_delta_vs_legacy'))})"
                ),
                "",
                "| Variable | Status | Candidate | Legacy ECPS | Delta vs legacy | Candidate ref err | Legacy ref err |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in h5_rows:
            lines.append(
                "| {variable} | {status} | {candidate} | {legacy} | {delta} | {candidate_ref} | {legacy_ref} |".format(
                    variable=row["variable"],
                    status=row["status"],
                    candidate=_format_number(
                        row.get("candidate_value"),
                        row.get("display_type", "amount"),
                    ),
                    legacy=_format_number(
                        row.get("legacy_value"),
                        row.get("display_type", "amount"),
                    ),
                    delta=_format_pct(row.get("pct_delta_vs_legacy")),
                    candidate_ref=_format_pct(
                        row.get("candidate_pct_error_vs_reference")
                    ),
                    legacy_ref=_format_pct(row.get("legacy_pct_error_vs_reference")),
                )
            )
    else:
        lines.append("H5 comparison was skipped.")

    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, preserving a stable sorted field set."""
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def determine_overall_status(
    summaries: list[dict[str, Any]],
    h5_rows: list[dict[str, Any]],
) -> str:
    """Return PASS, WARN, or ERROR for the full report."""
    statuses = [summary["status"] for summary in summaries]
    if h5_rows:
        statuses.append(summarize_h5_rows(h5_rows)["status"])
    if "ERROR" in statuses:
        return "ERROR"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a run-scoped calibration comparison report for deciding "
            "whether the unified L0 pipeline can replace legacy Enhanced CPS."
        )
    )
    parser.add_argument("--run-id", required=True, help="Completed pipeline run ID.")
    parser.add_argument(
        "--version",
        default=DATA_PACKAGE_VERSION,
        help="Data package version segment for run-scoped HF staging paths.",
    )
    parser.add_argument(
        "--regional-diagnostics",
        help="Path to regional unified_diagnostics.csv. Defaults from --run-id.",
    )
    parser.add_argument(
        "--national-diagnostics",
        help="Path to national_unified_diagnostics.csv. Defaults from --run-id.",
    )
    parser.add_argument(
        "--candidate-h5",
        help="Candidate national US.h5 path. Defaults to run-scoped HF staging.",
    )
    parser.add_argument(
        "--legacy-h5",
        help="Legacy Enhanced CPS h5 path. Defaults to run-scoped HF staging.",
    )
    parser.add_argument(
        "--skip-h5",
        action="store_true",
        help="Only summarize calibration diagnostics; skip H5 aggregate comparison.",
    )
    parser.add_argument(
        "--variables",
        help="Comma-separated H5 variables to compare. Defaults to national validator variables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=DEFAULT_P95_THRESHOLD,
        help="Warn/fail when diagnostics p95 abs relative error exceeds this value.",
    )
    parser.add_argument(
        "--p99-threshold",
        type=float,
        default=DEFAULT_P99_THRESHOLD,
        help="Warn/fail when diagnostics p99 abs relative error exceeds this value.",
    )
    parser.add_argument(
        "--h5-delta-threshold",
        type=float,
        default=DEFAULT_H5_DELTA_THRESHOLD,
        help="Warn/fail when H5 aggregate delta vs legacy exceeds this value.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit nonzero when any threshold is exceeded or H5 variables fail.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    defaults = RunComparisonPaths(args.run_id, version=args.version)
    regional_path = args.regional_diagnostics or defaults.regional_diagnostics
    national_path = args.national_diagnostics or defaults.national_diagnostics
    candidate_h5 = args.candidate_h5 or defaults.candidate_h5
    legacy_h5 = args.legacy_h5 or defaults.legacy_h5

    summaries = [
        load_diagnostic_summary(
            regional_path,
            label="regional",
            p95_threshold=args.p95_threshold,
            p99_threshold=args.p99_threshold,
        ),
        load_diagnostic_summary(
            national_path,
            label="national",
            p95_threshold=args.p95_threshold,
            p99_threshold=args.p99_threshold,
        ),
    ]

    h5_rows: list[dict[str, Any]] = []
    if not args.skip_h5:
        variables = parse_variables(args.variables)
        candidate_totals = calculate_h5_totals(candidate_h5, variables)
        legacy_totals = calculate_h5_totals(legacy_h5, variables)
        h5_rows = build_h5_comparison_rows(
            candidate_totals=candidate_totals,
            legacy_totals=legacy_totals,
            reference_values=load_reference_values(),
            max_delta_vs_legacy=args.h5_delta_threshold,
        )

    overall_status = determine_overall_status(summaries, h5_rows)
    report = {
        "run_id": args.run_id,
        "overall_status": overall_status,
        "fail_on_threshold": args.fail_on_threshold,
        "inputs": {
            "regional_diagnostics": regional_path,
            "national_diagnostics": national_path,
            "candidate_h5": None if args.skip_h5 else candidate_h5,
            "legacy_h5": None if args.skip_h5 else legacy_h5,
        },
        "thresholds": {
            "p95_abs_rel_error": args.p95_threshold,
            "p99_abs_rel_error": args.p99_threshold,
            "h5_delta_vs_legacy": args.h5_delta_threshold,
        },
        "diagnostics": summaries,
        "h5_summary": summarize_h5_rows(h5_rows) if h5_rows else None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "diagnostics_summary.csv", summaries)
    write_csv(args.output_dir / "h5_comparison.csv", h5_rows)
    markdown = build_markdown_report(
        run_id=args.run_id,
        summaries=summaries,
        h5_rows=h5_rows,
        overall_status=overall_status,
        fail_on_threshold=args.fail_on_threshold,
    )
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote report artifacts to {args.output_dir}")

    if args.fail_on_threshold and overall_status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
