from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
import re
import sys

import numpy as np

from policyengine_us import Microsimulation

from calibration import build_calibration_audit, calibrate_weights
from calibration_profiles import (
    approximate_window_for_year,
    classify_calibration_quality,
    get_profile,
    validate_calibration_audit,
)
from projection_utils import (
    aggregate_age_targets,
    aggregate_household_age_matrix,
    build_age_bins,
    build_household_age_matrix,
)
from ssa_data import (
    get_long_term_target_source,
    load_hi_tob_projections,
    load_oasdi_tob_projections,
    load_ssa_age_projections,
    load_ssa_benefit_projections,
    load_taxable_payroll_projections,
    set_long_term_target_source,
)

try:
    from samplics.weighting import SampleWeight
except ImportError:  # pragma: no cover - only needed for greg profiles
    SampleWeight = None


DEFAULT_BASE_DATASET_PATH = (
    "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess publishable microsimulation horizon quality for selected years."
        )
    )
    parser.add_argument(
        "--profile",
        default="ss-payroll-tob",
        help="Named calibration profile to assess.",
    )
    parser.add_argument(
        "--target-source",
        default=get_long_term_target_source(),
        help="Named long-run target source package.",
    )
    parser.add_argument(
        "--years",
        default="2075,2080,2085,2090,2095,2100",
        help="Comma-separated years to assess.",
    )
    parser.add_argument(
        "--base-dataset",
        default=DEFAULT_BASE_DATASET_PATH,
        help="Base microsimulation dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path. Defaults to stdout.",
    )
    return parser.parse_args()


def parse_years(raw: str) -> list[int]:
    years = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not years:
        raise ValueError("At least one year must be provided.")
    return sorted(set(years))


def maybe_build_calibrator(method: str):
    if method != "greg":
        return None
    if SampleWeight is None:
        raise ImportError(
            "samplics is required for GREG calibration. "
            "Install with: pip install policyengine-us-data[calibration]"
        )
    return SampleWeight()


def benchmark_tob_values(
    year: int,
    weights: np.ndarray,
    *,
    oasdi_tob_values: np.ndarray | None,
    hi_tob_values: np.ndarray | None,
) -> dict[str, float] | None:
    if oasdi_tob_values is None or hi_tob_values is None:
        return None

    oasdi_target = float(load_oasdi_tob_projections(year))
    oasdi_achieved = float(np.sum(oasdi_tob_values * weights))
    hi_target = float(load_hi_tob_projections(year))
    hi_achieved = float(np.sum(hi_tob_values * weights))

    return {
        "oasdi_tob_benchmark_pct_error": (
            0.0
            if oasdi_target == 0
            else (oasdi_achieved - oasdi_target) / oasdi_target * 100
        ),
        "hi_tob_benchmark_pct_error": (
            0.0 if hi_target == 0 else (hi_achieved - hi_target) / hi_target * 100
        ),
    }


def assess_years(
    *,
    years: list[int],
    profile_name: str,
    target_source: str,
    base_dataset_path: str,
) -> list[dict[str, object]]:
    profile = get_profile(profile_name)
    if profile.use_h6_reform:
        raise NotImplementedError(
            "assess_publishable_horizon.py does not yet support H6-calibrated profiles."
        )

    set_long_term_target_source(target_source)
    calibrator = maybe_build_calibrator(profile.calibration_method)

    start_year = min(years)
    end_year = max(years)
    target_matrix = load_ssa_age_projections(start_year=start_year, end_year=end_year)
    n_ages = target_matrix.shape[0]

    sim = Microsimulation(dataset=base_dataset_path)
    X, _, _ = build_household_age_matrix(sim, n_ages)
    del sim
    gc.collect()

    aggregated_age_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    for year in years:
        print(f"[assess_publishable_horizon] year={year}", file=sys.stderr, flush=True)
        year_idx = year - start_year
        sim = Microsimulation(dataset=base_dataset_path)

        household_microseries = sim.calculate("household_id", map_to="household")
        baseline_weights = household_microseries.weights.values

        ss_values = None
        ss_target = None
        if profile.use_ss:
            ss_values = sim.calculate(
                "social_security",
                period=year,
                map_to="household",
            ).values
            ss_target = load_ssa_benefit_projections(year)

        payroll_values = None
        payroll_target = None
        if profile.use_payroll:
            taxable_wages = sim.calculate(
                "taxable_earnings_for_social_security",
                period=year,
                map_to="household",
            ).values
            taxable_self_employment = sim.calculate(
                "social_security_taxable_self_employment_income",
                period=year,
                map_to="household",
            ).values
            payroll_values = taxable_wages + taxable_self_employment
            payroll_target = load_taxable_payroll_projections(year)

        oasdi_tob_values = None
        hi_tob_values = None
        if profile.use_tob or profile.benchmark_tob:
            oasdi_tob_values = sim.calculate(
                "tob_revenue_oasdi",
                period=year,
                map_to="household",
            ).values
            hi_tob_values = sim.calculate(
                "tob_revenue_medicare_hi",
                period=year,
                map_to="household",
            ).values

        approximate_window = approximate_window_for_year(profile, year)
        age_bucket_size = (
            approximate_window.age_bucket_size
            if approximate_window is not None and approximate_window.age_bucket_size
            else 1
        )

        if age_bucket_size > 1:
            if age_bucket_size not in aggregated_age_cache:
                age_bins = build_age_bins(n_ages=n_ages, bucket_size=age_bucket_size)
                aggregated_age_cache[age_bucket_size] = (
                    aggregate_household_age_matrix(X, age_bins),
                    aggregate_age_targets(target_matrix, age_bins),
                )
            X_current, aggregated_target_matrix = aggregated_age_cache[age_bucket_size]
            y_target = aggregated_target_matrix[:, year_idx]
        else:
            X_current = X
            y_target = target_matrix[:, year_idx]

        try:
            weights, iterations, calibration_event = calibrate_weights(
                X=X_current,
                y_target=y_target,
                baseline_weights=baseline_weights,
                method=profile.calibration_method,
                calibrator=calibrator,
                ss_values=ss_values,
                ss_target=ss_target,
                payroll_values=payroll_values,
                payroll_target=payroll_target,
                h6_income_values=None,
                h6_revenue_target=None,
                oasdi_tob_values=oasdi_tob_values if profile.use_tob else None,
                oasdi_tob_target=load_oasdi_tob_projections(year)
                if profile.use_tob
                else None,
                hi_tob_values=hi_tob_values if profile.use_tob else None,
                hi_tob_target=load_hi_tob_projections(year)
                if profile.use_tob
                else None,
                n_ages=X_current.shape[1],
                max_iters=100,
                tol=1e-6,
                verbose=False,
                allow_fallback_to_ipf=profile.allow_greg_fallback,
                allow_approximate_entropy=approximate_window is not None,
                approximate_max_error_pct=(
                    approximate_window.max_constraint_error_pct
                    if approximate_window is not None
                    else None
                ),
            )
        except RuntimeError as error:
            row: dict[str, object] = {
                "year": year,
                "target_source": target_source,
                "profile": profile.name,
                "calibration_quality": "failed",
                "approximation_method": "runtime_error",
                "iterations": None,
                "age_bucket_size": age_bucket_size,
                "window_max_constraint_error_pct": (
                    approximate_window.max_constraint_error_pct
                    if approximate_window is not None
                    else profile.max_constraint_error_pct
                ),
                "window_max_age_error_pct": (
                    approximate_window.max_age_error_pct
                    if approximate_window is not None
                    else profile.max_age_error_pct
                ),
                "max_constraint_pct_error": None,
                "age_max_pct_error": None,
                "positive_weight_count": None,
                "effective_sample_size": None,
                "top_10_weight_share_pct": None,
                "top_100_weight_share_pct": None,
                "negative_weight_pct": None,
                "validation_passed": False,
                "validation_issue_count": 1,
                "validation_issues": str(error),
                "runtime_error": str(error),
            }
            best_case_match = re.search(r"([0-9.]+)%\s*>\s*([0-9.]+)%", str(error))
            if best_case_match:
                row["reported_best_case_constraint_error_pct"] = float(
                    best_case_match.group(1)
                )
                row["reported_allowed_constraint_error_pct"] = float(
                    best_case_match.group(2)
                )
            rows.append(row)
            del sim
            gc.collect()
            continue

        audit = build_calibration_audit(
            X=X_current,
            y_target=y_target,
            weights=weights,
            baseline_weights=baseline_weights,
            calibration_event=calibration_event,
            ss_values=ss_values,
            ss_target=ss_target,
            payroll_values=payroll_values,
            payroll_target=payroll_target,
            h6_income_values=None,
            h6_revenue_target=None,
            oasdi_tob_values=oasdi_tob_values if profile.use_tob else None,
            oasdi_tob_target=load_oasdi_tob_projections(year)
            if profile.use_tob
            else None,
            hi_tob_values=hi_tob_values if profile.use_tob else None,
            hi_tob_target=load_hi_tob_projections(year) if profile.use_tob else None,
        )
        audit["calibration_quality"] = classify_calibration_quality(
            audit,
            profile,
            year=year,
        )
        audit["age_bucket_size"] = age_bucket_size
        audit["age_bucket_count"] = int(X_current.shape[1])

        validation_issues = validate_calibration_audit(
            audit,
            profile,
            year=year,
        )
        audit["validation_issues"] = validation_issues
        audit["validation_passed"] = not bool(validation_issues)

        row: dict[str, object] = {
            "year": year,
            "target_source": target_source,
            "profile": profile.name,
            "calibration_quality": audit["calibration_quality"],
            "approximation_method": audit.get("approximation_method")
            or audit.get("method_used"),
            "iterations": iterations,
            "age_bucket_size": age_bucket_size,
            "window_max_constraint_error_pct": (
                approximate_window.max_constraint_error_pct
                if approximate_window is not None
                else profile.max_constraint_error_pct
            ),
            "window_max_age_error_pct": (
                approximate_window.max_age_error_pct
                if approximate_window is not None
                else profile.max_age_error_pct
            ),
            "max_constraint_pct_error": audit.get("max_constraint_pct_error"),
            "age_max_pct_error": audit.get("age_max_pct_error"),
            "positive_weight_count": audit.get("positive_weight_count"),
            "effective_sample_size": audit.get("effective_sample_size"),
            "top_10_weight_share_pct": audit.get("top_10_weight_share_pct"),
            "top_100_weight_share_pct": audit.get("top_100_weight_share_pct"),
            "negative_weight_pct": audit.get("negative_weight_pct"),
            "validation_passed": audit["validation_passed"],
            "validation_issue_count": len(validation_issues),
            "validation_issues": "; ".join(validation_issues),
        }

        tob_benchmarks = benchmark_tob_values(
            year,
            weights,
            oasdi_tob_values=oasdi_tob_values,
            hi_tob_values=hi_tob_values,
        )
        if tob_benchmarks is not None:
            row.update(tob_benchmarks)

        rows.append(row)

        del sim
        gc.collect()

    return rows


def write_rows(rows: list[dict[str, object]], output: Path | None) -> None:
    if not rows:
        raise SystemExit("No rows to write.")

    fieldnames = [
        "year",
        "target_source",
        "profile",
        "calibration_quality",
        "approximation_method",
        "iterations",
        "age_bucket_size",
        "window_max_constraint_error_pct",
        "window_max_age_error_pct",
        "max_constraint_pct_error",
        "age_max_pct_error",
        "positive_weight_count",
        "effective_sample_size",
        "top_10_weight_share_pct",
        "top_100_weight_share_pct",
        "negative_weight_pct",
        "validation_passed",
        "validation_issue_count",
        "validation_issues",
        "runtime_error",
        "reported_best_case_constraint_error_pct",
        "reported_allowed_constraint_error_pct",
        "oasdi_tob_benchmark_pct_error",
        "hi_tob_benchmark_pct_error",
    ]

    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
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
    rows = assess_years(
        years=parse_years(args.years),
        profile_name=args.profile,
        target_source=args.target_source,
        base_dataset_path=args.base_dataset,
    )
    write_rows(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
