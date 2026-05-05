from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from policyengine_us import Microsimulation

from calibration import (
    _build_constraint_dataframe_and_controls,
    assess_nonnegative_feasibility,
)
from calibration_profiles import approximate_window_for_year, get_profile
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
from support_augmentation import build_augmented_dataset


DEFAULT_DATASET = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
BASE_YEAR = 2024


def _evaluate_dataset(
    *,
    dataset: str | object,
    dataset_label: str,
    year: int,
    profile_name: str,
) -> dict[str, object]:
    profile = get_profile(profile_name)
    sim = Microsimulation(dataset=dataset)
    target_matrix = load_ssa_age_projections(start_year=year, end_year=year)
    n_ages = target_matrix.shape[0]
    X, _, _ = build_household_age_matrix(sim, n_ages=n_ages)

    approximate_window = approximate_window_for_year(profile, year)
    age_bucket_size = (
        approximate_window.age_bucket_size if approximate_window is not None else None
    )
    if age_bucket_size and age_bucket_size > 1:
        age_bins = build_age_bins(n_ages=n_ages, bucket_size=age_bucket_size)
        X_current = aggregate_household_age_matrix(X, age_bins)
        y_target = aggregate_age_targets(target_matrix, age_bins)[:, 0]
    else:
        X_current = X
        y_target = target_matrix[:, 0]
        age_bucket_size = 1

    household_series = sim.calculate("household_id", period=year, map_to="household")
    baseline_weights = household_series.weights.values

    ss_values = None
    ss_target = None
    if profile.use_ss:
        ss_values = sim.calculate(
            "social_security", period=year, map_to="household"
        ).values
        ss_target = load_ssa_benefit_projections(year)

    payroll_values = None
    payroll_target = None
    if profile.use_payroll:
        payroll_values = (
            sim.calculate(
                "taxable_earnings_for_social_security",
                period=year,
                map_to="household",
            ).values
            + sim.calculate(
                "social_security_taxable_self_employment_income",
                period=year,
                map_to="household",
            ).values
        )
        payroll_target = load_taxable_payroll_projections(year)

    oasdi_tob_values = None
    oasdi_tob_target = None
    hi_tob_values = None
    hi_tob_target = None
    if profile.use_tob:
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
        oasdi_tob_target = load_oasdi_tob_projections(year)
        hi_tob_target = load_hi_tob_projections(year)

    aux_df, controls = _build_constraint_dataframe_and_controls(
        X_current,
        y_target,
        ss_values=ss_values,
        ss_target=ss_target,
        payroll_values=payroll_values,
        payroll_target=payroll_target,
        oasdi_tob_values=oasdi_tob_values,
        oasdi_tob_target=oasdi_tob_target,
        hi_tob_values=hi_tob_values,
        hi_tob_target=hi_tob_target,
        n_ages=X_current.shape[1],
    )
    targets = np.array(list(controls.values()), dtype=float)
    feasibility = assess_nonnegative_feasibility(
        aux_df.to_numpy(dtype=float),
        targets,
    )

    return {
        "dataset": dataset_label,
        "year": year,
        "profile": profile.name,
        "target_source": get_long_term_target_source(),
        "household_count": int(len(baseline_weights)),
        "age_bucket_size": int(age_bucket_size),
        "constraint_count": int(len(targets)),
        "best_case_max_pct_error": feasibility["best_case_max_pct_error"],
        "feasibility_status": feasibility["status"],
        "feasibility_message": feasibility["message"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare late-year nonnegative feasibility before and after support augmentation."
        )
    )
    parser.add_argument("year", type=int, help="Projection year to evaluate.")
    parser.add_argument(
        "--profile",
        default="ss-payroll",
        help="Calibration profile to evaluate.",
    )
    parser.add_argument(
        "--target-source",
        default="trustees_2025_current_law",
        help="Named long-run target source package.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Base dataset path or HF reference.",
    )
    parser.add_argument(
        "--support-augmentation",
        default="late-clone-v1",
        help="Support augmentation profile name.",
    )
    args = parser.parse_args()

    set_long_term_target_source(args.target_source)

    base_result = _evaluate_dataset(
        dataset=args.dataset,
        dataset_label="base",
        year=args.year,
        profile_name=args.profile,
    )
    augmented_dataset, augmentation_report = build_augmented_dataset(
        base_dataset=args.dataset,
        base_year=BASE_YEAR,
        profile=args.support_augmentation,
    )
    augmented_result = _evaluate_dataset(
        dataset=augmented_dataset,
        dataset_label=args.support_augmentation,
        year=args.year,
        profile_name=args.profile,
    )

    report = {
        "year": args.year,
        "profile": args.profile,
        "target_source": args.target_source,
        "augmentation": augmentation_report,
        "results": {
            "base": base_result,
            "augmented": augmented_result,
            "delta_best_case_max_pct_error": (
                None
                if base_result["best_case_max_pct_error"] is None
                or augmented_result["best_case_max_pct_error"] is None
                else augmented_result["best_case_max_pct_error"]
                - base_result["best_case_max_pct_error"]
            ),
            "delta_household_count": (
                augmented_result["household_count"] - base_result["household_count"]
            ),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
