from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
from policyengine_us import Microsimulation

from calibration import (
    _build_constraint_dataframe_and_controls,
    assess_nonnegative_feasibility,
)
from calibration_profiles import get_profile
from projection_utils import build_household_age_matrix
from ssa_data import (
    load_hi_tob_projections,
    load_oasdi_tob_projections,
    load_ssa_age_projections,
    load_ssa_benefit_projections,
    load_taxable_payroll_projections,
)


DATASET_OPTIONS = {
    "enhanced_cps_2024": {
        "path": "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        "base_year": 2024,
    },
}
SELECTED_DATASET = "enhanced_cps_2024"
BASE_DATASET_PATH = DATASET_OPTIONS[SELECTED_DATASET]["path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess the nonnegative calibration frontier for a named long-term "
            "profile by solving the minimax relative-error LP."
        )
    )
    parser.add_argument(
        "--profile",
        default="ss-payroll-tob",
        help="Named calibration profile to assess.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2035,
        help="First year to assess when --years is not provided.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2100,
        help="Last year to assess when --years is not provided.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Year increment when --years is not provided.",
    )
    parser.add_argument(
        "--years",
        help="Optional comma-separated list of explicit years to assess.",
    )
    parser.add_argument(
        "--output",
        help="Optional CSV path for the frontier table.",
    )
    return parser.parse_args()


def parse_years(args: argparse.Namespace) -> list[int]:
    if args.years:
        return [int(value.strip()) for value in args.years.split(",") if value.strip()]
    return list(range(args.start_year, args.end_year + 1, args.step))


def reorder_to_households(values, order, n_households: int) -> np.ndarray:
    ordered = np.zeros(n_households, dtype=float)
    ordered[order] = np.asarray(values, dtype=float)
    return ordered


def build_constraint_inputs(year: int, hh_id_to_idx: dict, n_households: int, profile) -> dict:
    sim = Microsimulation(dataset=BASE_DATASET_PATH)
    if profile.use_h6_reform:
        raise NotImplementedError(
            "Frontier assessment for H6-enabled profiles is not yet implemented."
        )
    household_ids = sim.calculate("household_id", period=year, map_to="household").values
    if len(household_ids) != n_households:
        raise ValueError(
            f"Household count mismatch for {year}: {len(household_ids)} vs {n_households}"
        )
    order = np.fromiter(
        (hh_id_to_idx[hh_id] for hh_id in household_ids),
        dtype=int,
        count=len(household_ids),
    )
    inputs: dict[str, np.ndarray | float | None] = {
        "ss_values": None,
        "ss_target": None,
        "payroll_values": None,
        "payroll_target": None,
        "h6_income_values": None,
        "h6_revenue_target": None,
        "oasdi_tob_values": None,
        "oasdi_tob_target": None,
        "hi_tob_values": None,
        "hi_tob_target": None,
    }

    if profile.use_ss:
        inputs["ss_values"] = reorder_to_households(
            sim.calculate("social_security", period=year, map_to="household").values,
            order,
            n_households,
        )
        inputs["ss_target"] = load_ssa_benefit_projections(year)

    if profile.use_payroll:
        inputs["payroll_values"] = reorder_to_households(
            sim.calculate(
                "taxable_earnings_for_social_security",
                period=year,
                map_to="household",
            ).values
            + sim.calculate(
                "social_security_taxable_self_employment_income",
                period=year,
                map_to="household",
            ).values,
            order,
            n_households,
        )
        inputs["payroll_target"] = load_taxable_payroll_projections(year)

    if profile.use_tob:
        inputs["oasdi_tob_values"] = reorder_to_households(
            sim.calculate(
                "tob_revenue_oasdi",
                period=year,
                map_to="household",
            ).values,
            order,
            n_households,
        )
        inputs["hi_tob_values"] = reorder_to_households(
            sim.calculate(
                "tob_revenue_medicare_hi",
                period=year,
                map_to="household",
            ).values,
            order,
            n_households,
        )
        inputs["oasdi_tob_target"] = load_oasdi_tob_projections(year)
        inputs["hi_tob_target"] = load_hi_tob_projections(year)

    del sim
    gc.collect()
    return inputs


def main() -> int:
    args = parse_args()
    years = parse_years(args)
    if not years:
        raise ValueError("No years requested.")

    profile = get_profile(args.profile)
    start_year = min(years)
    end_year = max(years)
    target_matrix = load_ssa_age_projections(start_year=start_year, end_year=end_year)

    base_sim = Microsimulation(dataset=BASE_DATASET_PATH)
    X, household_ids_unique, hh_id_to_idx = build_household_age_matrix(
        base_sim,
        n_ages=target_matrix.shape[0],
    )
    del base_sim
    gc.collect()

    rows: list[dict[str, object]] = []
    print(
        f"Assessing profile {profile.name!r} for {len(years)} years "
        f"using {len(household_ids_unique):,} fixed households."
    )
    for year in years:
        year_idx = year - start_year
        y_target = target_matrix[:, year_idx]
        inputs = build_constraint_inputs(
            year,
            hh_id_to_idx,
            len(household_ids_unique),
            profile,
        )
        aux_df, controls = _build_constraint_dataframe_and_controls(
            X,
            y_target,
            n_ages=target_matrix.shape[0],
            **inputs,
        )
        targets = np.array(list(controls.values()), dtype=float)
        feasibility = assess_nonnegative_feasibility(
            aux_df.to_numpy(dtype=float),
            targets,
        )
        best_case = feasibility["best_case_max_pct_error"]
        within_tolerance = (
            best_case is not None and best_case <= profile.max_constraint_error_pct
        )
        row = {
            "year": year,
            "profile": profile.name,
            "best_case_max_pct_error": best_case,
            "within_profile_tolerance": within_tolerance,
            "quality": "exact" if within_tolerance else "approximate",
            "status": feasibility["status"],
            "message": feasibility["message"],
        }
        rows.append(row)
        best_case_display = "n/a" if best_case is None else f"{best_case:.3f}%"
        print(
            f"{year}: best-case max error {best_case_display} -> "
            f"{row['quality']}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
