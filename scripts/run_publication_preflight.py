"""Run fast artifact checks before launching the full publication pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from policyengine_core.data import Dataset
from policyengine_us import Microsimulation

from policyengine_us_data.db.etl_national_targets import (
    BEA_NIPA_WAGES_AND_SALARIES_2024,
)
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.storage.upload_completed_datasets import (
    DatasetValidationError,
    validate_dataset,
)
from policyengine_us_data.utils import ABSOLUTE_ERROR_SCALE_TARGETS

DEFAULT_ENHANCED_CPS_PATH = STORAGE_FOLDER / "enhanced_cps_2024.h5"
DEFAULT_CALIBRATION_LOG_PATH = Path("calibration_log.csv")
DEFAULT_PERIOD = 2024
DEFAULT_EMPLOYMENT_TOLERANCE = 0.01
DEFAULT_FINAL_EPOCH_TARGET_SHARE = 60.0
MEDICAID_VALIDATION_PERIOD = 2025
MEDICAID_STATE_TOLERANCE = 10.0
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PreflightResult:
    enhanced_cps_path: str
    calibration_log_path: str | None
    period: int
    baseline_spm: float
    employment_income: float
    employment_income_target: float
    employment_income_relative_error: float
    dataset_validation_passed: bool
    jct_diagnostics_passed: bool | None
    final_epoch_target_share_within_tolerance: float | None
    aca_state_calibration_passed: bool | None
    medicaid_state_calibration_passed: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a built enhanced CPS artifact before spending publication "
            "time on local-area outputs."
        )
    )
    parser.add_argument(
        "--enhanced-cps",
        type=Path,
        default=DEFAULT_ENHANCED_CPS_PATH,
        help="Path to enhanced_cps_2024.h5.",
    )
    parser.add_argument(
        "--calibration-log",
        type=Path,
        default=DEFAULT_CALIBRATION_LOG_PATH,
        help="Path to calibration_log.csv.",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=DEFAULT_PERIOD,
        help="PolicyEngine year for SPM and income aggregates.",
    )
    parser.add_argument(
        "--employment-tolerance",
        type=float,
        default=DEFAULT_EMPLOYMENT_TOLERANCE,
        help="Allowed relative error against the BEA NIPA wages target.",
    )
    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help="Skip upload-contract validation of the enhanced CPS H5.",
    )
    parser.add_argument(
        "--skip-calibration-log",
        action="store_true",
        help="Skip calibration_log.csv diagnostics.",
    )
    parser.add_argument(
        "--skip-state-health",
        action="store_true",
        help="Skip ACA and Medicaid state calibration checks.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a JSON summary.",
    )
    return parser.parse_args()


def load_simulation(path: Path) -> Microsimulation:
    return Microsimulation(dataset=Dataset.from_file(path))


def ensure_repo_root_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def calculate_baseline_spm(sim: Microsimulation, period: int) -> float:
    try:
        return float(sim.calculate("in_poverty", period, map_to="person").mean())
    except ValueError:
        return float(sim.calculate("person_in_poverty", period, map_to="person").mean())


def calculate_employment_income(sim: Microsimulation, period: int) -> float:
    return float(sim.calculate("employment_income", period, map_to="person").sum())


def validate_employment_income(
    value: float,
    *,
    target: float,
    tolerance: float,
) -> float:
    relative_error = (value - target) / target
    if abs(relative_error) > tolerance:
        raise AssertionError(
            "employment_income is outside the NIPA wages tolerance: "
            f"value={value:,.0f}, target={target:,.0f}, "
            f"relative_error={relative_error:.4%}, tolerance={tolerance:.2%}"
        )
    return relative_error


def final_epoch_target_share_within_tolerance(calibration_log: pd.DataFrame) -> float:
    final_epoch = calibration_log["epoch"].max()
    final_rows = calibration_log[calibration_log["epoch"] == final_epoch].copy()
    if final_rows.empty:
        raise AssertionError("No final-epoch calibration diagnostics found.")

    tolerance = 0.10 * final_rows["target"].abs()
    for target_name, scale in ABSOLUTE_ERROR_SCALE_TARGETS.items():
        tolerance.loc[final_rows["target_name"] == target_name] = 0.10 * scale
    return float((final_rows["abs_error"] <= tolerance).mean() * 100)


def validate_calibration_log(path: Path) -> float:
    ensure_repo_root_on_path()
    from validation.stage_1.jct_calibration import (
        assert_no_unexpected_high_error_jct_diagnostics,
    )

    calibration_log = pd.read_csv(path)
    assert_no_unexpected_high_error_jct_diagnostics(calibration_log)
    share = final_epoch_target_share_within_tolerance(calibration_log)
    if share <= DEFAULT_FINAL_EPOCH_TARGET_SHARE:
        raise AssertionError(
            "Too few final-epoch calibration targets are within tolerance: "
            f"{share:.1f}% <= {DEFAULT_FINAL_EPOCH_TARGET_SHARE:.1f}%"
        )
    return share


def validate_medicaid_state_calibration(sim: Microsimulation) -> None:
    targets_path = (
        Path("policyengine_us_data/storage/calibration_targets")
        / f"medicaid_enrollment_{MEDICAID_VALIDATION_PERIOD}.csv"
    )
    targets = pd.read_csv(targets_path)
    state_code_hh = sim.calculate("state_code", map_to="household").values
    medicaid_enrolled = sim.calculate(
        "medicaid_enrolled",
        MEDICAID_VALIDATION_PERIOD,
        map_to="household",
    )

    failures = []
    for row in targets.itertuples(index=False):
        target_enrollment = float(row.enrollment)
        simulated = float(medicaid_enrolled[state_code_hh == row.state].sum())
        pct_error = (
            np.inf
            if target_enrollment <= 0
            else abs(simulated - target_enrollment) / target_enrollment
        )
        if pct_error > MEDICAID_STATE_TOLERANCE:
            failures.append(
                f"{row.state}: simulated {simulated:,.0f}, "
                f"target {target_enrollment:,.0f}, error {pct_error:.2%}"
            )

    if failures:
        raise AssertionError(
            "One or more Medicaid state targets exceeded tolerance of "
            f"{MEDICAID_STATE_TOLERANCE:.0%}:\n" + "\n".join(failures)
        )


def write_summary(result: PreflightResult, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    enhanced_cps_path = args.enhanced_cps.expanduser().resolve()
    calibration_log_path = args.calibration_log.expanduser().resolve()

    if not enhanced_cps_path.exists():
        raise FileNotFoundError(enhanced_cps_path)

    dataset_validation_passed = False
    if not args.skip_dataset_validation:
        try:
            validate_dataset(enhanced_cps_path)
        except DatasetValidationError:
            raise
        dataset_validation_passed = True

    sim = load_simulation(enhanced_cps_path)
    baseline_spm = calculate_baseline_spm(sim, args.period)
    employment_income = calculate_employment_income(sim, args.period)
    employment_relative_error = validate_employment_income(
        employment_income,
        target=BEA_NIPA_WAGES_AND_SALARIES_2024,
        tolerance=args.employment_tolerance,
    )

    jct_diagnostics_passed = None
    target_share = None
    if not args.skip_calibration_log:
        if not calibration_log_path.exists():
            raise FileNotFoundError(calibration_log_path)
        target_share = validate_calibration_log(calibration_log_path)
        jct_diagnostics_passed = True

    aca_state_calibration_passed = None
    medicaid_state_calibration_passed = None
    if not args.skip_state_health:
        ensure_repo_root_on_path()
        from validation.stage_1.aca_calibration import assert_aca_ptc_calibration

        assert_aca_ptc_calibration(sim, emit=print)
        aca_state_calibration_passed = True
        validate_medicaid_state_calibration(sim)
        medicaid_state_calibration_passed = True

    result = PreflightResult(
        enhanced_cps_path=str(enhanced_cps_path),
        calibration_log_path=(
            str(calibration_log_path) if not args.skip_calibration_log else None
        ),
        period=args.period,
        baseline_spm=baseline_spm,
        employment_income=employment_income,
        employment_income_target=BEA_NIPA_WAGES_AND_SALARIES_2024,
        employment_income_relative_error=employment_relative_error,
        dataset_validation_passed=dataset_validation_passed,
        jct_diagnostics_passed=jct_diagnostics_passed,
        final_epoch_target_share_within_tolerance=target_share,
        aca_state_calibration_passed=aca_state_calibration_passed,
        medicaid_state_calibration_passed=medicaid_state_calibration_passed,
    )
    write_summary(result, args.json_output)

    print("\nPublication preflight passed.")
    print(f"  enhanced CPS: {enhanced_cps_path}")
    print(f"  baseline SPM ({args.period}): {baseline_spm:.6f}")
    print(f"  employment_income ({args.period}): {employment_income:,.0f}")
    print(f"  employment_income vs NIPA wages: {employment_relative_error:.4%}")
    if target_share is not None:
        print(
            f"  final-epoch calibration targets within tolerance: {target_share:.1f}%"
        )


if __name__ == "__main__":
    main()
