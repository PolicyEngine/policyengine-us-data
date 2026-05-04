"""Diagnose state ACA target fit for local-area H5 files.

Usage:
    python -m policyengine_us_data.calibration.diagnose_aca_state_targets
    python -m policyengine_us_data.calibration.diagnose_aca_state_targets \
        --run-id 1.89.1_a2f3bb36_20260430_205113
    python -m policyengine_us_data.calibration.diagnose_aca_state_targets \
        --h5-prefix /tmp/staging/states --states NY,MN,FL,TX
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_us_data.calibration.calibration_utils import STATE_CODES
from policyengine_us_data.datasets.cps.enhanced_cps import (
    _get_base_aca_takeup,
    _get_period_array,
    create_aca_2025_takeup_override,
)
from policyengine_us_data.storage import STORAGE_FOLDER

DEFAULT_HF_PREFIX = "hf://policyengine/policyengine-us-data/staging/states"
STATE_ABBRS = sorted(STATE_CODES.values())
OPTIONAL_BLOCKERS = {
    "marketplace_people": "has_marketplace_health_coverage",
    "premium_people": "pays_aca_premium",
    "medicaid_blocked_people": "is_medicaid_eligible",
    "chip_blocked_people": "is_chip_eligible",
    "eshi_blocked_people": "is_aca_eshi_eligible",
    "medicare_blocked_people": "is_medicare_eligible",
    "bhp_blocked_people": "is_basic_health_program_eligible",
    "immigration_blocked_people": "is_aca_ptc_immigration_status_eligible",
    "taxpayer_tin_blocked_people": "taxpayer_has_tin",
}


def _array(values, dtype=None) -> np.ndarray:
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(values, dtype=dtype)


def _state_code_array(values) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ]
    )


def _weighted_count(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(np.asarray(mask, dtype=np.float64), weights))


def _weighted_sum(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    return float(np.dot(np.where(mask, values, 0), weights))


def _percent_error(value: float, target: float) -> float:
    if target == 0:
        return np.nan
    return 100 * (value - target) / target


def _target_path(period: int) -> Path:
    return (
        STORAGE_FOLDER
        / "calibration_targets"
        / f"aca_spending_and_enrollment_{period}.csv"
    )


def _load_targets(period: int) -> pd.DataFrame:
    targets = pd.read_csv(_target_path(period))
    targets["annual_spending"] = targets["spending"] * 12
    return targets


def _resolve_h5_path(prefix: str, state: str) -> str:
    if prefix.startswith("hf://"):
        return f"{prefix.rstrip('/')}/{state}.h5"
    return str(Path(prefix) / f"{state}.h5")


def _parse_states(states: str) -> list[str]:
    if states == "":
        return STATE_ABBRS
    return [state.strip().upper() for state in states.split(",") if state.strip()]


def _person_tax_unit_indices(
    person_tax_unit_ids: np.ndarray,
    tax_unit_ids: np.ndarray,
) -> np.ndarray:
    tax_unit_id_to_idx = {
        int(tax_unit_id): idx for idx, tax_unit_id in enumerate(tax_unit_ids)
    }
    return np.array(
        [tax_unit_id_to_idx[int(tax_unit_id)] for tax_unit_id in person_tax_unit_ids],
        dtype=np.int64,
    )


def _optional_person_bool(sim, variable: str, period: int) -> np.ndarray | None:
    try:
        return _array(
            sim.calculate(
                variable,
                map_to="person",
                period=period,
                use_weights=False,
            ),
            dtype=bool,
        )
    except Exception:
        return None


def _delete_if_cached(sim, variable: str) -> None:
    try:
        sim.delete_arrays(variable)
    except Exception:
        pass


def _assigned_aca_spending(
    sim,
    period: int,
    takeup: np.ndarray,
    household_weights: np.ndarray,
    household_in_state: np.ndarray,
) -> float:
    sim.set_input("takes_up_aca_if_eligible", period, takeup.astype(bool, copy=False))
    _delete_if_cached(sim, "assigned_aca_ptc")
    assigned_aca_ptc = _array(
        sim.calculate(
            "assigned_aca_ptc",
            map_to="household",
            period=period,
            use_weights=False,
        ),
        dtype=np.float64,
    )
    return _weighted_sum(assigned_aca_ptc, household_weights, household_in_state)


def _diagnose_state(
    state: str,
    h5_path: str,
    targets_by_state: pd.DataFrame,
    period: int,
):
    from policyengine_us import Microsimulation

    target_row = targets_by_state.loc[state]
    target_enrollment = float(target_row.enrollment)
    target_annual_spending = float(target_row.annual_spending)

    sim = Microsimulation(dataset=h5_path)
    data = sim.dataset.load_dataset()
    base_year = int(str(sim.default_calculation_period))

    tax_unit_ids = _get_period_array(data["tax_unit_id"], base_year)
    person_tax_unit_ids = _get_period_array(data["person_tax_unit_id"], base_year)
    person_tax_unit_idx = _person_tax_unit_indices(
        person_tax_unit_ids=person_tax_unit_ids,
        tax_unit_ids=tax_unit_ids,
    )
    base_takeup = _get_base_aca_takeup(
        data=data,
        base_year=base_year,
        tax_unit_count=len(tax_unit_ids),
    )

    if "household_weight" in data:
        sim.set_input(
            "household_weight",
            base_year,
            _get_period_array(data["household_weight"], base_year).astype(np.float32),
        )

    sim.set_input(
        "takes_up_aca_if_eligible",
        period,
        np.ones(len(tax_unit_ids), dtype=bool),
    )
    _delete_if_cached(sim, "aca_ptc")
    _delete_if_cached(sim, "assigned_aca_ptc")

    person_weights = _array(
        sim.calculate("person_weight", period=period, use_weights=False),
        dtype=np.float64,
    )
    household_weights = _array(
        sim.calculate(
            "household_weight",
            map_to="household",
            period=period,
            use_weights=False,
        ),
        dtype=np.float64,
    )
    person_state = _state_code_array(
        _array(
            sim.calculate(
                "state_code",
                map_to="person",
                period=period,
                use_weights=False,
            )
        )
    )
    household_state = _state_code_array(
        _array(
            sim.calculate(
                "state_code",
                map_to="household",
                period=period,
                use_weights=False,
            )
        )
    )
    person_in_state = person_state == state
    household_in_state = household_state == state

    aca_ptc_person = _array(
        sim.calculate(
            "aca_ptc",
            map_to="person",
            period=period,
            use_weights=False,
        ),
        dtype=np.float64,
    )
    aca_ptc_household = _array(
        sim.calculate(
            "aca_ptc",
            map_to="household",
            period=period,
            use_weights=False,
        ),
        dtype=np.float64,
    )
    aca_ptc_tax_unit = _array(
        sim.calculate(
            "aca_ptc",
            period=period,
            use_weights=False,
        ),
        dtype=np.float64,
    )
    tax_unit_weights = _array(
        sim.calculate(
            "tax_unit_weight",
            period=period,
            use_weights=False,
        ),
        dtype=np.float64,
    )
    is_aca_eligible = _array(
        sim.calculate(
            "is_aca_ptc_eligible",
            map_to="person",
            period=period,
            use_weights=False,
        ),
        dtype=bool,
    )

    potential = person_in_state & (aca_ptc_person > 0)
    loss_concept = potential & is_aca_eligible
    base_selected = potential & base_takeup[person_tax_unit_idx]
    adjusted_takeup = create_aca_2025_takeup_override(
        base_takeup=base_takeup,
        person_enrolled_if_takeup=aca_ptc_person > 0,
        person_weights=person_weights,
        person_tax_unit_ids=person_tax_unit_ids,
        tax_unit_ids=tax_unit_ids,
        person_state_codes=person_state,
        target_people_by_state={state: target_enrollment},
        tax_unit_aca_ptc=aca_ptc_tax_unit,
        tax_unit_weights=tax_unit_weights,
        target_spending_by_state={state: target_annual_spending},
    )
    adjusted_selected = potential & adjusted_takeup[person_tax_unit_idx]
    potential_people = _weighted_count(potential, person_weights)
    base_selected_people = _weighted_count(base_selected, person_weights)
    adjusted_selected_people = _weighted_count(adjusted_selected, person_weights)
    loss_concept_people = _weighted_count(loss_concept, person_weights)

    row = {
        "state": state,
        "status": "ok",
        "h5_path": h5_path,
        "target_enrollment": target_enrollment,
        "target_annual_spending": target_annual_spending,
        "potential_people": potential_people,
        "potential_gap": potential_people - target_enrollment,
        "potential_error_pct": _percent_error(potential_people, target_enrollment),
        "loss_concept_people": loss_concept_people,
        "loss_concept_error_pct": _percent_error(
            loss_concept_people,
            target_enrollment,
        ),
        "base_selected_people": base_selected_people,
        "base_selected_error_pct": _percent_error(
            base_selected_people,
            target_enrollment,
        ),
        "adjusted_selected_people": adjusted_selected_people,
        "adjusted_selected_error_pct": _percent_error(
            adjusted_selected_people,
            target_enrollment,
        ),
        "aca_ptc_spending": _weighted_sum(
            aca_ptc_household,
            household_weights,
            household_in_state,
        ),
        "base_assigned_aca_ptc_spending": _assigned_aca_spending(
            sim,
            period,
            base_takeup,
            household_weights,
            household_in_state,
        ),
        "adjusted_assigned_aca_ptc_spending": _assigned_aca_spending(
            sim,
            period,
            adjusted_takeup,
            household_weights,
            household_in_state,
        ),
    }
    row["aca_ptc_spending_error_pct"] = _percent_error(
        row["aca_ptc_spending"],
        target_annual_spending,
    )
    row["adjusted_assigned_spending_error_pct"] = _percent_error(
        row["adjusted_assigned_aca_ptc_spending"],
        target_annual_spending,
    )

    for column, variable in OPTIONAL_BLOCKERS.items():
        values = _optional_person_bool(sim, variable, period)
        if values is None:
            row[column] = np.nan
            row[f"marketplace_{column}"] = np.nan
            continue
        if variable in {
            "is_aca_ptc_immigration_status_eligible",
            "taxpayer_has_tin",
        }:
            values = ~values
        row[column] = _weighted_count(person_in_state & values, person_weights)
        marketplace = _optional_person_bool(
            sim,
            "has_marketplace_health_coverage",
            period,
        )
        if marketplace is None:
            row[f"marketplace_{column}"] = np.nan
        else:
            row[f"marketplace_{column}"] = _weighted_count(
                person_in_state & marketplace & values,
                person_weights,
            )

    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose state ACA enrollment and spending target fit."
    )
    parser.add_argument(
        "--h5-prefix",
        "--hf-prefix",
        default=DEFAULT_HF_PREFIX,
        help=f"Path prefix for state H5 files (default: {DEFAULT_HF_PREFIX})",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run ID to scope HF staging prefix (e.g. staging/{run_id}/states/...)",
    )
    parser.add_argument(
        "--states",
        default="",
        help="Comma-separated states to diagnose. Defaults to all states.",
    )
    parser.add_argument("--period", type=int, default=2025)
    parser.add_argument(
        "--output",
        default="aca_state_diagnostics.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    if args.run_id and args.h5_prefix == DEFAULT_HF_PREFIX:
        args.h5_prefix = (
            f"hf://policyengine/policyengine-us-data/staging/{args.run_id}/states"
        )

    targets = _load_targets(args.period).set_index("state")
    states = _parse_states(args.states)
    rows = []
    for index, state in enumerate(states, start=1):
        h5_path = _resolve_h5_path(args.h5_prefix, state)
        print(f"[{index}/{len(states)}] {state}...", end=" ", flush=True)
        try:
            rows.append(
                _diagnose_state(
                    state=state,
                    h5_path=h5_path,
                    targets_by_state=targets,
                    period=args.period,
                )
            )
            print("OK")
        except Exception as exc:
            print(f"FAILED: {exc}")
            rows.append(
                {
                    "state": state,
                    "status": "failed",
                    "h5_path": h5_path,
                    "error": str(exc),
                }
            )
        gc.collect()

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    ok = df[df["status"] == "ok"].copy()

    if not ok.empty:
        summary_columns = [
            "state",
            "target_enrollment",
            "potential_people",
            "adjusted_selected_people",
            "adjusted_selected_error_pct",
            "aca_ptc_spending_error_pct",
            "adjusted_assigned_spending_error_pct",
        ]
        print("\nACA state diagnostics:")
        print(
            ok[summary_columns]
            .sort_values("adjusted_selected_error_pct", key=np.abs, ascending=False)
            .to_string(index=False)
        )

        shortages = ok.sort_values("potential_gap").head(10)
        print("\nLargest potential shortfalls:")
        print(
            shortages[
                [
                    "state",
                    "target_enrollment",
                    "potential_people",
                    "potential_gap",
                    "potential_error_pct",
                ]
            ].to_string(index=False)
        )

    failures = df[df["status"] != "ok"]
    if not failures.empty:
        print("\nFailures:")
        print(failures[["state", "error"]].to_string(index=False))

    print(f"\nSaved diagnostics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
