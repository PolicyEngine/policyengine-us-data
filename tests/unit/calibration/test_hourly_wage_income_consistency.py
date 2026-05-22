import h5py
import numpy as np

from policyengine_us_data.calibration.sanity_checks import (
    build_hourly_wage_income_consistency_diagnostics,
    run_sanity_checks,
)


def _write_period_dataset(h5, name, values, period=2024):
    group = h5.create_group(name)
    group.create_dataset(str(period), data=np.asarray(values))


def test_hourly_wage_income_consistency_warns_on_overtime_mismatch():
    diagnostics = build_hourly_wage_income_consistency_diagnostics(
        employment_income=np.array([65_000.0, 57_200.0, 40_000.0, 65_000.0]),
        hourly_wage=np.array([25.0, 20.0, 0.0, 25.0]),
        hours_worked_last_week=np.array([50.0, 50.0, 50.0, 50.0]),
        is_paid_hourly=np.array([True, True, True, False]),
        weights=np.array([2.0, 1.0, 1.0, 1.0]),
    )

    by_check = {diagnostic["check"]: diagnostic for diagnostic in diagnostics}

    assert by_check["hourly_wage_income_consistency"]["status"] == "WARN"
    assert by_check["hourly_wage_income_consistency_overtime"]["status"] == "WARN"
    assert (
        "66.7% weighted mismatch share"
        in by_check["hourly_wage_income_consistency_overtime"]["detail"]
    )
    assert (
        "imply annual wages above employment_income"
        in by_check["hourly_wage_income_consistency_overtime"]["detail"]
    )


def test_hourly_wage_income_consistency_passes_when_hourly_facts_reconcile():
    diagnostics = build_hourly_wage_income_consistency_diagnostics(
        employment_income=np.array([57_200.0, 41_600.0]),
        hourly_wage=np.array([20.0, 20.0]),
        hours_worked_last_week=np.array([50.0, 40.0]),
        is_paid_hourly=np.array([True, True]),
    )

    assert [diagnostic["status"] for diagnostic in diagnostics] == ["PASS", "PASS"]


def test_run_sanity_checks_adds_hourly_wage_income_consistency(tmp_path):
    h5_path = tmp_path / "sample.h5"
    with h5py.File(h5_path, "w") as h5:
        _write_period_dataset(h5, "household_weight", [2.0, 1.0])
        _write_period_dataset(h5, "employment_income", [65_000.0, 57_200.0])
        _write_period_dataset(h5, "hourly_wage", [25.0, 20.0])
        _write_period_dataset(h5, "hours_worked_last_week", [50.0, 50.0])
        _write_period_dataset(h5, "is_paid_hourly", [True, True])

    diagnostics = run_sanity_checks(str(h5_path), period=2024)
    by_check = {diagnostic["check"]: diagnostic for diagnostic in diagnostics}

    assert by_check["hourly_wage_income_consistency"]["status"] == "WARN"
    assert by_check["hourly_wage_income_consistency_overtime"]["status"] == "WARN"


def test_run_sanity_checks_keeps_raw_ssi_and_checks_computed_outlays(
    tmp_path, monkeypatch
):
    h5_path = tmp_path / "sample.h5"
    with h5py.File(h5_path, "w") as h5:
        _write_period_dataset(h5, "household_weight", [1.0, 1.0])
        _write_period_dataset(h5, "ssi", [100.0, 0.0])

    monkeypatch.setattr(
        "policyengine_us_data.calibration.sanity_checks._computed_key_monetary_values",
        lambda h5_path, period: {
            "ssi_federal_fiscal_year_outlays": np.array([100.0, np.inf])
        },
    )

    diagnostics = run_sanity_checks(str(h5_path), period=2024)
    by_check = {diagnostic["check"]: diagnostic for diagnostic in diagnostics}

    assert by_check["no_nan_inf_ssi"]["status"] == "PASS"
    assert by_check["no_nan_inf_ssi_federal_fiscal_year_outlays"]["status"] == "FAIL"
