from __future__ import annotations

import json
import sys
import types

import pytest

from scripts import run_publication_preflight as preflight


class FakeSimulation:
    pass


def _patch_fast_simulation(monkeypatch, *, spm=0.125, employment_income=None):
    if employment_income is None:
        employment_income = preflight.BEA_NIPA_WAGES_AND_SALARIES_2024

    monkeypatch.setattr(preflight, "load_simulation", lambda _: FakeSimulation())
    monkeypatch.setattr(preflight, "calculate_baseline_spm", lambda *_: spm)
    monkeypatch.setattr(
        preflight,
        "calculate_employment_income",
        lambda *_: employment_income,
    )


def test_main_runs_preflight_and_writes_json_summary(tmp_path, monkeypatch):
    enhanced_cps = tmp_path / "enhanced_cps_2024.h5"
    calibration_log = tmp_path / "calibration_log.csv"
    json_output = tmp_path / "preflight.json"
    enhanced_cps.write_text("not a real h5")
    calibration_log.write_text("epoch,target_name,target,abs_error\n")
    calls = []

    monkeypatch.setattr(preflight, "validate_dataset", lambda path: calls.append(path))
    monkeypatch.setattr(preflight, "validate_calibration_log", lambda _: 75.0)
    _patch_fast_simulation(
        monkeypatch,
        spm=0.182489,
        employment_income=preflight.BEA_NIPA_WAGES_AND_SALARIES_2024 * 0.999,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_publication_preflight.py",
            "--enhanced-cps",
            str(enhanced_cps),
            "--calibration-log",
            str(calibration_log),
            "--skip-state-health",
            "--json-output",
            str(json_output),
        ],
    )

    preflight.main()

    assert calls == [enhanced_cps.resolve()]
    result = json.loads(json_output.read_text())
    assert result["enhanced_cps_path"] == str(enhanced_cps.resolve())
    assert result["calibration_log_path"] == str(calibration_log.resolve())
    assert result["baseline_spm"] == pytest.approx(0.182489)
    assert result["dataset_validation_passed"] is True
    assert result["jct_diagnostics_passed"] is True
    assert result["final_epoch_target_share_within_tolerance"] == 75.0
    assert result["aca_state_calibration_passed"] is None
    assert result["medicaid_state_calibration_passed"] is None


def test_main_honors_skip_flags_without_optional_artifacts(tmp_path, monkeypatch):
    enhanced_cps = tmp_path / "enhanced_cps_2024.h5"
    enhanced_cps.write_text("not a real h5")
    monkeypatch.setattr(
        preflight,
        "validate_dataset",
        lambda _: pytest.fail("dataset validation should be skipped"),
    )
    monkeypatch.setattr(
        preflight,
        "validate_calibration_log",
        lambda _: pytest.fail("calibration log validation should be skipped"),
    )
    _patch_fast_simulation(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_publication_preflight.py",
            "--enhanced-cps",
            str(enhanced_cps),
            "--calibration-log",
            str(tmp_path / "missing_calibration_log.csv"),
            "--skip-dataset-validation",
            "--skip-calibration-log",
            "--skip-state-health",
        ],
    )

    preflight.main()


def test_main_runs_state_health_checks_by_default(tmp_path, monkeypatch):
    enhanced_cps = tmp_path / "enhanced_cps_2024.h5"
    calibration_log = tmp_path / "calibration_log.csv"
    enhanced_cps.write_text("not a real h5")
    calibration_log.write_text("epoch,target_name,target,abs_error\n")
    calls = []

    fake_aca_module = types.ModuleType("validation.stage_1.aca_calibration")

    def fake_aca_check(sim, emit):
        calls.append(("aca", sim, emit))

    fake_aca_module.assert_aca_ptc_calibration = fake_aca_check
    monkeypatch.setitem(
        sys.modules,
        "validation.stage_1.aca_calibration",
        fake_aca_module,
    )
    monkeypatch.setattr(preflight, "validate_dataset", lambda _: None)
    monkeypatch.setattr(preflight, "validate_calibration_log", lambda _: 80.0)
    monkeypatch.setattr(
        preflight,
        "validate_medicaid_state_calibration",
        lambda sim: calls.append(("medicaid", sim)),
    )
    _patch_fast_simulation(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_publication_preflight.py",
            "--enhanced-cps",
            str(enhanced_cps),
            "--calibration-log",
            str(calibration_log),
        ],
    )

    preflight.main()

    sim = calls[0][1]
    assert calls == [
        ("aca", sim, print),
        ("medicaid", sim),
    ]


def test_main_raises_for_missing_enhanced_cps(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_publication_preflight.py",
            "--enhanced-cps",
            str(tmp_path / "missing.h5"),
            "--skip-dataset-validation",
            "--skip-calibration-log",
            "--skip-state-health",
        ],
    )

    with pytest.raises(FileNotFoundError):
        preflight.main()
