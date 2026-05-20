from pathlib import Path

import pytest

from policyengine_us_data.fit_weights import FitScope, fit_artifacts_for_scope


def test_regional_artifact_specs_match_current_filenames() -> None:
    artifacts = fit_artifacts_for_scope(FitScope.REGIONAL)

    assert artifacts.weights.filename == "calibration_weights.npy"
    assert artifacts.geography.filename == "geography_assignment.npz"
    assert artifacts.run_config.filename == "unified_run_config.json"
    assert artifacts.diagnostics.filename == "unified_diagnostics.csv"
    assert artifacts.epoch_log.filename == "calibration_log.csv"


def test_national_artifact_specs_match_current_filenames() -> None:
    artifacts = fit_artifacts_for_scope("national")

    assert artifacts.weights.filename == "national_calibration_weights.npy"
    assert artifacts.geography.filename == "national_geography_assignment.npz"
    assert artifacts.run_config.filename == "national_unified_run_config.json"
    assert artifacts.diagnostics.filename == "national_unified_diagnostics.csv"
    assert artifacts.epoch_log.filename == "national_calibration_log.csv"


def test_result_key_mappings_cover_current_remote_result_shape() -> None:
    artifacts = fit_artifacts_for_scope(FitScope.REGIONAL)

    assert artifacts.diagnostic_result_filenames() == {
        "log": "unified_diagnostics.csv",
        "cal_log": "calibration_log.csv",
        "config": "unified_run_config.json",
    }
    assert {artifact.result_key for artifact in artifacts.artifact_specs()} == {
        "weights",
        "geography",
        "config",
    }


def test_artifact_paths_are_under_supplied_root() -> None:
    artifacts = fit_artifacts_for_scope(FitScope.NATIONAL)

    assert artifacts.artifact_paths("/pipeline/artifacts/run-1") == [
        Path("/pipeline/artifacts/run-1/national_calibration_weights.npy"),
        Path("/pipeline/artifacts/run-1/national_geography_assignment.npz"),
        Path("/pipeline/artifacts/run-1/national_unified_run_config.json"),
    ]


def test_unknown_artifact_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown fit scope"):
        fit_artifacts_for_scope("zip")
