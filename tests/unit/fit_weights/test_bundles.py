from pathlib import Path

import pytest

from policyengine_us_data.fit_weights import (
    FitScope,
    FittedWeightsInputBundle,
    FittedWeightsOutputBundle,
    MissingFitWeightsOutputError,
)


def test_input_bundle_exposes_calibration_package_identity_path(
    calibration_package_path: Path,
) -> None:
    bundle = FittedWeightsInputBundle(
        scope="regional",
        calibration_package_path=calibration_package_path,
    )

    assert bundle.scope == FitScope.REGIONAL
    assert bundle.artifact_identity_paths() == {
        "calibration_package": calibration_package_path
    }


def test_regional_output_bundle_writes_expected_paths(
    artifacts_rel: str,
    fake_batch,
    regional_output_bundle: FittedWeightsOutputBundle,
) -> None:
    written = regional_output_bundle.write_artifacts(fake_batch, artifacts_rel)

    assert written == [
        "artifacts/run-1/calibration_weights.npy",
        "artifacts/run-1/geography_assignment.npz",
        "artifacts/run-1/unified_run_config.json",
    ]
    assert fake_batch.files["artifacts/run-1/calibration_weights.npy"] == b"weights"
    assert regional_output_bundle.artifact_paths("/pipeline/artifacts/run-1") == [
        Path("/pipeline/artifacts/run-1/calibration_weights.npy"),
        Path("/pipeline/artifacts/run-1/geography_assignment.npz"),
        Path("/pipeline/artifacts/run-1/unified_run_config.json"),
    ]


def test_national_output_bundle_writes_expected_paths(
    artifacts_rel: str,
    fake_batch,
    national_output_bundle: FittedWeightsOutputBundle,
) -> None:
    written = national_output_bundle.write_artifacts(fake_batch, artifacts_rel)

    assert written == [
        "artifacts/run-1/national_calibration_weights.npy",
        "artifacts/run-1/national_geography_assignment.npz",
        "artifacts/run-1/national_unified_run_config.json",
    ]
    assert (
        fake_batch.files["artifacts/run-1/national_calibration_weights.npy"]
        == b"weights"
    )


def test_missing_optional_epoch_log_is_allowed(
    regional_result_bytes: dict[str, bytes],
) -> None:
    result_bytes = dict(regional_result_bytes)
    result_bytes.pop("cal_log")
    bundle = FittedWeightsOutputBundle.from_result_bytes(
        scope=FitScope.REGIONAL,
        result_bytes=result_bytes,
    )

    assert bundle.diagnostic_result_bytes() == {
        "log": b"regional-log",
        "cal_log": None,
        "config": b"regional-config",
    }


def test_missing_weights_is_a_hard_failure() -> None:
    with pytest.raises(MissingFitWeightsOutputError, match="weights"):
        FittedWeightsOutputBundle.from_result_bytes(
            scope=FitScope.REGIONAL,
            result_bytes={"geography": b"geo"},
        )


@pytest.mark.parametrize(
    ("missing_key", "expected_role"),
    [
        ("geography", "geography"),
        ("config", "run_config"),
    ],
)
def test_missing_required_primary_artifacts_fail_before_writes(
    missing_key: str,
    expected_role: str,
    artifacts_rel: str,
    fake_batch,
    regional_result_bytes: dict[str, bytes],
) -> None:
    result_bytes = dict(regional_result_bytes)
    result_bytes.pop(missing_key)
    bundle = FittedWeightsOutputBundle.from_result_bytes(
        scope=FitScope.REGIONAL,
        result_bytes=result_bytes,
    )

    with pytest.raises(MissingFitWeightsOutputError, match=expected_role):
        bundle.write_artifacts(fake_batch, artifacts_rel)


def test_diagnostics_are_scoped_to_the_output_bundle(
    regional_output_bundle: FittedWeightsOutputBundle,
    national_output_bundle: FittedWeightsOutputBundle,
) -> None:
    assert (
        regional_output_bundle.artifacts.diagnostics.filename
        == "unified_diagnostics.csv"
    )
    assert (
        national_output_bundle.artifacts.diagnostics.filename
        == "national_unified_diagnostics.csv"
    )
    assert regional_output_bundle.diagnostic_result_bytes()["log"] == b"regional-log"
    assert national_output_bundle.diagnostic_result_bytes()["log"] == b"national-log"
