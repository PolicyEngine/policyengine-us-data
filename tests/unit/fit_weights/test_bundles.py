from pathlib import Path

import pytest

from policyengine_us_data.fit_weights import (
    FitScope,
    FittedWeightsInputBundle,
    FittedWeightsOutputBundle,
    MissingFitWeightsOutputError,
)


class FakeBatch:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def put_file(self, file_obj, destination: str) -> None:
        self.files[destination] = file_obj.read()


def test_input_bundle_exposes_calibration_package_identity_path() -> None:
    bundle = FittedWeightsInputBundle(
        scope="regional",
        calibration_package_path=Path(
            "/pipeline/artifacts/run/calibration_package.pkl"
        ),
    )

    assert bundle.scope == FitScope.REGIONAL
    assert bundle.artifact_identity_paths() == {
        "calibration_package": Path("/pipeline/artifacts/run/calibration_package.pkl")
    }


def test_regional_output_bundle_writes_expected_paths() -> None:
    bundle = FittedWeightsOutputBundle.from_result_bytes(
        scope=FitScope.REGIONAL,
        result_bytes={
            "weights": b"weights",
            "geography": b"geo",
            "config": b"config",
            "log": b"log",
            "cal_log": b"epoch",
        },
        run_id="run-1",
    )
    batch = FakeBatch()

    written = bundle.write_artifacts(batch, "artifacts/run-1")

    assert written == [
        "artifacts/run-1/calibration_weights.npy",
        "artifacts/run-1/geography_assignment.npz",
        "artifacts/run-1/unified_run_config.json",
    ]
    assert batch.files["artifacts/run-1/calibration_weights.npy"] == b"weights"
    assert bundle.artifact_paths("/pipeline/artifacts/run-1") == [
        Path("/pipeline/artifacts/run-1/calibration_weights.npy"),
        Path("/pipeline/artifacts/run-1/geography_assignment.npz"),
        Path("/pipeline/artifacts/run-1/unified_run_config.json"),
    ]


def test_national_output_bundle_writes_expected_paths() -> None:
    bundle = FittedWeightsOutputBundle.from_result_bytes(
        scope="national",
        result_bytes={
            "weights": b"weights",
            "geography": b"geo",
            "config": b"config",
        },
    )
    batch = FakeBatch()

    written = bundle.write_artifacts(batch, "artifacts/run-1")

    assert written == [
        "artifacts/run-1/national_calibration_weights.npy",
        "artifacts/run-1/national_geography_assignment.npz",
        "artifacts/run-1/national_unified_run_config.json",
    ]
    assert batch.files["artifacts/run-1/national_calibration_weights.npy"] == b"weights"


def test_missing_optional_epoch_log_is_allowed() -> None:
    bundle = FittedWeightsOutputBundle.from_result_bytes(
        scope=FitScope.REGIONAL,
        result_bytes={
            "weights": b"weights",
            "log": b"log",
        },
    )

    assert bundle.diagnostic_result_bytes() == {
        "log": b"log",
        "cal_log": None,
        "config": None,
    }


def test_missing_weights_is_a_hard_failure() -> None:
    with pytest.raises(MissingFitWeightsOutputError, match="weights"):
        FittedWeightsOutputBundle.from_result_bytes(
            scope=FitScope.REGIONAL,
            result_bytes={"geography": b"geo"},
        )


def test_diagnostics_are_scoped_to_the_output_bundle() -> None:
    regional = FittedWeightsOutputBundle.from_result_bytes(
        scope=FitScope.REGIONAL,
        result_bytes={
            "weights": b"weights",
            "log": b"regional-log",
            "cal_log": b"regional-epoch",
        },
    )
    national = FittedWeightsOutputBundle.from_result_bytes(
        scope=FitScope.NATIONAL,
        result_bytes={
            "weights": b"weights",
            "log": b"national-log",
            "cal_log": b"national-epoch",
        },
    )

    assert regional.artifacts.diagnostics.filename == "unified_diagnostics.csv"
    assert national.artifacts.diagnostics.filename == "national_unified_diagnostics.csv"
    assert regional.diagnostic_result_bytes()["log"] == b"regional-log"
    assert national.diagnostic_result_bytes()["log"] == b"national-log"
