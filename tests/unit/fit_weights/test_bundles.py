from pathlib import Path

import pytest

from policyengine_us_data.fit_weights import (
    FitScope,
    FittedWeightsInputContractError,
    FittedWeightsInputBundle,
    FittedWeightsOutputBundle,
    MissingFitWeightsOutputError,
)
from policyengine_us_data.utils.step_manifest import sha256_file
from tests.unit.fixtures.calibration_package_stage_contract import (
    calibration_package_payload_with_block_geoids,
    write_calibration_package_payload,
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
        "calibration_package": calibration_package_path,
        "calibration_package_contract": calibration_package_path.with_name(
            "calibration_package_contract.json"
        ),
    }


def test_input_bundle_records_stage_2_contract_identity(
    stage2_contract_fixture,
) -> None:
    bundle = FittedWeightsInputBundle(
        scope=FitScope.REGIONAL,
        calibration_package_path=stage2_contract_fixture.package_path,
        calibration_package_contract_path=stage2_contract_fixture.contract_path,
    )

    assert bundle.artifact_identity_paths() == {
        "calibration_package": stage2_contract_fixture.package_path,
        "calibration_package_contract": stage2_contract_fixture.contract_path,
    }
    assert bundle.stage2_identity_parameters() == {
        "calibration_package_sha256": stage2_contract_fixture.contract.outputs[
            0
        ].sha256,
        "calibration_package_size_bytes": (
            stage2_contract_fixture.package_path.stat().st_size
        ),
        "stage2_contract_mode": "stage2_contract",
        "calibration_package_contract_sha256": (
            f"sha256:{sha256_file(stage2_contract_fixture.contract_path)}"
        ),
        "calibration_package_contract_size_bytes": (
            stage2_contract_fixture.contract_path.stat().st_size
        ),
        "calibration_package_contract_fingerprint": (
            stage2_contract_fixture.contract.fingerprint.value
        ),
        "calibration_package_contract_run_id": stage2_contract_fixture.contract.run_id,
    }


def test_input_bundle_rejects_package_contract_checksum_mismatch(
    stage2_contract_fixture,
) -> None:
    write_calibration_package_payload(
        stage2_contract_fixture.package_path,
        calibration_package_payload_with_block_geoids(),
    )
    bundle = FittedWeightsInputBundle(
        scope=FitScope.REGIONAL,
        calibration_package_path=stage2_contract_fixture.package_path,
        calibration_package_contract_path=stage2_contract_fixture.contract_path,
    )

    with pytest.raises(
        FittedWeightsInputContractError,
        match="checksum mismatch",
    ) as exc_info:
        bundle.stage2_identity_parameters()

    assert exc_info.value.code == "stage2_contract_package_mismatch"


def test_input_bundle_rejects_missing_package_artifact(tmp_path: Path) -> None:
    bundle = FittedWeightsInputBundle(
        scope=FitScope.REGIONAL,
        calibration_package_path=tmp_path / "missing.pkl",
    )

    with pytest.raises(FittedWeightsInputContractError, match="Missing") as exc_info:
        bundle.stage2_identity_parameters()

    assert exc_info.value.code == "missing_calibration_package"


def test_input_bundle_requires_contract_unless_legacy_fallback(
    stage2_contract_fixture,
) -> None:
    stage2_contract_fixture.contract_path.unlink()
    bundle = FittedWeightsInputBundle(
        scope=FitScope.REGIONAL,
        calibration_package_path=stage2_contract_fixture.package_path,
        calibration_package_contract_path=stage2_contract_fixture.contract_path,
    )

    with pytest.raises(FittedWeightsInputContractError) as exc_info:
        bundle.stage2_identity_parameters()

    assert exc_info.value.code == "missing_stage2_contract"


def test_input_bundle_legacy_no_contract_fallback_warns(
    stage2_contract_fixture,
) -> None:
    stage2_contract_fixture.contract_path.unlink()
    bundle = FittedWeightsInputBundle(
        scope=FitScope.REGIONAL,
        calibration_package_path=stage2_contract_fixture.package_path,
        calibration_package_contract_path=stage2_contract_fixture.contract_path,
        allow_legacy_no_contract=True,
    )

    with pytest.warns(RuntimeWarning, match="legacy manual fallback"):
        identity = bundle.stage2_identity_parameters()

    assert identity["stage2_contract_mode"] == "legacy_no_contract"
    assert identity["calibration_package_sha256"].startswith("sha256:")
    assert "calibration_package_contract_sha256" not in identity
    assert bundle.artifact_identity_paths() == {
        "calibration_package": stage2_contract_fixture.package_path
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
