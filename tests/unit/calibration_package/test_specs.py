from pathlib import Path

import pytest

from policyengine_us_data.calibration_package.specs import (
    CALIBRATION_PACKAGE_CONTRACT_FILENAME,
    CALIBRATION_PACKAGE_FILENAME,
    CALIBRATION_PACKAGE_METADATA_FILENAME,
    DEFAULT_TARGET_CONFIG_PATH,
    MATRIX_BUILD_DIRNAME,
    TargetConfigIdentity,
    calibration_package_artifact_paths,
    resolve_target_config_identity,
)
from policyengine_us_data.utils.manifest import compute_file_checksum


def _write_default_target_config(repo_root: Path, body: str = "include: []\n") -> Path:
    config_path = repo_root / DEFAULT_TARGET_CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_default_target_config_identity_resolution(tmp_path):
    config_path = _write_default_target_config(tmp_path)

    identity = resolve_target_config_identity(repo_root=tmp_path)

    assert identity == TargetConfigIdentity(
        path=DEFAULT_TARGET_CONFIG_PATH,
        sha256=compute_file_checksum(config_path),
        mode="default",
        resolved_path=str(config_path.resolve()),
    )
    assert identity.to_parameters() == {
        "target_config": DEFAULT_TARGET_CONFIG_PATH,
        "target_config_sha256": compute_file_checksum(config_path),
        "target_config_mode": "default",
    }


def test_explicit_target_config_identity_resolution(tmp_path):
    config_path = _write_default_target_config(tmp_path)

    identity = resolve_target_config_identity(
        DEFAULT_TARGET_CONFIG_PATH,
        repo_root=tmp_path,
    )

    assert identity.path == DEFAULT_TARGET_CONFIG_PATH
    assert identity.sha256 == compute_file_checksum(config_path)
    assert identity.mode == "explicit"
    assert identity.resolved_path == str(config_path.resolve())


def test_all_active_targets_identity_resolution():
    identity = resolve_target_config_identity(all_active_targets=True)

    assert identity.to_parameters() == {
        "target_config": None,
        "target_config_sha256": None,
        "target_config_mode": "all_active_targets",
    }


def test_all_active_targets_rejects_config_path():
    with pytest.raises(ValueError, match="all-active-targets"):
        resolve_target_config_identity(
            DEFAULT_TARGET_CONFIG_PATH,
            all_active_targets=True,
        )


def test_calibration_package_artifact_paths():
    paths = calibration_package_artifact_paths("/pipeline/artifacts/run-a")

    assert paths.package == Path("/pipeline/artifacts/run-a") / (
        CALIBRATION_PACKAGE_FILENAME
    )
    assert paths.metadata == Path("/pipeline/artifacts/run-a") / (
        CALIBRATION_PACKAGE_METADATA_FILENAME
    )
    assert paths.contract == Path("/pipeline/artifacts/run-a") / (
        CALIBRATION_PACKAGE_CONTRACT_FILENAME
    )
    assert paths.matrix_build_dir == Path("/pipeline/artifacts/run-a") / (
        MATRIX_BUILD_DIRNAME
    )
    assert paths.manifest_outputs == (paths.package, paths.contract)
