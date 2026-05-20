from pathlib import Path
from types import SimpleNamespace

import pytest

from policyengine_us_data.calibration_package.specs import (
    CALIBRATION_PACKAGE_CONTRACT_FILENAME,
    CALIBRATION_PACKAGE_FILENAME,
    CALIBRATION_PACKAGE_METADATA_FILENAME,
    CALIBRATION_REPORTS_DIRNAME,
    DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
    DEFAULT_TARGET_CONFIG_PATH,
    MATRIX_BUILD_DIRNAME,
    SOURCE_DATASET_FILENAME,
    TARGET_DATABASE_FILENAME,
    Stage2InputBundleError,
    TargetConfigIdentity,
    calibration_package_artifact_paths,
    resolve_target_config_identity,
    stage2_build_context_for_run,
    stage2_input_bundle_from_artifacts_dir,
    stage2_input_bundle_from_stage1_contract,
)
from policyengine_us_data.stage_contracts.calibration_package import (
    CalibrationPackageParameters,
)
from policyengine_us_data.stage_contracts.dataset_build import (
    build_dataset_build_output_contract,
)
from policyengine_us_data.stage_contracts.io import write_contract
from policyengine_us_data.utils.manifest import compute_file_checksum


def _sha256_digest(path: Path) -> str:
    return f"sha256:{compute_file_checksum(path)}"


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
        sha256=_sha256_digest(config_path),
        mode="default",
        resolved_path=str(config_path.resolve()),
    )
    assert identity.to_parameters() == {
        "target_config": DEFAULT_TARGET_CONFIG_PATH,
        "target_config_sha256": _sha256_digest(config_path),
        "target_config_mode": "default",
    }


def test_explicit_target_config_identity_resolution(tmp_path):
    config_path = _write_default_target_config(tmp_path)

    identity = resolve_target_config_identity(
        DEFAULT_TARGET_CONFIG_PATH,
        repo_root=tmp_path,
    )

    assert identity.path == DEFAULT_TARGET_CONFIG_PATH
    assert identity.sha256 == _sha256_digest(config_path)
    assert identity.mode == "explicit"
    assert identity.resolved_path == str(config_path.resolve())


def test_resolved_target_config_identity_is_contract_compatible(tmp_path):
    _write_default_target_config(tmp_path)
    identity = resolve_target_config_identity(repo_root=tmp_path)

    params = CalibrationPackageParameters.from_runtime_args(
        workers=8,
        n_clones=430,
        target_config_path=identity.path,
        target_config_sha256=identity.sha256,
        target_config_mode=identity.mode,
        skip_county=True,
        skip_source_impute=True,
        skip_takeup_rerandomize=False,
        chunked_matrix=False,
        chunk_size=25_000,
        parallel=False,
        num_matrix_workers=50,
    )

    assert params.target_config_sha256 == identity.sha256


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
    assert paths.reports_dir == Path("/pipeline/artifacts/run-a") / (
        CALIBRATION_REPORTS_DIRNAME
    )
    assert paths.matrix_build_dir == Path("/pipeline/artifacts/run-a") / (
        MATRIX_BUILD_DIRNAME
    )
    assert paths.manifest_outputs == (paths.package, paths.contract)


def test_stage2_input_bundle_from_artifacts_dir(tmp_path):
    (tmp_path / SOURCE_DATASET_FILENAME).write_bytes(b"h5")
    (tmp_path / TARGET_DATABASE_FILENAME).write_bytes(b"db")

    bundle = stage2_input_bundle_from_artifacts_dir(tmp_path)

    assert bundle.source == "artifacts_dir_fallback"
    assert bundle.compatibility_only is True
    assert bundle.manifest_inputs == {
        "dataset": tmp_path / SOURCE_DATASET_FILENAME,
        "database": tmp_path / TARGET_DATABASE_FILENAME,
    }
    assert bundle.validation_report().status == "pass"


def test_stage2_input_bundle_from_fake_stage1_contract(tmp_path):
    dataset = tmp_path / SOURCE_DATASET_FILENAME
    database = tmp_path / TARGET_DATABASE_FILENAME
    dataset.write_bytes(b"h5")
    database.write_bytes(b"db")
    contract = SimpleNamespace(
        stage_id="1_build_datasets",
        run_id="run-a",
        outputs=(
            SimpleNamespace(
                logical_name="source_imputed_stratified_extended_cps",
                uri=dataset.resolve().as_uri(),
                sha256=_sha256_digest(dataset),
                size_bytes=dataset.stat().st_size,
            ),
            SimpleNamespace(
                logical_name="policy_data_db",
                uri=database.resolve().as_uri(),
                sha256=_sha256_digest(database),
                size_bytes=database.stat().st_size,
            ),
        ),
    )

    bundle = stage2_input_bundle_from_stage1_contract(
        contract,
        artifacts_dir=tmp_path,
        contract_path=tmp_path / DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
    )

    assert bundle.source == "stage1_contract"
    assert bundle.compatibility_only is False
    assert bundle.stage1_contract_run_id == "run-a"
    assert bundle.stage1_contract_path == (
        tmp_path / DATASET_BUILD_OUTPUT_CONTRACT_FILENAME
    )
    assert bundle.source_dataset == dataset
    assert bundle.target_database == database
    assert bundle.expected_input_identities == {
        "dataset": {
            "sha256": _sha256_digest(dataset),
            "size_bytes": dataset.stat().st_size,
        },
        "database": {
            "sha256": _sha256_digest(database),
            "size_bytes": database.stat().st_size,
        },
    }
    assert bundle.validation_report().status == "pass"


def test_stage2_input_bundle_validates_stage1_contract_identity(tmp_path):
    dataset = tmp_path / SOURCE_DATASET_FILENAME
    database = tmp_path / TARGET_DATABASE_FILENAME
    dataset.write_bytes(b"h5")
    database.write_bytes(b"db")
    contract = SimpleNamespace(
        stage_id="1_build_datasets",
        run_id="run-a",
        outputs=(
            SimpleNamespace(
                logical_name="source_imputed_stratified_extended_cps",
                uri=dataset.resolve().as_uri(),
                sha256="sha256:not-the-dataset",
                size_bytes=dataset.stat().st_size,
            ),
            SimpleNamespace(
                logical_name="policy_data_db",
                uri=database.resolve().as_uri(),
                sha256=_sha256_digest(database),
                size_bytes=database.stat().st_size + 1,
            ),
        ),
    )

    bundle = stage2_input_bundle_from_stage1_contract(contract)
    report = bundle.validation_report()

    assert report.status == "fail"
    assert [finding.check_id for finding in report.findings] == [
        "stage2_input_identity:dataset:sha256",
        "stage2_input_identity:database:size_bytes",
    ]
    assert report.metadata["expected_identities"] == {
        "dataset": {
            "sha256": "sha256:not-the-dataset",
            "size_bytes": dataset.stat().st_size,
        },
        "database": {
            "sha256": _sha256_digest(database),
            "size_bytes": database.stat().st_size + 1,
        },
    }
    with pytest.raises(Stage2InputBundleError, match="checksum mismatch"):
        bundle.require_existing()


def test_stage2_input_bundle_missing_required_artifacts_are_actionable(tmp_path):
    (tmp_path / SOURCE_DATASET_FILENAME).write_bytes(b"h5")
    bundle = stage2_input_bundle_from_artifacts_dir(tmp_path)

    report = bundle.validation_report()

    assert report.status == "fail"
    assert [finding.check_id for finding in report.findings] == [
        "stage2_input_exists:database"
    ]
    assert str(tmp_path / TARGET_DATABASE_FILENAME) in report.findings[0].message
    with pytest.raises(Stage2InputBundleError, match="database"):
        bundle.require_existing()


def test_stage2_build_context_prefers_stage1_contract(tmp_path):
    artifacts_dir = tmp_path / "artifacts" / "run-a"
    artifacts_dir.mkdir(parents=True)
    for filename in (
        "acs_2022.h5",
        "irs_puf_2015.h5",
        "cps_2024.h5",
        "puf_2024.h5",
        "extended_cps_2024.h5",
        "enhanced_cps_2024.h5",
        "small_enhanced_cps_2024.h5",
        "stratified_extended_cps_2024.h5",
        "source_imputed_stratified_extended_cps_2024.h5",
        SOURCE_DATASET_FILENAME,
        TARGET_DATABASE_FILENAME,
        "build_log.txt",
        "data_build_checkpoint_stats.json",
    ):
        (artifacts_dir / filename).write_bytes(filename.encode("utf-8"))
    contract_path = artifacts_dir / DATASET_BUILD_OUTPUT_CONTRACT_FILENAME
    write_contract(
        build_dataset_build_output_contract(
            artifacts_dir=artifacts_dir,
            run_id="run-a",
            code_sha="abc123",
            package_version="1.0.0",
            checkpoint_stats={},
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            duration_s=1.0,
        ),
        contract_path,
    )

    context = stage2_build_context_for_run(tmp_path, "run-a")

    assert context.input_bundle.source == "stage1_contract"
    assert (
        context.input_bundle.source_dataset == artifacts_dir / SOURCE_DATASET_FILENAME
    )
    assert (
        context.input_bundle.target_database == artifacts_dir / TARGET_DATABASE_FILENAME
    )
    assert context.input_bundle.expected_source_dataset_sha256 == _sha256_digest(
        artifacts_dir / SOURCE_DATASET_FILENAME
    )
    assert (
        context.input_bundle.expected_target_database_size_bytes
        == (artifacts_dir / TARGET_DATABASE_FILENAME).stat().st_size
    )
    assert context.output_bundle.package == artifacts_dir / CALIBRATION_PACKAGE_FILENAME


def test_stage2_build_context_rejects_explicit_missing_stage1_contract(tmp_path):
    explicit_contract = tmp_path / "missing-dataset-build-output.json"

    with pytest.raises(FileNotFoundError, match="Stage 1 contract not found"):
        stage2_build_context_for_run(
            tmp_path,
            "run-a",
            stage1_contract_path=explicit_contract,
        )
