import inspect
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from policyengine_us_data.fit_weights import FittedWeightsInputContractError


def _load_remote_calibration_runner_module():
    fake_modal = ModuleType("modal")

    class _FakeApp:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    fake_modal.App = _FakeApp
    fake_modal.Secret = SimpleNamespace(from_name=lambda *args, **kwargs: object())
    fake_modal.Volume = SimpleNamespace(from_name=lambda *args, **kwargs: object())

    fake_images = ModuleType("modal_app.images")
    fake_images.gpu_image = object()

    sys.modules["modal"] = fake_modal
    sys.modules["modal_app.images"] = fake_images
    sys.modules.pop("modal_app.remote_calibration_runner", None)
    return importlib.import_module("modal_app.remote_calibration_runner")


def test_remote_runner_does_not_expose_optimizer_checkpoint_contract():
    remote_runner = _load_remote_calibration_runner_module()

    assert not hasattr(remote_runner, "_append_checkpoint_args")
    for func in (
        remote_runner._fit_weights_impl,
        remote_runner._fit_from_package_impl,
        remote_runner.fit_weights_t4,
        remote_runner.fit_from_package_t4,
        remote_runner.main,
    ):
        assert "checkpoint_name" not in inspect.signature(func).parameters


def test_fit_from_package_impl_requires_stage_2_contract_before_calibration(
    monkeypatch,
    tmp_path,
):
    remote_runner = _load_remote_calibration_runner_module()
    package_path = tmp_path / "calibration_package.pkl"
    package_path.write_bytes(b"package")
    run_streaming = Mock()

    monkeypatch.setattr(remote_runner, "_setup_repo", lambda: None)
    monkeypatch.setattr(remote_runner, "_run_streaming", run_streaming)

    with pytest.raises(FittedWeightsInputContractError) as exc_info:
        remote_runner._fit_from_package_impl(
            branch="main",
            epochs=1,
            volume_package_path=str(package_path),
        )

    assert exc_info.value.code == "missing_stage2_contract"
    run_streaming.assert_not_called()


def test_collect_outputs_returns_pipeline_artifact_bytes(tmp_path):
    remote_runner = _load_remote_calibration_runner_module()
    weights = tmp_path / "weights.npy"
    geography = tmp_path / "geography.npz"
    log_path = tmp_path / "diag.csv"
    cal_log = tmp_path / "calibration.csv"
    config = tmp_path / "config.json"

    paths_and_bytes = {
        weights: b"weights",
        geography: b"geography",
        log_path: b"log",
        cal_log: b"cal-log",
        config: b"config",
    }
    for path, content in paths_and_bytes.items():
        path.write_bytes(content)

    result = remote_runner._collect_outputs(
        [
            f"OUTPUT_PATH:{weights}",
            f"GEOGRAPHY_PATH:{geography}",
            f"LOG_PATH:{log_path}",
            f"CAL_LOG_PATH:{cal_log}",
            f"CONFIG_PATH:{config}",
        ]
    )

    assert result == {
        "weights": b"weights",
        "geography": b"geography",
        "log": b"log",
        "cal_log": b"cal-log",
        "config": b"config",
    }


def test_fit_output_filenames_match_scoped_artifacts():
    remote_runner = _load_remote_calibration_runner_module()
    regional = remote_runner.fit_artifacts_for_scope(remote_runner.FitScope.REGIONAL)
    national = remote_runner.fit_artifacts_for_scope(remote_runner.FitScope.NATIONAL)

    assert remote_runner._fit_output_filenames(
        scope=remote_runner.FitScope.REGIONAL,
        output=regional.weights.filename,
        log_output=regional.diagnostics.filename,
    ) == {
        "output": regional.weights.filename,
        "log_output": regional.diagnostics.filename,
        "geography": regional.geography.filename,
        "calibration_log": regional.epoch_log.filename,
        "run_config": regional.run_config.filename,
        "pipeline_weights": regional.weights.filename,
        "pipeline_geography": regional.geography.filename,
        "pipeline_run_config": regional.run_config.filename,
    }
    assert remote_runner._fit_output_filenames(
        scope=remote_runner.FitScope.NATIONAL,
        output=regional.weights.filename,
        log_output=regional.diagnostics.filename,
    ) == {
        "output": national.weights.filename,
        "log_output": national.diagnostics.filename,
        "geography": national.geography.filename,
        "calibration_log": national.epoch_log.filename,
        "run_config": national.run_config.filename,
        "pipeline_weights": national.weights.filename,
        "pipeline_geography": national.geography.filename,
        "pipeline_run_config": national.run_config.filename,
    }


def test_national_fit_output_filenames_preserve_explicit_overrides():
    remote_runner = _load_remote_calibration_runner_module()
    national = remote_runner.fit_artifacts_for_scope(remote_runner.FitScope.NATIONAL)

    filenames = remote_runner._fit_output_filenames(
        scope=remote_runner.FitScope.NATIONAL,
        output="custom_weights.npy",
        log_output="custom_diagnostics.csv",
    )

    assert filenames["output"] == "national_custom_weights.npy"
    assert filenames["log_output"] == "national_custom_diagnostics.csv"
    assert filenames["pipeline_weights"] == national.weights.filename


def test_ensure_geography_prerequisites_downloads_only_geography(monkeypatch):
    remote_runner = _load_remote_calibration_runner_module()
    from policyengine_us_data.storage import download_prerequisites as prereq_module

    captured = {}

    def fake_download_prerequisites(artifacts):
        captured["artifacts"] = artifacts

    monkeypatch.setattr(
        prereq_module,
        "download_prerequisites",
        fake_download_prerequisites,
    )

    remote_runner._ensure_geography_prerequisites()

    artifacts = captured["artifacts"]
    assert {artifact.local_filename for artifact in artifacts} == {
        "block_cd_distributions.csv.gz",
        "block_crosswalk.csv.gz",
    }
    assert all(artifact.repo == prereq_module.GEOGRAPHY_REPO for artifact in artifacts)


def test_fit_weights_impl_does_not_use_optimizer_checkpoint_artifacts(
    monkeypatch,
    tmp_path,
):
    remote_runner = _load_remote_calibration_runner_module()
    (tmp_path / "policy_data.db").write_bytes(b"db")
    (tmp_path / "source_imputed_stratified_extended_cps.h5").write_bytes(b"h5")
    weights = tmp_path / "weights.npy"

    volume = SimpleNamespace(reload=Mock(), commit=Mock())
    ensure_prereqs = Mock()
    monkeypatch.setattr(remote_runner, "pipeline_vol", volume)
    monkeypatch.setattr(remote_runner, "_setup_repo", lambda: None)
    monkeypatch.setattr(
        remote_runner,
        "_ensure_geography_prerequisites",
        ensure_prereqs,
    )

    def fake_run_streaming(cmd, env=None, label=""):
        assert "--resume-from" not in cmd
        assert "--checkpoint-output" not in cmd
        weights.write_bytes(b"weights")
        return 0, [f"OUTPUT_PATH:{weights}"]

    monkeypatch.setattr(remote_runner, "_run_streaming", fake_run_streaming)

    result = remote_runner._fit_weights_impl(
        branch="main",
        epochs=1,
        artifacts_dir=str(tmp_path),
    )

    assert result["weights"] == b"weights"
    assert "checkpoint" not in result
    ensure_prereqs.assert_called_once()
    volume.reload.assert_called_once()
    volume.commit.assert_not_called()


def test_build_package_impl_sets_volume_chunk_dir_for_parallel_matrix(
    monkeypatch,
    tmp_path,
):
    remote_runner = _load_remote_calibration_runner_module()
    artifacts_dir = tmp_path / "artifacts" / "bench-run"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "policy_data.db").write_bytes(b"db")
    (artifacts_dir / "source_imputed_stratified_extended_cps.h5").write_bytes(b"h5")

    volume = SimpleNamespace(reload=Mock(), commit=Mock())
    ensure_prereqs = Mock()
    monkeypatch.setattr(remote_runner, "PIPELINE_MOUNT", str(tmp_path))
    monkeypatch.setattr(remote_runner, "pipeline_vol", volume)
    monkeypatch.setattr(remote_runner, "_setup_repo", lambda: None)
    monkeypatch.setattr(
        remote_runner,
        "_ensure_geography_prerequisites",
        ensure_prereqs,
    )
    monkeypatch.setattr(remote_runner, "_write_package_sidecar", lambda _: True)
    from policyengine_us_data.stage_contracts import calibration_package

    captured = {}

    def fake_validate_persisted_contract(**kwargs):
        captured["contract_validation"] = kwargs

    monkeypatch.setattr(
        calibration_package,
        "validate_persisted_calibration_package_contract",
        fake_validate_persisted_contract,
    )

    def fake_run_streaming(cmd, env=None, label=""):
        captured["cmd"] = cmd
        captured["env"] = env
        (artifacts_dir / "calibration_package.pkl").write_bytes(b"pkg")
        return 0, []

    monkeypatch.setattr(remote_runner, "_run_streaming", fake_run_streaming)

    result = remote_runner._build_package_impl(
        branch="main",
        workers=1,
        n_clones=10,
        run_id="bench-run",
        modal_app_name="us-data-bench-run",
        modal_environment="main",
        pipeline_volume_name="pipeline-artifacts-bench-run",
        chunked_matrix=True,
        parallel_matrix=True,
        num_matrix_workers=1,
    )

    assert result == str(artifacts_dir / "calibration_package.pkl")
    assert str(artifacts_dir / "policy_data.db") in captured["cmd"]
    assert (
        str(artifacts_dir / "source_imputed_stratified_extended_cps.h5")
        in (captured["cmd"])
    )
    assert str(artifacts_dir / "calibration_package.pkl") in captured["cmd"]
    assert "--chunk-dir" in captured["cmd"]
    chunk_dir_idx = captured["cmd"].index("--chunk-dir") + 1
    assert captured["cmd"][chunk_dir_idx] == str(artifacts_dir / "matrix_build")
    assert captured["env"]["POLICYENGINE_US_DATA_RUN_ID"] == "bench-run"
    assert captured["env"]["US_DATA_RUN_ID"] == "bench-run"
    assert captured["env"]["US_DATA_MODAL_APP_NAME"] == "us-data-bench-run"
    assert captured["env"]["MODAL_APP_NAME"] == "us-data-bench-run"
    assert captured["env"]["US_DATA_MODAL_ENVIRONMENT"] == "main"
    assert captured["env"]["MODAL_ENVIRONMENT"] == "main"
    assert (
        captured["env"]["US_DATA_PIPELINE_VOLUME_NAME"]
        == "pipeline-artifacts-bench-run"
    )
    assert captured["contract_validation"]["package_path"] == (
        artifacts_dir / "calibration_package.pkl"
    )
    assert captured["contract_validation"]["contract_path"] == (
        artifacts_dir / "calibration_package_contract.json"
    )
    assert captured["contract_validation"]["dataset_path"] == (
        artifacts_dir / "source_imputed_stratified_extended_cps.h5"
    )
    assert captured["contract_validation"]["db_path"] == (
        artifacts_dir / "policy_data.db"
    )
    ensure_prereqs.assert_called_once()
    volume.reload.assert_called_once()
    volume.commit.assert_called_once()
