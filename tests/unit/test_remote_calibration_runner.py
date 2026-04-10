import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock


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


def test_collect_outputs_reads_checkpoint_bytes(tmp_path):
    remote_runner = _load_remote_calibration_runner_module()
    weights = tmp_path / "weights.npy"
    geography = tmp_path / "geography.npz"
    log_path = tmp_path / "diag.csv"
    cal_log = tmp_path / "calibration.csv"
    config = tmp_path / "config.json"
    checkpoint = tmp_path / "weights.checkpoint.pt"

    paths_and_bytes = {
        weights: b"weights",
        geography: b"geography",
        log_path: b"log",
        cal_log: b"cal-log",
        config: b"config",
        checkpoint: b"checkpoint",
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
            f"CHECKPOINT_PATH:{checkpoint}",
        ]
    )

    assert result == {
        "weights": b"weights",
        "geography": b"geography",
        "log": b"log",
        "cal_log": b"cal-log",
        "config": b"config",
        "checkpoint": b"checkpoint",
    }


def test_fit_weights_impl_saves_and_resumes_checkpoint_on_volume(
    monkeypatch,
    tmp_path,
):
    remote_runner = _load_remote_calibration_runner_module()
    (tmp_path / "policy_data.db").write_bytes(b"db")
    (tmp_path / "source_imputed_stratified_extended_cps.h5").write_bytes(b"h5")
    checkpoint = tmp_path / "test.checkpoint.pt"
    checkpoint.write_bytes(b"old-checkpoint")
    weights = tmp_path / "weights.npy"

    volume = SimpleNamespace(reload=Mock(), commit=Mock())
    monkeypatch.setattr(remote_runner, "pipeline_vol", volume)
    monkeypatch.setattr(remote_runner, "_setup_repo", lambda: None)

    def fake_run_streaming(cmd, env=None, label=""):
        assert "--resume-from" in cmd
        assert cmd[cmd.index("--resume-from") + 1] == str(checkpoint)
        assert "--checkpoint-output" in cmd
        assert cmd[cmd.index("--checkpoint-output") + 1] == str(checkpoint)
        weights.write_bytes(b"weights")
        checkpoint.write_bytes(b"new-checkpoint")
        return 0, [f"OUTPUT_PATH:{weights}", f"CHECKPOINT_PATH:{checkpoint}"]

    monkeypatch.setattr(remote_runner, "_run_streaming", fake_run_streaming)

    result = remote_runner._fit_weights_impl(
        branch="main",
        epochs=1,
        artifacts_dir=str(tmp_path),
        checkpoint_name="test.checkpoint.pt",
    )

    assert result["weights"] == b"weights"
    assert result["checkpoint"] == b"new-checkpoint"
    volume.reload.assert_called_once()
    volume.commit.assert_called_once()
