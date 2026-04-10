import importlib
import sys
from types import ModuleType, SimpleNamespace


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
