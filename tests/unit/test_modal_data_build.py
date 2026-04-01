import importlib
import sys
from types import ModuleType, SimpleNamespace


def _load_data_build_module():
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
    fake_images.cpu_image = object()

    sys.modules["modal"] = fake_modal
    sys.modules["modal_app.images"] = fake_images
    sys.modules.pop("modal_app.data_build", None)
    return importlib.import_module("modal_app.data_build")


def test_validate_and_maybe_upload_datasets_validates_before_upload(monkeypatch):
    data_build = _load_data_build_module()
    calls = []

    def fake_run_script(script_path, args=None, env=None, log_file=None):
        calls.append((script_path, args or [], env))
        return script_path

    monkeypatch.setattr(data_build, "run_script", fake_run_script)

    data_build.validate_and_maybe_upload_datasets(
        upload=True,
        skip_enhanced_cps=False,
        env={"TEST_ENV": "1"},
    )

    assert calls == [
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            ["--validate-only"],
            {"TEST_ENV": "1"},
        ),
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            [],
            {"TEST_ENV": "1"},
        ),
    ]


def test_validate_and_maybe_upload_datasets_skips_upload_when_disabled(monkeypatch):
    data_build = _load_data_build_module()
    calls = []

    def fake_run_script(script_path, args=None, env=None, log_file=None):
        calls.append((script_path, args or [], env))
        return script_path

    monkeypatch.setattr(data_build, "run_script", fake_run_script)

    data_build.validate_and_maybe_upload_datasets(
        upload=False,
        skip_enhanced_cps=True,
        env={"TEST_ENV": "1"},
    )

    assert calls == [
        (
            "policyengine_us_data/storage/upload_completed_datasets.py",
            ["--validate-only", "--no-require-enhanced-cps"],
            {"TEST_ENV": "1"},
        ),
    ]
