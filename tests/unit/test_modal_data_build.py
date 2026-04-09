import importlib
import sys
from pathlib import Path
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


def test_mirror_source_imputed_artifact_uploads_when_present(monkeypatch, tmp_path):
    data_build = _load_data_build_module()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    source_imputed = artifacts_dir / "source_imputed_stratified_extended_cps.h5"
    source_imputed.write_text("placeholder")

    calls = []

    def fake_mirror(stage_name, files, **kwargs):
        calls.append((stage_name, files, kwargs))

    fake_pipeline_artifacts = ModuleType(
        "policyengine_us_data.utils.pipeline_artifacts"
    )
    fake_pipeline_artifacts.mirror_to_pipeline = fake_mirror
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.utils.pipeline_artifacts",
        fake_pipeline_artifacts,
    )

    data_build.mirror_source_imputed_artifact(artifacts_dir, run_id="run-123")

    assert calls == [
        (
            "stage_4_source_imputed",
            [source_imputed],
            {"run_id": "run-123"},
        )
    ]


def test_mirror_source_imputed_artifact_skips_when_missing(monkeypatch, tmp_path):
    data_build = _load_data_build_module()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    calls = []

    def fake_mirror(stage_name, files, **kwargs):
        calls.append((stage_name, files, kwargs))

    fake_pipeline_artifacts = ModuleType(
        "policyengine_us_data.utils.pipeline_artifacts"
    )
    fake_pipeline_artifacts.mirror_to_pipeline = fake_mirror
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.utils.pipeline_artifacts",
        fake_pipeline_artifacts,
    )

    data_build.mirror_source_imputed_artifact(Path(artifacts_dir))

    assert calls == []
