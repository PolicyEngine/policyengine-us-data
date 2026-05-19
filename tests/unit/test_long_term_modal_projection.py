import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_long_term_projection_module(monkeypatch):
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

    class _FakeVolume:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

        def commit(self):
            pass

    fake_modal.App = _FakeApp
    fake_modal.Secret = SimpleNamespace(from_name=lambda *args, **kwargs: object())
    fake_modal.Volume = _FakeVolume

    fake_images = ModuleType("modal_app.images")
    fake_images.cpu_image = object()

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "modal_app.images", fake_images)
    sys.modules.pop("modal_app.long_term_projection", None)
    return importlib.import_module("modal_app.long_term_projection")


def test_build_command_forwards_production_flags(monkeypatch, tmp_path):
    long_term = _load_long_term_projection_module(monkeypatch)

    command = long_term._build_command(
        years="2026,2075",
        jobs=2,
        output_dir=tmp_path / "out",
        profile="ss-payroll-tob",
        target_source="trustees_2025_current_law",
        tax_assumption="trustees-2025-core-thresholds-v1",
        run_id="run-123",
        source_sha="abc123",
        upload_to_hf_staging=True,
        base_dataset="hf://example/base.h5",
        allow_validation_failures=True,
        keep_temp=True,
        support_augmentation_profile="donor-backed-composite-v1",
        support_augmentation_target_year=2100,
        support_augmentation_align_to_run_year=False,
        support_augmentation_start_year=2075,
        support_augmentation_top_n_targets=120,
        support_augmentation_donors_per_target=20,
        support_augmentation_max_distance=5.0,
        support_augmentation_clone_weight_scale=2.0,
        support_augmentation_blueprint_base_weight_scale=5.0,
        support_augmentation_sanitize_worker_non_target_income=False,
        support_augmentation_sanitize_clone_non_target_income=True,
    )

    assert command[:2] == [
        sys.executable,
        "-u",
    ]
    assert command[2:4] == ["-m", long_term._LONG_TERM_PRODUCTION_MODULE]
    expected_flags = {
        "--years": "2026,2075",
        "--jobs": "2",
        "--profile": "ss-payroll-tob",
        "--target-source": "trustees_2025_current_law",
        "--tax-assumption": "trustees-2025-core-thresholds-v1",
        "--run-id": "run-123",
        "--source-sha": "abc123",
        "--base-dataset": "hf://example/base.h5",
        "--support-augmentation-profile": "donor-backed-composite-v1",
        "--support-augmentation-target-year": "2100",
        "--support-augmentation-start-year": "2075",
        "--support-augmentation-top-n-targets": "120",
        "--support-augmentation-donors-per-target": "20",
        "--support-augmentation-max-distance": "5.0",
        "--support-augmentation-clone-weight-scale": "2.0",
        "--support-augmentation-blueprint-base-weight-scale": "5.0",
    }
    for flag, value in expected_flags.items():
        assert command[command.index(flag) + 1] == value
    assert "--support-augmentation-sanitize-clone-non-target-income" in command
    assert "--support-augmentation-sanitize-worker-non-target-income" not in command
    assert "--support-augmentation-align-to-run-year" not in command
    assert "--allow-validation-failures" in command
    assert "--keep-temp" in command
    assert "--upload-to-hf-staging" in command


def test_build_command_does_not_upload_to_hf_by_default(monkeypatch, tmp_path):
    long_term = _load_long_term_projection_module(monkeypatch)

    command = long_term._build_command(
        years="2026",
        jobs=1,
        output_dir=tmp_path / "out",
        profile="ss-payroll-tob",
        target_source="trustees_2025_current_law",
        tax_assumption="trustees-2025-core-thresholds-v1",
        run_id="run-123",
        source_sha="abc123",
        upload_to_hf_staging=False,
        base_dataset="",
        allow_validation_failures=False,
        keep_temp=False,
        support_augmentation_profile="",
        support_augmentation_target_year=None,
        support_augmentation_align_to_run_year=False,
        support_augmentation_start_year=None,
        support_augmentation_top_n_targets=None,
        support_augmentation_donors_per_target=None,
        support_augmentation_max_distance=None,
        support_augmentation_clone_weight_scale=None,
        support_augmentation_blueprint_base_weight_scale=None,
        support_augmentation_sanitize_worker_non_target_income=False,
        support_augmentation_sanitize_clone_non_target_income=False,
    )

    assert "--upload-to-hf-staging" not in command


def test_main_spawn_resolves_git_sha_sanitizes_run_id_and_reports_volume(
    monkeypatch,
    capsys,
):
    long_term = _load_long_term_projection_module(monkeypatch)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(long_term, "_local_git_sha", lambda: "abc123")
    monkeypatch.setattr(long_term, "_local_git_dirty", lambda: False)

    captured_kwargs = {}

    def fake_spawn(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(object_id="fc-123")

    long_term.build_long_term_projection = SimpleNamespace(spawn=fake_spawn)

    long_term.main(
        years="2026",
        run_id="../CRFB Sentinel",
        source_sha="",
        spawn=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "function_call_id": "fc-123",
        "modal_volume": long_term._OUTPUT_VOLUME_NAME,
        "run_id": "crfb-sentinel",
        "source_sha": "abc123",
        "volume_output_prefix": "crfb-sentinel",
    }
    assert captured_kwargs["run_id"] == "crfb-sentinel"
    assert captured_kwargs["source_sha"] == "abc123"
    assert captured_kwargs["upload_to_hf_staging"] is False
    assert captured_kwargs["clear_output"] is False


def test_main_refuses_dirty_source_by_default(monkeypatch):
    long_term = _load_long_term_projection_module(monkeypatch)
    monkeypatch.setattr(long_term, "_local_git_dirty", lambda: True)

    with pytest.raises(ValueError, match="uncommitted changes"):
        long_term.main(
            years="2026",
            run_id="run-123",
            source_sha="abc123",
            spawn=True,
        )


def test_main_refuses_source_sha_that_does_not_match_local_checkout(monkeypatch):
    long_term = _load_long_term_projection_module(monkeypatch)
    monkeypatch.setattr(long_term, "_local_git_dirty", lambda: False)
    monkeypatch.setattr(long_term, "_local_git_sha", lambda: "localabc")

    with pytest.raises(ValueError, match="does not match the local checkout"):
        long_term.main(
            years="2026",
            run_id="run-123",
            source_sha="otherabc",
            spawn=True,
        )


def test_main_allows_dirty_source_only_when_explicit(monkeypatch, capsys):
    long_term = _load_long_term_projection_module(monkeypatch)
    monkeypatch.setattr(long_term, "_local_git_dirty", lambda: True)

    captured_kwargs = {}

    def fake_spawn(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(object_id="fc-123")

    long_term.build_long_term_projection = SimpleNamespace(spawn=fake_spawn)

    long_term.main(
        years="2026",
        run_id="run-123",
        source_sha="abc123",
        spawn=True,
        allow_dirty_source=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["function_call_id"] == "fc-123"
    assert captured_kwargs["source_sha"] == "abc123"


def test_remote_result_reports_hf_staging_prefix(monkeypatch, tmp_path):
    long_term = _load_long_term_projection_module(monkeypatch)
    monkeypatch.setattr(long_term, "_OUTPUT_MOUNT", tmp_path)
    monkeypatch.setattr(long_term, "_stream_command", lambda command, env: None)

    result = long_term.build_long_term_projection(
        years="2026",
        run_id="Run 123",
        source_sha="abc123",
        upload_to_hf_staging=True,
    )

    assert result["run_id"] == "run-123"
    assert result["hf_staging_prefix"] == "staging/abc123-run-123/long_term"
    assert Path(result["output_dir"]) == tmp_path / "run-123"


def test_remote_preserves_resume_artifacts_when_clear_output_requested(
    monkeypatch,
    tmp_path,
):
    long_term = _load_long_term_projection_module(monkeypatch)
    monkeypatch.setattr(long_term, "_OUTPUT_MOUNT", tmp_path)

    output_dir = tmp_path / "run-123"
    resume_dir = output_dir / ".parallel_tmp" / "2026"
    resume_dir.mkdir(parents=True)
    completed_h5 = resume_dir / "2026.h5"
    completed_h5.write_text("already-complete", encoding="utf-8")
    (resume_dir / "2026.h5.metadata.json").write_text("{}", encoding="utf-8")
    (resume_dir / "calibration_manifest.json").write_text("{}", encoding="utf-8")

    def fake_stream_command(command, env):
        assert completed_h5.exists()

    monkeypatch.setattr(long_term, "_stream_command", fake_stream_command)

    result = long_term.build_long_term_projection(
        years="2026-2100",
        run_id="Run 123",
        source_sha="abc123",
        clear_output=True,
    )

    assert completed_h5.read_text(encoding="utf-8") == "already-complete"
    assert ".parallel_tmp/2026/2026.h5" in result["files"]


def test_remote_clear_output_can_remove_non_resumable_scratch(
    monkeypatch,
    tmp_path,
):
    long_term = _load_long_term_projection_module(monkeypatch)
    monkeypatch.setattr(long_term, "_OUTPUT_MOUNT", tmp_path)

    output_dir = tmp_path / "run-123"
    output_dir.mkdir(parents=True)
    scratch_file = output_dir / "scratch.txt"
    scratch_file.write_text("scratch", encoding="utf-8")

    def fake_stream_command(command, env):
        assert not scratch_file.exists()

    monkeypatch.setattr(long_term, "_stream_command", fake_stream_command)

    result = long_term.build_long_term_projection(
        years="2026-2100",
        run_id="Run 123",
        source_sha="abc123",
        clear_output=True,
    )

    assert "scratch.txt" not in result["files"]
