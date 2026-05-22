"""Tests for pipeline orchestrator metadata and helpers."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

modal = pytest.importorskip("modal")

from policyengine_us_data.calibration_package.specs import (  # noqa: E402
    DEFAULT_TARGET_CONFIG_PATH,
)
from policyengine_us_data.utils.manifest import compute_file_checksum  # noqa: E402
from modal_app.pipeline import (  # noqa: E402
    NATIONAL_FIT_LAMBDA_L0,
    _build_diagnostics_upload_script,
    _calibration_package_parameters,
    _new_run_metadata,
    _pipeline_error_summary,
    _promotion_result_from_stdout,
    _release_artifact_metadata_by_path,
    _run_required_promotion_subprocess,
    _stage4_output_contract_repo_path_if_available,
    _traceback_text_for_pipeline_failure,
    _try_reload_pipeline_volume_after_h5_builds,
)
from modal_app.step_manifests.state import RunMetadata  # noqa: E402
from modal_app.step_manifests.store import (  # noqa: E402
    read_run_meta,
    write_run_meta,
)
from policyengine_us_data.build_datasets.commands import DatasetCommandError  # noqa: E402
from policyengine_us_data.build_datasets.results import DatasetCommandResult  # noqa: E402
from policyengine_us_data.build_datasets.status import Stage1ErrorRecord  # noqa: E402
from policyengine_us_data.utils.run_context import RunContext  # noqa: E402
from policyengine_us_data.utils.step_manifest import ArtifactReference  # noqa: E402


# -- RunMetadata tests ------------------------------------------


def test_calibration_package_parameters_track_matrix_mode():
    params = _calibration_package_parameters(
        workers=50,
        n_clones=430,
        target_config=None,
        skip_county=True,
        chunked_matrix=True,
        chunk_size=10_000,
        parallel_matrix=True,
        num_matrix_workers=25,
    )

    assert params["chunked_matrix"] is True
    assert "workers" not in params
    assert params["target_config"] == DEFAULT_TARGET_CONFIG_PATH
    assert params["target_config_sha256"] == (
        f"sha256:{compute_file_checksum(DEFAULT_TARGET_CONFIG_PATH)}"
    )
    assert params["target_config_mode"] == "default"
    assert params["chunk_size"] == 10_000
    assert params["parallel_matrix"] is True
    assert params["num_matrix_workers"] == 25


def test_calibration_package_parameters_ignore_unused_matrix_options():
    params = _calibration_package_parameters(
        workers=50,
        n_clones=430,
        target_config=None,
        skip_county=True,
        chunked_matrix=False,
        chunk_size=10_000,
        parallel_matrix=True,
        num_matrix_workers=25,
    )

    assert params["chunked_matrix"] is False
    assert params["workers"] == 50
    assert params["target_config"] == DEFAULT_TARGET_CONFIG_PATH
    assert params["target_config_sha256"] == (
        f"sha256:{compute_file_checksum(DEFAULT_TARGET_CONFIG_PATH)}"
    )
    assert params["target_config_mode"] == "default"
    assert "chunk_size" not in params
    assert params["parallel_matrix"] is False
    assert "num_matrix_workers" not in params


def test_national_fit_lambda_matches_national_preset():
    assert NATIONAL_FIT_LAMBDA_L0 == pytest.approx(1e-4)


def test_try_reload_pipeline_volume_after_h5_builds_tolerates_modal_open_file():
    class VolumeWithOpenFileConflict:
        def reload(self):
            raise RuntimeError(
                "there are open files preventing the operation: "
                "path artifacts/run/policy_data.db is open"
            )

    assert (
        _try_reload_pipeline_volume_after_h5_builds(VolumeWithOpenFileConflict())
        is False
    )


def test_try_reload_pipeline_volume_after_h5_builds_reraises_other_errors():
    class BrokenVolume:
        def reload(self):
            raise RuntimeError("volume service unavailable")

    with pytest.raises(RuntimeError, match="volume service unavailable"):
        _try_reload_pipeline_volume_after_h5_builds(BrokenVolume())


def test_pipeline_error_summary_uses_traceback_ref_when_available():
    ref = ArtifactReference(
        path="runs/run-1/errors/error.json",
        size_bytes=10,
        sha256="abc",
        role="error",
        media_type="application/json",
    )

    summary = _pipeline_error_summary(
        RuntimeError("boom"),
        traceback_ref=ref,
        traceback_text="full traceback should not be duplicated",
    )

    assert summary == "RuntimeError: boom; traceback_ref=runs/run-1/errors/error.json"


def test_pipeline_error_summary_falls_back_to_bounded_traceback(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "secret-value")
    traceback_text = "old traceback\n" + ("x" * 30_000) + "\nnewest secret-value"

    summary = _pipeline_error_summary(
        RuntimeError("failed with secret-value"),
        traceback_text=traceback_text,
    )

    assert "secret-value" not in summary
    assert summary.startswith("\n[truncated older error text; omitted ")
    assert summary.endswith("newest <redacted:API_TOKEN>")
    assert "old traceback" not in summary


def test_pipeline_failure_traceback_prefers_stage_1_command_tail():
    result = DatasetCommandResult(
        command_name="policyengine_us_data/datasets/cps/extended_cps.py",
        argv=("python", "-m", "policyengine_us_data.datasets.cps.extended_cps"),
        status="failed",
        returncode=1,
        started_at="2026-05-22T12:00:00Z",
        completed_at="2026-05-22T12:00:01Z",
        duration_s=1.0,
        combined_output_tail=("actual ecps failure\n",),
        error=Stage1ErrorRecord(
            substep_id="1c_extended_cps_puf_clone",
            command_name="policyengine_us_data/datasets/cps/extended_cps.py",
            error_type="RuntimeError",
            message="Command failed",
            returncode=1,
            metadata={
                "argv": [
                    "python",
                    "-m",
                    "policyengine_us_data.datasets.cps.extended_cps",
                ],
                "output_tail": ["actual ecps failure\n"],
            },
        ),
    )

    traceback_text = _traceback_text_for_pipeline_failure(
        DatasetCommandError(result),
        "fallback traceback",
    )

    assert "fallback traceback" not in traceback_text
    assert "policyengine_us_data.datasets.cps.extended_cps" in traceback_text
    assert "actual ecps failure" in traceback_text


def test_promotion_result_from_stdout_returns_typed_result():
    result = _promotion_result_from_stdout(
        json.dumps(
            {
                "run_id": "run-123",
                "candidate_version": "1.73.0rc1",
                "release_version": "1.73.0",
                "rel_paths": ("states/AL.h5",),
                "artifact_count": 1,
                "hf_repo_name": "policyengine/policyengine-us-data",
                "hf_repo_type": "model",
                "hf_staging_prefix": "staging/1.73.0rc1-run-123",
                "hf_promoted": 1,
                "hf_promoted_paths": ("states/AL.h5",),
                "hf_commit_id": None,
                "hf_noop_paths": (),
                "gcs_bucket_name": "policyengine-us-data",
                "gcs_uploaded": 1,
                "gcs_object_paths": ("states/AL.h5",),
                "gcs_skipped_paths": (),
                "gcs_failures": (),
                "release_manifest_path": "release_manifest.json",
                "versioned_release_manifest_path": (
                    "releases/1.73.0/release_manifest.json"
                ),
                "trace_tro_path": "trace.tro.jsonld",
                "versioned_trace_tro_path": "releases/1.73.0/trace.tro.jsonld",
                "release_manifest_sha256": None,
                "release_manifest_artifacts": 1,
                "version_manifest_path": "version_manifest.json",
                "version_manifest_version": "1.73.0",
                "version_manifest_current_version": "1.73.0",
                "version_manifest_updated": True,
                "release_completion_marker": ("releases/1.73.0/release-complete.json"),
                "release_completion_tag": "1.73.0",
                "release_completion_valid": True,
                "staging_cleaned": 2,
                "staging_cleanup_attempted": True,
                "staging_cleanup_status": "completed",
            }
        )
    )

    assert result.run_id == "run-123"
    assert result.artifact_count == 1
    assert result.hf.promoted_count == 1


def test_release_artifact_metadata_by_path_uses_local_files(tmp_path, monkeypatch):
    artifact = tmp_path / "states" / "AL.h5"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("state fixture", encoding="utf-8")

    monkeypatch.setattr(
        "modal_app.pipeline._full_release_manifest_files",
        lambda run_id, rel_paths: [(artifact, "states/AL.h5")],
    )

    metadata = _release_artifact_metadata_by_path("run-123", ["states/AL.h5"])

    assert metadata["states/AL.h5"]["sha256"].startswith("sha256:")
    assert metadata["states/AL.h5"]["size_bytes"] == artifact.stat().st_size


def test_stage4_output_contract_repo_path_detects_run_local_contract(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "run-123"
    contract_path = run_dir / "diagnostics" / "contracts" / "output_build_contract.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("modal_app.pipeline._run_dir", lambda run_id: run_dir)

    assert _stage4_output_contract_repo_path_if_available("run-123") == (
        "calibration/runs/run-123/diagnostics/contracts/output_build_contract.json"
    )


def test_new_run_metadata_accepts_release_context_fields_once():
    context = RunContext.from_mapping(
        {
            "run_id": "run-123",
            "candidate_version": "1.73.0-minor",
            "release_version": "",
            "base_release_version": "1.73.0",
            "release_bump": "minor",
            "modal_app_name": "us-data-1-73-0-minor-run-123",
            "modal_environment": "main",
            "hf_staging_prefix": "staging/1.73.0-minor-run-123",
        }
    )

    meta = _new_run_metadata(
        run_id=context.run_id,
        branch="main",
        sha="abc123",
        candidate_version=context.candidate_version,
        release_version=context.release_version,
        run_context=context,
    )

    assert meta.base_release_version == "1.73.0"
    assert meta.release_bump == "minor"
    assert meta.modal_app_name == "us-data-1-73-0-minor-run-123"
    assert meta.hf_staging_prefix == "staging/1.73.0-minor-run-123"
    assert meta.run_context["base_release_version"] == "1.73.0"
    assert meta.run_context["release_bump"] == "minor"


class TestRunMetadata:
    def test_to_dict(self):
        meta = RunMetadata(
            run_id="1.72.3_abc12345_20260319_120000",
            branch="main",
            sha="abc12345deadbeef",
            version="1.72.3",
            start_time="2026-03-19T12:00:00Z",
            status="running",
        )
        d = meta.to_dict()

        assert d["run_id"] == ("1.72.3_abc12345_20260319_120000")
        assert d["branch"] == "main"
        assert d["sha"] == "abc12345deadbeef"
        assert d["version"] == "1.72.3"
        assert d["status"] == "running"
        assert d["error"] is None

    def test_from_dict(self):
        data = {
            "run_id": "1.72.3_abc12345_20260319_120000",
            "branch": "main",
            "sha": "abc12345deadbeef",
            "version": "1.72.3",
            "start_time": "2026-03-19T12:00:00Z",
            "status": "completed",
            "error": None,
        }
        meta = RunMetadata.from_dict(data)

        assert meta.run_id == ("1.72.3_abc12345_20260319_120000")
        assert meta.status == "completed"

    def test_from_dict_maps_legacy_fingerprint_to_regional_scope(self):
        meta = RunMetadata.from_dict(
            {
                "run_id": "test",
                "branch": "main",
                "sha": "abc12345deadbeef",
                "version": "1.72.3",
                "start_time": "2026-03-19T12:00:00Z",
                "status": "running",
                "fingerprint": "legacy-fingerprint",
            }
        )

        assert meta.fingerprint == "legacy-fingerprint"
        assert meta.regional_fingerprint == "legacy-fingerprint"

    def test_from_dict_keeps_explicit_regional_fingerprint_when_both_present(self):
        meta = RunMetadata.from_dict(
            {
                "run_id": "test",
                "branch": "main",
                "sha": "abc12345deadbeef",
                "version": "1.72.3",
                "start_time": "2026-03-19T12:00:00Z",
                "status": "running",
                "fingerprint": "legacy-fingerprint",
                "regional_fingerprint": "regional-fingerprint",
            }
        )

        assert meta.fingerprint == "legacy-fingerprint"
        assert meta.regional_fingerprint == "regional-fingerprint"

    def test_roundtrip(self):
        meta = RunMetadata(
            run_id="1.72.3_abc12345_20260319_120000",
            branch="main",
            sha="abc12345deadbeef",
            version="1.72.3",
            start_time="2026-03-19T12:00:00Z",
            status="failed",
            error="RuntimeError: test",
        )
        roundtripped = RunMetadata.from_dict(meta.to_dict())

        assert roundtripped.run_id == meta.run_id
        assert roundtripped.status == meta.status
        assert roundtripped.error == meta.error

    def test_to_dict_keeps_legacy_fingerprint_alias_in_sync(self):
        meta = RunMetadata(
            run_id="test",
            branch="main",
            sha="abc",
            version="1.0.0",
            start_time="now",
            status="running",
            regional_fingerprint="regional-fp",
        )

        payload = meta.to_dict()

        assert payload["fingerprint"] == "regional-fp"
        assert payload["regional_fingerprint"] == "regional-fp"

    def test_to_dict_preserves_distinct_explicit_regional_fingerprint(self):
        meta = RunMetadata(
            run_id="test",
            branch="main",
            sha="abc",
            version="1.0.0",
            start_time="now",
            status="running",
            fingerprint="legacy-fp",
            regional_fingerprint="regional-fp",
        )

        payload = meta.to_dict()

        assert payload["fingerprint"] == "legacy-fp"
        assert payload["regional_fingerprint"] == "regional-fp"


# -- write/read_run_meta tests --------------------------------


class TestRunMetaIO:
    def test_write_and_read(self, tmp_path):
        meta = RunMetadata(
            run_id="test_run",
            branch="main",
            sha="abc123",
            version="1.0.0",
            start_time="2026-03-19T12:00:00Z",
            status="running",
        )
        mock_vol = MagicMock()

        runs_dir = tmp_path / "runs"

        with patch(
            "modal_app.step_manifests.state.RUNS_DIR",
            str(runs_dir),
        ):
            write_run_meta(meta, mock_vol)
            mock_vol.commit.assert_called_once()

            manifest_path = runs_dir / "test_run" / "run_manifest.json"
            assert manifest_path.exists()
            assert not (runs_dir / "test_run" / "meta.json").exists()

            with open(manifest_path) as f:
                data = json.load(f)
            assert data["run_id"] == "test_run"
            assert data["status"] == "running"
            assert data["known_step_ids"]

            roundtripped = read_run_meta("test_run", mock_vol)
            assert roundtripped.run_id == meta.run_id
            assert roundtripped.start_time == meta.start_time

    def test_read_nonexistent_raises(self):
        mock_vol = MagicMock()

        with patch(
            "modal_app.step_manifests.state.RUNS_DIR",
            "/nonexistent",
        ):
            with pytest.raises(FileNotFoundError):
                read_run_meta("fake_run", mock_vol)


def test_diagnostics_upload_script_is_valid_python(monkeypatch, capsys):
    entries = [
        (
            "/pipeline/runs/test/diagnostics/unified_diagnostics.csv",
            "calibration/runs/test/diagnostics/unified_diagnostics.csv",
        )
    ]
    entries_json = json.dumps(entries)

    script = _build_diagnostics_upload_script(entries_json)

    compile(script, "<diagnostics-upload>", "exec")
    assert "\t" not in script
    assert "api.upload_file(" in script

    calls = []

    class FakeHfApi:
        def upload_file(self, **kwargs):
            calls.append(kwargs)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "token")

    exec(compile(script, "<diagnostics-upload>", "exec"), {})

    assert calls == [
        {
            "path_or_fileobj": entries[0][0],
            "path_in_repo": entries[0][1],
            "repo_id": "policyengine/policyengine-us-data",
            "repo_type": "model",
            "token": "token",
        }
    ]
    assert capsys.readouterr().out == f"Uploaded {entries[0][1]}\n"


def test_required_promotion_subprocess_raises_on_failure(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="missing staged files",
        )

    monkeypatch.setattr("modal_app.pipeline.subprocess.run", fake_run)

    with pytest.raises(
        RuntimeError,
        match="Base dataset promotion failed: missing staged files",
    ):
        _run_required_promotion_subprocess("Base dataset promotion", "print('x')")

    assert captured["cmd"][-1] == "print('x')"
    assert captured["kwargs"]["capture_output"] is True
