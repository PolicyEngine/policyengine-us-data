"""Tests for pipeline orchestrator metadata and helpers."""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

modal = pytest.importorskip("modal")

from modal_app.pipeline import (  # noqa: E402
    _build_diagnostics_upload_script,
)
from modal_app.step_manifests.state import RunMetadata  # noqa: E402
from modal_app.step_manifests.store import (  # noqa: E402
    read_run_meta,
    write_run_meta,
)


# -- RunMetadata tests ------------------------------------------


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
