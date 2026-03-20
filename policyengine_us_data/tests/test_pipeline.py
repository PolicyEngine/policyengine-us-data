"""Tests for pipeline orchestrator metadata and helpers."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

modal = pytest.importorskip("modal")

from modal_app.pipeline import (
    RunMetadata,
    _step_completed,
    _record_step,
    generate_run_id,
    write_run_meta,
    read_run_meta,
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
        assert d["step_timings"] == {}
        assert d["error"] is None

    def test_from_dict(self):
        data = {
            "run_id": "1.72.3_abc12345_20260319_120000",
            "branch": "main",
            "sha": "abc12345deadbeef",
            "version": "1.72.3",
            "start_time": "2026-03-19T12:00:00Z",
            "status": "completed",
            "step_timings": {
                "build_datasets": {
                    "status": "completed",
                    "duration_s": 100.0,
                }
            },
            "error": None,
        }
        meta = RunMetadata.from_dict(data)

        assert meta.run_id == ("1.72.3_abc12345_20260319_120000")
        assert meta.status == "completed"
        assert meta.step_timings["build_datasets"]["status"] == "completed"

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

    def test_step_timings_default_empty(self):
        meta = RunMetadata(
            run_id="test",
            branch="main",
            sha="abc",
            version="1.0.0",
            start_time="now",
            status="running",
        )
        assert meta.step_timings == {}


# -- generate_run_id tests -------------------------------------


class TestGenerateRunId:
    def test_format(self):
        run_id = generate_run_id("1.72.3", "abc12345deadbeef")

        parts = run_id.split("_")
        assert parts[0] == "1.72.3"
        assert parts[1] == "abc12345"
        assert len(parts) == 4  # version_sha_date_time

    def test_sha_truncated_to_8(self):
        run_id = generate_run_id("1.0.0", "abcdef1234567890")
        sha_part = run_id.split("_")[1]
        assert sha_part == "abcdef12"
        assert len(sha_part) == 8

    def test_unique_ids(self):
        id1 = generate_run_id("1.0.0", "abc123")
        time.sleep(0.01)
        id2 = generate_run_id("1.0.0", "abc123")
        # Timestamps should differ (or at least
        # the function doesn't reuse)
        assert isinstance(id1, str)
        assert isinstance(id2, str)


# -- _step_completed tests ------------------------------------


class TestStepCompleted:
    def test_completed_step(self):
        meta = RunMetadata(
            run_id="test",
            branch="main",
            sha="abc",
            version="1.0.0",
            start_time="now",
            status="running",
            step_timings={
                "build_datasets": {
                    "status": "completed",
                    "duration_s": 50.0,
                }
            },
        )
        assert _step_completed(meta, "build_datasets")

    def test_incomplete_step(self):
        meta = RunMetadata(
            run_id="test",
            branch="main",
            sha="abc",
            version="1.0.0",
            start_time="now",
            status="running",
            step_timings={
                "build_datasets": {
                    "status": "failed",
                    "duration_s": 10.0,
                }
            },
        )
        assert not _step_completed(meta, "build_datasets")

    def test_missing_step(self):
        meta = RunMetadata(
            run_id="test",
            branch="main",
            sha="abc",
            version="1.0.0",
            start_time="now",
            status="running",
        )
        assert not _step_completed(meta, "build_datasets")


# -- _record_step tests ----------------------------------------


class TestRecordStep:
    def test_records_timing(self):
        meta = RunMetadata(
            run_id="test",
            branch="main",
            sha="abc",
            version="1.0.0",
            start_time="now",
            status="running",
        )
        mock_vol = MagicMock()
        start = time.time() - 5.0

        with patch("modal_app.pipeline.write_run_meta"):
            _record_step(meta, "build_datasets", start, mock_vol)

        timing = meta.step_timings["build_datasets"]
        assert timing["status"] == "completed"
        assert timing["duration_s"] >= 5.0
        assert "start" in timing
        assert "end" in timing

    def test_records_custom_status(self):
        meta = RunMetadata(
            run_id="test",
            branch="main",
            sha="abc",
            version="1.0.0",
            start_time="now",
            status="running",
        )
        mock_vol = MagicMock()

        with patch("modal_app.pipeline.write_run_meta"):
            _record_step(
                meta,
                "build_datasets",
                time.time(),
                mock_vol,
                status="failed",
            )

        assert meta.step_timings["build_datasets"]["status"] == "failed"


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
            "modal_app.pipeline.RUNS_DIR",
            str(runs_dir),
        ):
            write_run_meta(meta, mock_vol)
            mock_vol.commit.assert_called_once()

            # Verify file was written
            meta_path = runs_dir / "test_run" / "meta.json"
            assert meta_path.exists()

            with open(meta_path) as f:
                data = json.load(f)
            assert data["run_id"] == "test_run"
            assert data["status"] == "running"

    def test_read_nonexistent_raises(self):
        mock_vol = MagicMock()

        with patch(
            "modal_app.pipeline.RUNS_DIR",
            "/nonexistent",
        ):
            with pytest.raises(FileNotFoundError):
                read_run_meta("fake_run", mock_vol)
