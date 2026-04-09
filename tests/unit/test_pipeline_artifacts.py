"""Tests for pipeline artifact utilities."""

import re
from unittest.mock import patch

from policyengine_us_data.utils.pipeline_artifacts import (
    PIPELINE_REPO,
    PIPELINE_REPO_TYPE,
    generate_stage_manifest,
    get_pipeline_run_id,
    mirror_to_pipeline,
)


class TestGetPipelineRunId:
    def test_format(self):
        run_id = get_pipeline_run_id()
        assert re.match(r"^\d{8}T\d{6}Z$", run_id)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_RUN_ID", "20260101T000000Z")
        assert get_pipeline_run_id() == "20260101T000000Z"

    def test_env_var_not_set(self, monkeypatch):
        monkeypatch.delenv("PIPELINE_RUN_ID", raising=False)
        run_id = get_pipeline_run_id()
        assert re.match(r"^\d{8}T\d{6}Z$", run_id)


class TestGenerateStageManifest:
    def test_schema(self, tmp_path):
        f1 = tmp_path / "data.h5"
        f1.write_bytes(b"fake h5 content")
        f2 = tmp_path / "weights.npy"
        f2.write_bytes(b"fake weights")

        manifest = generate_stage_manifest("stage_1_base", "20260317T143000Z", [f1, f2])

        assert manifest["stage"] == "stage_1_base"
        assert manifest["run_id"] == "20260317T143000Z"
        assert "created_at" in manifest
        assert "git_commit" in manifest
        assert "git_branch" in manifest
        assert "git_dirty" in manifest
        assert "data.h5" in manifest["files"]
        assert "weights.npy" in manifest["files"]

    def test_file_checksums(self, tmp_path):
        f1 = tmp_path / "test.bin"
        f1.write_bytes(b"deterministic content")

        manifest = generate_stage_manifest("stage_0_raw", "20260317T143000Z", [f1])

        entry = manifest["files"]["test.bin"]
        assert "sha256" in entry
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] == len(b"deterministic content")

    def test_missing_file_skipped(self, tmp_path):
        existing = tmp_path / "exists.h5"
        existing.write_bytes(b"data")
        missing = tmp_path / "missing.h5"

        manifest = generate_stage_manifest(
            "stage_1_base",
            "20260317T143000Z",
            [existing, missing],
        )

        assert "exists.h5" in manifest["files"]
        assert "missing.h5" not in manifest["files"]

    def test_empty_files_list(self):
        manifest = generate_stage_manifest("stage_0_raw", "20260317T143000Z", [])
        assert manifest["files"] == {}


class TestMirrorToPipeline:
    @patch("policyengine_us_data.utils.pipeline_artifacts.hf_create_commit_with_retry")
    def test_uploads_files_and_manifest(self, mock_commit, tmp_path):
        f1 = tmp_path / "cps_2024.h5"
        f1.write_bytes(b"cps data")

        run_id = mirror_to_pipeline(
            "stage_1_base",
            [f1],
            run_id="20260317T143000Z",
        )

        assert run_id == "20260317T143000Z"
        mock_commit.assert_called_once()
        call_kwargs = mock_commit.call_args
        ops = call_kwargs.kwargs.get("operations", call_kwargs[1].get("operations"))

        paths = [op.path_in_repo for op in ops]
        assert any("manifest.json" in p for p in paths)
        assert any("cps_2024.h5" in p for p in paths)

        assert (
            call_kwargs.kwargs.get("repo_id", call_kwargs[1].get("repo_id"))
            == PIPELINE_REPO
        )
        assert (
            call_kwargs.kwargs.get("repo_type", call_kwargs[1].get("repo_type"))
            == PIPELINE_REPO_TYPE
        )

    @patch("policyengine_us_data.utils.pipeline_artifacts.hf_create_commit_with_retry")
    def test_manifest_only(self, mock_commit, tmp_path):
        f1 = tmp_path / "SC.h5"
        f1.write_bytes(b"state data")

        mirror_to_pipeline(
            "stage_7_local_area",
            [f1],
            run_id="20260317T143000Z",
            manifest_only=True,
        )

        call_kwargs = mock_commit.call_args
        ops = call_kwargs.kwargs.get("operations", call_kwargs[1].get("operations"))

        paths = [op.path_in_repo for op in ops]
        assert len(ops) == 1
        assert "manifest.json" in paths[0]

    @patch("policyengine_us_data.utils.pipeline_artifacts.hf_create_commit_with_retry")
    def test_returns_run_id_when_none(self, mock_commit, tmp_path):
        f1 = tmp_path / "test.bin"
        f1.write_bytes(b"data")

        run_id = mirror_to_pipeline("stage_0_raw", [f1])
        assert re.match(r"^\d{8}T\d{6}Z$", run_id)

    @patch(
        "policyengine_us_data.utils.pipeline_artifacts.hf_create_commit_with_retry",
        side_effect=Exception("No token"),
    )
    def test_error_does_not_raise(self, mock_commit, tmp_path):
        f1 = tmp_path / "test.bin"
        f1.write_bytes(b"data")

        # Should not raise.
        run_id = mirror_to_pipeline(
            "stage_0_raw",
            [f1],
            run_id="20260317T143000Z",
        )
        assert run_id == "20260317T143000Z"

    @patch("policyengine_us_data.utils.pipeline_artifacts.hf_create_commit_with_retry")
    def test_folder_structure(self, mock_commit, tmp_path):
        f1 = tmp_path / "weights.npy"
        f1.write_bytes(b"weights")

        mirror_to_pipeline(
            "stage_6_weights",
            [f1],
            run_id="20260317T143000Z",
        )

        call_kwargs = mock_commit.call_args
        ops = call_kwargs.kwargs.get("operations", call_kwargs[1].get("operations"))

        for op in ops:
            assert op.path_in_repo.startswith("20260317T143000Z/stage_6_weights/")
