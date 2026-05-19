from pathlib import Path

import pytest

from policyengine_us_data.release_promotion import FullPromotionResult
from policyengine_us_data.utils.release_promotion import (
    FullReleasePromotionConfig,
    FullReleasePromotionDependencies,
    promote_full_release_with_result,
)


def _make_files(tmp_path, rel_paths):
    files = []
    for rel_path in rel_paths:
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel_path, encoding="utf-8")
        files.append((path, rel_path))
    return tuple(files)


class FakeFullReleasePromotionDependencies:
    def __init__(
        self,
        *,
        finalized_manifest=None,
        marker_exists=True,
        missing_staged_artifacts=(),
        preflight_result=(True, []),
        cleanup_error: Exception | None = None,
    ) -> None:
        self.finalized_manifest = finalized_manifest
        self.marker_exists = marker_exists
        self.missing_staged_artifacts = tuple(missing_staged_artifacts)
        self.preflight_result = preflight_result
        self.cleanup_error = cleanup_error
        self.calls = []

    def as_dependencies(self) -> FullReleasePromotionDependencies:
        return FullReleasePromotionDependencies(
            dedupe_preserving_order=self.dedupe_preserving_order,
            download_staged_artifacts_for_manifest=(
                self.download_staged_artifacts_for_manifest
            ),
            get_matching_finalized_release_manifest=(
                self.get_matching_finalized_release_manifest
            ),
            list_missing_staged_artifacts=self.list_missing_staged_artifacts,
            preflight_release_manifest_publish=self.preflight_release_manifest_publish,
            promote_staging_to_production_hf=self.promote_staging_to_production_hf,
            upload_from_hf_staging_to_gcs=self.upload_from_hf_staging_to_gcs,
            publish_release_manifest_to_hf=self.publish_release_manifest_to_hf,
            upload_final_version_manifest=self.upload_final_version_manifest,
            upload_release_completion_marker=self.upload_release_completion_marker,
            release_completion_marker_exists=self.release_completion_marker_exists,
            cleanup_staging_hf=self.cleanup_staging_hf,
        )

    def dedupe_preserving_order(self, paths):
        seen = set()
        deduped = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                deduped.append(path)
        return deduped

    def download_staged_artifacts_for_manifest(self, *args, **kwargs):
        self.calls.append("download")
        return []

    def get_matching_finalized_release_manifest(self, *args, **kwargs):
        self.calls.append("check_finalized")
        return self.finalized_manifest

    def list_missing_staged_artifacts(self, *args, **kwargs):
        self.calls.append("validate_staging")
        return list(self.missing_staged_artifacts)

    def preflight_release_manifest_publish(self, *args, **kwargs):
        self.calls.append("preflight_manifest")
        return self.preflight_result

    def promote_staging_to_production_hf(self, paths, *args, **kwargs):
        self.calls.append("promote_hf")
        return len(paths)

    def upload_from_hf_staging_to_gcs(self, paths, *args, **kwargs):
        self.calls.append("upload_gcs")
        return len(paths)

    def publish_release_manifest_to_hf(self, files_with_paths, *args, **kwargs):
        self.calls.append("release_manifest")
        return {
            "artifacts": {
                Path(repo_path).with_suffix("").as_posix(): {
                    "path": repo_path,
                    "sha256": f"sha256:{repo_path}",
                }
                for _, repo_path in files_with_paths
            }
        }

    def upload_final_version_manifest(self, *args, **kwargs):
        self.calls.append("version_manifest")

    def upload_release_completion_marker(self, *args, **kwargs):
        self.calls.append("release_complete")
        return {"marker_path": "releases/1.73.0/release-complete.json"}

    def release_completion_marker_exists(self, *args, **kwargs):
        self.calls.append("check_marker")
        return self.marker_exists

    def cleanup_staging_hf(self, paths, *args, **kwargs):
        self.calls.append("cleanup_staging")
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return len(paths)


def _config(
    rel_paths,
    files_with_paths,
    *,
    cleanup_staging=True,
) -> FullReleasePromotionConfig:
    return FullReleasePromotionConfig(
        rel_paths=rel_paths,
        candidate_version="1.73.0rc1",
        release_version="1.73.0",
        run_id="run-123",
        files_with_paths=files_with_paths,
        extra_cleanup_paths=("_run_context.json",),
        cleanup_staging=cleanup_staging,
    )


def test_full_promotion_result_wraps_legacy_dict() -> None:
    result = FullPromotionResult.from_legacy_dict(
        {
            "run_id": "run-123",
            "candidate_version": "1.73.0rc1",
            "release_version": "1.73.0",
            "artifact_count": 2,
            "hf_promoted": 2,
            "gcs_uploaded": 2,
            "release_manifest_artifacts": 2,
            "release_completion_marker": "releases/1.73.0/release-complete.json",
            "staging_cleaned": 3,
        }
    )

    assert result.hf.promoted_count == 2
    assert result.gcs.uploaded_count == 2
    assert result.release_manifest.artifact_count == 2
    assert result.version_manifest.updated is True
    assert result.completion_marker.marker_path == (
        "releases/1.73.0/release-complete.json"
    )
    assert result.cleanup.cleaned_count == 3
    assert FullPromotionResult.from_dict(result.to_dict()) == result


def test_promote_full_release_with_result_preserves_transaction_order(tmp_path) -> None:
    rel_paths = ("cps_2024.h5", "states/AL.h5", "national/US.h5")
    files = _make_files(tmp_path, rel_paths)
    fake_deps = FakeFullReleasePromotionDependencies()

    result = promote_full_release_with_result(
        _config(rel_paths, files),
        fake_deps.as_dependencies(),
    )

    assert fake_deps.calls == [
        "check_finalized",
        "validate_staging",
        "preflight_manifest",
        "promote_hf",
        "upload_gcs",
        "release_manifest",
        "version_manifest",
        "release_complete",
        "cleanup_staging",
    ]
    assert isinstance(result, FullPromotionResult)
    assert result.run_id == "run-123"
    assert result.artifact_count == 3
    assert result.hf.promoted_count == 3
    assert result.gcs.uploaded_count == 3
    assert result.release_manifest.artifact_count == 3
    assert result.version_manifest.updated is True
    assert result.cleanup.cleaned_count == 4
    assert result.already_finalized is False


def test_promote_full_release_with_result_handles_already_finalized(tmp_path) -> None:
    rel_paths = ("states/AL.h5",)
    files = _make_files(tmp_path, rel_paths)
    fake_deps = FakeFullReleasePromotionDependencies(
        finalized_manifest={"artifacts": {"states/AL": {"path": "states/AL.h5"}}},
        marker_exists=True,
    )

    result = promote_full_release_with_result(
        _config(rel_paths, files),
        fake_deps.as_dependencies(),
    )

    assert fake_deps.calls == ["check_finalized", "check_marker", "cleanup_staging"]
    assert result.already_finalized is True
    assert result.hf.promoted_count == 0
    assert result.gcs.uploaded_count == 0
    assert result.release_manifest.artifact_count == 1
    assert result.version_manifest.updated is False
    assert result.completion_marker.marker_path == (
        "releases/1.73.0/release-complete.json"
    )


def test_promote_full_release_with_result_represents_cleanup_failure(tmp_path) -> None:
    rel_paths = ("states/AL.h5",)
    files = _make_files(tmp_path, rel_paths)
    fake_deps = FakeFullReleasePromotionDependencies(
        cleanup_error=RuntimeError("cleanup unavailable"),
    )

    result = promote_full_release_with_result(
        _config(rel_paths, files),
        fake_deps.as_dependencies(),
    )

    assert "cleanup_staging" in fake_deps.calls
    assert result.cleanup.attempted is True
    assert result.cleanup.cleaned_count == 0
    assert result.hf.promoted_count == 1
    assert result.gcs.uploaded_count == 1


def test_promote_full_release_with_result_represents_skipped_cleanup(tmp_path) -> None:
    rel_paths = ("states/AL.h5",)
    files = _make_files(tmp_path, rel_paths)
    fake_deps = FakeFullReleasePromotionDependencies()

    result = promote_full_release_with_result(
        _config(rel_paths, files, cleanup_staging=False),
        fake_deps.as_dependencies(),
    )

    assert "cleanup_staging" not in fake_deps.calls
    assert result.cleanup.attempted is False
    assert result.cleanup.cleaned_count == 0


def test_promote_full_release_with_result_fails_before_public_writes(tmp_path) -> None:
    rel_paths = ("states/AL.h5",)
    files = _make_files(tmp_path, rel_paths)
    fake_deps = FakeFullReleasePromotionDependencies(
        missing_staged_artifacts=("staging/1.73.0rc1-run-123/states/AL.h5",),
    )

    with pytest.raises(FileNotFoundError, match="Missing staged release artifacts"):
        promote_full_release_with_result(
            _config(rel_paths, files),
            fake_deps.as_dependencies(),
        )

    assert fake_deps.calls == ["check_finalized", "validate_staging"]
