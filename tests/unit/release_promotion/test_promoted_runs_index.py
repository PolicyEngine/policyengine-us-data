import pytest

from policyengine_us_data.release_promotion import (
    FullPromotionResult,
    ReleasePromotionContext,
    build_promoted_run_index_entry,
    load_promoted_runs_index,
    promoted_runs_index_artifact_ref,
    promoted_runs_index_from_json,
    promoted_runs_index_path,
    promoted_runs_index_repo_path,
    promoted_runs_index_to_json,
    read_promoted_runs_index,
    update_promoted_runs_index,
)


def _context(
    *,
    run_id: str = "run-123",
    candidate_version: str = "1.73.0rc1",
    release_version: str = "1.73.0",
) -> ReleasePromotionContext:
    return ReleasePromotionContext(
        run_id=run_id,
        candidate_version=candidate_version,
        release_version=release_version,
        hf_repo_name="policyengine/policyengine-us-data",
        gcs_bucket_name="policyengine-us-data",
        base_release_version="1.72.0",
        release_bump="minor",
    )


def _promotion_result(
    *,
    run_id: str = "run-123",
    candidate_version: str = "1.73.0rc1",
    release_version: str = "1.73.0",
    already_finalized: bool = False,
) -> FullPromotionResult:
    return FullPromotionResult.from_legacy_dict(
        {
            "run_id": run_id,
            "candidate_version": candidate_version,
            "release_version": release_version,
            "artifact_count": 2,
            "hf_promoted": 0 if already_finalized else 2,
            "gcs_uploaded": 0 if already_finalized else 2,
            "release_manifest_artifacts": 2,
            "version_manifest_updated": not already_finalized,
            "release_completion_marker": "releases/1.73.0/release-complete.json",
            "staging_cleaned": 3,
            "staging_cleanup_attempted": True,
            "already_finalized": already_finalized,
        }
    )


def _entry(
    *,
    run_id: str = "run-123",
    candidate_version: str = "1.73.0rc1",
    release_version: str = "1.73.0",
    already_finalized: bool = False,
    promoted_at: str = "2026-05-20T12:00:00+00:00",
):
    return build_promoted_run_index_entry(
        context=_context(
            run_id=run_id,
            candidate_version=candidate_version,
            release_version=release_version,
        ),
        promotion_result=_promotion_result(
            run_id=run_id,
            candidate_version=candidate_version,
            release_version=release_version,
            already_finalized=already_finalized,
        ),
        promoted_at=promoted_at,
        release_promotion_contract_path=(
            f"calibration/runs/{run_id}/diagnostics/contracts/"
            "release_promotion_contract.json"
        ),
        published_artifact_index_path=(
            f"calibration/runs/{run_id}/diagnostics/published_artifact_index.jsonl"
        ),
        run_manifest_path=f"calibration/runs/{run_id}/run_manifest.json",
        step_manifest_path=(
            f"calibration/runs/{run_id}/steps/5_validate_and_promote_release.json"
        ),
        metadata={"branch": "main", "package_version": release_version},
    )


def test_promoted_runs_index_creates_run_oriented_discovery_file(tmp_path) -> None:
    path = promoted_runs_index_path(tmp_path / "runs")
    entry = _entry()

    index, update = update_promoted_runs_index(
        path=path,
        entry=entry,
        updated_at="2026-05-20T12:00:01+00:00",
    )

    assert update.status == "created"
    assert update.run_count == 1
    assert update.release_version_run_count == 1
    assert index.runs["run-123"].run_id == "run-123"
    assert index.runs["run-123"].status == "promoted"
    assert index.release_versions["1.73.0"].latest_run_id == "run-123"
    assert index.release_versions["1.73.0"].run_ids == ("run-123",)
    assert read_promoted_runs_index(path) == index
    assert promoted_runs_index_from_json(promoted_runs_index_to_json(index)) == index
    assert promoted_runs_index_repo_path() == "calibration/runs/index.json"


def test_promoted_runs_index_updates_same_run_without_duplicates(tmp_path) -> None:
    path = promoted_runs_index_path(tmp_path / "runs")
    update_promoted_runs_index(
        path=path,
        entry=_entry(),
        updated_at="2026-05-20T12:00:01+00:00",
    )

    index, update = update_promoted_runs_index(
        path=path,
        entry=_entry(
            already_finalized=True,
            promoted_at="2026-05-20T12:05:00+00:00",
        ),
        updated_at="2026-05-20T12:05:01+00:00",
    )

    assert update.status == "updated"
    assert update.already_finalized is True
    assert len(index.runs) == 1
    assert index.runs["run-123"].already_finalized is True
    assert index.release_versions["1.73.0"].run_ids == ("run-123",)


def test_promoted_runs_index_tracks_duplicate_release_versions_once_per_run(
    tmp_path,
) -> None:
    path = promoted_runs_index_path(tmp_path / "runs")
    update_promoted_runs_index(
        path=path,
        entry=_entry(),
        updated_at="2026-05-20T12:00:01+00:00",
    )
    update_promoted_runs_index(
        path=path,
        entry=_entry(
            run_id="run-456",
            candidate_version="1.73.0rc2",
            promoted_at="2026-05-20T13:00:00+00:00",
        ),
        updated_at="2026-05-20T13:00:01+00:00",
    )

    index, update = update_promoted_runs_index(
        path=path,
        entry=_entry(
            run_id="run-456",
            candidate_version="1.73.0rc2",
            promoted_at="2026-05-20T13:05:00+00:00",
        ),
        updated_at="2026-05-20T13:05:01+00:00",
    )

    release = index.release_versions["1.73.0"]
    assert update.status == "updated"
    assert len(index.runs) == 2
    assert release.latest_run_id == "run-456"
    assert release.run_ids == ("run-123", "run-456")
    assert update.release_version_run_count == 2


def test_promoted_runs_index_artifact_ref_records_update_status(tmp_path) -> None:
    index, update = update_promoted_runs_index(
        path=promoted_runs_index_path(tmp_path / "runs"),
        entry=_entry(),
        updated_at="2026-05-20T12:00:01+00:00",
    )

    artifact = promoted_runs_index_artifact_ref(
        _context(),
        update,
        sha256="sha256:index",
        size_bytes=123,
    )

    assert len(index.runs) == 1
    assert artifact.logical_name == "promoted_runs_index"
    assert artifact.media_type == "application/json"
    assert artifact.metadata["update_status"] == "created"
    assert artifact.metadata["relative_path"] == "calibration/runs/index.json"


def test_load_promoted_runs_index_returns_empty_index_for_missing_file(
    tmp_path,
) -> None:
    index = load_promoted_runs_index(
        promoted_runs_index_path(tmp_path / "runs"),
        updated_at="2026-05-20T12:00:00+00:00",
    )

    assert index.runs == {}
    assert index.release_versions == {}


def test_promoted_run_entry_rejects_mismatched_result_identity() -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_promoted_run_index_entry(
            context=_context(),
            promotion_result=_promotion_result(run_id="other-run"),
            promoted_at="2026-05-20T12:00:00+00:00",
        )
