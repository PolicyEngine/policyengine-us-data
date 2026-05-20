import pytest

from policyengine_us_data.release_promotion import (
    DEFAULT_RELEASE_DIAGNOSTICS_SOURCES,
    FullPromotionResult,
    ReleasePromotionContext,
    build_release_diagnostics_summary_from_run_dir,
    read_release_diagnostics_summary,
    release_diagnostics_summary_artifact_ref,
    release_diagnostics_summary_from_json,
    release_diagnostics_summary_path,
    release_diagnostics_summary_repo_path,
    release_diagnostics_summary_to_json,
    write_release_diagnostics_summary,
)
from policyengine_us_data.stage_contracts import ArtifactRef
from policyengine_us_data.utils.canonical_json import canonical_json_dumps


def _context() -> ReleasePromotionContext:
    return ReleasePromotionContext(
        run_id="run-123",
        candidate_version="1.73.0rc1",
        release_version="1.73.0",
        hf_repo_name="policyengine/policyengine-us-data",
        gcs_bucket_name="policyengine-us-data",
        base_release_version="1.72.0",
        release_bump="minor",
    )


def _promotion_result(*, run_id: str = "run-123") -> FullPromotionResult:
    return FullPromotionResult.from_legacy_dict(
        {
            "run_id": run_id,
            "candidate_version": "1.73.0rc1",
            "release_version": "1.73.0",
            "artifact_count": 2,
            "hf_promoted": 2,
            "gcs_uploaded": 2,
            "release_manifest_artifacts": 2,
            "version_manifest_updated": True,
            "release_completion_marker": "releases/1.73.0/release-complete.json",
            "staging_cleaned": 3,
            "staging_cleanup_attempted": True,
        }
    )


def _artifact(name: str, relative_path: str) -> ArtifactRef:
    return ArtifactRef(
        logical_name=name,
        uri=f"hf://policyengine/policyengine-us-data/{relative_path}",
        media_type="application/json",
        metadata={
            "artifact_family": name,
            "source_stage_id": "5_validate_and_promote_release",
            "relative_path": relative_path,
        },
    )


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_dumps(payload), encoding="utf-8")


def _run_manifest() -> dict:
    return {
        "run_id": "run-123",
        "branch": "main",
        "sha": "abc123",
        "version": "1.73.0",
        "candidate_version": "1.73.0rc1",
        "release_version": "1.73.0",
        "status": "completed",
        "started_at": "2026-05-20T12:00:00+00:00",
        "completed_at": "2026-05-20T13:00:00+00:00",
        "known_step_ids": ["2_build_calibration_package"],
    }


def _step_manifest(step_id: str, *, outputs: int = 1) -> dict:
    return {
        "run_id": "run-123",
        "step_id": step_id,
        "status": "completed",
        "attempt": 1,
        "started_at": "2026-05-20T12:00:00+00:00",
        "completed_at": "2026-05-20T12:05:00+00:00",
        "duration_s": 300.0,
        "outputs": [
            {"path": f"{step_id}.json", "size_bytes": 1, "sha256": "abc"}
            for _ in range(outputs)
        ],
        "diagnostics": [],
        "reuse_decision": "computed",
        "reuse_measurement": {
            "expected_outputs": outputs,
            "recomputed_outputs": outputs,
            "valid_reused_outputs": 0,
        },
    }


def _stage4_contract() -> dict:
    return {
        "contract_type": "policyengine_us_data.stage_contracts.4_build_outputs",
        "stage_id": "4_build_outputs",
        "run_id": "run-123",
        "created_at": "2026-05-20T12:30:00+00:00",
        "inputs": [],
        "outputs": [{"logical_name": "regional_h5", "uri": "hf://example"}],
        "parameters": {},
        "fingerprint": "fingerprint-123",
        "substages": [],
        "execution": {"status": "completed", "reuse_decision": "computed"},
        "metadata": {},
    }


def _write_complete_sources(run_dir) -> None:
    _write_json(run_dir / "run_manifest.json", _run_manifest())
    for name, source_kind, relative_path in DEFAULT_RELEASE_DIAGNOSTICS_SOURCES:
        if source_kind == "run_manifest":
            continue
        payload = (
            _stage4_contract()
            if source_kind == "stage_contract"
            else _step_manifest(relative_path.removesuffix(".json").split("/")[-1])
        )
        _write_json(run_dir / relative_path, payload)


def test_release_diagnostics_summary_marks_missing_upstream_sources(tmp_path) -> None:
    summary = build_release_diagnostics_summary_from_run_dir(
        run_dir=tmp_path / "run-123",
        context=_context(),
        promotion_result=_promotion_result(),
        generated_at="2026-05-20T13:00:00+00:00",
    )

    assert summary.status == "partial"
    assert "stage_2_calibration_package" in summary.missing_sources
    assert summary.sources["stage_5_promotion_result"].status == "available"
    assert summary.release_promotion["artifact_count"] == 2


def test_release_diagnostics_summary_reads_partial_structured_sources(tmp_path) -> None:
    run_dir = tmp_path / "run-123"
    _write_json(run_dir / "run_manifest.json", _run_manifest())
    _write_json(
        run_dir / "steps" / "2_build_calibration_package.json",
        _step_manifest("2_build_calibration_package", outputs=2),
    )

    summary_path = release_diagnostics_summary_path(run_dir)
    summary = build_release_diagnostics_summary_from_run_dir(
        run_dir=run_dir,
        context=_context(),
        promotion_result=_promotion_result(),
        generated_at="2026-05-20T13:00:00+00:00",
        artifact_refs=(
            _artifact(
                "release_promotion_contract",
                "calibration/runs/run-123/diagnostics/contracts/release_promotion_contract.json",
            ),
        ),
    )
    written = write_release_diagnostics_summary(summary, summary_path)

    assert written == summary
    assert read_release_diagnostics_summary(summary_path) == summary
    assert summary.sources["run_manifest"].facts["status"] == "completed"
    assert summary.sources["stage_2_calibration_package"].facts["output_count"] == 2
    assert summary.sources["stage_3_weight_fitting_regional"].status == "missing"
    assert "release_promotion_contract" in summary.artifacts


def test_release_diagnostics_summary_records_complete_sources(tmp_path) -> None:
    run_dir = tmp_path / "run-123"
    _write_complete_sources(run_dir)

    summary = build_release_diagnostics_summary_from_run_dir(
        run_dir=run_dir,
        context=_context(),
        promotion_result=_promotion_result(),
        generated_at="2026-05-20T13:00:00+00:00",
    )

    payload = release_diagnostics_summary_to_json(summary)
    restored = release_diagnostics_summary_from_json(payload)

    assert summary.status == "complete"
    assert summary.missing_sources == ()
    assert summary.sources["stage_4_output_contract"].facts["fingerprint"] == (
        "fingerprint-123"
    )
    assert restored == summary


def test_release_diagnostics_summary_artifact_ref_records_source_counts() -> None:
    summary = build_release_diagnostics_summary_from_run_dir(
        run_dir="/missing/run-123",
        context=_context(),
        promotion_result=_promotion_result(),
        generated_at="2026-05-20T13:00:00+00:00",
    )

    artifact = release_diagnostics_summary_artifact_ref(
        _context(),
        summary,
        sha256="sha256:summary",
        size_bytes=123,
    )

    assert artifact.logical_name == "release_diagnostics_summary"
    assert artifact.media_type == "application/json"
    assert artifact.metadata["summary_status"] == "partial"
    assert artifact.metadata["missing_source_count"] == len(summary.missing_sources)
    assert release_diagnostics_summary_repo_path("run-123") == (
        "calibration/runs/run-123/diagnostics/release_diagnostics_summary.json"
    )


def test_release_diagnostics_summary_rejects_mismatched_result_identity() -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_release_diagnostics_summary_from_run_dir(
            run_dir="/missing/run-123",
            context=_context(),
            promotion_result=_promotion_result(run_id="other-run"),
            generated_at="2026-05-20T13:00:00+00:00",
        )
