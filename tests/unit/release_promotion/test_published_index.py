import json

import pytest

from policyengine_us_data.release_promotion import (
    FullPromotionResult,
    ReleasePromotionContext,
    build_legacy_release_candidate_bundle,
    build_published_artifact_index,
    published_artifact_index_artifact_ref,
    published_artifact_index_from_jsonl,
    published_artifact_index_path,
    published_artifact_index_repo_path,
    published_artifact_index_to_jsonl,
    read_published_artifact_index,
    write_published_artifact_index,
)
from policyengine_us_data.stage_contracts import ArtifactRef


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


def _rel_paths() -> list[str]:
    return [
        "policy_data.db",
        "states/AL.h5",
        "districts/NC-01.h5",
        "cities/NYC.h5",
        "national/US.h5",
    ]


def _candidate_bundle():
    return build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=_rel_paths(),
        artifact_metadata_by_path={
            path: {"sha256": f"sha256:candidate-{index}", "size_bytes": index}
            for index, path in enumerate(_rel_paths(), start=1)
        },
    )


def _legacy_promotion_payload(**overrides):
    rel_paths = tuple(_rel_paths())
    payload = {
        "run_id": "run-123",
        "candidate_version": "1.73.0rc1",
        "release_version": "1.73.0",
        "rel_paths": rel_paths,
        "artifact_count": len(rel_paths),
        "hf_repo_name": "policyengine/policyengine-us-data",
        "hf_repo_type": "model",
        "hf_staging_prefix": "staging/1.73.0rc1-run-123",
        "hf_promoted": len(rel_paths),
        "hf_promoted_paths": rel_paths,
        "hf_commit_id": None,
        "hf_noop_paths": (),
        "gcs_bucket_name": "policyengine-us-data",
        "gcs_uploaded": len(rel_paths),
        "gcs_object_paths": rel_paths,
        "gcs_skipped_paths": (),
        "gcs_failures": (),
        "release_manifest_path": "release_manifest.json",
        "versioned_release_manifest_path": "releases/1.73.0/release_manifest.json",
        "trace_tro_path": "trace.tro.jsonld",
        "versioned_trace_tro_path": "releases/1.73.0/trace.tro.jsonld",
        "release_manifest_sha256": None,
        "release_manifest_artifacts": len(rel_paths),
        "version_manifest_path": "version_manifest.json",
        "version_manifest_version": "1.73.0",
        "version_manifest_current_version": "1.73.0",
        "version_manifest_updated": True,
        "release_completion_marker": "releases/1.73.0/release-complete.json",
        "release_completion_tag": "1.73.0",
        "release_completion_valid": True,
        "staging_cleaned": len(rel_paths) + 1,
        "staging_cleanup_attempted": True,
        "staging_cleanup_status": "completed",
    }
    payload.update(overrides)
    return payload


def _promotion_result() -> FullPromotionResult:
    return FullPromotionResult.from_legacy_dict(_legacy_promotion_payload())


def _release_manifest() -> dict:
    return {
        "artifacts": {
            path.removesuffix(".h5").removesuffix(".db"): {
                "kind": "database" if path.endswith(".db") else "microdata",
                "path": path,
                "repo_id": "policyengine/policyengine-us-data",
                "revision": "1.73.0",
                "sha256": f"manifest-{index}",
                "size_bytes": 100 + index,
            }
            for index, path in enumerate(_rel_paths(), start=1)
        }
    }


def _diagnostic_artifact() -> ArtifactRef:
    return ArtifactRef(
        logical_name="release_promotion_contract",
        uri=(
            "hf://policyengine/policyengine-us-data/calibration/runs/"
            "run-123/diagnostics/contracts/release_promotion_contract.json"
        ),
        media_type="application/json",
        metadata={
            "artifact_family": "stage_contract",
            "source_stage_id": "5_validate_and_promote_release",
            "relative_path": (
                "calibration/runs/run-123/diagnostics/contracts/"
                "release_promotion_contract.json"
            ),
        },
    )


def _rows():
    return build_published_artifact_index(
        candidate_bundle=_candidate_bundle(),
        promotion_result=_promotion_result(),
        release_manifest=_release_manifest(),
        diagnostic_artifacts=(_diagnostic_artifact(),),
    )


def _by_path(rows):
    return {row.relative_path: row for row in rows}


def test_published_artifact_index_records_release_artifact_destinations() -> None:
    rows = _by_path(_rows())

    state = rows["states/AL.h5"]
    district = rows["districts/NC-01.h5"]
    city = rows["cities/NYC.h5"]
    national = rows["national/US.h5"]
    base = rows["policy_data.db"]

    assert state.artifact_family == "state_h5"
    assert state.area_type == "state"
    assert state.area_id == "AL"
    assert state.hf_uri == "hf://policyengine/policyengine-us-data/states/AL.h5"
    assert state.gcs_uri == "gs://policyengine-us-data/states/AL.h5"
    assert state.release_manifest_key == "states/AL"
    assert state.release_manifest_revision == "1.73.0"
    assert state.sha256 == "manifest-2"
    assert district.artifact_family == "district_h5"
    assert district.area_id == "NC-01"
    assert city.artifact_family == "city_h5"
    assert city.area_id == "NYC"
    assert national.artifact_family == "national_h5"
    assert national.area_type == "national"
    assert base.artifact_family == "base_dataset"
    assert base.source_stage_id == "1_build_datasets"
    assert base.gcs_uri == "gs://policyengine-us-data/policy_data.db"


def test_published_artifact_index_records_manifest_and_diagnostic_rows() -> None:
    rows = _by_path(_rows())

    release_manifest = rows["release_manifest.json"]
    version_manifest = rows["version_manifest.json"]
    completion_marker = rows["releases/1.73.0/release-complete.json"]
    diagnostic = rows[
        "calibration/runs/run-123/diagnostics/contracts/release_promotion_contract.json"
    ]

    assert release_manifest.artifact_role == "release_metadata"
    assert release_manifest.artifact_family == "release_manifest"
    assert release_manifest.gcs_uri is None
    assert version_manifest.artifact_family == "version_manifest"
    assert completion_marker.artifact_family == "release_completion_marker"
    assert diagnostic.artifact_role == "diagnostic"
    assert diagnostic.artifact_family == "stage_contract"
    assert diagnostic.hf_uri.endswith("release_promotion_contract.json")


def test_published_artifact_index_uses_typed_promotion_metadata_paths() -> None:
    result = FullPromotionResult.from_legacy_dict(
        _legacy_promotion_payload(
            release_manifest_path="manifests/root-release.json",
            versioned_release_manifest_path="release-ledger/1.73.0/root-release.json",
            trace_tro_path="manifests/root-trace.jsonld",
            versioned_trace_tro_path="release-ledger/1.73.0/root-trace.jsonld",
            release_manifest_sha256="sha256:manifest",
            version_manifest_path="registries/version-index.json",
            release_completion_marker="release-ledger/1.73.0/complete.json",
        )
    )

    rows = build_published_artifact_index(
        candidate_bundle=_candidate_bundle(),
        promotion_result=result,
    )
    metadata_paths = {
        row.logical_name: row.relative_path
        for row in rows
        if row.artifact_role == "release_metadata"
    }

    assert metadata_paths == {
        "release_manifest": "manifests/root-release.json",
        "versioned_release_manifest": "release-ledger/1.73.0/root-release.json",
        "trace_tro": "manifests/root-trace.jsonld",
        "versioned_trace_tro": "release-ledger/1.73.0/root-trace.jsonld",
        "version_manifest": "registries/version-index.json",
        "release_completion_marker": "release-ledger/1.73.0/complete.json",
    }
    release_manifest = next(
        row for row in rows if row.logical_name == "release_manifest"
    )
    assert release_manifest.sha256 == "sha256:manifest"
    assert release_manifest.hf_uri == (
        "hf://policyengine/policyengine-us-data/manifests/root-release.json"
    )


def test_published_artifact_index_jsonl_round_trips_deterministically() -> None:
    rows = _rows()

    payload = published_artifact_index_to_jsonl(rows)
    restored = published_artifact_index_from_jsonl(payload)

    assert payload.endswith("\n")
    assert len(payload.splitlines()) == len(rows)
    assert restored == tuple(rows)
    assert published_artifact_index_to_jsonl(restored) == payload


def test_write_published_artifact_index_writes_explicit_path(tmp_path) -> None:
    path = published_artifact_index_path(tmp_path / "run-123")
    rows = _rows()

    written = write_published_artifact_index(rows, path)

    assert written == tuple(rows)
    assert read_published_artifact_index(path) == tuple(rows)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert published_artifact_index_repo_path("run-123") == (
        "calibration/runs/run-123/diagnostics/published_artifact_index.jsonl"
    )


def test_published_artifact_index_artifact_ref_records_row_count() -> None:
    artifact = published_artifact_index_artifact_ref(
        _context(),
        row_count=12,
        sha256="sha256:index",
        size_bytes=123,
    )

    assert artifact.logical_name == "published_artifact_index"
    assert artifact.media_type == "application/jsonl"
    assert artifact.metadata["row_count"] == 12
    assert artifact.metadata["relative_path"] == (
        "calibration/runs/run-123/diagnostics/published_artifact_index.jsonl"
    )


def test_published_artifact_index_rejects_mismatched_result_identity() -> None:
    result = FullPromotionResult.from_legacy_dict(
        _legacy_promotion_payload(run_id="other-run")
    )

    with pytest.raises(ValueError, match="run_id"):
        build_published_artifact_index(
            candidate_bundle=_candidate_bundle(),
            promotion_result=result,
        )
