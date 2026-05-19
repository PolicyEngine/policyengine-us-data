import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from policyengine_us_data.release_promotion import (
    ReleaseCandidateInputBundle,
    ReleaseArtifactSpec,
    ReleasePromotionContext,
    build_legacy_release_candidate_bundle,
    build_release_candidate_bundle_from_stage4_contract,
    build_release_candidate_shape_report,
    infer_release_artifact_spec,
    normalize_release_path,
    read_stage4_release_candidate_bundle,
)
from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    DiagnosticRef,
    ExecutionRecord,
    StageContract,
    ValidationReport,
    contract_to_json,
)
from policyengine_us_data.stage_contracts.fingerprints import fingerprint_material


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


def _stage4_contract(
    *,
    fingerprint_marker: str = "default",
    relative_path: str = "states/AL.h5",
    run_id: str = "run-123",
    execution_status: str = "completed",
) -> StageContract:
    outputs = (
        ArtifactRef(
            logical_name="state_al_h5",
            uri="hf://policyengine/policyengine-us-data/staging/1.73.0rc1-run-123/states/AL.h5",
            sha256="sha256:state-al",
            size_bytes=12,
            metadata={
                "relative_path": relative_path,
                "artifact_family": "state_h5",
                "source_stage_id": "4_build_outputs",
                "area_type": "state",
                "area_id": "AL",
            },
        ),
    )
    return StageContract(
        contract_type="output_build",
        stage_id="4_build_outputs",
        run_id=run_id,
        created_at="2026-05-18T12:00:00Z",
        outputs=outputs,
        fingerprint=fingerprint_material(
            {
                "stage_id": "4_build_outputs",
                "outputs": [output.to_dict() for output in outputs],
                "fingerprint_marker": fingerprint_marker,
            }
        ),
        execution=ExecutionRecord(status=execution_status, reuse_decision="computed"),
    )


def _stage4_contract_with_outputs(
    outputs: Sequence[ArtifactRef],
    *,
    diagnostics: Sequence[DiagnosticRef] = (),
    validation: ValidationReport | None = None,
    fingerprint_payload: Mapping[str, Any] | None = None,
) -> StageContract:
    output_tuple = tuple(outputs)
    return StageContract(
        contract_type="output_build",
        stage_id="4_build_outputs",
        run_id="run-123",
        created_at="2026-05-18T12:00:00Z",
        outputs=output_tuple,
        diagnostics=tuple(diagnostics),
        validation=validation,
        fingerprint=fingerprint_material(
            fingerprint_payload
            or {"outputs": [output.to_dict() for output in output_tuple]}
        ),
        execution=ExecutionRecord(status="completed", reuse_decision="computed"),
    )


def _inventory_record(
    path: str,
    *,
    key: str = "path",
    logical_name: str = "district_nc_01_h5",
    artifact_family: str = "district_h5",
    area_type: str = "district",
    area_id: str = "NC-01",
    sha256: str = "sha256:nc-01",
    size_bytes: int = 42,
    run_id: str = "run-123",
) -> dict:
    return {
        key: path,
        "logical_name": logical_name,
        "artifact_family": artifact_family,
        "source_stage_id": "4_build_outputs",
        "area_type": area_type,
        "area_id": area_id,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "run_id": run_id,
        "stage_id": "4_build_outputs",
    }


def _legacy_identity_metadata() -> dict[str, dict]:
    return {
        "states/AL.h5": {"sha256": "sha256:state-al", "size_bytes": 12},
        "policy_data.db": {"sha256": "sha256:policy-db", "size_bytes": 24},
    }


def test_release_path_normalization_rejects_parent_paths() -> None:
    assert normalize_release_path("./states//AL.h5") == "states/AL.h5"

    with pytest.raises(ValueError, match="parent traversal"):
        normalize_release_path("../states/AL.h5")
    with pytest.raises(ValueError, match="parent traversal"):
        normalize_release_path("states/../release_manifest.json")


@pytest.mark.parametrize(
    "path",
    [
        "hf://repo/states/AL.h5",
        "s3://bucket/states/AL.h5",
        "/states/AL.h5",
        "C:\\tmp\\AL.h5",
        "..\\states\\AL.h5",
    ],
)
def test_release_path_normalization_rejects_external_or_absolute_paths(
    path,
) -> None:
    with pytest.raises(ValueError):
        normalize_release_path(path)


def test_release_artifact_spec_infers_area_and_source_stage() -> None:
    state = infer_release_artifact_spec("states/AL.h5")
    base = infer_release_artifact_spec("policy_data.db")
    national = infer_release_artifact_spec("national/US.h5")

    assert state.artifact_family == "state_h5"
    assert state.area_type == "state"
    assert state.area_id == "AL"
    assert state.source_stage_id == "4_build_outputs"
    assert base.logical_name == "policy_data_db"
    assert base.source_stage_id == "1_build_datasets"
    assert national.area_type == "national"


def test_release_promotion_context_round_trips_with_staging_prefix() -> None:
    context = _context()

    restored = ReleasePromotionContext.from_dict(context.to_dict())

    assert restored == context
    assert restored.candidate_scope == "1.73.0rc1"
    assert restored.hf_staging_prefix == "staging/1.73.0rc1-run-123"
    assert restored.schema_version


def test_release_promotion_context_serializes_canonical_identity() -> None:
    context = ReleasePromotionContext(
        run_id="Run ID",
        candidate_version="Candidate Scope",
        release_version="1.73.0rc1",
        hf_repo_name="policyengine/policyengine-us-data",
        gcs_bucket_name="policyengine-us-data",
        base_release_version="1.72.0rc1",
        release_bump="MINOR",
    )

    assert context.run_id == "run-id"
    assert context.candidate_version == "Candidate-Scope"
    assert context.release_version == "1.73.0"
    assert context.base_release_version == "1.72.0"
    assert context.release_bump == "minor"
    assert context.hf_staging_prefix == "staging/Candidate-Scope-run-id"


def test_release_promotion_context_rejects_mismatched_staging_prefix() -> None:
    with pytest.raises(ValueError, match="hf_staging_prefix"):
        ReleasePromotionContext(
            run_id="run-123",
            candidate_version="1.73.0rc1",
            release_version="1.73.0",
            hf_repo_name="policyengine/policyengine-us-data",
            gcs_bucket_name="policyengine-us-data",
            hf_staging_prefix="staging/other-run",
        )


def test_legacy_candidate_bundle_dedupes_and_strips_staging_prefix() -> None:
    bundle = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=[
            "states/AL.h5",
            "staging/1.73.0rc1-run-123/national/US.h5",
            "policy_data.db",
            "states/AL.h5",
        ],
        validation_report_paths=[
            "calibration/runs/run-123/diagnostics/validation_report.json"
        ],
    )

    assert isinstance(bundle, ReleaseCandidateInputBundle)
    assert [artifact.relative_path for artifact in bundle.artifacts] == [
        "national/US.h5",
        "policy_data.db",
        "states/AL.h5",
    ]
    assert bundle.release_candidate_fingerprint is None
    assert bundle.metadata["fingerprint_status"] == (
        "path_only_missing_artifact_identity"
    )
    assert bundle.validation_report_paths == (
        "calibration/runs/run-123/diagnostics/validation_report.json",
    )
    assert bundle.metadata["reader"] == "legacy_staged_paths"


def test_legacy_candidate_bundle_rejects_wrong_run_staging_prefix() -> None:
    with pytest.raises(ValueError, match="expected staging prefix"):
        build_legacy_release_candidate_bundle(
            context=_context(),
            rel_paths=["staging/other-run/states/AL.h5"],
        )


def test_legacy_candidate_fingerprint_tracks_semantic_material() -> None:
    base = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["states/AL.h5", "policy_data.db", "states/AL.h5"],
        artifact_metadata_by_path=_legacy_identity_metadata(),
        validation_report_paths=[
            "calibration/runs/run-123/diagnostics/validation_report.json"
        ],
        diagnostics_manifest_path="calibration/runs/run-123/diagnostics/manifest.json",
    )
    reordered_duplicate = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["policy_data.db", "states/AL.h5", "states/AL.h5"],
        artifact_metadata_by_path=_legacy_identity_metadata(),
        validation_report_paths=[
            "calibration/runs/run-123/diagnostics/validation_report.json"
        ],
        diagnostics_manifest_path="calibration/runs/run-123/diagnostics/manifest.json",
    )
    changed_report = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["states/AL.h5", "policy_data.db"],
        artifact_metadata_by_path=_legacy_identity_metadata(),
        validation_report_paths=[
            "calibration/runs/run-123/diagnostics/validation_summary.json"
        ],
        diagnostics_manifest_path="calibration/runs/run-123/diagnostics/manifest.json",
    )
    changed_diagnostics = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["states/AL.h5", "policy_data.db"],
        artifact_metadata_by_path=_legacy_identity_metadata(),
        validation_report_paths=[
            "calibration/runs/run-123/diagnostics/validation_report.json"
        ],
        diagnostics_manifest_path="calibration/runs/run-123/diagnostics/other.json",
    )
    changed_artifacts = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["states/AL.h5"],
        artifact_metadata_by_path=_legacy_identity_metadata(),
        validation_report_paths=[
            "calibration/runs/run-123/diagnostics/validation_report.json"
        ],
        diagnostics_manifest_path="calibration/runs/run-123/diagnostics/manifest.json",
    )

    assert [artifact.relative_path for artifact in reordered_duplicate.artifacts] == [
        "policy_data.db",
        "states/AL.h5",
    ]
    assert reordered_duplicate.release_candidate_fingerprint == (
        base.release_candidate_fingerprint
    )
    assert changed_report.release_candidate_fingerprint != (
        base.release_candidate_fingerprint
    )
    assert changed_diagnostics.release_candidate_fingerprint != (
        base.release_candidate_fingerprint
    )
    assert changed_artifacts.release_candidate_fingerprint != (
        base.release_candidate_fingerprint
    )


def test_candidate_fingerprint_excludes_arbitrary_metadata() -> None:
    first = build_legacy_release_candidate_bundle(
        context=ReleasePromotionContext(
            run_id="run-123",
            candidate_version="1.73.0rc1",
            release_version="1.73.0",
            hf_repo_name="policyengine/policyengine-us-data",
            gcs_bucket_name="policyengine-us-data",
            metadata={"attempt": 1},
        ),
        rel_paths=["states/AL.h5"],
        artifact_metadata_by_path={
            "states/AL.h5": {
                "sha256": "sha256:state-al",
                "size_bytes": 12,
                "provenance": "first",
            }
        },
    )
    second = build_legacy_release_candidate_bundle(
        context=ReleasePromotionContext(
            run_id="run-123",
            candidate_version="1.73.0rc1",
            release_version="1.73.0",
            hf_repo_name="policyengine/policyengine-us-data",
            gcs_bucket_name="policyengine-us-data",
            metadata={"attempt": 2},
        ),
        rel_paths=["states/AL.h5"],
        artifact_metadata_by_path={
            "states/AL.h5": {
                "sha256": "sha256:state-al",
                "size_bytes": 12,
                "provenance": "second",
            }
        },
    )

    assert first.release_candidate_fingerprint == second.release_candidate_fingerprint


def test_candidate_fingerprint_uses_normalized_paths() -> None:
    base = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["states/AL.h5"],
        artifact_metadata_by_path={
            "states/AL.h5": _legacy_identity_metadata()["states/AL.h5"]
        },
        validation_report_paths=[
            "./calibration/runs/run-123/diagnostics/validation_report.json"
        ],
        diagnostics_manifest_path="./calibration/runs/run-123/diagnostics/manifest.json",
    )
    equivalent = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["./states/AL.h5"],
        artifact_metadata_by_path={
            "./states/AL.h5": _legacy_identity_metadata()["states/AL.h5"]
        },
        validation_report_paths=[
            "calibration/runs/run-123/diagnostics/validation_report.json"
        ],
        diagnostics_manifest_path="calibration/runs/run-123/diagnostics/manifest.json",
    )

    assert (
        base.release_candidate_fingerprint == equivalent.release_candidate_fingerprint
    )


def test_stage4_candidate_reader_accepts_inventory_records() -> None:
    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=_stage4_contract(),
        inventory_records=[
            _inventory_record("staging/1.73.0rc1-run-123/districts/NC-01.h5")
        ],
        source_output_contract_path="calibration/runs/run-123/output_build_contract.json",
    )

    assert [artifact.relative_path for artifact in bundle.artifacts] == [
        "districts/NC-01.h5",
        "states/AL.h5",
    ]
    assert bundle.artifacts[0].artifact_family == "district_h5"
    assert bundle.artifacts[0].sha256 == "sha256:nc-01"
    assert bundle.source_output_contract_path == (
        "calibration/runs/run-123/output_build_contract.json"
    )
    assert bundle.metadata["reader"] == "stage4_contract"


@pytest.mark.parametrize(
    ("record", "expected_path"),
    [
        (
            _inventory_record(
                "national/US.h5",
                key="expected_release_path",
                logical_name="national_us_h5",
                artifact_family="national_h5",
                area_type="national",
                area_id="US",
            ),
            "national/US.h5",
        ),
        (
            _inventory_record(
                "staging/1.73.0rc1-run-123/national/US.h5",
                key="staging_path",
                logical_name="national_us_h5",
                artifact_family="national_h5",
                area_type="national",
                area_id="US",
            ),
            "national/US.h5",
        ),
        (
            _inventory_record(
                "national/US.h5",
                key="output_relative_path",
                logical_name="national_us_h5",
                artifact_family="national_h5",
                area_type="national",
                area_id="US",
            ),
            "national/US.h5",
        ),
        (
            _inventory_record(
                "states/AL.h5",
                key="repo_path",
                logical_name="state_al_h5",
                artifact_family="state_h5",
                area_type="state",
                area_id="AL",
                sha256="sha256:state-al",
                size_bytes=12,
            ),
            "states/AL.h5",
        ),
        (
            {
                "artifact": _inventory_record(
                    "districts/NC-01.h5",
                    key="output_relative_path",
                )
            },
            "districts/NC-01.h5",
        ),
    ],
)
def test_stage4_candidate_reader_accepts_supported_inventory_path_shapes(
    record,
    expected_path,
) -> None:
    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=_stage4_contract(),
        inventory_records=[record],
    )

    assert expected_path in {artifact.relative_path for artifact in bundle.artifacts}
    assert "states/AL.h5" in {artifact.relative_path for artifact in bundle.artifacts}


def test_stage4_candidate_reader_requires_inventory_path_fields_to_agree() -> None:
    matching_record = _inventory_record(
        "national/US.h5",
        key="expected_release_path",
        logical_name="national_us_h5",
        artifact_family="national_h5",
        area_type="national",
        area_id="US",
    )
    matching_record["staging_path"] = "staging/1.73.0rc1-run-123/national/US.h5"
    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=_stage4_contract(),
        inventory_records=[matching_record],
    )

    assert "national/US.h5" in {artifact.relative_path for artifact in bundle.artifacts}

    conflicting_record = {
        **matching_record,
        "staging_path": "staging/1.73.0rc1-run-123/cities/NYC.h5",
    }
    with pytest.raises(ValueError, match="path fields must agree"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            inventory_records=[conflicting_record],
        )

    wrong_prefix_record = {
        **matching_record,
        "staging_path": "staging/other-run/national/US.h5",
    }
    with pytest.raises(ValueError, match="expected staging prefix"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            inventory_records=[wrong_prefix_record],
        )


def test_stage4_candidate_reader_rejects_run_mismatches() -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(run_id="other-run"),
        )

    with pytest.raises(ValueError, match="run_id"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            inventory_records=[_inventory_record("states/AL.h5", run_id="other-run")],
        )


def test_stage4_candidate_reader_rejects_stage_mismatches() -> None:
    with pytest.raises(ValueError, match="Stage 4"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=StageContract(
                contract_type="dataset_build_output",
                stage_id="1_build_datasets",
                run_id="run-123",
                created_at="2026-05-18T12:00:00Z",
                outputs=(),
                fingerprint=fingerprint_material({"stage_id": "1_build_datasets"}),
                execution=ExecutionRecord(status="completed"),
            ),
        )

    with pytest.raises(ValueError, match="stage_id"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            inventory_records=[
                {
                    **_inventory_record("states/AL.h5"),
                    "stage_id": "3_fit_weights",
                }
            ],
        )


def test_stage4_candidate_reader_rejects_incomplete_contracts() -> None:
    for execution_status in ("pending", "running", "failed", "skipped"):
        with pytest.raises(ValueError, match="execution.status"):
            build_release_candidate_bundle_from_stage4_contract(
                context=_context(),
                output_contract=_stage4_contract(execution_status=execution_status),
            )


@pytest.mark.parametrize(
    "execution_status", ["completed", "reused", "partially_reused"]
)
def test_stage4_candidate_reader_accepts_release_safe_execution_statuses(
    execution_status,
) -> None:
    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=_stage4_contract(execution_status=execution_status),
    )

    assert [artifact.relative_path for artifact in bundle.artifacts] == ["states/AL.h5"]


def test_stage4_candidate_reader_strips_or_rejects_staged_contract_paths() -> None:
    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=_stage4_contract(
            relative_path="staging/1.73.0rc1-run-123/states/AL.h5"
        ),
    )

    assert [artifact.relative_path for artifact in bundle.artifacts] == ["states/AL.h5"]

    with pytest.raises(ValueError, match="expected staging prefix"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(
                relative_path="staging/other-run/states/AL.h5"
            ),
        )


def test_stage4_candidate_reader_rejects_duplicate_identity_conflicts() -> None:
    with pytest.raises(ValueError, match="sha256"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            inventory_records=[
                _inventory_record(
                    "states/AL.h5",
                    logical_name="state_al_h5",
                    artifact_family="state_h5",
                    area_type="state",
                    area_id="AL",
                    sha256="sha256:different",
                    size_bytes=12,
                )
            ],
        )

    with pytest.raises(ValueError, match="sha256"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            inventory_records=[
                _inventory_record("districts/NC-01.h5", sha256="sha256:first"),
                _inventory_record("districts/NC-01.h5", sha256="sha256:second"),
            ],
        )


def test_stage4_candidate_fingerprint_tracks_source_contract_fingerprint() -> None:
    first = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=_stage4_contract(fingerprint_marker="first"),
    )
    second = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=_stage4_contract(fingerprint_marker="second"),
    )

    assert first.release_candidate_fingerprint != second.release_candidate_fingerprint


def test_stage4_candidate_reader_falls_back_to_contract_output_paths() -> None:
    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=_stage4_contract(),
    )

    assert [artifact.relative_path for artifact in bundle.artifacts] == ["states/AL.h5"]
    assert bundle.artifacts[0].sha256 == "sha256:state-al"


def test_stage4_candidate_reader_can_use_artifact_uri_without_path_metadata() -> None:
    output = ArtifactRef(
        logical_name="state_al_h5",
        uri="hf://policyengine/policyengine-us-data/staging/1.73.0rc1-run-123/states/AL.h5",
        sha256="sha256:state-al",
        size_bytes=12,
    )
    contract = _stage4_contract_with_outputs((output,))

    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=contract,
    )

    assert [artifact.relative_path for artifact in bundle.artifacts] == ["states/AL.h5"]
    assert bundle.artifacts[0].artifact_family == "state_h5"


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        (
            "hf://policyengine/policyengine-us-data/staging/1.73.0rc1-other-run/states/AL.h5",
            "expected staging prefix",
        ),
        (
            "hf://policyengine/policyengine-us-data/states/AL.h5",
            "expected staging prefix",
        ),
        (
            "hf://other/policyengine-us-data/staging/1.73.0rc1-run-123/states/AL.h5",
            "hf_repo_name",
        ),
        (
            "hf://policyengine/policyengine-us-data/staging/1.73.0rc1-run-123/districts/NC-01.h5",
            "metadata path must match",
        ),
    ],
)
def test_stage4_candidate_reader_validates_contract_artifact_uri_against_metadata(
    uri,
    match,
) -> None:
    output = ArtifactRef(
        logical_name="state_al_h5",
        uri=uri,
        sha256="sha256:state-al",
        size_bytes=12,
        metadata={
            "relative_path": "states/AL.h5",
            "artifact_family": "state_h5",
            "source_stage_id": "4_build_outputs",
            "area_type": "state",
            "area_id": "AL",
        },
    )

    with pytest.raises(ValueError, match=match):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract_with_outputs((output,)),
        )


def test_stage4_candidate_reader_rejects_external_production_uris() -> None:
    output = ArtifactRef(
        logical_name="state_al_h5",
        uri="hf://policyengine/policyengine-us-data/states/AL.h5",
        sha256="sha256:state-al",
        size_bytes=12,
    )
    contract = _stage4_contract_with_outputs((output,))

    with pytest.raises(ValueError, match="expected staging prefix"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=contract,
        )


def test_stage4_candidate_reader_rejects_wrong_repo_staged_uris() -> None:
    output = ArtifactRef(
        logical_name="state_al_h5",
        uri="hf://other/policyengine-us-data/staging/1.73.0rc1-run-123/states/AL.h5",
        sha256="sha256:state-al",
        size_bytes=12,
    )
    contract = _stage4_contract_with_outputs((output,))

    with pytest.raises(ValueError, match="hf_repo_name"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=contract,
        )


def test_stage4_candidate_reader_rejects_uri_only_base_artifact() -> None:
    output = ArtifactRef(
        logical_name="policy_data_db",
        uri="hf://policyengine/policyengine-us-data/policy_data.db",
        sha256="sha256:policy-db",
        size_bytes=24,
    )

    with pytest.raises(ValueError, match="expected staging prefix"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract_with_outputs((output,)),
        )


def test_stage4_candidate_bundle_can_read_contract_and_inventory_files(
    tmp_path,
) -> None:
    contract_path = tmp_path / "output_build_contract.json"
    inventory_path = tmp_path / "output_inventory.jsonl"
    contract_path.write_text(contract_to_json(_stage4_contract()), encoding="utf-8")
    inventory_path.write_text(
        json.dumps(
            _inventory_record(
                "cities/NYC.h5",
                key="relative_path",
                logical_name="city_nyc_h5",
                artifact_family="city_h5",
                area_type="city",
                area_id="NYC",
                sha256="sha256:nyc",
                size_bytes=6,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = read_stage4_release_candidate_bundle(
        context=_context(),
        output_contract_path=contract_path,
        output_inventory_path=inventory_path,
        source_output_contract_path="calibration/runs/run-123/output_build_contract.json",
    )

    assert bundle.source_output_contract_path == (
        "calibration/runs/run-123/output_build_contract.json"
    )
    assert [artifact.relative_path for artifact in bundle.artifacts] == [
        "cities/NYC.h5",
        "states/AL.h5",
    ]
    assert bundle.artifacts[0].area_type == "city"


def test_stage4_candidate_reader_uses_named_diagnostics_manifest() -> None:
    diagnostics_manifest = ArtifactRef(
        logical_name="diagnostics_manifest",
        uri="hf://policyengine/policyengine-us-data/calibration/runs/run-123/diagnostics/manifest.json",
        metadata={
            "relative_path": "calibration/runs/run-123/diagnostics/manifest.json"
        },
    )
    other_diagnostic = ArtifactRef(
        logical_name="worker_log",
        uri="hf://policyengine/policyengine-us-data/calibration/runs/run-123/diagnostics/worker.log",
        metadata={"relative_path": "calibration/runs/run-123/diagnostics/worker.log"},
    )
    contract = _stage4_contract_with_outputs(
        _stage4_contract().outputs,
        diagnostics=(
            DiagnosticRef(name="worker_log", kind="log", artifact=other_diagnostic),
            DiagnosticRef(
                name="diagnostics_manifest",
                kind="json",
                artifact=diagnostics_manifest,
            ),
        ),
        fingerprint_payload={"diagnostics": "manifest"},
    )

    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=contract,
    )

    assert bundle.diagnostics_manifest_path == (
        "calibration/runs/run-123/diagnostics/manifest.json"
    )


def test_stage4_candidate_reader_uses_uri_only_and_validation_diagnostics() -> None:
    diagnostics_manifest = ArtifactRef(
        logical_name="diagnostics_manifest",
        uri="hf://policyengine/policyengine-us-data/calibration/runs/run-123/diagnostics/manifest.json",
    )
    contract = _stage4_contract_with_outputs(
        _stage4_contract().outputs,
        validation=ValidationReport(
            status="pass",
            diagnostics=(
                DiagnosticRef(
                    name="diagnostics_manifest",
                    kind="json",
                    artifact=diagnostics_manifest,
                ),
            ),
        ),
        fingerprint_payload={"diagnostics": "validation"},
    )

    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=contract,
    )

    assert bundle.diagnostics_manifest_path == (
        "calibration/runs/run-123/diagnostics/manifest.json"
    )


def test_stage4_candidate_reader_scopes_diagnostics_and_validation_paths() -> None:
    with pytest.raises(ValueError, match="context.run_id"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            diagnostics_manifest_path=(
                "calibration/runs/other-run/diagnostics/manifest.json"
            ),
        )

    with pytest.raises(ValueError, match="context.run_id"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            validation_report_paths=(
                "calibration/runs/other-run/diagnostics/validation_report.json",
            ),
        )


def test_stage4_candidate_reader_scopes_source_contract_path() -> None:
    with pytest.raises(ValueError, match="context.run_id"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=_stage4_contract(),
            source_output_contract_path=(
                "calibration/runs/other-run/output_build_contract.json"
            ),
        )


def test_stage4_candidate_reader_validates_diagnostics_manifest_uri() -> None:
    diagnostics_manifest = ArtifactRef(
        logical_name="diagnostics_manifest",
        uri="hf://policyengine/policyengine-us-data/calibration/runs/run-123/diagnostics/other.json",
        metadata={
            "relative_path": "calibration/runs/run-123/diagnostics/manifest.json"
        },
    )
    contract = _stage4_contract_with_outputs(
        _stage4_contract().outputs,
        diagnostics=(
            DiagnosticRef(
                name="diagnostics_manifest",
                kind="json",
                artifact=diagnostics_manifest,
            ),
        ),
        fingerprint_payload={"diagnostics": "manifest"},
    )

    with pytest.raises(ValueError, match="metadata path must match"):
        build_release_candidate_bundle_from_stage4_contract(
            context=_context(),
            output_contract=contract,
        )


def test_stage4_candidate_fingerprint_tracks_diagnostics_manifest_identity() -> None:
    def contract_with_manifest_sha(sha256: str) -> StageContract:
        diagnostics_manifest = ArtifactRef(
            logical_name="diagnostics_manifest",
            uri="hf://policyengine/policyengine-us-data/calibration/runs/run-123/diagnostics/manifest.json",
            sha256=sha256,
            size_bytes=100,
            metadata={
                "relative_path": "calibration/runs/run-123/diagnostics/manifest.json"
            },
        )
        return _stage4_contract_with_outputs(
            _stage4_contract().outputs,
            diagnostics=(
                DiagnosticRef(
                    name="diagnostics_manifest",
                    kind="json",
                    artifact=diagnostics_manifest,
                ),
            ),
            fingerprint_payload={"stage4": "same"},
        )

    first = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=contract_with_manifest_sha("sha256:first"),
    )
    second = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=contract_with_manifest_sha("sha256:second"),
    )

    assert first.release_candidate_fingerprint != second.release_candidate_fingerprint


def test_stage4_candidate_reader_keeps_diagnostics_out_of_release_artifacts() -> None:
    diagnostics_output = ArtifactRef(
        logical_name="diagnostics_manifest",
        uri="hf://policyengine/policyengine-us-data/calibration/runs/run-123/diagnostics/manifest.json",
        metadata={
            "relative_path": "calibration/runs/run-123/diagnostics/manifest.json",
            "artifact_family": "diagnostics",
            "source_stage_id": "4_build_outputs",
        },
    )
    base_contract = _stage4_contract()
    contract = _stage4_contract_with_outputs(
        (*base_contract.outputs, diagnostics_output),
        fingerprint_payload={"outputs": "with_diagnostics"},
    )

    bundle = build_release_candidate_bundle_from_stage4_contract(
        context=_context(),
        output_contract=contract,
    )

    assert [artifact.relative_path for artifact in bundle.artifacts] == ["states/AL.h5"]
    assert bundle.diagnostics_manifest_path == (
        "calibration/runs/run-123/diagnostics/manifest.json"
    )


def test_release_candidate_bundle_round_trips_through_dict_and_json() -> None:
    artifact = infer_release_artifact_spec(
        "states/AL.h5",
        sha256="sha256:state-al",
        size_bytes=12,
        metadata={"source": "fixture"},
    )
    bundle = ReleaseCandidateInputBundle(
        context=_context(),
        artifacts=(artifact,),
        source_output_contract_path="calibration/runs/run-123/output_build_contract.json",
        release_candidate_fingerprint="sha256:fixture",
        validation_report_paths=(
            "calibration/runs/run-123/diagnostics/validation_report.json",
        ),
        diagnostics_manifest_path="calibration/runs/run-123/diagnostics/manifest.json",
        metadata={"reader": "fixture"},
    )

    payload = bundle.to_dict()
    restored = ReleaseCandidateInputBundle.from_dict(json.loads(json.dumps(payload)))

    assert payload["bundle_type"] == "release_candidate_input_bundle"
    assert payload["stage_id"] == "5_validate_and_promote_release"
    assert payload["schema_version"]
    assert ReleaseArtifactSpec.from_dict(artifact.to_dict()) == artifact
    assert restored.to_dict() == payload


def test_release_candidate_shape_report_uses_canonical_validation_schema() -> None:
    bundle = build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["states/AL.h5"],
    )

    report = build_release_candidate_shape_report(bundle)

    assert isinstance(report, ValidationReport)
    assert report.status == "pass"
    assert [finding.check_id for finding in report.findings] == [
        "release_candidate_identity_declared",
        "release_candidate_artifacts_declared",
    ]
    assert report.metadata["stage_id"] == "5_validate_and_promote_release"
    assert report.metadata["release_candidate_fingerprint"] == (
        bundle.release_candidate_fingerprint
    )
