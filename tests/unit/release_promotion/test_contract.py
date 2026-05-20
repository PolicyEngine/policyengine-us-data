import json

import pytest

from policyengine_us_data.release_promotion import (
    RELEASE_PROMOTION_CONTRACT_FILENAME,
    RELEASE_PROMOTION_CONTRACT_TYPE,
    FullPromotionResult,
    ReleasePromotionContext,
    build_legacy_release_candidate_bundle,
    build_release_promotion_contract,
    published_artifact_index_artifact_ref,
    promoted_runs_index_artifact_ref,
    PromotedRunsIndexUpdate,
    release_promotion_contract_path,
    release_promotion_contract_repo_path,
    write_release_promotion_contract,
)
from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    DiagnosticRef,
    StageContract,
    ValidationFinding,
    ValidationReport,
    contract_to_json,
    read_contract,
)


def _context() -> ReleasePromotionContext:
    return ReleasePromotionContext(
        run_id="run-123",
        candidate_version="1.73.0rc1",
        release_version="1.73.0",
        hf_repo_name="policyengine/policyengine-us-data",
        gcs_bucket_name="policyengine-us-data",
        base_release_version="1.72.0",
        release_bump="minor",
        modal_app_name="us-data-run-123",
        modal_environment="main",
    )


def _candidate_bundle():
    return build_legacy_release_candidate_bundle(
        context=_context(),
        rel_paths=["states/AL.h5", "policy_data.db"],
        artifact_metadata_by_path={
            "states/AL.h5": {"sha256": "sha256:state-al", "size_bytes": 12},
            "policy_data.db": {"sha256": "sha256:policy-db", "size_bytes": 24},
        },
        validation_report_paths=[
            "calibration/runs/run-123/diagnostics/validation_report.json"
        ],
        diagnostics_manifest_path="calibration/runs/run-123/diagnostics/manifest.json",
        source_output_contract_path=(
            "calibration/runs/run-123/diagnostics/contracts/output_build_contract.json"
        ),
    )


def _promotion_result(*, already_finalized: bool = False) -> FullPromotionResult:
    return FullPromotionResult.from_legacy_dict(
        {
            "run_id": "run-123",
            "candidate_version": "1.73.0rc1",
            "release_version": "1.73.0",
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


def _validation_report() -> ValidationReport:
    diagnostic = DiagnosticRef(
        name="validation_report",
        kind="json",
        artifact=ArtifactRef(
            logical_name="validation_report",
            uri=(
                "hf://policyengine/policyengine-us-data/calibration/runs/"
                "run-123/diagnostics/validation_report.json"
            ),
            media_type="application/json",
        ),
    )
    return ValidationReport(
        status="pass",
        findings=(
            ValidationFinding(
                check_id="release_candidate_identity_declared",
                status="pass",
                message="candidate identity is declared",
            ),
        ),
        diagnostics=(diagnostic,),
        metadata={"stage_id": "5_validate_and_promote_release"},
    )


def _promoted_runs_index_update() -> PromotedRunsIndexUpdate:
    return PromotedRunsIndexUpdate(
        status="created",
        run_id="run-123",
        release_version="1.73.0",
        run_count=1,
        release_version_run_count=1,
        already_finalized=False,
        updated_at="2026-05-18T12:00:00+00:00",
    )


def test_release_promotion_contract_records_candidate_and_public_refs() -> None:
    promoted_runs_update = _promoted_runs_index_update()
    contract = build_release_promotion_contract(
        candidate_bundle=_candidate_bundle(),
        promotion_result=_promotion_result(),
        created_at="2026-05-18T12:00:00+00:00",
        code_sha="abc123",
        package_version="1.73.0",
        validation=_validation_report(),
        published_artifact_index=published_artifact_index_artifact_ref(
            _context(),
            row_count=9,
            sha256="sha256:index",
            size_bytes=123,
        ),
        promoted_runs_index=promoted_runs_index_artifact_ref(
            _context(),
            promoted_runs_update,
            sha256="sha256:runs-index",
            size_bytes=456,
        ),
        promoted_runs_index_update=promoted_runs_update.to_dict(),
        metadata={"writer": "test"},
    )

    input_names = {artifact.logical_name for artifact in contract.inputs}
    output_names = {artifact.logical_name for artifact in contract.outputs}

    assert contract.contract_type == RELEASE_PROMOTION_CONTRACT_TYPE
    assert contract.stage_id == "5_validate_and_promote_release"
    assert contract.run_id == "run-123"
    assert "stage4_output_contract" in input_names
    assert "validation_report_1" in input_names
    assert "diagnostics_manifest" in input_names
    assert output_names == {
        "huggingface_release_artifacts",
        "gcs_release_artifacts",
        "release_manifest",
        "versioned_release_manifest",
        "trace_tro",
        "versioned_trace_tro",
        "version_manifest",
        "release_completion_marker",
        "published_artifact_index",
        "promoted_runs_index",
    }
    assert contract.execution.status == "completed"
    assert contract.execution.reuse_decision == "computed"
    assert contract.execution.reuse_summary.expected_outputs == 2
    assert contract.parameters["release_candidate_fingerprint"]
    assert contract.parameters["source_output_contract_path"] == (
        "calibration/runs/run-123/diagnostics/contracts/output_build_contract.json"
    )
    assert contract.parameters["published_artifact_index_path"] == (
        "calibration/runs/run-123/diagnostics/published_artifact_index.jsonl"
    )
    assert contract.parameters["promoted_runs_index_path"] == (
        "calibration/runs/index.json"
    )
    assert contract.metadata["contract_file"] == RELEASE_PROMOTION_CONTRACT_FILENAME
    assert contract.metadata["already_finalized"] is False
    assert contract.metadata["cleanup"]["cleaned_count"] == 3
    assert contract.metadata["published_artifact_index"]["metadata"]["row_count"] == 9
    assert (
        contract.metadata["promoted_runs_index"]["metadata"]["update_status"]
        == "created"
    )
    assert contract.metadata["promoted_runs_index_update"]["run_count"] == 1
    assert contract.metadata["public_refs"]["release_manifest"] == (
        "hf://policyengine/policyengine-us-data/release_manifest.json"
    )
    assert contract.metadata["public_refs"]["published_artifact_index"].endswith(
        "published_artifact_index.jsonl"
    )
    assert contract.metadata["public_refs"]["promoted_runs_index"].endswith(
        "calibration/runs/index.json"
    )
    assert [substage.substage_id for substage in contract.substages] == [
        "5a_validate_outputs",
        "5b_promote_huggingface",
        "5c_promote_gcs",
        "5d_write_version_manifest",
    ]
    assert StageContract.from_dict(json.loads(contract_to_json(contract))) == contract


def test_release_promotion_contract_records_already_finalized_reuse() -> None:
    contract = build_release_promotion_contract(
        candidate_bundle=_candidate_bundle(),
        promotion_result=_promotion_result(already_finalized=True),
        created_at="2026-05-18T12:00:00+00:00",
    )

    assert contract.execution.reuse_decision == "reused"
    assert contract.execution.reuse_reason == "already_finalized"
    assert contract.execution.reuse_summary.valid_reused_outputs == 2
    assert contract.execution.reuse_summary.recomputed_outputs == 0


def test_write_release_promotion_contract_writes_run_diagnostics_path(tmp_path) -> None:
    contract_path = release_promotion_contract_path(tmp_path / "run-123")

    written = write_release_promotion_contract(
        contract_path=contract_path,
        candidate_bundle=_candidate_bundle(),
        promotion_result=_promotion_result(),
        created_at="2026-05-18T12:00:00+00:00",
    )

    assert contract_path == (
        tmp_path
        / "run-123"
        / "diagnostics"
        / "contracts"
        / "release_promotion_contract.json"
    )
    assert read_contract(contract_path) == written
    assert release_promotion_contract_repo_path("run-123") == (
        "calibration/runs/run-123/diagnostics/contracts/release_promotion_contract.json"
    )


def test_release_promotion_contract_rejects_mismatched_result_identity() -> None:
    result = FullPromotionResult.from_legacy_dict(
        {
            "run_id": "other-run",
            "candidate_version": "1.73.0rc1",
            "release_version": "1.73.0",
            "artifact_count": 2,
            "hf_promoted": 2,
            "gcs_uploaded": 2,
            "release_manifest_artifacts": 2,
            "version_manifest_updated": True,
            "release_completion_marker": "releases/1.73.0/release-complete.json",
            "staging_cleaned": 3,
        }
    )

    with pytest.raises(ValueError, match="run_id"):
        build_release_promotion_contract(
            candidate_bundle=_candidate_bundle(),
            promotion_result=result,
            created_at="2026-05-18T12:00:00+00:00",
        )
