"""Fixture helpers for Stage 5 release promotion tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from policyengine_us_data.release_promotion import ReleasePromotionContext
from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    DiagnosticRef,
    ExecutionRecord,
    StageContract,
    ValidationReport,
)
from policyengine_us_data.stage_contracts.fingerprints import fingerprint_material

__test__ = False

RELEASE_RUN_ID = "run-123"
CANDIDATE_VERSION = "1.73.0rc1"
RELEASE_VERSION = "1.73.0"
HF_REPO_NAME = "policyengine/policyengine-us-data"
GCS_BUCKET_NAME = "policyengine-us-data"
BASE_RELEASE_VERSION = "1.72.0"
RELEASE_BUMP = "minor"
STAGING_PREFIX = f"staging/{CANDIDATE_VERSION}-{RELEASE_RUN_ID}"
RUN_DIAGNOSTICS_PREFIX = f"calibration/runs/{RELEASE_RUN_ID}/diagnostics"


def release_promotion_context() -> ReleasePromotionContext:
    """Return the canonical Stage 5 release-promotion context fixture."""

    return ReleasePromotionContext(
        run_id=RELEASE_RUN_ID,
        candidate_version=CANDIDATE_VERSION,
        release_version=RELEASE_VERSION,
        hf_repo_name=HF_REPO_NAME,
        gcs_bucket_name=GCS_BUCKET_NAME,
        base_release_version=BASE_RELEASE_VERSION,
        release_bump=RELEASE_BUMP,
    )


def stage4_contract(
    *,
    fingerprint_marker: str = "default",
    relative_path: str = "states/AL.h5",
    run_id: str = RELEASE_RUN_ID,
    execution_status: str = "completed",
) -> StageContract:
    """Return a minimal Stage 4 output contract for release-candidate tests."""

    outputs = (
        ArtifactRef(
            logical_name="state_al_h5",
            uri=f"hf://{HF_REPO_NAME}/{STAGING_PREFIX}/states/AL.h5",
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


def stage4_contract_with_outputs(
    outputs: Sequence[ArtifactRef],
    *,
    diagnostics: Sequence[DiagnosticRef] = (),
    validation: ValidationReport | None = None,
    fingerprint_payload: Mapping[str, Any] | None = None,
) -> StageContract:
    """Return a Stage 4 output contract with caller-supplied outputs."""

    output_tuple = tuple(outputs)
    return StageContract(
        contract_type="output_build",
        stage_id="4_build_outputs",
        run_id=RELEASE_RUN_ID,
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


def stage4_inventory_record(
    path: str,
    *,
    key: str = "path",
    logical_name: str = "district_nc_01_h5",
    artifact_family: str = "district_h5",
    area_type: str = "district",
    area_id: str = "NC-01",
    sha256: str = "sha256:nc-01",
    size_bytes: int = 42,
    run_id: str = RELEASE_RUN_ID,
) -> dict[str, Any]:
    """Return a Stage 4 inventory record fixture."""

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


def legacy_identity_metadata() -> dict[str, dict[str, Any]]:
    """Return checksum identity for legacy staged path candidate tests."""

    return {
        "states/AL.h5": {"sha256": "sha256:state-al", "size_bytes": 12},
        "policy_data.db": {"sha256": "sha256:policy-db", "size_bytes": 24},
    }


def validation_report_ref(
    *,
    path: str = f"{RUN_DIAGNOSTICS_PREFIX}/validation_report.json",
    sha256: str | None = "sha256:validation-report",
    size_bytes: int | None = 128,
) -> DiagnosticRef:
    """Return a validation report diagnostic ref with artifact identity."""

    return DiagnosticRef(
        name="stage4_validation_report",
        kind="validation_report",
        artifact=ArtifactRef(
            logical_name="stage4_validation_report",
            uri=f"hf://{HF_REPO_NAME}/{path}",
            sha256=sha256,
            size_bytes=size_bytes,
            metadata={"relative_path": path},
        ),
    )
