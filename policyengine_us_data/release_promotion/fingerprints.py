"""Release candidate fingerprint construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from policyengine_us_data.stage_contracts import DiagnosticRef
from policyengine_us_data.stage_contracts.fingerprints import fingerprint_material

from .artifacts import ReleaseArtifactSpec
from .candidate import ReleaseCandidateInputBundle
from .context import ReleasePromotionContext
from .diagnostics import (
    normalize_run_contract_path,
    normalize_run_diagnostic_path,
    validation_report_ref_path,
)


def candidate_bundle_with_fingerprint(
    *,
    context: ReleasePromotionContext,
    artifacts: tuple[ReleaseArtifactSpec, ...],
    source_output_contract_path: str | None,
    validation_report_paths: Sequence[str],
    validation_report_refs: Sequence[DiagnosticRef],
    diagnostics_manifest_path: str | None,
    reader: str,
    extra_fingerprint_material: Mapping[str, Any] | None = None,
) -> ReleaseCandidateInputBundle:
    """Build a candidate bundle with stable fingerprint material when complete."""

    sorted_artifacts = tuple(sorted(artifacts, key=lambda item: item.relative_path))
    normalized_source_output_contract_path = (
        normalize_run_contract_path(source_output_contract_path, context)
        if source_output_contract_path is not None
        else None
    )
    normalized_validation_report_paths = tuple(
        normalize_run_diagnostic_path(path, context) for path in validation_report_paths
    )
    normalized_validation_report_refs = tuple(validation_report_refs)
    normalized_validation_report_ref_paths = tuple(
        validation_report_ref_path(ref, context)
        for ref in normalized_validation_report_refs
    )
    normalized_diagnostics_manifest_path = (
        normalize_run_diagnostic_path(diagnostics_manifest_path, context)
        if diagnostics_manifest_path is not None
        else None
    )
    fingerprint_status, missing_artifact_identity_paths = (
        release_artifact_identity_status(sorted_artifacts)
    )
    missing_validation_report_identity_paths = validation_report_identity_missing_paths(
        normalized_validation_report_refs,
        context=context,
    )
    if missing_validation_report_identity_paths:
        fingerprint_status = (
            "path_only_missing_identity"
            if missing_artifact_identity_paths
            else "path_only_missing_validation_report_identity"
        )
    fingerprint = None
    if fingerprint_status == "complete":
        fingerprint = fingerprint_material(
            {
                "reader": reader,
                "context": context_fingerprint_material(context),
                "artifacts": [
                    artifact_fingerprint_material(artifact)
                    for artifact in sorted_artifacts
                ],
                "source_output_contract_path": normalized_source_output_contract_path,
                "validation_report_paths": sorted(normalized_validation_report_paths),
                "validation_report_refs": sorted(
                    (
                        validation_report_ref_fingerprint_material(ref, context)
                        for ref in normalized_validation_report_refs
                    ),
                    key=lambda item: (
                        item["path"],
                        item["name"],
                        item["kind"],
                    ),
                ),
                "diagnostics_manifest_path": normalized_diagnostics_manifest_path,
                **(extra_fingerprint_material or {}),
            }
        ).value
    return ReleaseCandidateInputBundle(
        context=context,
        artifacts=sorted_artifacts,
        source_output_contract_path=normalized_source_output_contract_path,
        release_candidate_fingerprint=fingerprint,
        validation_report_paths=normalized_validation_report_paths,
        validation_report_refs=normalized_validation_report_refs,
        diagnostics_manifest_path=normalized_diagnostics_manifest_path,
        metadata={
            "reader": reader,
            "fingerprint_status": fingerprint_status,
            "missing_fingerprint_identity_paths": missing_artifact_identity_paths,
            "missing_validation_report_identity_paths": (
                missing_validation_report_identity_paths
            ),
            "validation_report_ref_paths": normalized_validation_report_ref_paths,
        },
    )


def release_artifact_identity_status(
    artifacts: Sequence[ReleaseArtifactSpec],
) -> tuple[str, tuple[str, ...]]:
    """Return whether required release artifacts have checksum identity."""

    missing_identity_paths = tuple(
        artifact.relative_path
        for artifact in artifacts
        if artifact.required
        and (artifact.sha256 is None or artifact.size_bytes is None)
    )
    if missing_identity_paths:
        return "path_only_missing_artifact_identity", missing_identity_paths
    return "complete", ()


def validation_report_identity_missing_paths(
    refs: Sequence[DiagnosticRef],
    *,
    context: ReleasePromotionContext,
) -> tuple[str, ...]:
    """Return validation report paths missing checksum identity."""

    missing_paths: list[str] = []
    for ref in refs:
        path = validation_report_ref_path(ref, context)
        artifact = ref.artifact
        if artifact is None or artifact.sha256 is None or artifact.size_bytes is None:
            missing_paths.append(path)
    return tuple(missing_paths)


def validation_report_ref_fingerprint_material(
    ref: DiagnosticRef,
    context: ReleasePromotionContext,
) -> dict[str, Any]:
    """Return stable fingerprint material for a validation report ref."""

    artifact = ref.artifact
    path = validation_report_ref_path(ref, context)
    if artifact is None:
        raise ValueError("validation_report_refs entries must include artifacts")
    return {
        "name": ref.name,
        "kind": ref.kind,
        "path": path,
        "logical_name": artifact.logical_name,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def context_fingerprint_material(
    context: ReleasePromotionContext,
) -> dict[str, Any]:
    """Return candidate context fields that participate in the fingerprint."""

    return {
        "run_id": context.run_id,
        "candidate_version": context.candidate_version,
        "release_version": context.release_version,
        "hf_repo_name": context.hf_repo_name,
        "hf_repo_type": context.hf_repo_type,
        "gcs_bucket_name": context.gcs_bucket_name,
        "base_release_version": context.base_release_version,
        "release_bump": context.release_bump,
        "modal_app_name": context.modal_app_name,
        "modal_environment": context.modal_environment,
        "hf_staging_prefix": context.hf_staging_prefix,
        "schema_version": context.schema_version,
    }


def artifact_fingerprint_material(
    artifact: ReleaseArtifactSpec,
) -> dict[str, Any]:
    """Return artifact fields that participate in the candidate fingerprint."""

    return {
        "logical_name": artifact.logical_name,
        "relative_path": artifact.relative_path,
        "artifact_family": artifact.artifact_family,
        "source_stage_id": artifact.source_stage_id,
        "area_type": artifact.area_type,
        "area_id": artifact.area_id,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "required": artifact.required,
        "schema_version": artifact.schema_version,
    }
