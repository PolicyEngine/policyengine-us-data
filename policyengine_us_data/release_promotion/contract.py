"""Stage 5 release promotion contract assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    DiagnosticRef,
    ExecutionRecord,
    ReuseSummary,
    StageContract,
    SubstageRecord,
    ValidationReport,
    contract_type_for_stage,
    write_contract,
)
from policyengine_us_data.stage_contracts._coercion import freeze_sequence
from policyengine_us_data.stage_contracts.fingerprints import fingerprint_material
from policyengine_us_data.stage_contracts.stages import (
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
)

from .candidate import ReleaseCandidateInputBundle
from .context import ReleasePromotionContext
from .results import FullPromotionResult

RELEASE_PROMOTION_CONTRACT_FILENAME = "release_promotion_contract.json"
RELEASE_PROMOTION_CONTRACT_TYPE = contract_type_for_stage(
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE
)


def release_promotion_contract_repo_path(run_id: str) -> str:
    """Return the run-scoped repository path for the Stage 5 contract."""

    return (
        f"calibration/runs/{run_id}/diagnostics/contracts/"
        f"{RELEASE_PROMOTION_CONTRACT_FILENAME}"
    )


def release_promotion_contract_path(run_dir: str | Path) -> Path:
    """Return the run-local diagnostics/contracts path for the Stage 5 contract."""

    return (
        Path(run_dir)
        / "diagnostics"
        / "contracts"
        / RELEASE_PROMOTION_CONTRACT_FILENAME
    )


@pipeline_node(
    id="release_promotion_contract_builder",
    label="ReleasePromotionContractBuilder",
    node_type="library",
    description="Build the canonical Stage 5 release promotion contract.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    artifacts_in=["release candidate bundle", "typed promotion result"],
    artifacts_out=["release_promotion_contract.json"],
    validation_commands=["uv run pytest tests/unit/release_promotion/test_contract.py"],
)
@dataclass(frozen=True, kw_only=True)
class ReleasePromotionContractBuilder:
    """Build a Stage 5 contract from candidate identity and promotion results."""

    candidate_bundle: ReleaseCandidateInputBundle
    promotion_result: FullPromotionResult
    created_at: str
    code_sha: str | None = None
    package_version: str | None = None
    validation: ValidationReport | None = None
    diagnostics: Sequence[DiagnosticRef] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_bundle, ReleaseCandidateInputBundle):
            raise ValueError("candidate_bundle must be ReleaseCandidateInputBundle")
        if not isinstance(self.promotion_result, FullPromotionResult):
            raise ValueError("promotion_result must be FullPromotionResult")
        object.__setattr__(
            self,
            "diagnostics",
            freeze_sequence(self.diagnostics, "diagnostics", DiagnosticRef),
        )
        _validate_result_matches_candidate(
            self.promotion_result,
            self.candidate_bundle,
        )

    def build(self) -> StageContract:
        """Return the canonical Stage 5 release promotion contract."""

        context = self.candidate_bundle.context
        inputs = _contract_inputs(self.candidate_bundle)
        outputs = _contract_outputs(self.promotion_result)
        parameters = _contract_parameters(
            self.candidate_bundle,
            self.promotion_result,
        )
        return StageContract(
            contract_type=RELEASE_PROMOTION_CONTRACT_TYPE,
            stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
            run_id=context.run_id,
            created_at=self.created_at,
            code_sha=self.code_sha,
            package_version=self.package_version,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            fingerprint=fingerprint_material(
                {
                    "stage_id": STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
                    "contract_type": RELEASE_PROMOTION_CONTRACT_TYPE,
                    "context": context.to_dict(),
                    "candidate_bundle": self.candidate_bundle.to_dict(),
                    "promotion_result": self.promotion_result.to_dict(),
                    "outputs": [output.to_dict() for output in outputs],
                }
            ),
            substages=_substage_records(
                candidate_inputs=inputs,
                public_outputs=outputs,
                promotion_result=self.promotion_result,
            ),
            execution=_execution_record(self.promotion_result),
            validation=self.validation,
            diagnostics=tuple(self.diagnostics),
            metadata=_contract_metadata(
                context=context,
                candidate_bundle=self.candidate_bundle,
                promotion_result=self.promotion_result,
                outputs=outputs,
                extra=self.metadata,
            ),
        )


def build_release_promotion_contract(
    *,
    candidate_bundle: ReleaseCandidateInputBundle,
    promotion_result: FullPromotionResult,
    created_at: str,
    code_sha: str | None = None,
    package_version: str | None = None,
    validation: ValidationReport | None = None,
    diagnostics: Sequence[DiagnosticRef] = (),
    metadata: Mapping[str, Any] | None = None,
) -> StageContract:
    """Build the Stage 5 release promotion contract."""

    return ReleasePromotionContractBuilder(
        candidate_bundle=candidate_bundle,
        promotion_result=promotion_result,
        created_at=created_at,
        code_sha=code_sha,
        package_version=package_version,
        validation=validation,
        diagnostics=diagnostics,
        metadata=metadata or {},
    ).build()


def write_release_promotion_contract(
    *,
    contract_path: str | Path,
    candidate_bundle: ReleaseCandidateInputBundle,
    promotion_result: FullPromotionResult,
    created_at: str,
    code_sha: str | None = None,
    package_version: str | None = None,
    validation: ValidationReport | None = None,
    diagnostics: Sequence[DiagnosticRef] = (),
    metadata: Mapping[str, Any] | None = None,
) -> StageContract:
    """Build, write, and return the Stage 5 release promotion contract."""

    contract = build_release_promotion_contract(
        candidate_bundle=candidate_bundle,
        promotion_result=promotion_result,
        created_at=created_at,
        code_sha=code_sha,
        package_version=package_version,
        validation=validation,
        diagnostics=diagnostics,
        metadata=metadata,
    )
    write_contract(contract, contract_path)
    return contract


def _validate_result_matches_candidate(
    result: FullPromotionResult,
    candidate_bundle: ReleaseCandidateInputBundle,
) -> None:
    context = candidate_bundle.context
    if result.run_id != context.run_id:
        raise ValueError("promotion_result.run_id must match context.run_id")
    if result.candidate_version != context.candidate_version:
        raise ValueError(
            "promotion_result.candidate_version must match context.candidate_version"
        )
    if result.release_version != context.release_version:
        raise ValueError(
            "promotion_result.release_version must match context.release_version"
        )
    if result.hf.repo_name != context.hf_repo_name:
        raise ValueError(
            "promotion_result.hf.repo_name must match context.hf_repo_name"
        )
    if result.gcs.bucket_name != context.gcs_bucket_name:
        raise ValueError(
            "promotion_result.gcs.bucket_name must match context.gcs_bucket_name"
        )
    if result.artifact_count != len(candidate_bundle.artifacts):
        raise ValueError(
            "promotion_result.artifact_count must match candidate artifacts"
        )


def _contract_inputs(
    candidate_bundle: ReleaseCandidateInputBundle,
) -> tuple[ArtifactRef, ...]:
    context = candidate_bundle.context
    inputs = [
        artifact.to_artifact_ref(
            uri_prefix=f"hf://{context.hf_repo_name}/{context.hf_staging_prefix}",
        )
        for artifact in candidate_bundle.artifacts
    ]
    if candidate_bundle.source_output_contract_path is not None:
        inputs.append(
            ArtifactRef(
                logical_name="stage4_output_contract",
                uri=f"hf://{context.hf_repo_name}/{candidate_bundle.source_output_contract_path}",
                media_type="application/json",
                metadata={
                    "artifact_family": "stage_contract",
                    "source_stage_id": "4_build_outputs",
                },
            )
        )
    for index, path in enumerate(candidate_bundle.validation_report_paths, start=1):
        inputs.append(
            ArtifactRef(
                logical_name=f"validation_report_{index}",
                uri=f"hf://{context.hf_repo_name}/{path}",
                media_type=_diagnostic_media_type(path),
                metadata={"artifact_family": "validation_report"},
            )
        )
    if candidate_bundle.diagnostics_manifest_path is not None:
        inputs.append(
            ArtifactRef(
                logical_name="diagnostics_manifest",
                uri=(
                    f"hf://{context.hf_repo_name}/"
                    f"{candidate_bundle.diagnostics_manifest_path}"
                ),
                media_type="application/json",
                metadata={"artifact_family": "diagnostics_manifest"},
            )
        )
    return tuple(inputs)


def _diagnostic_media_type(path: str) -> str:
    """Return a conservative media type for run diagnostics referenced by contract."""

    if path.endswith(".csv"):
        return "text/csv"
    if path.endswith(".txt"):
        return "text/plain"
    return "application/json"


def _contract_outputs(result: FullPromotionResult) -> tuple[ArtifactRef, ...]:
    hf_base = f"hf://{result.hf.repo_name}"
    return (
        ArtifactRef(
            logical_name="huggingface_release_artifacts",
            uri=f"{hf_base}/",
            metadata={
                "artifact_family": "release_artifact_collection",
                "artifact_count": result.artifact_count,
                "promoted_count": result.hf.promoted_count,
                "already_finalized": result.already_finalized,
                "repo_type": result.hf.repo_type,
                "hf_commit": result.hf.commit_id,
                "promoted_paths": list(result.hf.promoted_paths),
                "noop_paths": list(result.hf.noop_paths),
            },
        ),
        ArtifactRef(
            logical_name="gcs_release_artifacts",
            uri=f"gs://{result.gcs.bucket_name}/",
            metadata={
                "artifact_family": "release_artifact_collection",
                "artifact_count": result.artifact_count,
                "uploaded_count": result.gcs.uploaded_count,
                "already_finalized": result.already_finalized,
                "object_paths": list(result.gcs.object_paths),
                "skipped_paths": list(result.gcs.skipped_paths),
            },
        ),
        ArtifactRef(
            logical_name="release_manifest",
            uri=_hf_artifact_uri(
                result.hf.repo_name, result.release_manifest.root_path
            ),
            sha256=result.release_manifest.manifest_sha256,
            media_type="application/json",
            metadata={
                "artifact_family": "release_manifest",
                "artifact_count": result.release_manifest.artifact_count,
            },
        ),
        ArtifactRef(
            logical_name="versioned_release_manifest",
            uri=_hf_artifact_uri(
                result.hf.repo_name,
                result.release_manifest.versioned_path,
            ),
            sha256=result.release_manifest.manifest_sha256,
            media_type="application/json",
            metadata={
                "artifact_family": "release_manifest",
                "artifact_count": result.release_manifest.artifact_count,
            },
        ),
        ArtifactRef(
            logical_name="trace_tro",
            uri=_hf_artifact_uri(
                result.hf.repo_name, result.release_manifest.trace_tro_path
            ),
            media_type="application/ld+json",
            metadata={"artifact_family": "trace_tro"},
        ),
        ArtifactRef(
            logical_name="versioned_trace_tro",
            uri=_hf_artifact_uri(
                result.hf.repo_name,
                result.release_manifest.versioned_trace_tro_path,
            ),
            media_type="application/ld+json",
            metadata={"artifact_family": "trace_tro"},
        ),
        ArtifactRef(
            logical_name="version_manifest",
            uri=_hf_artifact_uri(result.hf.repo_name, result.version_manifest.path),
            media_type="application/json",
            metadata={
                "artifact_family": "version_manifest",
                "updated": result.version_manifest.updated,
            },
        ),
        ArtifactRef(
            logical_name="release_completion_marker",
            uri=_hf_artifact_uri(
                result.hf.repo_name, result.completion_marker.marker_path
            ),
            media_type="application/json",
            metadata={"artifact_family": "release_completion_marker"},
        ),
    )


def _hf_artifact_uri(repo_name: str, repo_path: str) -> str:
    """Return a Hugging Face URI from typed promotion result path material."""

    return f"hf://{repo_name}/{repo_path.lstrip('/')}"


def _contract_parameters(
    candidate_bundle: ReleaseCandidateInputBundle,
    result: FullPromotionResult,
) -> dict[str, Any]:
    context = candidate_bundle.context
    return {
        "run_id": context.run_id,
        "candidate_version": context.candidate_version,
        "release_version": context.release_version,
        "base_release_version": context.base_release_version,
        "release_bump": context.release_bump,
        "hf_repo_name": context.hf_repo_name,
        "hf_repo_type": context.hf_repo_type,
        "gcs_bucket_name": context.gcs_bucket_name,
        "hf_staging_prefix": context.hf_staging_prefix,
        "artifact_count": result.artifact_count,
        "release_candidate_fingerprint": (
            candidate_bundle.release_candidate_fingerprint
        ),
        "source_output_contract_path": candidate_bundle.source_output_contract_path,
        "validation_report_paths": list(candidate_bundle.validation_report_paths),
        "diagnostics_manifest_path": candidate_bundle.diagnostics_manifest_path,
    }


def _contract_metadata(
    *,
    context: ReleasePromotionContext,
    candidate_bundle: ReleaseCandidateInputBundle,
    promotion_result: FullPromotionResult,
    outputs: Sequence[ArtifactRef],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(extra),
        "contract_file": RELEASE_PROMOTION_CONTRACT_FILENAME,
        "contract_repo_path": release_promotion_contract_repo_path(context.run_id),
        "candidate_bundle_type": candidate_bundle.bundle_type,
        "candidate_metadata": candidate_bundle.metadata,
        "cleanup": promotion_result.cleanup.to_dict(),
        "already_finalized": promotion_result.already_finalized,
        "promotion_result": promotion_result.to_dict(),
        "public_refs": {output.logical_name: output.uri for output in outputs},
    }


def _execution_record(result: FullPromotionResult) -> ExecutionRecord:
    return ExecutionRecord(
        status="completed",
        reuse_decision="reused" if result.already_finalized else "computed",
        reuse_reason=(
            "already_finalized" if result.already_finalized else "fresh_promotion"
        ),
        reuse_summary=ReuseSummary(
            expected_outputs=result.artifact_count,
            valid_reused_outputs=(
                result.artifact_count if result.already_finalized else 0
            ),
            recomputed_outputs=0 if result.already_finalized else result.artifact_count,
        ),
    )


def _substage_records(
    *,
    candidate_inputs: Sequence[ArtifactRef],
    public_outputs: Sequence[ArtifactRef],
    promotion_result: FullPromotionResult,
) -> tuple[SubstageRecord, ...]:
    outputs_by_name = {artifact.logical_name: artifact for artifact in public_outputs}
    return (
        SubstageRecord(
            substage_id="5a_validate_outputs",
            status="completed",
            inputs=tuple(candidate_inputs),
            reuse_mode="observed_only",
            metadata={"artifact_count": promotion_result.artifact_count},
        ),
        SubstageRecord(
            substage_id="5b_promote_huggingface",
            status="completed",
            outputs=(outputs_by_name["huggingface_release_artifacts"],),
            reuse_mode="handoff",
            metadata={
                "promoted_count": promotion_result.hf.promoted_count,
                "already_finalized": promotion_result.already_finalized,
            },
        ),
        SubstageRecord(
            substage_id="5c_promote_gcs",
            status="completed",
            outputs=(outputs_by_name["gcs_release_artifacts"],),
            reuse_mode="handoff",
            metadata={
                "uploaded_count": promotion_result.gcs.uploaded_count,
                "already_finalized": promotion_result.already_finalized,
            },
        ),
        SubstageRecord(
            substage_id="5d_write_version_manifest",
            status="completed",
            outputs=(
                outputs_by_name["release_manifest"],
                outputs_by_name["versioned_release_manifest"],
                outputs_by_name["trace_tro"],
                outputs_by_name["versioned_trace_tro"],
                outputs_by_name["version_manifest"],
                outputs_by_name["release_completion_marker"],
            ),
            reuse_mode="handoff",
            metadata={
                "version_manifest_updated": promotion_result.version_manifest.updated,
                "cleanup": promotion_result.cleanup.to_dict(),
                "already_finalized": promotion_result.already_finalized,
            },
        ),
    )
