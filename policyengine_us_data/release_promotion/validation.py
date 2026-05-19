"""Canonical validation-report adapters for Stage 5 release candidates."""

from __future__ import annotations

from collections import Counter

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import ValidationFinding, ValidationReport
from policyengine_us_data.stage_contracts.stages import (
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
)

from .candidate import ReleaseCandidateInputBundle


@pipeline_node(
    id="release_candidate_shape_report",
    label="Release Candidate Shape Report",
    node_type="validation",
    description="Adapt Stage 5 release-candidate shape checks into the canonical validation report schema.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
def build_release_candidate_shape_report(
    bundle: ReleaseCandidateInputBundle,
) -> ValidationReport:
    """Describe candidate-bundle shape using the shared validation schema."""

    families = Counter(artifact.artifact_family for artifact in bundle.artifacts)
    return ValidationReport(
        status="pass",
        findings=(
            ValidationFinding(
                check_id="release_candidate_identity_declared",
                status="pass",
                message="Release candidate declares one canonical run and release identity.",
                metadata={
                    "run_id": bundle.context.run_id,
                    "candidate_version": bundle.context.candidate_version,
                    "release_version": bundle.context.release_version,
                    "hf_staging_prefix": bundle.context.hf_staging_prefix,
                    "release_candidate_fingerprint": (
                        bundle.release_candidate_fingerprint
                    ),
                },
            ),
            ValidationFinding(
                check_id="release_candidate_artifacts_declared",
                status="pass",
                message="Release candidate declares typed release artifacts.",
                metric="artifact_count",
                value=len(bundle.artifacts),
                threshold=1,
                metadata={
                    "artifact_families": dict(sorted(families.items())),
                    "required_artifacts": sum(
                        1 for artifact in bundle.artifacts if artifact.required
                    ),
                },
            ),
        ),
        metadata={
            "stage_id": STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
            "substage_id": "5a_validate_outputs",
            "run_id": bundle.context.run_id,
            "release_candidate_fingerprint": bundle.release_candidate_fingerprint,
            "validation_kind": "candidate_shape",
        },
    )
