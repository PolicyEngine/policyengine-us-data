"""High-level Stage 5 release candidate bundle builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import DiagnosticRef
from policyengine_us_data.stage_contracts._coercion import jsonable_value

from .artifacts import (
    dedupe_normalized_release_paths,
    infer_release_artifact_spec,
    strip_staging_prefix,
)
from .candidate import ReleaseCandidateInputBundle
from .context import ReleasePromotionContext
from .fingerprints import candidate_bundle_with_fingerprint
from .stage4_inventory import (
    optional_record_int,
    optional_record_string,
)


@pipeline_node(
    id="legacy_release_candidate_bundle_builder",
    label="Legacy Release Candidate Bundle Builder",
    node_type="library",
    description="Compatibility builder for Stage 5 candidates from legacy staged path sets.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
def build_legacy_release_candidate_bundle(
    *,
    context: ReleasePromotionContext,
    rel_paths: Sequence[str],
    artifact_metadata_by_path: Mapping[str, Mapping[str, Any]] | None = None,
    validation_report_paths: Sequence[str] = (),
    validation_report_refs: Sequence[DiagnosticRef] = (),
    source_output_contract_path: str | None = None,
    diagnostics_manifest_path: str | None = None,
) -> ReleaseCandidateInputBundle:
    """Build a candidate bundle from the current legacy staged relative paths."""

    artifact_metadata_by_path = normalize_artifact_metadata_by_path(
        artifact_metadata_by_path or {},
        staging_prefix=context.hf_staging_prefix,
    )
    artifacts = tuple(
        legacy_artifact_spec(
            path,
            artifact_metadata_by_path=artifact_metadata_by_path,
        )
        for path in dedupe_normalized_release_paths(
            rel_paths,
            staging_prefix=context.hf_staging_prefix,
        )
    )
    return candidate_bundle_with_fingerprint(
        context=context,
        artifacts=artifacts,
        source_output_contract_path=source_output_contract_path,
        validation_report_paths=validation_report_paths,
        validation_report_refs=validation_report_refs,
        diagnostics_manifest_path=diagnostics_manifest_path,
        reader="legacy_staged_paths",
    )


def normalize_artifact_metadata_by_path(
    artifact_metadata_by_path: Mapping[str, Mapping[str, Any]],
    *,
    staging_prefix: str | None,
) -> dict[str, Mapping[str, Any]]:
    """Normalize artifact metadata keys by release-relative artifact path."""

    return {
        strip_staging_prefix(path, staging_prefix): metadata
        for path, metadata in artifact_metadata_by_path.items()
    }


def legacy_artifact_spec(
    path: str,
    *,
    artifact_metadata_by_path: Mapping[str, Mapping[str, Any]],
):
    """Build a release artifact spec from legacy path and metadata records."""

    metadata = artifact_metadata_by_path.get(path, {})
    return infer_release_artifact_spec(
        path,
        sha256=optional_record_string(metadata, "sha256"),
        size_bytes=optional_record_int(metadata, "size_bytes"),
        metadata={
            key: value
            for key, value in jsonable_value(metadata).items()
            if key not in {"sha256", "size_bytes"}
        },
    )
