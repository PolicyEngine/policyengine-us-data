"""Typed full release-promotion result aggregate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.release_promotion.results._coercion import (
    bool_value,
    nonnegative_int,
    require_type,
    string_tuple,
)
from policyengine_us_data.release_promotion.results.cleanup import (
    CLEANUP_STATUS_COMPLETED,
    CLEANUP_STATUS_SKIPPED,
    CleanupPromotionResult,
    cleanup_status,
)
from policyengine_us_data.release_promotion.results.destinations import (
    GcsPromotionResult,
    HuggingFacePromotionResult,
)
from policyengine_us_data.release_promotion.results.manifests import (
    CompletionMarkerPromotionResult,
    ReleaseManifestPromotionResult,
    VersionManifestPromotionResult,
)
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
    optional_string,
    require_non_empty,
    required_string,
    schema_version,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION


def _default_hf_staging_prefix(candidate_version: str, run_id: str) -> str:
    return f"staging/{candidate_version}-{run_id}"


@pipeline_node(
    id="full_promotion_result",
    label="FullPromotionResult",
    node_type="library",
    description="Typed Stage 5 result model for full release promotion outcomes.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    api_refs=["policyengine_us_data.release_promotion.results.FullPromotionResult"],
    artifacts_in=["release promotion transaction output"],
    artifacts_out=["typed promotion result"],
    validation_commands=["uv run pytest tests/unit/release_promotion/test_results.py"],
)
@dataclass(frozen=True, kw_only=True)
class FullPromotionResult:
    """Typed result for a full Stage 5 release promotion transaction."""

    run_id: str
    candidate_version: str
    release_version: str
    artifact_count: int
    hf: HuggingFacePromotionResult
    gcs: GcsPromotionResult
    release_manifest: ReleaseManifestPromotionResult
    version_manifest: VersionManifestPromotionResult
    completion_marker: CompletionMarkerPromotionResult
    cleanup: CleanupPromotionResult
    already_finalized: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(self, "run_id", require_non_empty(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "candidate_version",
            require_non_empty(self.candidate_version, "candidate_version"),
        )
        object.__setattr__(
            self,
            "release_version",
            require_non_empty(self.release_version, "release_version"),
        )
        object.__setattr__(
            self,
            "artifact_count",
            nonnegative_int(self.artifact_count, "artifact_count"),
        )
        require_type(self.hf, "hf", HuggingFacePromotionResult)
        require_type(self.gcs, "gcs", GcsPromotionResult)
        require_type(
            self.release_manifest,
            "release_manifest",
            ReleaseManifestPromotionResult,
        )
        require_type(
            self.version_manifest,
            "version_manifest",
            VersionManifestPromotionResult,
        )
        require_type(
            self.completion_marker,
            "completion_marker",
            CompletionMarkerPromotionResult,
        )
        require_type(self.cleanup, "cleanup", CleanupPromotionResult)
        object.__setattr__(
            self,
            "already_finalized",
            bool_value(self.already_finalized, "already_finalized"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "run_id": self.run_id,
            "candidate_version": self.candidate_version,
            "release_version": self.release_version,
            "artifact_count": self.artifact_count,
            "hf": self.hf.to_dict(),
            "gcs": self.gcs.to_dict(),
            "release_manifest": self.release_manifest.to_dict(),
            "version_manifest": self.version_manifest.to_dict(),
            "completion_marker": self.completion_marker.to_dict(),
            "cleanup": self.cleanup.to_dict(),
            "already_finalized": self.already_finalized,
            "metadata": jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FullPromotionResult":
        """Restore a full promotion result from a mapping."""

        return cls(
            run_id=required_string(data, "run_id"),
            candidate_version=required_string(data, "candidate_version"),
            release_version=required_string(data, "release_version"),
            artifact_count=nonnegative_int(
                data.get("artifact_count"),
                "artifact_count",
            ),
            hf=HuggingFacePromotionResult.from_dict(mapping_value(data, "hf")),
            gcs=GcsPromotionResult.from_dict(mapping_value(data, "gcs")),
            release_manifest=ReleaseManifestPromotionResult.from_dict(
                mapping_value(data, "release_manifest"),
            ),
            version_manifest=VersionManifestPromotionResult.from_dict(
                mapping_value(data, "version_manifest"),
            ),
            completion_marker=CompletionMarkerPromotionResult.from_dict(
                mapping_value(data, "completion_marker"),
            ),
            cleanup=CleanupPromotionResult.from_dict(mapping_value(data, "cleanup")),
            already_finalized=bool_value(
                data.get("already_finalized", False),
                "already_finalized",
            ),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )

    @classmethod
    def from_legacy_dict(cls, data: Mapping[str, Any]) -> "FullPromotionResult":
        """Build a typed result from the existing promotion dictionary output."""

        run_id = required_string(data, "run_id")
        candidate_version = required_string(data, "candidate_version")
        release_version = required_string(data, "release_version")
        rel_paths = string_tuple(data.get("rel_paths"), "rel_paths")
        already_finalized = bool_value(
            data.get("already_finalized", False),
            "already_finalized",
        )
        cleanup_attempted = bool_value(
            data.get("staging_cleanup_attempted", True),
            "staging_cleanup_attempted",
        )
        cleanup_status_value = data.get("staging_cleanup_status")
        if cleanup_status_value is None:
            cleanup_status_value = (
                CLEANUP_STATUS_SKIPPED
                if not cleanup_attempted
                else CLEANUP_STATUS_COMPLETED
            )

        return cls(
            run_id=run_id,
            candidate_version=candidate_version,
            release_version=release_version,
            artifact_count=nonnegative_int(
                data.get("artifact_count"),
                "artifact_count",
            ),
            hf=HuggingFacePromotionResult(
                repo_name=required_string(data, "hf_repo_name"),
                repo_type=required_string(data, "hf_repo_type"),
                source_staging_prefix=(
                    optional_string(data, "hf_staging_prefix")
                    or _default_hf_staging_prefix(candidate_version, run_id)
                ),
                promoted_paths=string_tuple(
                    data.get("hf_promoted_paths", rel_paths),
                    "hf_promoted_paths",
                ),
                promoted_count=nonnegative_int(
                    data.get("hf_promoted"),
                    "hf_promoted",
                ),
                commit_id=optional_string(data, "hf_commit_id"),
                noop_paths=string_tuple(
                    data.get("hf_noop_paths", rel_paths if already_finalized else ()),
                    "hf_noop_paths",
                ),
                already_finalized=already_finalized,
            ),
            gcs=GcsPromotionResult(
                bucket_name=required_string(data, "gcs_bucket_name"),
                object_paths=string_tuple(
                    data.get("gcs_object_paths", rel_paths),
                    "gcs_object_paths",
                ),
                release_version=release_version,
                uploaded_count=nonnegative_int(
                    data.get("gcs_uploaded"),
                    "gcs_uploaded",
                ),
                skipped_paths=string_tuple(
                    data.get(
                        "gcs_skipped_paths",
                        rel_paths if already_finalized else (),
                    ),
                    "gcs_skipped_paths",
                ),
                failures=string_tuple(data.get("gcs_failures"), "gcs_failures"),
            ),
            release_manifest=ReleaseManifestPromotionResult(
                root_path=required_string(data, "release_manifest_path"),
                versioned_path=required_string(
                    data,
                    "versioned_release_manifest_path",
                ),
                trace_tro_path=required_string(data, "trace_tro_path"),
                versioned_trace_tro_path=required_string(
                    data,
                    "versioned_trace_tro_path",
                ),
                artifact_count=nonnegative_int(
                    data.get("release_manifest_artifacts"),
                    "release_manifest_artifacts",
                ),
                manifest_sha256=optional_string(data, "release_manifest_sha256"),
            ),
            version_manifest=VersionManifestPromotionResult(
                path=required_string(data, "version_manifest_path"),
                version=required_string(data, "version_manifest_version"),
                updated=bool_value(
                    data.get("version_manifest_updated", not already_finalized),
                    "version_manifest_updated",
                ),
                current_version=(
                    optional_string(data, "version_manifest_current_version")
                    or release_version
                ),
            ),
            completion_marker=CompletionMarkerPromotionResult(
                marker_path=required_string(data, "release_completion_marker"),
                tag=required_string(data, "release_completion_tag"),
                valid=bool_value(data.get("release_completion_valid"), "valid"),
            ),
            cleanup=CleanupPromotionResult(
                cleaned_count=nonnegative_int(
                    data.get("staging_cleaned"),
                    "staging_cleaned",
                ),
                attempted=cleanup_attempted,
                status=cleanup_status(cleanup_status_value),
            ),
            already_finalized=already_finalized,
        )


def parse_full_promotion_result_json(payload: str) -> FullPromotionResult:
    """Parse a JSON legacy promotion payload into a typed promotion result."""

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("promotion result payload must be JSON") from exc
    if not isinstance(data, Mapping):
        raise ValueError("promotion result payload must be a JSON object")
    return FullPromotionResult.from_legacy_dict(data)
