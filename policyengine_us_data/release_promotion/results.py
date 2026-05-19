"""Typed Stage 5 promotion result models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
    optional_string,
    optional_string_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _bool_value(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


@dataclass(frozen=True, kw_only=True)
class HuggingFacePromotionResult:
    """Result for copying staged Hugging Face artifacts to production paths."""

    promoted_count: int
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "promoted_count",
            _nonnegative_int(self.promoted_count, "promoted_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "promoted_count": self.promoted_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HuggingFacePromotionResult":
        """Restore a Hugging Face promotion result from a mapping."""

        return cls(
            promoted_count=_nonnegative_int(
                data.get("promoted_count"),
                "promoted_count",
            ),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class GcsPromotionResult:
    """Result for uploading staged Hugging Face artifacts to GCS."""

    uploaded_count: int
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "uploaded_count",
            _nonnegative_int(self.uploaded_count, "uploaded_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "uploaded_count": self.uploaded_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GcsPromotionResult":
        """Restore a GCS promotion result from a mapping."""

        return cls(
            uploaded_count=_nonnegative_int(
                data.get("uploaded_count"),
                "uploaded_count",
            ),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class ReleaseManifestPromotionResult:
    """Result for writing the release manifest and TRACE TRO artifacts."""

    artifact_count: int
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "artifact_count",
            _nonnegative_int(self.artifact_count, "artifact_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "artifact_count": self.artifact_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseManifestPromotionResult":
        """Restore a release manifest result from a mapping."""

        return cls(
            artifact_count=_nonnegative_int(
                data.get("artifact_count"),
                "artifact_count",
            ),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class VersionManifestPromotionResult:
    """Result for updating the public version manifest."""

    updated: bool
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(self, "updated", _bool_value(self.updated, "updated"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "updated": self.updated,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionManifestPromotionResult":
        """Restore a version manifest result from a mapping."""

        return cls(
            updated=_bool_value(data.get("updated"), "updated"),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class CompletionMarkerPromotionResult:
    """Result for writing or verifying the release completion marker."""

    marker_path: str | None
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "marker_path",
            optional_string_value(self.marker_path, "marker_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "marker_path": self.marker_path,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompletionMarkerPromotionResult":
        """Restore a completion marker result from a mapping."""

        return cls(
            marker_path=optional_string(data, "marker_path"),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class CleanupPromotionResult:
    """Result for post-certification staging cleanup."""

    cleaned_count: int
    attempted: bool = True
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "cleaned_count",
            _nonnegative_int(self.cleaned_count, "cleaned_count"),
        )
        object.__setattr__(self, "attempted", _bool_value(self.attempted, "attempted"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "cleaned_count": self.cleaned_count,
            "attempted": self.attempted,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CleanupPromotionResult":
        """Restore a cleanup result from a mapping."""

        return cls(
            cleaned_count=_nonnegative_int(
                data.get("cleaned_count"),
                "cleaned_count",
            ),
            attempted=_bool_value(data.get("attempted", True), "attempted"),
            schema_version=schema_version(data),
        )


@pipeline_node(
    id="full_promotion_result",
    label="FullPromotionResult",
    node_type="library",
    description="Typed Stage 5 result model for full release promotion outcomes.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
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
            _nonnegative_int(self.artifact_count, "artifact_count"),
        )
        _require_type(self.hf, "hf", HuggingFacePromotionResult)
        _require_type(self.gcs, "gcs", GcsPromotionResult)
        _require_type(
            self.release_manifest,
            "release_manifest",
            ReleaseManifestPromotionResult,
        )
        _require_type(
            self.version_manifest,
            "version_manifest",
            VersionManifestPromotionResult,
        )
        _require_type(
            self.completion_marker,
            "completion_marker",
            CompletionMarkerPromotionResult,
        )
        _require_type(self.cleanup, "cleanup", CleanupPromotionResult)
        object.__setattr__(
            self,
            "already_finalized",
            _bool_value(self.already_finalized, "already_finalized"),
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
            artifact_count=_nonnegative_int(
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
            already_finalized=_bool_value(
                data.get("already_finalized", False),
                "already_finalized",
            ),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )

    @classmethod
    def from_legacy_dict(cls, data: Mapping[str, Any]) -> "FullPromotionResult":
        """Build a typed result from the existing promotion dictionary output."""

        already_finalized = _bool_value(
            data.get("already_finalized", False),
            "already_finalized",
        )
        return cls(
            run_id=required_string(data, "run_id"),
            candidate_version=required_string(data, "candidate_version"),
            release_version=required_string(data, "release_version"),
            artifact_count=_nonnegative_int(
                data.get("artifact_count"),
                "artifact_count",
            ),
            hf=HuggingFacePromotionResult(
                promoted_count=_nonnegative_int(
                    data.get("hf_promoted"),
                    "hf_promoted",
                ),
            ),
            gcs=GcsPromotionResult(
                uploaded_count=_nonnegative_int(
                    data.get("gcs_uploaded"),
                    "gcs_uploaded",
                ),
            ),
            release_manifest=ReleaseManifestPromotionResult(
                artifact_count=_nonnegative_int(
                    data.get("release_manifest_artifacts"),
                    "release_manifest_artifacts",
                ),
            ),
            version_manifest=VersionManifestPromotionResult(
                updated=_bool_value(
                    data.get("version_manifest_updated", not already_finalized),
                    "version_manifest_updated",
                ),
            ),
            completion_marker=CompletionMarkerPromotionResult(
                marker_path=optional_string(data, "release_completion_marker"),
            ),
            cleanup=CleanupPromotionResult(
                cleaned_count=_nonnegative_int(
                    data.get("staging_cleaned"),
                    "staging_cleaned",
                ),
                attempted=_bool_value(
                    data.get("staging_cleanup_attempted", True),
                    "staging_cleanup_attempted",
                ),
            ),
            already_finalized=already_finalized,
        )


def _require_type(value: Any, field_name: str, expected_type: type) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field_name} must be {expected_type.__name__}")
