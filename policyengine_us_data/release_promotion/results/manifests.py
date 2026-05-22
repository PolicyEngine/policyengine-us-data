"""Typed manifest result models for release promotion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from policyengine_us_data.release_promotion.results._coercion import (
    bool_value,
    nonnegative_int,
)
from policyengine_us_data.stage_contracts._coercion import (
    optional_string,
    optional_string_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True, kw_only=True)
class ReleaseManifestPromotionResult:
    """Result for writing the release manifest and TRACE TRO artifacts."""

    root_path: str
    versioned_path: str
    trace_tro_path: str
    versioned_trace_tro_path: str
    artifact_count: int
    manifest_sha256: str | None = None
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self, "root_path", require_non_empty(self.root_path, "root_path")
        )
        object.__setattr__(
            self,
            "versioned_path",
            require_non_empty(self.versioned_path, "versioned_path"),
        )
        object.__setattr__(
            self,
            "trace_tro_path",
            require_non_empty(self.trace_tro_path, "trace_tro_path"),
        )
        object.__setattr__(
            self,
            "versioned_trace_tro_path",
            require_non_empty(
                self.versioned_trace_tro_path,
                "versioned_trace_tro_path",
            ),
        )
        object.__setattr__(
            self,
            "artifact_count",
            nonnegative_int(self.artifact_count, "artifact_count"),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            optional_string_value(self.manifest_sha256, "manifest_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "root_path": self.root_path,
            "versioned_path": self.versioned_path,
            "trace_tro_path": self.trace_tro_path,
            "versioned_trace_tro_path": self.versioned_trace_tro_path,
            "artifact_count": self.artifact_count,
            "manifest_sha256": self.manifest_sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseManifestPromotionResult":
        """Restore a release manifest result from a mapping."""

        return cls(
            root_path=required_string(data, "root_path"),
            versioned_path=required_string(data, "versioned_path"),
            trace_tro_path=required_string(data, "trace_tro_path"),
            versioned_trace_tro_path=required_string(data, "versioned_trace_tro_path"),
            artifact_count=nonnegative_int(
                data.get("artifact_count"),
                "artifact_count",
            ),
            manifest_sha256=optional_string(data, "manifest_sha256"),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class VersionManifestPromotionResult:
    """Result for updating the public version manifest."""

    path: str
    version: str
    updated: bool
    current_version: str | None = None
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(self, "path", require_non_empty(self.path, "path"))
        object.__setattr__(self, "version", require_non_empty(self.version, "version"))
        object.__setattr__(self, "updated", bool_value(self.updated, "updated"))
        object.__setattr__(
            self,
            "current_version",
            optional_string_value(self.current_version, "current_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "path": self.path,
            "version": self.version,
            "updated": self.updated,
            "current_version": self.current_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionManifestPromotionResult":
        """Restore a version manifest result from a mapping."""

        return cls(
            path=required_string(data, "path"),
            version=required_string(data, "version"),
            updated=bool_value(data.get("updated"), "updated"),
            current_version=optional_string(data, "current_version"),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class CompletionMarkerPromotionResult:
    """Result for writing or verifying the release completion marker."""

    marker_path: str
    tag: str
    valid: bool
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "marker_path",
            require_non_empty(self.marker_path, "marker_path"),
        )
        object.__setattr__(self, "tag", require_non_empty(self.tag, "tag"))
        object.__setattr__(self, "valid", bool_value(self.valid, "valid"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "marker_path": self.marker_path,
            "tag": self.tag,
            "valid": self.valid,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompletionMarkerPromotionResult":
        """Restore a completion marker result from a mapping."""

        return cls(
            marker_path=required_string(data, "marker_path"),
            tag=required_string(data, "tag"),
            valid=bool_value(data.get("valid"), "valid"),
            schema_version=schema_version(data),
        )
