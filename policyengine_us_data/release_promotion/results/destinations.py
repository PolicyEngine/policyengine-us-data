"""Typed destination result models for release promotion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from policyengine_us_data.release_promotion.results._coercion import (
    bool_value,
    nonnegative_int,
    string_tuple,
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
class HuggingFacePromotionResult:
    """Result for copying staged Hugging Face artifacts to production paths."""

    repo_name: str
    repo_type: str
    source_staging_prefix: str
    promoted_paths: tuple[str, ...]
    promoted_count: int
    commit_id: str | None = None
    noop_paths: tuple[str, ...] = ()
    already_finalized: bool = False
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self, "repo_name", require_non_empty(self.repo_name, "repo_name")
        )
        object.__setattr__(
            self, "repo_type", require_non_empty(self.repo_type, "repo_type")
        )
        object.__setattr__(
            self,
            "source_staging_prefix",
            require_non_empty(self.source_staging_prefix, "source_staging_prefix"),
        )
        object.__setattr__(
            self,
            "promoted_paths",
            string_tuple(self.promoted_paths, "promoted_paths"),
        )
        object.__setattr__(
            self,
            "promoted_count",
            nonnegative_int(self.promoted_count, "promoted_count"),
        )
        object.__setattr__(
            self,
            "commit_id",
            optional_string_value(self.commit_id, "commit_id"),
        )
        object.__setattr__(
            self,
            "noop_paths",
            string_tuple(self.noop_paths, "noop_paths"),
        )
        object.__setattr__(
            self,
            "already_finalized",
            bool_value(self.already_finalized, "already_finalized"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "repo_name": self.repo_name,
            "repo_type": self.repo_type,
            "source_staging_prefix": self.source_staging_prefix,
            "promoted_paths": list(self.promoted_paths),
            "promoted_count": self.promoted_count,
            "commit_id": self.commit_id,
            "noop_paths": list(self.noop_paths),
            "already_finalized": self.already_finalized,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HuggingFacePromotionResult":
        """Restore a Hugging Face promotion result from a mapping."""

        return cls(
            repo_name=required_string(data, "repo_name"),
            repo_type=required_string(data, "repo_type"),
            source_staging_prefix=required_string(data, "source_staging_prefix"),
            promoted_paths=string_tuple(data.get("promoted_paths"), "promoted_paths"),
            promoted_count=nonnegative_int(
                data.get("promoted_count"),
                "promoted_count",
            ),
            commit_id=optional_string(data, "commit_id"),
            noop_paths=string_tuple(data.get("noop_paths"), "noop_paths"),
            already_finalized=bool_value(
                data.get("already_finalized", False),
                "already_finalized",
            ),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class GcsPromotionResult:
    """Result for uploading staged Hugging Face artifacts to GCS."""

    bucket_name: str
    object_paths: tuple[str, ...]
    release_version: str
    uploaded_count: int
    skipped_paths: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "bucket_name",
            require_non_empty(self.bucket_name, "bucket_name"),
        )
        object.__setattr__(
            self,
            "object_paths",
            string_tuple(self.object_paths, "object_paths"),
        )
        object.__setattr__(
            self,
            "release_version",
            require_non_empty(self.release_version, "release_version"),
        )
        object.__setattr__(
            self,
            "uploaded_count",
            nonnegative_int(self.uploaded_count, "uploaded_count"),
        )
        object.__setattr__(
            self,
            "skipped_paths",
            string_tuple(self.skipped_paths, "skipped_paths"),
        )
        object.__setattr__(
            self,
            "failures",
            string_tuple(self.failures, "failures"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to JSON-compatible primitives."""

        return {
            "bucket_name": self.bucket_name,
            "object_paths": list(self.object_paths),
            "release_version": self.release_version,
            "uploaded_count": self.uploaded_count,
            "skipped_paths": list(self.skipped_paths),
            "failures": list(self.failures),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GcsPromotionResult":
        """Restore a GCS promotion result from a mapping."""

        return cls(
            bucket_name=required_string(data, "bucket_name"),
            object_paths=string_tuple(data.get("object_paths"), "object_paths"),
            release_version=required_string(data, "release_version"),
            uploaded_count=nonnegative_int(
                data.get("uploaded_count"),
                "uploaded_count",
            ),
            skipped_paths=string_tuple(data.get("skipped_paths"), "skipped_paths"),
            failures=string_tuple(data.get("failures"), "failures"),
            schema_version=schema_version(data),
        )
