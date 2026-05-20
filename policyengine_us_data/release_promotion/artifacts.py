"""Release artifact identity helpers for Stage 5 candidate bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
import posixpath
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import ArtifactRef
from policyengine_us_data.stage_contracts._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
    optional_int_value,
    optional_string,
    optional_string_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_optional_int,
    validate_schema_version,
)
from policyengine_us_data.stage_contracts.constants import CONTRACT_SCHEMA_VERSION
from policyengine_us_data.stage_contracts.stages import (
    STAGE_1_BUILD_DATASETS,
    STAGE_4_BUILD_OUTPUTS,
    is_canonical_stage_id,
)

BASE_RELEASE_ARTIFACT_PATHS = (
    "cps_2024.h5",
    "policy_data.db",
    "enhanced_cps_2024.h5",
    "small_enhanced_cps_2024.h5",
)

BASE_RELEASE_LOGICAL_NAMES = {
    "cps_2024.h5": "cps_2024",
    "policy_data.db": "policy_data_db",
    "enhanced_cps_2024.h5": "enhanced_cps_2024",
    "small_enhanced_cps_2024.h5": "small_enhanced_cps_2024",
}


@pipeline_node(
    id="release_artifact_spec",
    label="ReleaseArtifactSpec",
    node_type="library",
    description="Normalized per-artifact identity for a Stage 5 release candidate.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class ReleaseArtifactSpec:
    """Normalized identity for one artifact in a Stage 5 release candidate."""

    logical_name: str
    relative_path: str
    artifact_family: str
    source_stage_id: str
    area_type: str | None = None
    area_id: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "logical_name",
            require_non_empty(self.logical_name, "logical_name"),
        )
        object.__setattr__(
            self,
            "relative_path",
            normalize_release_path(self.relative_path),
        )
        object.__setattr__(
            self,
            "artifact_family",
            require_non_empty(self.artifact_family, "artifact_family"),
        )
        object.__setattr__(
            self,
            "source_stage_id",
            require_non_empty(self.source_stage_id, "source_stage_id"),
        )
        if not is_canonical_stage_id(self.source_stage_id):
            raise ValueError(f"Invalid source_stage_id: {self.source_stage_id!r}")
        object.__setattr__(
            self,
            "area_type",
            optional_string_value(self.area_type, "area_type"),
        )
        object.__setattr__(
            self,
            "area_id",
            optional_string_value(self.area_id, "area_id"),
        )
        object.__setattr__(
            self,
            "sha256",
            optional_string_value(self.sha256, "sha256"),
        )
        validate_optional_int(self.size_bytes, "size_bytes")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not isinstance(self.required, bool):
            raise ValueError("required must be a boolean")
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_artifact_ref(self, *, uri_prefix: str = "") -> ArtifactRef:
        """Return a generic stage-contract artifact reference for this artifact."""

        uri = (
            f"{uri_prefix.rstrip('/')}/{self.relative_path}"
            if uri_prefix
            else self.relative_path
        )
        return ArtifactRef(
            logical_name=self.logical_name,
            uri=uri,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            metadata={
                **jsonable_value(self.metadata),
                "artifact_family": self.artifact_family,
                "source_stage_id": self.source_stage_id,
                "area_type": self.area_type,
                "area_id": self.area_id,
                "required": self.required,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact spec to JSON-compatible primitives."""

        return {
            "logical_name": self.logical_name,
            "relative_path": self.relative_path,
            "artifact_family": self.artifact_family,
            "source_stage_id": self.source_stage_id,
            "area_type": self.area_type,
            "area_id": self.area_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "required": self.required,
            "metadata": jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseArtifactSpec":
        """Restore a release artifact spec from serialized data."""

        return cls(
            logical_name=required_string(data, "logical_name"),
            relative_path=required_string(data, "relative_path"),
            artifact_family=required_string(data, "artifact_family"),
            source_stage_id=required_string(data, "source_stage_id"),
            area_type=optional_string(data, "area_type"),
            area_id=optional_string(data, "area_id"),
            sha256=optional_string(data, "sha256"),
            size_bytes=optional_int_value(data, "size_bytes"),
            required=data.get("required", True),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )


def normalize_release_path(path: str) -> str:
    """Normalize a release repo path and reject absolute or parent paths."""

    if not isinstance(path, str):
        raise ValueError("path must be a non-empty string")
    value = require_non_empty(path.strip().replace("\\", "/"), "path")
    if "://" in value:
        raise ValueError("release paths must be repo-relative, not URIs")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"release path must not contain parent traversal: {path!r}")
    normalized = posixpath.normpath(value)
    if normalized in {"", "."}:
        raise ValueError("release path must not be empty")
    if ":" in normalized.split("/", 1)[0]:
        raise ValueError("release path must be repo-relative, not a drive path")
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"release path must stay inside the repo: {path!r}")
    return normalized


def strip_staging_prefix(path: str, staging_prefix: str | None) -> str:
    """Return a production-relative path from a staged HF path when possible."""

    normalized = normalize_release_path(path)
    if not staging_prefix:
        return normalized
    prefix = normalize_release_path(staging_prefix)
    if normalized.startswith(f"{prefix}/"):
        return normalized[len(prefix) + 1 :]
    if normalized.startswith("staging/"):
        raise ValueError(
            "staged release path does not match the expected staging prefix"
        )
    return normalized


def dedupe_normalized_release_paths(
    paths: Sequence[str],
    *,
    staging_prefix: str | None = None,
) -> tuple[str, ...]:
    """Normalize and deduplicate release paths while preserving first mention."""

    seen: set[str] = set()
    deduped: list[str] = []
    for path in paths:
        normalized = strip_staging_prefix(path, staging_prefix)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return tuple(deduped)


def infer_release_artifact_spec(
    relative_path: str,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
    required: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ReleaseArtifactSpec:
    """Infer a Stage 5 artifact spec from a production-relative repo path."""

    path = normalize_release_path(relative_path)
    family, area_type, area_id, source_stage_id = infer_artifact_identity(path)
    return ReleaseArtifactSpec(
        logical_name=logical_name_for_release_path(path),
        relative_path=path,
        artifact_family=family,
        source_stage_id=source_stage_id,
        area_type=area_type,
        area_id=area_id,
        sha256=sha256,
        size_bytes=size_bytes,
        required=required,
        metadata=metadata or {},
    )


def infer_artifact_identity(
    relative_path: str,
) -> tuple[str, str | None, str | None, str]:
    """Infer artifact family, area identity, and source stage from a repo path."""

    path = normalize_release_path(relative_path)
    parts = PurePosixPath(path).parts
    if path in BASE_RELEASE_ARTIFACT_PATHS:
        return "base_dataset", None, None, STAGE_1_BUILD_DATASETS
    if parts == ("national", "US.h5"):
        return "national_h5", "national", "US", STAGE_4_BUILD_OUTPUTS
    if len(parts) == 2 and parts[0] in {"states", "districts", "cities"}:
        area_type = {
            "states": "state",
            "districts": "district",
            "cities": "city",
        }[parts[0]]
        return (
            f"{area_type}_h5",
            area_type,
            PurePosixPath(parts[1]).stem,
            STAGE_4_BUILD_OUTPUTS,
        )
    return "release_artifact", None, None, STAGE_4_BUILD_OUTPUTS


def logical_name_for_release_path(relative_path: str) -> str:
    """Return a stable logical name for a release repo path."""

    path = normalize_release_path(relative_path)
    if path in BASE_RELEASE_LOGICAL_NAMES:
        return BASE_RELEASE_LOGICAL_NAMES[path]
    pure_path = PurePosixPath(path)
    if pure_path.suffix:
        return str(pure_path.with_suffix(""))
    return path
