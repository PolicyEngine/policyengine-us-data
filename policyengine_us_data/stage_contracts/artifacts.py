"""Artifact references used by stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._coercion import (
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
from .constants import CONTRACT_SCHEMA_VERSION, ContractPayload


@dataclass(frozen=True, kw_only=True)
class ArtifactRef:
    """Semantic pointer to a physical artifact."""

    logical_name: str
    uri: str
    sha256: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    schema_version: str = CONTRACT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "logical_name",
            require_non_empty(self.logical_name, "logical_name"),
        )
        object.__setattr__(self, "uri", require_non_empty(self.uri, "uri"))
        object.__setattr__(
            self,
            "sha256",
            optional_string_value(self.sha256, "sha256"),
        )
        object.__setattr__(
            self,
            "media_type",
            optional_string_value(self.media_type, "media_type"),
        )
        validate_optional_int(self.size_bytes, "size_bytes")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "logical_name": self.logical_name,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "schema_version": self.schema_version,
            "metadata": jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            logical_name=required_string(data, "logical_name"),
            uri=required_string(data, "uri"),
            sha256=optional_string(data, "sha256"),
            size_bytes=optional_int_value(data, "size_bytes"),
            media_type=optional_string(data, "media_type"),
            schema_version=schema_version(data),
            metadata=mapping_value(data, "metadata"),
        )
