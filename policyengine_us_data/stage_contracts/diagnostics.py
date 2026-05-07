"""Diagnostic references used by stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_optional_instance,
    validate_schema_version,
)
from .artifacts import ArtifactRef
from .constants import (
    CONTRACT_SCHEMA_VERSION,
    DIAGNOSTIC_SEVERITIES,
    ContractPayload,
    DiagnosticSeverity,
)


@dataclass(frozen=True, kw_only=True)
class DiagnosticRef:
    """Reference to a diagnostic artifact or embedded diagnostic summary."""

    name: str
    kind: str
    artifact: ArtifactRef | None = None
    summary: Mapping[str, Any] = field(default_factory=dict)
    severity: DiagnosticSeverity = "info"
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(self, "name", require_non_empty(self.name, "name"))
        object.__setattr__(self, "kind", require_non_empty(self.kind, "kind"))
        validate_optional_instance(self.artifact, "artifact", ArtifactRef)
        if self.severity not in DIAGNOSTIC_SEVERITIES:
            raise ValueError(f"Invalid diagnostic severity: {self.severity!r}")
        object.__setattr__(
            self,
            "summary",
            freeze_mapping(self.summary, "summary"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "name": self.name,
            "kind": self.kind,
            "artifact": (
                self.artifact.to_dict() if self.artifact is not None else None
            ),
            "summary": jsonable_value(self.summary),
            "severity": self.severity,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiagnosticRef":
        artifact = data.get("artifact")
        return cls(
            name=required_string(data, "name"),
            kind=required_string(data, "kind"),
            artifact=ArtifactRef.from_dict(artifact) if artifact is not None else None,
            summary=mapping_value(data, "summary"),
            severity=data.get("severity", "info"),
            schema_version=schema_version(data),
        )
