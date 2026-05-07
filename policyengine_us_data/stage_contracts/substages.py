"""Substage records embedded inside stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._coercion import (
    freeze_mapping,
    freeze_sequence,
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
    SUBSTAGE_REUSE_MODES,
    SUBSTAGE_STATUSES,
    ContractPayload,
    SubstageReuseMode,
    SubstageStatus,
)
from .diagnostics import DiagnosticRef
from .fingerprints import Fingerprint
from .validation import ValidationReport


@dataclass(frozen=True, kw_only=True)
class SubstageRecord:
    """Record of work inside a canonical stage."""

    substage_id: str
    status: SubstageStatus
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: Fingerprint | None = None
    reuse_mode: SubstageReuseMode = "observed_only"
    validation: ValidationReport | None = None
    diagnostics: tuple[DiagnosticRef, ...] = ()
    schema_version: str = CONTRACT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "substage_id",
            require_non_empty(self.substage_id, "substage_id"),
        )
        if self.status not in SUBSTAGE_STATUSES:
            raise ValueError(f"Invalid substage status: {self.status!r}")
        if self.reuse_mode not in SUBSTAGE_REUSE_MODES:
            raise ValueError(f"Invalid substage reuse mode: {self.reuse_mode!r}")
        object.__setattr__(
            self,
            "inputs",
            freeze_sequence(self.inputs, "inputs", ArtifactRef),
        )
        object.__setattr__(
            self,
            "outputs",
            freeze_sequence(self.outputs, "outputs", ArtifactRef),
        )
        validate_optional_instance(self.fingerprint, "fingerprint", Fingerprint)
        validate_optional_instance(self.validation, "validation", ValidationReport)
        object.__setattr__(
            self,
            "diagnostics",
            freeze_sequence(self.diagnostics, "diagnostics", DiagnosticRef),
        )
        object.__setattr__(
            self,
            "parameters",
            freeze_mapping(self.parameters, "parameters"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "substage_id": self.substage_id,
            "status": self.status,
            "inputs": [jsonable_value(item) for item in self.inputs],
            "outputs": [jsonable_value(item) for item in self.outputs],
            "parameters": jsonable_value(self.parameters),
            "fingerprint": (
                self.fingerprint.to_dict() if self.fingerprint is not None else None
            ),
            "reuse_mode": self.reuse_mode,
            "validation": (
                self.validation.to_dict() if self.validation is not None else None
            ),
            "diagnostics": [jsonable_value(item) for item in self.diagnostics],
            "schema_version": self.schema_version,
            "metadata": jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubstageRecord":
        fingerprint = data.get("fingerprint")
        validation = data.get("validation")
        return cls(
            substage_id=required_string(data, "substage_id"),
            status=data["status"],
            inputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("inputs", ())
            ),
            outputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("outputs", ())
            ),
            parameters=mapping_value(data, "parameters"),
            fingerprint=(
                Fingerprint.from_dict(fingerprint) if fingerprint is not None else None
            ),
            reuse_mode=data.get("reuse_mode", "observed_only"),
            validation=(
                ValidationReport.from_dict(validation)
                if validation is not None
                else None
            ),
            diagnostics=tuple(
                DiagnosticRef.from_dict(item) for item in data.get("diagnostics", ())
            ),
            schema_version=schema_version(data),
            metadata=mapping_value(data, "metadata"),
        )
