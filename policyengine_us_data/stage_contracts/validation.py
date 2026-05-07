"""Structured validation records for stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._coercion import (
    freeze_mapping,
    freeze_sequence,
    freeze_value,
    jsonable_value,
    mapping_value,
    optional_string,
    optional_string_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_schema_version,
)
from .constants import (
    CONTRACT_SCHEMA_VERSION,
    VALIDATION_FINDING_STATUSES,
    VALIDATION_REPORT_STATUSES,
    ContractPayload,
    ValidationFindingStatus,
    ValidationReportStatus,
)
from .diagnostics import DiagnosticRef


@dataclass(frozen=True, kw_only=True)
class ValidationFinding:
    """One structured validation check outcome."""

    check_id: str
    status: ValidationFindingStatus
    message: str
    metric: str | None = None
    value: Any | None = None
    threshold: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "check_id",
            require_non_empty(self.check_id, "check_id"),
        )
        object.__setattr__(
            self,
            "message",
            require_non_empty(self.message, "message"),
        )
        object.__setattr__(
            self,
            "metric",
            optional_string_value(self.metric, "metric"),
        )
        if self.status not in VALIDATION_FINDING_STATUSES:
            raise ValueError(f"Invalid validation finding status: {self.status!r}")
        object.__setattr__(self, "value", freeze_value(self.value))
        object.__setattr__(self, "threshold", freeze_value(self.threshold))
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
            "metric": self.metric,
            "value": jsonable_value(self.value),
            "threshold": jsonable_value(self.threshold),
            "metadata": jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationFinding":
        return cls(
            check_id=required_string(data, "check_id"),
            status=data["status"],
            message=required_string(data, "message"),
            metric=optional_string(data, "metric"),
            value=data.get("value"),
            threshold=data.get("threshold"),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class ValidationReport:
    """Structured validation status with findings and diagnostic references."""

    status: ValidationReportStatus = "not_run"
    findings: tuple[ValidationFinding, ...] = ()
    diagnostics: tuple[DiagnosticRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        if self.status not in VALIDATION_REPORT_STATUSES:
            raise ValueError(f"Invalid validation report status: {self.status!r}")
        object.__setattr__(
            self,
            "findings",
            freeze_sequence(self.findings, "findings", ValidationFinding),
        )
        object.__setattr__(
            self,
            "diagnostics",
            freeze_sequence(self.diagnostics, "diagnostics", DiagnosticRef),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "status": self.status,
            "findings": [jsonable_value(item) for item in self.findings],
            "diagnostics": [jsonable_value(item) for item in self.diagnostics],
            "metadata": jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        return cls(
            status=data.get("status", "not_run"),
            findings=tuple(
                ValidationFinding.from_dict(item) for item in data.get("findings", ())
            ),
            diagnostics=tuple(
                DiagnosticRef.from_dict(item) for item in data.get("diagnostics", ())
            ),
            metadata=mapping_value(data, "metadata"),
            schema_version=schema_version(data),
        )
