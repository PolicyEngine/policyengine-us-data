"""Core dataclasses for semantic stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, get_args

CONTRACT_SCHEMA_VERSION = "1"
CONTRACT_FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"

SubstageStatus = Literal[
    "planned",
    "running",
    "completed",
    "skipped",
    "failed",
    "not_run",
]
SubstageReuseMode = Literal[
    "observed_only",
    "checkpointable",
    "reusable",
    "handoff",
]
ExecutionStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "reused",
    "partially_reused",
    "skipped",
]
ReuseDecision = Literal[
    "computed",
    "reused",
    "partially_reused",
    "invalidated",
    "failed",
    "not_applicable",
]
DiagnosticSeverity = Literal["info", "warning", "error"]
ValidationFindingStatus = Literal["pass", "warn", "fail"]
ValidationReportStatus = Literal["pass", "warn", "fail", "not_run"]

SUBSTAGE_STATUSES = frozenset(get_args(SubstageStatus))
SUBSTAGE_REUSE_MODES = frozenset(get_args(SubstageReuseMode))
EXECUTION_STATUSES = frozenset(get_args(ExecutionStatus))
REUSE_DECISIONS = frozenset(get_args(ReuseDecision))
DIAGNOSTIC_SEVERITIES = frozenset(get_args(DiagnosticSeverity))
VALIDATION_FINDING_STATUSES = frozenset(get_args(ValidationFindingStatus))
VALIDATION_REPORT_STATUSES = frozenset(get_args(ValidationReportStatus))

ContractPayload = dict[str, Any]

__all__ = [
    "CONTRACT_FINGERPRINT_ALGORITHM",
    "CONTRACT_SCHEMA_VERSION",
    "DIAGNOSTIC_SEVERITIES",
    "EXECUTION_STATUSES",
    "REUSE_DECISIONS",
    "SUBSTAGE_REUSE_MODES",
    "SUBSTAGE_STATUSES",
    "VALIDATION_FINDING_STATUSES",
    "VALIDATION_REPORT_STATUSES",
    "ArtifactRef",
    "ContractPayload",
    "DiagnosticRef",
    "DiagnosticSeverity",
    "ExecutionRecord",
    "ExecutionStatus",
    "Fingerprint",
    "ReuseDecision",
    "ReuseSummary",
    "StageContract",
    "SubstageRecord",
    "SubstageReuseMode",
    "SubstageStatus",
    "ValidationFinding",
    "ValidationFindingStatus",
    "ValidationReport",
    "ValidationReportStatus",
]


def _validate_schema_version(schema_version: str, owner: str) -> None:
    if schema_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"{owner} schema_version must be {CONTRACT_SCHEMA_VERSION!r}"
        )


def _require_non_empty(value: str | None, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _required_string(data: Mapping[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(data: Mapping[str, Any], field_name: str) -> str | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return value


def _schema_version(data: Mapping[str, Any]) -> str:
    value = data.get("schema_version", CONTRACT_SCHEMA_VERSION)
    if not isinstance(value, str) or not value:
        raise ValueError("schema_version must be a non-empty string")
    return value


def _mapping_value(data: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = data.get(field_name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _optional_mapping_value(
    data: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any] | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    return value


def _validate_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def _validate_optional_int(value: Any, field_name: str) -> None:
    if value is not None:
        _validate_int(value, field_name)


def _validate_optional_float(value: Any, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _int_value(
    data: Mapping[str, Any],
    field_name: str,
    default: int,
) -> int:
    value = data.get(field_name, default)
    _validate_int(value, field_name)
    return value


def _optional_int_value(
    data: Mapping[str, Any],
    field_name: str,
) -> int | None:
    value = data.get(field_name)
    _validate_optional_int(value, field_name)
    return value


def _optional_float_value(
    data: Mapping[str, Any],
    field_name: str,
) -> float | None:
    value = data.get(field_name)
    _validate_optional_float(value, field_name)
    return float(value) if value is not None else None


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, tuple | list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in value.items()}
    )


def _jsonable_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, tuple | list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    return value


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
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        _require_non_empty(self.logical_name, "logical_name")
        _require_non_empty(self.uri, "uri")
        _validate_optional_int(self.size_bytes, "size_bytes")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "logical_name": self.logical_name,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "schema_version": self.schema_version,
            "metadata": _jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            logical_name=_required_string(data, "logical_name"),
            uri=_required_string(data, "uri"),
            sha256=_optional_string(data, "sha256"),
            size_bytes=_optional_int_value(data, "size_bytes"),
            media_type=_optional_string(data, "media_type"),
            schema_version=_schema_version(data),
            metadata=_mapping_value(data, "metadata"),
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
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        _require_non_empty(self.name, "name")
        _require_non_empty(self.kind, "kind")
        if self.severity not in DIAGNOSTIC_SEVERITIES:
            raise ValueError(f"Invalid diagnostic severity: {self.severity!r}")
        object.__setattr__(
            self,
            "summary",
            _freeze_mapping(self.summary, "summary"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "name": self.name,
            "kind": self.kind,
            "artifact": (
                self.artifact.to_dict() if self.artifact is not None else None
            ),
            "summary": _jsonable_value(self.summary),
            "severity": self.severity,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiagnosticRef":
        artifact = data.get("artifact")
        return cls(
            name=_required_string(data, "name"),
            kind=_required_string(data, "kind"),
            artifact=(
                ArtifactRef.from_dict(artifact)
                if artifact is not None
                else None
            ),
            summary=_mapping_value(data, "summary"),
            severity=data.get("severity", "info"),
            schema_version=_schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class Fingerprint:
    """Deterministic semantic identity for stage inputs and parameters."""

    value: str
    material: Mapping[str, Any]
    algorithm: str = CONTRACT_FINGERPRINT_ALGORITHM
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        _require_non_empty(self.algorithm, "algorithm")
        _require_non_empty(self.value, "value")
        object.__setattr__(
            self,
            "material",
            _freeze_mapping(self.material, "material"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "algorithm": self.algorithm,
            "value": self.value,
            "material": _jsonable_value(self.material),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fingerprint":
        return cls(
            algorithm=(
                _required_string(data, "algorithm")
                if "algorithm" in data
                else CONTRACT_FINGERPRINT_ALGORITHM
            ),
            value=_required_string(data, "value"),
            material=_mapping_value(data, "material"),
            schema_version=_schema_version(data),
        )


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
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        _require_non_empty(self.check_id, "check_id")
        _require_non_empty(self.message, "message")
        if self.status not in VALIDATION_FINDING_STATUSES:
            raise ValueError(
                f"Invalid validation finding status: {self.status!r}"
            )
        object.__setattr__(self, "value", _freeze_value(self.value))
        object.__setattr__(self, "threshold", _freeze_value(self.threshold))
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
            "metric": self.metric,
            "value": _jsonable_value(self.value),
            "threshold": _jsonable_value(self.threshold),
            "metadata": _jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationFinding":
        return cls(
            check_id=_required_string(data, "check_id"),
            status=data["status"],
            message=_required_string(data, "message"),
            metric=_optional_string(data, "metric"),
            value=data.get("value"),
            threshold=data.get("threshold"),
            metadata=_mapping_value(data, "metadata"),
            schema_version=_schema_version(data),
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
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        if self.status not in VALIDATION_REPORT_STATUSES:
            raise ValueError(f"Invalid validation report status: {self.status!r}")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "status": self.status,
            "findings": [_jsonable_value(item) for item in self.findings],
            "diagnostics": [_jsonable_value(item) for item in self.diagnostics],
            "metadata": _jsonable_value(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        return cls(
            status=data.get("status", "not_run"),
            findings=tuple(
                ValidationFinding.from_dict(item)
                for item in data.get("findings", ())
            ),
            diagnostics=tuple(
                DiagnosticRef.from_dict(item)
                for item in data.get("diagnostics", ())
            ),
            metadata=_mapping_value(data, "metadata"),
            schema_version=_schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class ReuseSummary:
    """Numeric summary of reused and recomputed stage outputs."""

    expected_outputs: int = 0
    valid_reused_outputs: int = 0
    recomputed_outputs: int = 0
    invalid_outputs: int = 0
    saved_duration_s: float | None = None
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        for field_name in (
            "expected_outputs",
            "valid_reused_outputs",
            "recomputed_outputs",
            "invalid_outputs",
        ):
            value = getattr(self, field_name)
            _validate_int(value, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        _validate_optional_float(self.saved_duration_s, "saved_duration_s")
        if self.saved_duration_s is not None and self.saved_duration_s < 0:
            raise ValueError("saved_duration_s must be non-negative")

    def to_dict(self) -> ContractPayload:
        return {
            "expected_outputs": self.expected_outputs,
            "valid_reused_outputs": self.valid_reused_outputs,
            "recomputed_outputs": self.recomputed_outputs,
            "invalid_outputs": self.invalid_outputs,
            "saved_duration_s": self.saved_duration_s,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReuseSummary":
        return cls(
            expected_outputs=_int_value(data, "expected_outputs", 0),
            valid_reused_outputs=_int_value(data, "valid_reused_outputs", 0),
            recomputed_outputs=_int_value(data, "recomputed_outputs", 0),
            invalid_outputs=_int_value(data, "invalid_outputs", 0),
            saved_duration_s=_optional_float_value(data, "saved_duration_s"),
            schema_version=_schema_version(data),
        )


@dataclass(frozen=True, kw_only=True)
class ExecutionRecord:
    """Runtime bookkeeping for one canonical stage execution."""

    status: ExecutionStatus = "pending"
    attempt: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None
    modal_call_id: str | None = None
    reuse_decision: ReuseDecision = "not_applicable"
    reuse_reason: str | None = None
    reuse_summary: ReuseSummary = field(default_factory=ReuseSummary)
    error: Mapping[str, Any] | None = None
    schema_version: str = CONTRACT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        if self.status not in EXECUTION_STATUSES:
            raise ValueError(f"Invalid execution status: {self.status!r}")
        if self.reuse_decision not in REUSE_DECISIONS:
            raise ValueError(f"Invalid reuse decision: {self.reuse_decision!r}")
        _validate_int(self.attempt, "attempt")
        if self.attempt < 0:
            raise ValueError("attempt must be non-negative")
        _validate_optional_float(self.duration_s, "duration_s")
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        object.__setattr__(
            self,
            "error",
            (
                _freeze_mapping(self.error, "error")
                if self.error is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "status": self.status,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "modal_call_id": self.modal_call_id,
            "reuse_decision": self.reuse_decision,
            "reuse_reason": self.reuse_reason,
            "reuse_summary": self.reuse_summary.to_dict(),
            "error": _jsonable_value(self.error) if self.error else None,
            "schema_version": self.schema_version,
            "metadata": _jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionRecord":
        return cls(
            status=data.get("status", "pending"),
            attempt=_int_value(data, "attempt", 0),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            duration_s=_optional_float_value(data, "duration_s"),
            modal_call_id=data.get("modal_call_id"),
            reuse_decision=data.get("reuse_decision", "not_applicable"),
            reuse_reason=data.get("reuse_reason"),
            reuse_summary=ReuseSummary.from_dict(
                data.get("reuse_summary", {})
            ),
            error=_optional_mapping_value(data, "error"),
            schema_version=_schema_version(data),
            metadata=_mapping_value(data, "metadata"),
        )


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
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        _require_non_empty(self.substage_id, "substage_id")
        if self.status not in SUBSTAGE_STATUSES:
            raise ValueError(f"Invalid substage status: {self.status!r}")
        if self.reuse_mode not in SUBSTAGE_REUSE_MODES:
            raise ValueError(f"Invalid substage reuse mode: {self.reuse_mode!r}")
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, "parameters"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "substage_id": self.substage_id,
            "status": self.status,
            "inputs": [_jsonable_value(item) for item in self.inputs],
            "outputs": [_jsonable_value(item) for item in self.outputs],
            "parameters": _jsonable_value(self.parameters),
            "fingerprint": (
                self.fingerprint.to_dict()
                if self.fingerprint is not None
                else None
            ),
            "reuse_mode": self.reuse_mode,
            "validation": (
                self.validation.to_dict()
                if self.validation is not None
                else None
            ),
            "diagnostics": [_jsonable_value(item) for item in self.diagnostics],
            "schema_version": self.schema_version,
            "metadata": _jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubstageRecord":
        fingerprint = data.get("fingerprint")
        validation = data.get("validation")
        return cls(
            substage_id=_required_string(data, "substage_id"),
            status=data["status"],
            inputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("inputs", ())
            ),
            outputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("outputs", ())
            ),
            parameters=_mapping_value(data, "parameters"),
            fingerprint=(
                Fingerprint.from_dict(fingerprint)
                if fingerprint is not None
                else None
            ),
            reuse_mode=data.get("reuse_mode", "observed_only"),
            validation=(
                ValidationReport.from_dict(validation)
                if validation is not None
                else None
            ),
            diagnostics=tuple(
                DiagnosticRef.from_dict(item)
                for item in data.get("diagnostics", ())
            ),
            schema_version=_schema_version(data),
            metadata=_mapping_value(data, "metadata"),
        )


@dataclass(frozen=True, kw_only=True)
class StageContract:
    """Durable semantic contract at a canonical stage boundary."""

    contract_type: str
    stage_id: str
    created_at: str
    fingerprint: Fingerprint
    execution: ExecutionRecord
    run_id: str | None = None
    code_sha: str | None = None
    package_version: str | None = None
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    substages: tuple[SubstageRecord, ...] = ()
    validation: ValidationReport | None = None
    diagnostics: tuple[DiagnosticRef, ...] = ()
    schema_version: str = CONTRACT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        _require_non_empty(self.contract_type, "contract_type")
        _require_non_empty(self.stage_id, "stage_id")
        _require_non_empty(self.created_at, "created_at")
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, "parameters"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "code_sha": self.code_sha,
            "package_version": self.package_version,
            "inputs": [_jsonable_value(item) for item in self.inputs],
            "outputs": [_jsonable_value(item) for item in self.outputs],
            "parameters": _jsonable_value(self.parameters),
            "fingerprint": self.fingerprint.to_dict(),
            "substages": [_jsonable_value(item) for item in self.substages],
            "execution": self.execution.to_dict(),
            "validation": (
                self.validation.to_dict()
                if self.validation is not None
                else None
            ),
            "diagnostics": [_jsonable_value(item) for item in self.diagnostics],
            "metadata": _jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageContract":
        return cls(
            contract_type=_required_string(data, "contract_type"),
            schema_version=_schema_version(data),
            stage_id=_required_string(data, "stage_id"),
            run_id=_optional_string(data, "run_id"),
            created_at=_required_string(data, "created_at"),
            code_sha=_optional_string(data, "code_sha"),
            package_version=_optional_string(data, "package_version"),
            inputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("inputs", ())
            ),
            outputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("outputs", ())
            ),
            parameters=_mapping_value(data, "parameters"),
            fingerprint=Fingerprint.from_dict(data["fingerprint"]),
            substages=tuple(
                SubstageRecord.from_dict(item)
                for item in data.get("substages", ())
            ),
            execution=ExecutionRecord.from_dict(data["execution"]),
            validation=(
                ValidationReport.from_dict(data["validation"])
                if data.get("validation") is not None
                else None
            ),
            diagnostics=tuple(
                DiagnosticRef.from_dict(item)
                for item in data.get("diagnostics", ())
            ),
            metadata=_mapping_value(data, "metadata"),
        )
