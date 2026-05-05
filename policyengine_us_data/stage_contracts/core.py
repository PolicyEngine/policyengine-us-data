"""Core dataclasses for semantic stage contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, get_args

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

SUBSTAGE_STATUSES = frozenset(get_args(SubstageStatus))
SUBSTAGE_REUSE_MODES = frozenset(get_args(SubstageReuseMode))
EXECUTION_STATUSES = frozenset(get_args(ExecutionStatus))
REUSE_DECISIONS = frozenset(get_args(ReuseDecision))

ContractPayload = dict[str, Any]

__all__ = [
    "CONTRACT_FINGERPRINT_ALGORITHM",
    "CONTRACT_SCHEMA_VERSION",
    "EXECUTION_STATUSES",
    "REUSE_DECISIONS",
    "SUBSTAGE_REUSE_MODES",
    "SUBSTAGE_STATUSES",
    "ArtifactRef",
    "ContractPayload",
    "ExecutionRecord",
    "ExecutionStatus",
    "Fingerprint",
    "ReuseDecision",
    "ReuseSummary",
    "StageContract",
    "SubstageRecord",
    "SubstageReuseMode",
    "SubstageStatus",
]


def _validate_schema_version(schema_version: str, owner: str) -> None:
    if schema_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"{owner} schema_version must be {CONTRACT_SCHEMA_VERSION!r}"
        )


def _require_non_empty(value: str | None, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


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
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

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
            logical_name=str(data["logical_name"]),
            uri=str(data["uri"]),
            sha256=data.get("sha256"),
            size_bytes=data.get("size_bytes"),
            media_type=data.get("media_type"),
            schema_version=str(
                data.get("schema_version", CONTRACT_SCHEMA_VERSION)
            ),
            metadata=dict(data.get("metadata", {})),
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
            algorithm=str(
                data.get("algorithm", CONTRACT_FINGERPRINT_ALGORITHM)
            ),
            value=str(data["value"]),
            material=dict(data.get("material", {})),
            schema_version=str(
                data.get("schema_version", CONTRACT_SCHEMA_VERSION)
            ),
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
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
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
            expected_outputs=int(data.get("expected_outputs", 0)),
            valid_reused_outputs=int(data.get("valid_reused_outputs", 0)),
            recomputed_outputs=int(data.get("recomputed_outputs", 0)),
            invalid_outputs=int(data.get("invalid_outputs", 0)),
            saved_duration_s=data.get("saved_duration_s"),
            schema_version=str(
                data.get("schema_version", CONTRACT_SCHEMA_VERSION)
            ),
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
        if self.attempt < 0:
            raise ValueError("attempt must be non-negative")
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")

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
            attempt=int(data.get("attempt", 0)),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            duration_s=data.get("duration_s"),
            modal_call_id=data.get("modal_call_id"),
            reuse_decision=data.get("reuse_decision", "not_applicable"),
            reuse_reason=data.get("reuse_reason"),
            reuse_summary=ReuseSummary.from_dict(
                data.get("reuse_summary", {})
            ),
            error=data.get("error"),
            schema_version=str(
                data.get("schema_version", CONTRACT_SCHEMA_VERSION)
            ),
            metadata=dict(data.get("metadata", {})),
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
    schema_version: str = CONTRACT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        _require_non_empty(self.substage_id, "substage_id")
        if self.status not in SUBSTAGE_STATUSES:
            raise ValueError(f"Invalid substage status: {self.status!r}")
        if self.reuse_mode not in SUBSTAGE_REUSE_MODES:
            raise ValueError(f"Invalid substage reuse mode: {self.reuse_mode!r}")

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
            "schema_version": self.schema_version,
            "metadata": _jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubstageRecord":
        fingerprint = data.get("fingerprint")
        return cls(
            substage_id=str(data["substage_id"]),
            status=data["status"],
            inputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("inputs", ())
            ),
            outputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("outputs", ())
            ),
            parameters=dict(data.get("parameters", {})),
            fingerprint=(
                Fingerprint.from_dict(fingerprint)
                if fingerprint is not None
                else None
            ),
            reuse_mode=data.get("reuse_mode", "observed_only"),
            schema_version=str(
                data.get("schema_version", CONTRACT_SCHEMA_VERSION)
            ),
            metadata=dict(data.get("metadata", {})),
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
    schema_version: str = CONTRACT_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, self.__class__.__name__)
        _require_non_empty(self.contract_type, "contract_type")
        _require_non_empty(self.stage_id, "stage_id")
        _require_non_empty(self.created_at, "created_at")

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
            "metadata": _jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageContract":
        return cls(
            contract_type=str(data["contract_type"]),
            schema_version=str(
                data.get("schema_version", CONTRACT_SCHEMA_VERSION)
            ),
            stage_id=str(data["stage_id"]),
            run_id=data.get("run_id"),
            created_at=str(data["created_at"]),
            code_sha=data.get("code_sha"),
            package_version=data.get("package_version"),
            inputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("inputs", ())
            ),
            outputs=tuple(
                ArtifactRef.from_dict(item) for item in data.get("outputs", ())
            ),
            parameters=dict(data.get("parameters", {})),
            fingerprint=Fingerprint.from_dict(data["fingerprint"]),
            substages=tuple(
                SubstageRecord.from_dict(item)
                for item in data.get("substages", ())
            ),
            execution=ExecutionRecord.from_dict(data["execution"]),
            metadata=dict(data.get("metadata", {})),
        )
