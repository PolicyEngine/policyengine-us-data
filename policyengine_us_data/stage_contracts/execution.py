"""Execution and reuse records for stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._coercion import (
    freeze_mapping,
    int_value,
    jsonable_value,
    mapping_value,
    optional_float_value,
    optional_mapping_value,
    optional_string_value,
    schema_version,
    validate_int,
    validate_optional_float,
    validate_schema_version,
)
from .constants import (
    CONTRACT_SCHEMA_VERSION,
    EXECUTION_STATUSES,
    REUSE_DECISIONS,
    ContractPayload,
    ExecutionStatus,
    ReuseDecision,
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
        validate_schema_version(self.schema_version, self.__class__.__name__)
        for field_name in (
            "expected_outputs",
            "valid_reused_outputs",
            "recomputed_outputs",
            "invalid_outputs",
        ):
            value = getattr(self, field_name)
            validate_int(value, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        validate_optional_float(self.saved_duration_s, "saved_duration_s")
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
            expected_outputs=int_value(data, "expected_outputs", 0),
            valid_reused_outputs=int_value(data, "valid_reused_outputs", 0),
            recomputed_outputs=int_value(data, "recomputed_outputs", 0),
            invalid_outputs=int_value(data, "invalid_outputs", 0),
            saved_duration_s=optional_float_value(data, "saved_duration_s"),
            schema_version=schema_version(data),
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
        validate_schema_version(self.schema_version, self.__class__.__name__)
        if self.status not in EXECUTION_STATUSES:
            raise ValueError(f"Invalid execution status: {self.status!r}")
        if self.reuse_decision not in REUSE_DECISIONS:
            raise ValueError(f"Invalid reuse decision: {self.reuse_decision!r}")
        object.__setattr__(
            self,
            "started_at",
            optional_string_value(self.started_at, "started_at"),
        )
        object.__setattr__(
            self,
            "completed_at",
            optional_string_value(self.completed_at, "completed_at"),
        )
        object.__setattr__(
            self,
            "modal_call_id",
            optional_string_value(self.modal_call_id, "modal_call_id"),
        )
        object.__setattr__(
            self,
            "reuse_reason",
            optional_string_value(self.reuse_reason, "reuse_reason"),
        )
        if not isinstance(self.reuse_summary, ReuseSummary):
            raise ValueError("reuse_summary must be ReuseSummary")
        validate_int(self.attempt, "attempt")
        if self.attempt < 0:
            raise ValueError("attempt must be non-negative")
        validate_optional_float(self.duration_s, "duration_s")
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        object.__setattr__(
            self,
            "error",
            freeze_mapping(self.error, "error") if self.error is not None else None,
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "metadata"),
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
            "error": jsonable_value(self.error) if self.error else None,
            "schema_version": self.schema_version,
            "metadata": jsonable_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionRecord":
        return cls(
            status=data.get("status", "pending"),
            attempt=int_value(data, "attempt", 0),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            duration_s=optional_float_value(data, "duration_s"),
            modal_call_id=data.get("modal_call_id"),
            reuse_decision=data.get("reuse_decision", "not_applicable"),
            reuse_reason=data.get("reuse_reason"),
            reuse_summary=ReuseSummary.from_dict(data.get("reuse_summary", {})),
            error=optional_mapping_value(data, "error"),
            schema_version=schema_version(data),
            metadata=mapping_value(data, "metadata"),
        )
