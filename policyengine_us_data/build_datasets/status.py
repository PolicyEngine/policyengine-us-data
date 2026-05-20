"""Structured status records for Stage 1 dataset-build execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


Stage1SubstepStatus = Literal["started", "completed", "skipped", "failed"]


def utc_timestamp(value: datetime | None = None) -> str:
    """Render a UTC timestamp for Stage 1 execution status records."""

    value = value or datetime.now(timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, kw_only=True)
class Stage1StatusEvent:
    """A timestamped status transition for a Stage 1 substep or command."""

    substep_id: str
    status: Stage1SubstepStatus
    created_at: str
    message: str | None = None
    command_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible status event payload."""

        return {
            "substep_id": self.substep_id,
            "status": self.status,
            "created_at": self.created_at,
            "message": self.message,
            "command_name": self.command_name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class Stage1ErrorRecord:
    """Structured command or substep failure details."""

    substep_id: str | None
    command_name: str | None
    error_type: str
    message: str
    returncode: int | None = None
    created_at: str = field(default_factory=utc_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        substep_id: str | None = None,
        command_name: str | None = None,
        returncode: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Stage1ErrorRecord":
        """Build an error record from an exception without parsing logs."""

        return cls(
            substep_id=substep_id,
            command_name=command_name,
            error_type=type(exc).__name__,
            message=str(exc),
            returncode=returncode,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible error payload."""

        return {
            "substep_id": self.substep_id,
            "command_name": self.command_name,
            "error_type": self.error_type,
            "message": self.message,
            "returncode": self.returncode,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "Stage1ErrorRecord",
    "Stage1StatusEvent",
    "Stage1SubstepStatus",
    "utc_timestamp",
]
