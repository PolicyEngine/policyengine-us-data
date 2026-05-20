"""Structured execution results for Stage 1 dataset-build commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .status import Stage1ErrorRecord, Stage1SubstepStatus


CommandExecutionStatus = Literal["completed", "failed"]


@dataclass(frozen=True, kw_only=True)
class DatasetCommandResult:
    """Result of running one Stage 1 command."""

    command_name: str
    argv: tuple[str, ...]
    status: CommandExecutionStatus
    returncode: int | None
    started_at: str
    completed_at: str
    duration_s: float
    combined_output_tail: tuple[str, ...] = ()
    error: Stage1ErrorRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible command result payload."""

        return {
            "command_name": self.command_name,
            "argv": list(self.argv),
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "combined_output_tail": list(self.combined_output_tail),
            "error": self.error.to_dict() if self.error else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class DatasetSubstepResult:
    """Result of running or skipping a Stage 1 substep."""

    substep_id: str
    title: str
    status: Stage1SubstepStatus
    started_at: str | None
    completed_at: str
    duration_s: float | None
    command_names: tuple[str, ...] = ()
    command_results: tuple[DatasetCommandResult, ...] = ()
    artifact_paths: tuple[str, ...] = ()
    error: Stage1ErrorRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible substep result payload."""

        return {
            "substep_id": self.substep_id,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "command_names": list(self.command_names),
            "command_results": [result.to_dict() for result in self.command_results],
            "artifact_paths": list(self.artifact_paths),
            "error": self.error.to_dict() if self.error else None,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "CommandExecutionStatus",
    "DatasetCommandResult",
    "DatasetSubstepResult",
]
