"""Structured execution results for Stage 1 dataset-build commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetCommandResult":
        """Build a command result from a JSON-compatible payload."""

        error = _error_record_from_payload(data.get("error"))
        return cls(
            command_name=str(data["command_name"]),
            argv=_string_tuple(data.get("argv", ())),
            status=_command_execution_status(data["status"]),
            returncode=_optional_int(data.get("returncode")),
            started_at=str(data["started_at"]),
            completed_at=str(data["completed_at"]),
            duration_s=float(data["duration_s"]),
            combined_output_tail=_string_tuple(data.get("combined_output_tail", ())),
            error=error,
            metadata=_metadata_mapping(data.get("metadata", {})),
        )

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
    reuse_decision: Mapping[str, Any] | None = None
    checkpoint_decisions: tuple[Mapping[str, Any], ...] = ()
    error: Stage1ErrorRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetSubstepResult":
        """Build a substep result from a JSON-compatible payload."""

        command_results = data.get("command_results", ())
        if not isinstance(command_results, Sequence) or isinstance(
            command_results, str
        ):
            raise TypeError("command_results must be a sequence")
        return cls(
            substep_id=str(data["substep_id"]),
            title=str(data["title"]),
            status=_stage_1_substep_status(data["status"]),
            started_at=_optional_str(data.get("started_at")),
            completed_at=str(data["completed_at"]),
            duration_s=_optional_float(data.get("duration_s")),
            command_names=_string_tuple(data.get("command_names", ())),
            command_results=tuple(
                DatasetCommandResult.from_dict(_mapping_payload(result))
                for result in command_results
            ),
            artifact_paths=_string_tuple(data.get("artifact_paths", ())),
            reuse_decision=_optional_mapping(data.get("reuse_decision")),
            checkpoint_decisions=_mapping_tuple(data.get("checkpoint_decisions", ())),
            error=_error_record_from_payload(data.get("error")),
            metadata=_metadata_mapping(data.get("metadata", {})),
        )

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
            "reuse_decision": (
                dict(self.reuse_decision) if self.reuse_decision is not None else None
            ),
            "checkpoint_decisions": [
                dict(decision) for decision in self.checkpoint_decisions
            ],
            "error": self.error.to_dict() if self.error else None,
            "metadata": dict(self.metadata),
        }


def _command_execution_status(value: Any) -> CommandExecutionStatus:
    if value in ("completed", "failed"):
        return cast(CommandExecutionStatus, value)
    raise ValueError(f"Invalid command execution status: {value!r}")


def _stage_1_substep_status(value: Any) -> Stage1SubstepStatus:
    if value in ("started", "completed", "reused", "skipped", "failed"):
        return cast(Stage1SubstepStatus, value)
    raise ValueError(f"Invalid Stage 1 substep status: {value!r}")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError("Expected a sequence")
    return tuple(str(item) for item in value)


def _mapping_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return dict(_mapping_payload(value))


def _mapping_tuple(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError("Expected a sequence")
    return tuple(dict(_mapping_payload(item)) for item in value)


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return dict(value)


def _error_record_from_payload(value: Any) -> Stage1ErrorRecord | None:
    if value is None:
        return None
    return Stage1ErrorRecord.from_dict(_mapping_payload(value))


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "CommandExecutionStatus",
    "DatasetCommandResult",
    "DatasetSubstepResult",
]
