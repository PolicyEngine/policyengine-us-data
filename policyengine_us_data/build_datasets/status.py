"""Structured status records for Stage 1 dataset-build execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from modal_app.step_manifests.errors import PipelineErrorRecord


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
    """Structured command or substep failure details.

    Stage 1 keeps this as an in-memory adapter surface during the refactor.
    Durable pipeline status should use :class:`PipelineErrorRecord`, available
    through ``to_pipeline_error_record``.
    """

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

    def to_pipeline_error_record(
        self,
        *,
        run_id: str,
        branch: str | None = None,
        sha: str | None = None,
        version: str | None = None,
        modal_app_name: str | None = None,
        modal_environment: str | None = None,
        surface: str = "stage_1_dataset_build",
        env: Mapping[str, str] | None = None,
    ) -> "PipelineErrorRecord":
        """Adapt this Stage 1 error into the durable pipeline error schema."""

        from modal_app.step_manifests.errors import build_pipeline_error_record
        from policyengine_us_data.stage_contracts.stages import (
            STAGE_1_BUILD_DATASETS,
        )

        record = build_pipeline_error_record(
            RuntimeError(self.message),
            run_id=run_id,
            stage_id=STAGE_1_BUILD_DATASETS,
            substage_id=self.substep_id,
            surface=surface,
            traceback_text=_pipeline_traceback_text(self),
            occurred_at=self.created_at,
            env=env,
        )
        return replace(
            record,
            error_type=self.error_type,
            branch=branch,
            sha=sha,
            version=version,
            modal_app_name=modal_app_name,
            modal_environment=modal_environment,
        )


def _pipeline_traceback_text(error: Stage1ErrorRecord) -> str:
    parts: list[str] = []
    argv = error.metadata.get("argv")
    if isinstance(argv, list) and argv:
        parts.append("Command argv: " + " ".join(str(part) for part in argv))
    output_tail = error.metadata.get("output_tail")
    if isinstance(output_tail, list) and output_tail:
        parts.append("Output tail:\n" + "".join(str(line) for line in output_tail))
    if not parts:
        parts.append(error.message)
    return "\n\n".join(parts)


__all__ = [
    "Stage1ErrorRecord",
    "Stage1StatusEvent",
    "Stage1SubstepStatus",
    "utc_timestamp",
]
