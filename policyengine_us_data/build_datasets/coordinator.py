"""Substep coordination for Stage 1 dataset builds."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from policyengine_us_data.stage_contracts import ValidationReport

from .artifacts import stage_1_artifact_specs
from .results import DatasetSubstepResult
from .specs import STAGE_1_BUILD_STEP_SPECS
from .status import Stage1ErrorRecord, Stage1StatusEvent, utc_timestamp


class Stage1SubstepRunner(Protocol):
    """Callable runner for one Stage 1 substep."""

    substep_id: str
    title: str

    def run(self) -> Any:
        """Run the substep action."""


class Stage1ValidationAdapter(Protocol):
    """Adapter that validates a completed Stage 1 substep result."""

    def run_for_substep_result(
        self,
        result: DatasetSubstepResult,
    ) -> ValidationReport:
        """Run validation for one substep result."""

    def should_stop(self, report: ValidationReport) -> bool:
        """Return whether validation failure should stop downstream work."""


@dataclass(frozen=True, kw_only=True)
class CommandBackedSubstepRunner:
    """Run a Stage 1 substep backed by existing side-effecting commands."""

    substep_id: str
    title: str
    action: Callable[[], Any]

    def run(self) -> Any:
        """Run the wrapped substep action."""

        return self.action()


@dataclass
class Stage1Coordinator:
    """Collect Stage 1 substep status events, errors, and results."""

    validation_runner: Stage1ValidationAdapter | None = None
    results: list[DatasetSubstepResult] = field(default_factory=list)
    status_events: list[Stage1StatusEvent] = field(default_factory=list)
    error_records: list[Stage1ErrorRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def run_substep(
        self,
        substep_id: str,
        title: str | None,
        action: Callable[[], Any],
        *,
        command_names: Sequence[str] = (),
        artifact_paths: Sequence[str | Path] = (),
        reuse_decision: Mapping[str, Any] | None = None,
        checkpoint_decisions: Sequence[Mapping[str, Any]] = (),
        skip: bool = False,
        skip_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run one declared substep and record structured status."""

        runner = CommandBackedSubstepRunner(
            substep_id=substep_id,
            title=title or stage_1_substep_title(substep_id),
            action=action,
        )
        if skip:
            result = self._skipped_result(
                runner=runner,
                command_names=command_names,
                reuse_decision=reuse_decision,
                checkpoint_decisions=checkpoint_decisions,
                skip_reason=skip_reason,
                metadata=metadata,
            )
            self._record(result)
            return None

        started_dt = datetime.now(timezone.utc)
        started_at = utc_timestamp(started_dt)
        self._record_event(
            Stage1StatusEvent(
                substep_id=substep_id,
                status="started",
                created_at=started_at,
                message=f"Started {runner.title}",
                metadata=dict(metadata or {}),
            )
        )
        try:
            value = runner.run()
        except Exception as exc:
            error = Stage1ErrorRecord.from_exception(
                exc,
                substep_id=substep_id,
                command_name=command_names[0] if command_names else None,
                metadata=dict(metadata or {}),
            )
            result = self._result(
                runner=runner,
                status="failed",
                started_dt=started_dt,
                command_names=command_names,
                artifact_paths=artifact_paths,
                reuse_decision=reuse_decision,
                checkpoint_decisions=checkpoint_decisions,
                error=error,
                metadata=metadata,
            )
            self._record(result)
            raise

        result = self._result(
            runner=runner,
            status="completed",
            started_dt=started_dt,
            command_names=command_names,
            artifact_paths=artifact_paths,
            reuse_decision=reuse_decision,
            checkpoint_decisions=checkpoint_decisions,
            metadata=metadata,
        )
        result = self._validated_result(result)
        self._record(result)
        return value

    def _skipped_result(
        self,
        *,
        runner: CommandBackedSubstepRunner,
        command_names: Sequence[str],
        reuse_decision: Mapping[str, Any] | None,
        checkpoint_decisions: Sequence[Mapping[str, Any]],
        skip_reason: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> DatasetSubstepResult:
        completed_at = utc_timestamp()
        return DatasetSubstepResult(
            substep_id=runner.substep_id,
            title=runner.title,
            status="skipped",
            started_at=None,
            completed_at=completed_at,
            duration_s=None,
            command_names=tuple(command_names),
            reuse_decision=reuse_decision,
            checkpoint_decisions=tuple(checkpoint_decisions),
            metadata={**dict(metadata or {}), "skip_reason": skip_reason},
        )

    def _result(
        self,
        *,
        runner: CommandBackedSubstepRunner,
        status: str,
        started_dt: datetime,
        command_names: Sequence[str],
        artifact_paths: Sequence[str | Path],
        reuse_decision: Mapping[str, Any] | None,
        checkpoint_decisions: Sequence[Mapping[str, Any]],
        error: Stage1ErrorRecord | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetSubstepResult:
        completed_dt = datetime.now(timezone.utc)
        return DatasetSubstepResult(
            substep_id=runner.substep_id,
            title=runner.title,
            status=status,
            started_at=utc_timestamp(started_dt),
            completed_at=utc_timestamp(completed_dt),
            duration_s=(completed_dt - started_dt).total_seconds(),
            command_names=tuple(command_names),
            artifact_paths=_artifact_paths(artifact_paths),
            reuse_decision=reuse_decision,
            checkpoint_decisions=tuple(checkpoint_decisions),
            error=error,
            metadata=dict(metadata or {}),
        )

    def _validated_result(self, result: DatasetSubstepResult) -> DatasetSubstepResult:
        if self.validation_runner is None:
            return result

        report = self.validation_runner.run_for_substep_result(result)
        result = replace(result, validation_report=report.to_dict())
        if self.validation_runner.should_stop(report):
            exc = RuntimeError(f"Stage 1 validation failed for {result.substep_id}")
            error = Stage1ErrorRecord.from_exception(
                exc,
                substep_id=result.substep_id,
                command_name=result.command_names[0] if result.command_names else None,
                metadata={"validation_report": report.to_dict()},
            )
            return replace(result, status="failed", error=error)
        return result

    def _record(self, result: DatasetSubstepResult) -> None:
        with self._lock:
            self.results.append(result)
            self.status_events.append(
                Stage1StatusEvent(
                    substep_id=result.substep_id,
                    status=result.status,
                    created_at=result.completed_at,
                    message=f"{result.title}: {result.status}",
                    metadata={
                        **dict(result.metadata),
                        "reuse_decision": result.reuse_decision,
                        "checkpoint_decisions": [
                            dict(decision) for decision in result.checkpoint_decisions
                        ],
                        "validation_report": (
                            dict(result.validation_report)
                            if result.validation_report is not None
                            else None
                        ),
                    },
                )
            )
            if result.error is not None:
                self.error_records.append(result.error)
                if result.validation_report is not None:
                    raise RuntimeError(
                        f"Stage 1 validation failed for {result.substep_id}"
                    )

    def _record_event(self, event: Stage1StatusEvent) -> None:
        with self._lock:
            self.status_events.append(event)


def stage_1_substep_id_for_script(script_path: str) -> str:
    """Return the Stage 1 substep id associated with a script path."""

    for spec in stage_1_artifact_specs():
        if spec.script_path == script_path:
            return spec.substage_id
    return "1g_stage_base_datasets"


def stage_1_substep_title(substep_id: str) -> str:
    """Return the configured Stage 1 title for a substep id."""

    for spec in STAGE_1_BUILD_STEP_SPECS:
        if spec.id == substep_id:
            return spec.title
    return substep_id


def _artifact_paths(paths: Sequence[str | Path]) -> tuple[str, ...]:
    return tuple(str(Path(path)) for path in paths)


__all__ = [
    "CommandBackedSubstepRunner",
    "Stage1Coordinator",
    "Stage1SubstepRunner",
    "Stage1ValidationAdapter",
    "stage_1_substep_id_for_script",
    "stage_1_substep_title",
]
