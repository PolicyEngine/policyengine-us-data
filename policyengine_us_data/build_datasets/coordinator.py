"""Substep coordination for Stage 1 dataset builds."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .artifacts import stage_1_artifact_specs
from .commands import DatasetCommandError
from .results import DatasetCommandResult, DatasetSubstepResult
from .specs import STAGE_1_BUILD_STEP_SPECS
from .status import Stage1ErrorRecord, Stage1StatusEvent, utc_timestamp


class Stage1SubstepRunner(Protocol):
    """Callable runner for one Stage 1 substep."""

    substep_id: str
    title: str

    def run(self) -> Any:
        """Run the substep action."""


class Stage1StatusSink(Protocol):
    """Persistence sink for Stage 1 status transitions and results."""

    def record_event(self, event: Stage1StatusEvent) -> None:
        """Persist one status event."""

    def record_result(self, result: DatasetSubstepResult) -> None:
        """Persist one substep result."""


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
class _SubstepAggregate:
    substep_id: str
    title: str
    started_dt: datetime | None = None
    completed_dt: datetime | None = None
    command_names: list[str] = field(default_factory=list)
    command_results: list[DatasetCommandResult] = field(default_factory=list)
    artifact_paths: list[str | Path] = field(default_factory=list)
    reuse_decisions: list[Mapping[str, Any]] = field(default_factory=list)
    checkpoint_decisions: list[Mapping[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    skip_reasons: list[str] = field(default_factory=list)
    skipped: bool = False
    error: Stage1ErrorRecord | None = None
    finalized: bool = False


@dataclass
class Stage1Coordinator:
    """Collect Stage 1 substep status events, errors, and results."""

    status_recorder: Stage1StatusSink | None = None
    results: list[DatasetSubstepResult] = field(default_factory=list)
    status_events: list[Stage1StatusEvent] = field(default_factory=list)
    error_records: list[Stage1ErrorRecord] = field(default_factory=list)
    _substep_aggregates: dict[str, _SubstepAggregate] = field(
        default_factory=dict,
        repr=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def run_substep(
        self,
        substep_id: str,
        title: str | None,
        action: Callable[[], Any],
        *,
        command_names: Sequence[str] = (),
        command_results: Sequence[DatasetCommandResult] = (),
        artifact_paths: Sequence[str | Path] = (),
        reuse_decision: Mapping[str, Any] | None = None,
        checkpoint_decisions: Sequence[Mapping[str, Any]] = (),
        skip: bool = False,
        skip_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        aggregate: bool = False,
    ) -> Any:
        """Run one declared substep and record structured status."""

        runner = CommandBackedSubstepRunner(
            substep_id=substep_id,
            title=title or stage_1_substep_title(substep_id),
            action=action,
        )
        if aggregate:
            return self._run_aggregated_substep(
                runner,
                command_names=command_names,
                command_results=command_results,
                artifact_paths=artifact_paths,
                reuse_decision=reuse_decision,
                checkpoint_decisions=checkpoint_decisions,
                skip=skip,
                skip_reason=skip_reason,
                metadata=metadata,
            )
        if skip:
            result = self._skipped_result(
                runner=runner,
                command_names=command_names,
                command_results=command_results,
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
            captured_command_results = _command_results_with_exception(
                command_results,
                exc,
            )
            error = _error_record_from_exception(
                exc,
                substep_id=substep_id,
                command_name=command_names[0] if command_names else None,
                command_results=captured_command_results,
                metadata=dict(metadata or {}),
            )
            result = self._result(
                runner=runner,
                status="failed",
                started_dt=started_dt,
                command_names=command_names,
                command_results=captured_command_results,
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
            status=_successful_status(
                reuse_decision=reuse_decision,
                checkpoint_decisions=checkpoint_decisions,
                command_results=command_results,
            ),
            started_dt=started_dt,
            command_names=command_names,
            command_results=tuple(command_results),
            artifact_paths=artifact_paths,
            reuse_decision=reuse_decision,
            checkpoint_decisions=checkpoint_decisions,
            metadata=metadata,
        )
        self._record(result)
        return value

    def finalize_results(self) -> tuple[DatasetSubstepResult, ...]:
        """Record final results for aggregated canonical substeps."""

        with self._lock:
            pending = tuple(
                state
                for state in self._substep_aggregates.values()
                if not state.finalized
            )
            for state in pending:
                state.finalized = True

        for state in pending:
            self._record(self._result_from_aggregate(state))
        return tuple(self.results)

    def _run_aggregated_substep(
        self,
        runner: CommandBackedSubstepRunner,
        *,
        command_names: Sequence[str],
        command_results: Sequence[DatasetCommandResult],
        artifact_paths: Sequence[str | Path],
        reuse_decision: Mapping[str, Any] | None,
        checkpoint_decisions: Sequence[Mapping[str, Any]],
        skip: bool,
        skip_reason: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> Any:
        if skip:
            self._record_aggregate_skip(
                runner=runner,
                command_names=command_names,
                command_results=command_results,
                reuse_decision=reuse_decision,
                checkpoint_decisions=checkpoint_decisions,
                skip_reason=skip_reason,
                metadata=metadata,
            )
            return None

        started_dt = datetime.now(timezone.utc)
        self._record_aggregate_start(runner, started_dt, metadata=metadata)
        try:
            value = runner.run()
        except Exception as exc:
            completed_dt = datetime.now(timezone.utc)
            captured_command_results = _command_results_with_exception(
                command_results,
                exc,
            )
            error = _error_record_from_exception(
                exc,
                substep_id=runner.substep_id,
                command_name=command_names[0] if command_names else None,
                command_results=captured_command_results,
                metadata=dict(metadata or {}),
            )
            self._finish_aggregate_failure(
                runner=runner,
                completed_dt=completed_dt,
                command_names=command_names,
                command_results=captured_command_results,
                artifact_paths=artifact_paths,
                reuse_decision=reuse_decision,
                checkpoint_decisions=checkpoint_decisions,
                error=error,
                metadata=metadata,
            )
            raise

        completed_dt = datetime.now(timezone.utc)
        self._update_aggregate_success(
            runner=runner,
            completed_dt=completed_dt,
            command_names=command_names,
            command_results=tuple(command_results),
            artifact_paths=artifact_paths,
            reuse_decision=reuse_decision,
            checkpoint_decisions=checkpoint_decisions,
            metadata=metadata,
        )
        return value

    def _record_aggregate_start(
        self,
        runner: CommandBackedSubstepRunner,
        started_dt: datetime,
        *,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        event: Stage1StatusEvent | None = None
        with self._lock:
            state = self._aggregate_state(runner)
            if state.started_dt is None:
                state.started_dt = started_dt
                event = Stage1StatusEvent(
                    substep_id=runner.substep_id,
                    status="started",
                    created_at=utc_timestamp(started_dt),
                    message=f"Started {runner.title}",
                    metadata=dict(metadata or {}),
                )
                self.status_events.append(event)
            elif started_dt < state.started_dt:
                state.started_dt = started_dt
            state.metadata.update(dict(metadata or {}))
        if event is not None and self.status_recorder is not None:
            self.status_recorder.record_event(event)

    def _record_aggregate_skip(
        self,
        *,
        runner: CommandBackedSubstepRunner,
        command_names: Sequence[str],
        command_results: Sequence[DatasetCommandResult],
        reuse_decision: Mapping[str, Any] | None,
        checkpoint_decisions: Sequence[Mapping[str, Any]],
        skip_reason: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        with self._lock:
            state = self._aggregate_state(runner)
            _extend_unique(state.command_names, command_names)
            state.command_results.extend(command_results)
            _append_reuse_decision(state, reuse_decision)
            state.checkpoint_decisions.extend(checkpoint_decisions)
            state.metadata.update(dict(metadata or {}))
            state.skipped = True
            if skip_reason is not None and skip_reason not in state.skip_reasons:
                state.skip_reasons.append(skip_reason)

    def _update_aggregate_success(
        self,
        *,
        runner: CommandBackedSubstepRunner,
        completed_dt: datetime,
        command_names: Sequence[str],
        command_results: Sequence[DatasetCommandResult],
        artifact_paths: Sequence[str | Path],
        reuse_decision: Mapping[str, Any] | None,
        checkpoint_decisions: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        with self._lock:
            state = self._aggregate_state(runner)
            if state.completed_dt is None or completed_dt > state.completed_dt:
                state.completed_dt = completed_dt
            _extend_unique(state.command_names, command_names)
            state.command_results.extend(command_results)
            state.artifact_paths.extend(artifact_paths)
            _append_reuse_decision(state, reuse_decision)
            state.checkpoint_decisions.extend(checkpoint_decisions)
            state.metadata.update(dict(metadata or {}))

    def _finish_aggregate_failure(
        self,
        *,
        runner: CommandBackedSubstepRunner,
        completed_dt: datetime,
        command_names: Sequence[str],
        command_results: Sequence[DatasetCommandResult],
        artifact_paths: Sequence[str | Path],
        reuse_decision: Mapping[str, Any] | None,
        checkpoint_decisions: Sequence[Mapping[str, Any]],
        error: Stage1ErrorRecord,
        metadata: Mapping[str, Any] | None,
    ) -> _SubstepAggregate:
        with self._lock:
            state = self._aggregate_state(runner)
            if state.completed_dt is None or completed_dt > state.completed_dt:
                state.completed_dt = completed_dt
            _extend_unique(state.command_names, command_names)
            state.command_results.extend(command_results)
            state.artifact_paths.extend(artifact_paths)
            _append_reuse_decision(state, reuse_decision)
            state.checkpoint_decisions.extend(checkpoint_decisions)
            state.metadata.update(dict(metadata or {}))
            state.error = error
            return state

    def _aggregate_state(
        self,
        runner: CommandBackedSubstepRunner,
    ) -> _SubstepAggregate:
        state = self._substep_aggregates.get(runner.substep_id)
        if state is None:
            state = _SubstepAggregate(
                substep_id=runner.substep_id,
                title=runner.title,
            )
            self._substep_aggregates[runner.substep_id] = state
        return state

    def _result_from_aggregate(
        self,
        state: _SubstepAggregate,
    ) -> DatasetSubstepResult:
        completed_dt = state.completed_dt or datetime.now(timezone.utc)
        if state.error is not None:
            status = "failed"
        elif _aggregate_was_reused(state):
            status = "reused"
        elif state.started_dt is None and state.skipped:
            status = "skipped"
        else:
            status = "completed"
        metadata = dict(state.metadata)
        if state.skip_reasons:
            metadata["skip_reasons"] = list(state.skip_reasons)
            if len(state.skip_reasons) == 1:
                metadata["skip_reason"] = state.skip_reasons[0]
        if state.reuse_decisions:
            metadata["reuse_decisions"] = list(state.reuse_decisions)
        if state.checkpoint_decisions:
            metadata["checkpoint_decisions"] = list(state.checkpoint_decisions)
        duration_s = (
            (completed_dt - state.started_dt).total_seconds()
            if state.started_dt is not None
            else None
        )
        return DatasetSubstepResult(
            substep_id=state.substep_id,
            title=state.title,
            status=status,
            started_at=utc_timestamp(state.started_dt) if state.started_dt else None,
            completed_at=utc_timestamp(completed_dt),
            duration_s=duration_s,
            command_names=tuple(state.command_names),
            command_results=tuple(state.command_results),
            artifact_paths=_existing_artifact_paths(state.artifact_paths),
            reuse_decision=_aggregate_reuse_decision(state),
            checkpoint_decisions=tuple(state.checkpoint_decisions),
            error=state.error,
            metadata=metadata,
        )

    def _skipped_result(
        self,
        *,
        runner: CommandBackedSubstepRunner,
        command_names: Sequence[str],
        command_results: Sequence[DatasetCommandResult],
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
            command_results=tuple(command_results),
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
        command_results: Sequence[DatasetCommandResult],
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
            command_results=tuple(command_results),
            artifact_paths=_existing_artifact_paths(artifact_paths),
            reuse_decision=reuse_decision,
            checkpoint_decisions=tuple(checkpoint_decisions),
            error=error,
            metadata=dict(metadata or {}),
        )

    def _record(self, result: DatasetSubstepResult) -> None:
        event = Stage1StatusEvent(
            substep_id=result.substep_id,
            status=result.status,
            created_at=result.completed_at,
            message=f"{result.title}: {result.status}",
            metadata=_result_event_metadata(result),
        )
        with self._lock:
            self.results.append(result)
            self.status_events.append(event)
            if result.error is not None:
                self.error_records.append(result.error)
        if self.status_recorder is not None:
            self.status_recorder.record_result(result)
            self.status_recorder.record_event(event)

    def _record_event(self, event: Stage1StatusEvent) -> None:
        with self._lock:
            self.status_events.append(event)
        if self.status_recorder is not None:
            self.status_recorder.record_event(event)


def stage_1_substep_id_for_script(script_path: str) -> str:
    """Return the Stage 1 substep id associated with a script path."""

    for spec in stage_1_artifact_specs():
        if spec.script_path == script_path:
            return spec.substage_id
    raise ValueError(f"No Stage 1 substep mapping declared for script: {script_path}")


def stage_1_substep_title(substep_id: str) -> str:
    """Return the configured Stage 1 title for a substep id."""

    for spec in STAGE_1_BUILD_STEP_SPECS:
        if spec.id == substep_id:
            return spec.title
    return substep_id


def _existing_artifact_paths(paths: Sequence[str | Path]) -> tuple[str, ...]:
    return tuple(str(Path(path)) for path in paths if Path(path).exists())


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _append_reuse_decision(
    state: _SubstepAggregate,
    reuse_decision: Mapping[str, Any] | None,
) -> None:
    if reuse_decision is None:
        return
    decision = dict(reuse_decision)
    if decision not in state.reuse_decisions:
        state.reuse_decisions.append(decision)


def _result_event_metadata(result: DatasetSubstepResult) -> dict[str, Any]:
    metadata = dict(result.metadata)
    if result.reuse_decision is not None:
        metadata["reuse_decision"] = dict(result.reuse_decision)
    if result.checkpoint_decisions:
        metadata["checkpoint_decisions"] = [
            dict(decision) for decision in result.checkpoint_decisions
        ]
    return metadata


def _aggregate_reuse_decision(
    state: _SubstepAggregate,
) -> Mapping[str, Any] | None:
    if not state.reuse_decisions:
        return None
    if len(state.reuse_decisions) == 1:
        return state.reuse_decisions[0]
    return {
        "substep_id": state.substep_id,
        "decisions": list(state.reuse_decisions),
    }


def _successful_status(
    *,
    reuse_decision: Mapping[str, Any] | None,
    checkpoint_decisions: Sequence[Mapping[str, Any]],
    command_results: Sequence[DatasetCommandResult],
) -> str:
    if command_results:
        return "completed"
    if _reuse_action(reuse_decision) != "reuse":
        return "completed"
    if not checkpoint_decisions:
        return "completed"
    if all(
        dict(decision).get("action") == "reuse" for decision in checkpoint_decisions
    ):
        return "reused"
    return "completed"


def _aggregate_was_reused(state: _SubstepAggregate) -> bool:
    return (
        _successful_status(
            reuse_decision=_aggregate_reuse_decision(state),
            checkpoint_decisions=state.checkpoint_decisions,
            command_results=state.command_results,
        )
        == "reused"
    )


def _reuse_action(reuse_decision: Mapping[str, Any] | None) -> object:
    if reuse_decision is None:
        return None
    decision = dict(reuse_decision)
    if "action" in decision:
        return decision["action"]
    nested = decision.get("decisions")
    if isinstance(nested, list) and nested:
        actions = {
            item.get("action")
            for item in nested
            if isinstance(item, Mapping) and "action" in item
        }
        if actions == {"reuse"}:
            return "reuse"
    return None


def _command_results_with_exception(
    command_results: Sequence[DatasetCommandResult],
    exc: Exception,
) -> tuple[DatasetCommandResult, ...]:
    captured = list(command_results)
    if isinstance(exc, DatasetCommandError) and not any(
        result is exc.result for result in captured
    ):
        captured.append(exc.result)
    return tuple(captured)


def _error_record_from_exception(
    exc: Exception,
    *,
    substep_id: str,
    command_name: str | None,
    command_results: Sequence[DatasetCommandResult],
    metadata: Mapping[str, Any],
) -> Stage1ErrorRecord:
    if isinstance(exc, DatasetCommandError) and exc.result.error is not None:
        command_error = exc.result.error
        return replace(
            command_error,
            substep_id=substep_id,
            metadata={
                **dict(command_error.metadata),
                **dict(metadata),
            },
        )
    result_with_error = next(
        (result for result in reversed(command_results) if result.error is not None),
        None,
    )
    if result_with_error is not None:
        return replace(
            result_with_error.error,
            substep_id=substep_id,
            metadata={
                **dict(result_with_error.error.metadata),
                **dict(metadata),
            },
        )
    return Stage1ErrorRecord.from_exception(
        exc,
        substep_id=substep_id,
        command_name=command_name,
        metadata=dict(metadata),
    )


__all__ = [
    "CommandBackedSubstepRunner",
    "Stage1Coordinator",
    "Stage1StatusSink",
    "Stage1SubstepRunner",
    "stage_1_substep_id_for_script",
    "stage_1_substep_title",
]
