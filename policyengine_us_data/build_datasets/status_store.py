"""Durable Stage 1 status storage for pipeline status readers."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .results import DatasetSubstepResult
from .specs import STAGE_1_BUILD_STEP_SPECS
from .status import Stage1StatusEvent, Stage1SubstepStatus


STAGE_1_STATUS_DIRNAME = "stage_1"
STAGE_1_STATUS_EVENTS_FILENAME = "status_events.jsonl"
STAGE_1_SUBSTEP_RESULTS_FILENAME = "substep_results.jsonl"
STAGE_1_CURRENT_SUBSTEP_FILENAME = "current_substep.json"
DEFAULT_MAX_STAGE_1_STATUS_RECORDS = 500
_T = TypeVar("_T")


@dataclass(frozen=True, kw_only=True)
class Stage1StoredStatusEvent:
    """Durable status event record with display metadata for readers."""

    substep_id: str
    title: str
    status: Stage1SubstepStatus
    created_at: str
    message: str | None = None
    command_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_event(cls, event: Stage1StatusEvent) -> "Stage1StoredStatusEvent":
        """Add the canonical title to an in-memory Stage 1 status event."""

        return cls(
            substep_id=event.substep_id,
            title=_stage_1_substep_title(event.substep_id),
            status=event.status,
            created_at=event.created_at,
            message=event.message,
            command_name=event.command_name,
            metadata=dict(event.metadata),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Stage1StoredStatusEvent":
        """Build a stored status event from a JSON-compatible payload."""

        event = Stage1StatusEvent.from_dict(data)
        return cls(
            substep_id=event.substep_id,
            title=str(data.get("title") or _stage_1_substep_title(event.substep_id)),
            status=event.status,
            created_at=event.created_at,
            message=event.message,
            command_name=event.command_name,
            metadata=dict(event.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the event as the pipeline status endpoint payload."""

        return {
            "substep_id": self.substep_id,
            "status": self.status,
            "created_at": self.created_at,
            "message": self.message,
            "command_name": self.command_name,
            "metadata": dict(self.metadata or {}),
            "title": self.title,
        }


@dataclass(frozen=True, kw_only=True)
class Stage1StatusReadError:
    """A non-fatal read error from durable Stage 1 status files."""

    path: str
    error_type: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible read error payload."""

        payload: dict[str, Any] = {
            "path": self.path,
            "error_type": self.error_type,
            "message": self.message,
        }
        if self.line is not None:
            payload["line"] = self.line
        return payload


@dataclass(frozen=True, kw_only=True)
class Stage1StatusSnapshot:
    """A JSON-compatible snapshot of durable Stage 1 substep status."""

    current: Stage1StoredStatusEvent | None
    events: tuple[Stage1StoredStatusEvent, ...]
    results: tuple[DatasetSubstepResult, ...]
    read_errors: tuple[Stage1StatusReadError, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot as the pipeline status endpoint payload."""

        return {
            "current": self.current.to_dict() if self.current else None,
            "events": [event.to_dict() for event in self.events],
            "results": [result.to_dict() for result in self.results],
            "read_errors": [error.to_dict() for error in self.read_errors],
        }


class Stage1StatusRecorder:
    """Persist Stage 1 substep transitions into a run-scoped directory."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        commit_callback: Callable[[], None] | None = None,
        strict: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.status_dir = self.run_dir / STAGE_1_STATUS_DIRNAME
        self.commit_callback = commit_callback
        self.strict = strict
        self._lock = threading.Lock()

    def record_event(self, event: Stage1StatusEvent) -> None:
        """Persist one Stage 1 status event and mark it current."""

        payload = Stage1StoredStatusEvent.from_event(event).to_dict()

        def write() -> None:
            _append_jsonl(_status_events_path(self.run_dir), payload)
            _write_json(_current_substep_path(self.run_dir), payload)

        self._write_best_effort(write)

    def record_result(self, result: DatasetSubstepResult) -> None:
        """Persist one finalized Stage 1 substep result."""

        self._write_best_effort(
            lambda: _append_jsonl(_substep_results_path(self.run_dir), result.to_dict())
        )

    def _write_best_effort(self, write: Callable[[], None]) -> None:
        with self._lock:
            try:
                self.status_dir.mkdir(parents=True, exist_ok=True)
                write()
                if self.commit_callback is not None:
                    self.commit_callback()
            except Exception as exc:
                if self.strict:
                    raise
                print(f"Stage 1 status persistence failed: {type(exc).__name__}: {exc}")


def empty_stage_1_status_snapshot() -> Stage1StatusSnapshot:
    """Return an empty Stage 1 status snapshot."""

    return Stage1StatusSnapshot(current=None, events=(), results=())


def read_stage_1_status_snapshot(
    run_dir: str | Path,
    *,
    max_records: int = DEFAULT_MAX_STAGE_1_STATUS_RECORDS,
) -> Stage1StatusSnapshot:
    """Read persisted Stage 1 status without failing the caller."""

    run_dir = Path(run_dir)
    read_errors: list[Stage1StatusReadError] = []

    current = _read_json_best_effort(
        _current_substep_path(run_dir),
        parser=Stage1StoredStatusEvent.from_dict,
        read_errors=read_errors,
    )
    events = _read_jsonl_best_effort(
        _status_events_path(run_dir),
        parser=Stage1StoredStatusEvent.from_dict,
        max_records=max_records,
        read_errors=read_errors,
    )
    results = _read_jsonl_best_effort(
        _substep_results_path(run_dir),
        parser=DatasetSubstepResult.from_dict,
        max_records=max_records,
        read_errors=read_errors,
    )

    if current is None and events:
        current = events[-1]

    return Stage1StatusSnapshot(
        current=current,
        events=tuple(events),
        results=tuple(results),
        read_errors=tuple(read_errors),
    )


def _status_events_path(run_dir: Path) -> Path:
    return run_dir / STAGE_1_STATUS_DIRNAME / STAGE_1_STATUS_EVENTS_FILENAME


def _substep_results_path(run_dir: Path) -> Path:
    return run_dir / STAGE_1_STATUS_DIRNAME / STAGE_1_SUBSTEP_RESULTS_FILENAME


def _current_substep_path(run_dir: Path) -> Path:
    return run_dir / STAGE_1_STATUS_DIRNAME / STAGE_1_CURRENT_SUBSTEP_FILENAME


def _stage_1_substep_title(substep_id: str) -> str:
    for spec in STAGE_1_BUILD_STEP_SPECS:
        if spec.id == substep_id:
            return spec.title
    return substep_id


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json_best_effort(
    path: Path,
    *,
    parser: Callable[[Mapping[str, Any]], _T],
    read_errors: list[Stage1StatusReadError],
) -> _T | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _append_read_error(read_errors, path=path, exc=exc)
        return None
    if not isinstance(data, dict):
        read_errors.append(
            Stage1StatusReadError(
                path=str(path),
                error_type="TypeError",
                message="Expected JSON object.",
            )
        )
        return None
    try:
        return parser(data)
    except Exception as exc:
        _append_read_error(read_errors, path=path, exc=exc)
        return None


def _read_jsonl_best_effort(
    path: Path,
    *,
    parser: Callable[[Mapping[str, Any]], _T],
    max_records: int,
    read_errors: list[Stage1StatusReadError],
) -> list[_T]:
    if not path.exists():
        return []
    records: list[_T] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        _append_read_error(read_errors, path=path, exc=exc)
        return records
    for line_number, line in enumerate(lines[-max_records:], start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except Exception as exc:
            read_errors.append(
                Stage1StatusReadError(
                    path=str(path),
                    line=line_number,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue
        if isinstance(data, dict):
            try:
                records.append(parser(data))
            except Exception as exc:
                read_errors.append(
                    Stage1StatusReadError(
                        path=str(path),
                        line=line_number,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        else:
            read_errors.append(
                Stage1StatusReadError(
                    path=str(path),
                    line=line_number,
                    error_type="TypeError",
                    message="Expected JSON object.",
                )
            )
    return records


def _append_read_error(
    read_errors: list[Stage1StatusReadError],
    *,
    path: Path,
    exc: BaseException,
) -> None:
    read_errors.append(
        Stage1StatusReadError(
            path=str(path),
            error_type=type(exc).__name__,
            message=str(exc),
        )
    )


__all__ = [
    "DEFAULT_MAX_STAGE_1_STATUS_RECORDS",
    "STAGE_1_CURRENT_SUBSTEP_FILENAME",
    "STAGE_1_STATUS_DIRNAME",
    "STAGE_1_STATUS_EVENTS_FILENAME",
    "STAGE_1_SUBSTEP_RESULTS_FILENAME",
    "Stage1StatusRecorder",
    "Stage1StatusReadError",
    "Stage1StatusSnapshot",
    "Stage1StoredStatusEvent",
    "empty_stage_1_status_snapshot",
    "read_stage_1_status_snapshot",
]
