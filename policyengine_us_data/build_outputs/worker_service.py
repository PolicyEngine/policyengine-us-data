"""Worker chunk execution boundary for local H5 publication."""

from __future__ import annotations

import traceback as traceback_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from policyengine_us_data.pipeline_metadata import pipeline_node

from .requests import AreaBuildRequest
from .validation import AreaValidationService
from .worker_session import WorkerSession
from .writer import H5Writer

WorkerAreaStatus = Literal["completed", "failed", "skipped"]
WorkerIssuePhase = Literal["request", "build", "write", "validation"]
WorkerIssueSeverity = Literal["worker_failure", "validation"]
WorkerValidationStatus = Literal["not_run", "passed", "error"]

__all__ = [
    "LocalH5WorkerService",
    "WorkerAreaResult",
    "WorkerAreaStatus",
    "WorkerExecutionConfig",
    "WorkerIssue",
    "WorkerIssuePhase",
    "WorkerIssueSeverity",
    "WorkerResult",
    "WorkerValidationStatus",
]


@pipeline_node(
    id="local_h5_worker_execution_config",
    label="WorkerExecutionConfig",
    node_type="library",
    description="Runtime policy for one local H5 worker-service execution.",
    source_file="policyengine_us_data/build_outputs/worker_service.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_service.py"
    ],
)
@dataclass(frozen=True)
class WorkerExecutionConfig:
    """Execution policy for one worker chunk."""

    output_dir: Path
    takeup_filter: tuple[str, ...] = ()
    validate: bool = True
    fail_on_validation_error: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "takeup_filter",
            tuple(str(item) for item in self.takeup_filter),
        )


@pipeline_node(
    id="local_h5_worker_issue",
    label="WorkerIssue",
    node_type="library",
    description="Structured issue reported by one local H5 worker request.",
    source_file="policyengine_us_data/build_outputs/worker_service.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_service.py"
    ],
)
@dataclass(frozen=True)
class WorkerIssue:
    """Structured worker issue for request, build, write, or validation failures."""

    item: str
    phase: WorkerIssuePhase
    message: str
    traceback: str | None = None
    severity: WorkerIssueSeverity = "worker_failure"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the issue to worker JSON output."""

        payload: dict[str, Any] = {
            "item": self.item,
            "phase": self.phase,
            "error": self.message,
            "severity": self.severity,
        }
        if self.traceback:
            payload["traceback"] = self.traceback
        return payload


@pipeline_node(
    id="local_h5_worker_area_result",
    label="WorkerAreaResult",
    node_type="library",
    description="Structured result for one local H5 worker request.",
    source_file="policyengine_us_data/build_outputs/worker_service.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_service.py"
    ],
)
@dataclass(frozen=True)
class WorkerAreaResult:
    """Structured result for one area handled by a worker."""

    key: str
    request: AreaBuildRequest
    status: WorkerAreaStatus
    output_relative_path: str
    output_path: Path | None = None
    validation_status: WorkerValidationStatus = "not_run"
    validation_rows: tuple[Mapping[str, Any], ...] = ()
    validation_summary: Mapping[str, Any] = field(default_factory=dict)
    issues: tuple[WorkerIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_path",
            Path(self.output_path) if self.output_path is not None else None,
        )
        object.__setattr__(self, "validation_rows", tuple(self.validation_rows))
        object.__setattr__(self, "validation_summary", dict(self.validation_summary))
        object.__setattr__(self, "issues", tuple(self.issues))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the area result to worker JSON output."""

        return {
            "key": self.key,
            "request": self.request.to_dict(),
            "status": self.status,
            "output_relative_path": self.output_relative_path,
            "output_path": str(self.output_path) if self.output_path else None,
            "validation_status": self.validation_status,
            "validation_rows": [dict(row) for row in self.validation_rows],
            "validation_summary": dict(self.validation_summary),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@pipeline_node(
    id="local_h5_worker_result",
    label="WorkerResult",
    node_type="library",
    description="Structured result for one local H5 worker chunk.",
    source_file="policyengine_us_data/build_outputs/worker_service.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_service.py"
    ],
)
@dataclass(frozen=True)
class WorkerResult:
    """Structured result for a worker chunk."""

    area_results: tuple[WorkerAreaResult, ...] = ()
    issues: tuple[WorkerIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "area_results", tuple(self.area_results))
        object.__setattr__(self, "issues", tuple(self.issues))

    def to_legacy_dict(self) -> dict[str, Any]:
        """Serialize to the existing worker/coordinator JSON contract."""

        completed = [
            result.key for result in self.area_results if result.status == "completed"
        ]
        failed = [
            result.key for result in self.area_results if result.status == "failed"
        ]
        failed.extend(issue.item for issue in self.issues)
        validation_rows: list[dict[str, Any]] = []
        validation_summary: dict[str, Mapping[str, Any]] = {}
        legacy_errors = [issue.to_dict() for issue in self.issues]
        structured_issues = [issue.to_dict() for issue in self.issues]

        for result in self.area_results:
            validation_rows.extend(dict(row) for row in result.validation_rows)
            if result.validation_summary:
                validation_summary[result.key] = dict(result.validation_summary)
            issue_dicts = [issue.to_dict() for issue in result.issues]
            structured_issues.extend(issue_dicts)
            if result.status == "failed":
                legacy_errors.extend(issue_dicts)

        return {
            "completed": completed,
            "failed": failed,
            "errors": legacy_errors,
            "validation_rows": validation_rows,
            "validation_summary": validation_summary,
            "results": [result.to_dict() for result in self.area_results],
            "issues": structured_issues,
        }


@pipeline_node(
    id="local_h5_worker_service",
    label="LocalH5WorkerService",
    node_type="library",
    description="Execute one worker chunk of local H5 build requests.",
    source_file="policyengine_us_data/build_outputs/worker_service.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_service.py",
        "uv run pytest tests/integration/build_outputs/h5_worker_runtime/test_worker_script_tiny_fixture.py",
    ],
)
@dataclass(frozen=True)
class LocalH5WorkerService:
    """Execute typed local H5 requests for one prepared worker session."""

    builder: Any = field(default_factory=lambda: _default_builder())
    writer: Any = field(default_factory=H5Writer)
    validation_service: AreaValidationService = field(
        default_factory=AreaValidationService
    )

    def execute(
        self,
        *,
        session: WorkerSession,
        requests: Sequence[AreaBuildRequest],
        config: WorkerExecutionConfig,
    ) -> WorkerResult:
        """Build and optionally validate every request in one worker chunk."""

        area_results = tuple(
            self._execute_request(session=session, request=request, config=config)
            for request in requests
        )
        return WorkerResult(area_results=area_results)

    def _execute_request(
        self,
        *,
        session: WorkerSession,
        request: AreaBuildRequest,
        config: WorkerExecutionConfig,
    ) -> WorkerAreaResult:
        key = _request_key(request)
        try:
            output_path = _resolve_output_path(
                output_dir=config.output_dir,
                output_relative_path=request.output_relative_path,
            )
        except Exception as exc:
            return _failed_result(
                key=key,
                request=request,
                phase="request",
                error=exc,
            )

        try:
            if request.area_type == "national":
                _validate_national_weight_scope(session)
            source = session.load_source()
            build_result = self.builder.build(
                source=source,
                simulation=_source_simulation(source),
                weights=session.weights,
                geography=session.geography,
                request=request,
                takeup_filter=(
                    None
                    if request.area_type == "national"
                    else tuple(config.takeup_filter)
                ),
            )
        except Exception as exc:
            return _failed_result(
                key=key,
                request=request,
                phase="build",
                error=exc,
                output_path=output_path,
            )

        try:
            write_result = self.writer.write(
                payload=build_result.payload,
                output_path=output_path,
            )
            written_path = Path(getattr(write_result, "path", output_path))
        except Exception as exc:
            return _failed_result(
                key=key,
                request=request,
                phase="write",
                error=exc,
                output_path=output_path,
            )

        validation_status: WorkerValidationStatus = "not_run"
        validation_rows: tuple[Mapping[str, Any], ...] = ()
        validation_summary: Mapping[str, Any] = {}
        issues: tuple[WorkerIssue, ...] = ()
        if config.validate and session.validation_context is not None:
            try:
                validation_result = self.validation_service.validate_request(
                    context=session.validation_context,
                    h5_path=written_path,
                    request=request,
                )
                validation_rows = tuple(validation_result.rows)
                validation_summary = dict(validation_result.summary)
                validation_status = "passed"
            except Exception as exc:
                issue = _issue(
                    key=key,
                    phase="validation",
                    error=exc,
                    severity=(
                        "worker_failure"
                        if config.fail_on_validation_error
                        else "validation"
                    ),
                )
                issues = (issue,)
                validation_status = "error"
                if config.fail_on_validation_error:
                    return WorkerAreaResult(
                        key=key,
                        request=request,
                        status="failed",
                        output_relative_path=request.output_relative_path,
                        output_path=written_path,
                        validation_status=validation_status,
                        issues=issues,
                    )

        return WorkerAreaResult(
            key=key,
            request=request,
            status="completed",
            output_relative_path=request.output_relative_path,
            output_path=written_path,
            validation_status=validation_status,
            validation_rows=validation_rows,
            validation_summary=validation_summary,
            issues=issues,
        )


def _request_key(request: AreaBuildRequest) -> str:
    return f"{request.area_type}:{request.area_id}"


def _default_builder() -> Any:
    from .builder import LocalAreaDatasetBuilder
    from .us_augmentations import default_us_postprocessors

    return LocalAreaDatasetBuilder(postprocessors=default_us_postprocessors())


def _resolve_output_path(*, output_dir: Path, output_relative_path: str) -> Path:
    candidate_path = (Path(output_dir) / output_relative_path).resolve(strict=False)
    output_dir_path = Path(output_dir).resolve(strict=False)
    try:
        candidate_path.relative_to(output_dir_path)
    except ValueError as exc:
        raise ValueError(
            "output_relative_path must stay within the worker output_dir"
        ) from exc
    return candidate_path


def _source_simulation(source: Any) -> Any:
    provider = getattr(source, "variable_provider", None)
    simulation = getattr(provider, "simulation", None)
    if simulation is None:
        raise ValueError("Worker source does not expose a simulation")
    return simulation


def _validate_national_weight_scope(session: WorkerSession) -> None:
    if session.weights.n_clones != session.geography.n_clones:
        raise ValueError(
            f"National weights have {session.weights.n_clones} clones "
            f"but geography has {session.geography.n_clones}. "
            "Use the matching saved geography artifact."
        )


def _failed_result(
    *,
    key: str,
    request: AreaBuildRequest,
    phase: WorkerIssuePhase,
    error: Exception,
    output_path: Path | None = None,
) -> WorkerAreaResult:
    return WorkerAreaResult(
        key=key,
        request=request,
        status="failed",
        output_relative_path=request.output_relative_path,
        output_path=output_path,
        issues=(_issue(key=key, phase=phase, error=error),),
    )


def _issue(
    *,
    key: str,
    phase: WorkerIssuePhase,
    error: Exception,
    severity: WorkerIssueSeverity = "worker_failure",
) -> WorkerIssue:
    return WorkerIssue(
        item=key,
        phase=phase,
        message=str(error),
        traceback=traceback_module.format_exc(),
        severity=severity,
    )
