"""Coordinator-side normalization for local H5 worker JSON responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from policyengine_us_data.pipeline_metadata import pipeline_node

__all__ = [
    "CoordinatorWorkerResult",
    "normalize_worker_response",
]


@pipeline_node(
    id="local_h5_coordinator_worker_result",
    label="CoordinatorWorkerResult",
    node_type="library",
    description="Coordinator-normalized view of one local H5 worker response.",
    source_file="policyengine_us_data/build_outputs/worker_responses.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_responses.py"
    ],
)
@dataclass(frozen=True)
class CoordinatorWorkerResult:
    """Normalized worker response with explicit fatal and nonfatal issue classes."""

    completed: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    fatal_errors: tuple[dict[str, Any], ...] = ()
    issues: tuple[dict[str, Any], ...] = ()
    validation_rows: tuple[dict[str, Any], ...] = ()


def _coordinator_error(
    error: Mapping[str, Any],
    *,
    worker_index: int,
    severity: str,
) -> dict[str, Any]:
    payload = dict(error)
    payload.setdefault("worker", worker_index)
    payload["severity"] = severity
    return payload


def _issue_severity(
    issue: Mapping[str, Any],
    *,
    default_severity: str,
) -> str:
    severity = issue.get("severity")
    if isinstance(severity, str) and severity:
        return severity
    if issue.get("phase") == "validation":
        return "validation"
    return default_severity


def _is_fatal_severity(severity: str) -> bool:
    return severity in {"protocol", "worker_failure"}


def _string_tuple_field(
    result: Mapping[str, Any],
    *,
    worker_index: int,
    field_name: str,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    value = result.get(field_name)
    if not isinstance(value, list | tuple):
        return (), (
            _coordinator_error(
                {
                    "phase": "protocol",
                    "error": f"Worker result field {field_name!r} must be a list",
                },
                worker_index=worker_index,
                severity="protocol",
            ),
        )
    return tuple(str(item) for item in value), ()


def _dict_tuple_field(
    result: Mapping[str, Any],
    *,
    worker_index: int,
    field_name: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    value = result.get(field_name, [])
    if not isinstance(value, list | tuple):
        return (), (
            _coordinator_error(
                {
                    "phase": "protocol",
                    "error": f"Worker result field {field_name!r} must be a list",
                },
                worker_index=worker_index,
                severity="protocol",
            ),
        )

    items: list[dict[str, Any]] = []
    protocol_errors: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            items.append(dict(item))
        else:
            protocol_errors.append(
                _coordinator_error(
                    {
                        "phase": "protocol",
                        "error": (
                            f"Worker result field {field_name!r} contained "
                            "a non-object item"
                        ),
                    },
                    worker_index=worker_index,
                    severity="protocol",
                )
            )
    return tuple(items), tuple(protocol_errors)


def _issue_identity(issue: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return issue.get("item"), issue.get("phase"), issue.get("error")


@pipeline_node(
    id="normalize_local_h5_worker_response",
    label="Normalize Local H5 Worker Response",
    node_type="library",
    description="Normalize legacy worker JSON into explicit coordinator severity classes.",
    source_file="policyengine_us_data/build_outputs/worker_responses.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_worker_responses.py"
    ],
)
def normalize_worker_response(
    *,
    worker_index: int,
    result: object,
) -> CoordinatorWorkerResult:
    """Normalize worker JSON into explicit fatal and nonfatal coordinator issues."""

    if result is None:
        return CoordinatorWorkerResult(
            fatal_errors=(
                _coordinator_error(
                    {"phase": "protocol", "error": "Worker returned None"},
                    worker_index=worker_index,
                    severity="protocol",
                ),
            )
        )
    if not isinstance(result, dict):
        return CoordinatorWorkerResult(
            fatal_errors=(
                _coordinator_error(
                    {
                        "phase": "protocol",
                        "error": f"Worker returned non-object result: {type(result)!r}",
                    },
                    worker_index=worker_index,
                    severity="protocol",
                ),
            )
        )

    completed, completed_errors = _string_tuple_field(
        result,
        worker_index=worker_index,
        field_name="completed",
    )
    failed, failed_errors = _string_tuple_field(
        result,
        worker_index=worker_index,
        field_name="failed",
    )
    worker_errors, worker_error_protocol_errors = _dict_tuple_field(
        result,
        worker_index=worker_index,
        field_name="errors",
    )
    worker_issues, worker_issue_protocol_errors = _dict_tuple_field(
        result,
        worker_index=worker_index,
        field_name="issues",
    )
    validation_rows, validation_row_protocol_errors = _dict_tuple_field(
        result,
        worker_index=worker_index,
        field_name="validation_rows",
    )

    fatal_errors = [
        *completed_errors,
        *failed_errors,
        *worker_error_protocol_errors,
        *worker_issue_protocol_errors,
        *validation_row_protocol_errors,
    ]
    nonfatal_issues: list[dict[str, Any]] = []
    nonfatal_issue_keys: set[tuple[Any, Any, Any]] = set()
    for error in worker_errors:
        severity = _issue_severity(error, default_severity="worker_failure")
        normalized_error = _coordinator_error(
            error,
            worker_index=worker_index,
            severity=severity,
        )
        if _is_fatal_severity(severity):
            fatal_errors.append(normalized_error)
        else:
            nonfatal_issues.append(normalized_error)
            nonfatal_issue_keys.add(_issue_identity(normalized_error))

    for issue in worker_issues:
        severity = _issue_severity(issue, default_severity="worker_issue")
        normalized_issue = _coordinator_error(
            issue,
            worker_index=worker_index,
            severity=severity,
        )
        if _is_fatal_severity(severity):
            fatal_errors.append(normalized_issue)
        elif _issue_identity(normalized_issue) not in nonfatal_issue_keys:
            nonfatal_issues.append(normalized_issue)
            nonfatal_issue_keys.add(_issue_identity(normalized_issue))

    error_items = {
        str(error.get("item")) for error in fatal_errors if error.get("item")
    }
    fatal_errors.extend(
        _coordinator_error(
            {
                "item": item,
                "phase": "worker",
                "error": "Worker reported failed item without a matching error",
            },
            worker_index=worker_index,
            severity="worker_failure",
        )
        for item in failed
        if item not in error_items
    )

    return CoordinatorWorkerResult(
        completed=completed,
        failed=failed,
        fatal_errors=tuple(fatal_errors),
        issues=tuple(nonfatal_issues),
        validation_rows=validation_rows,
    )
