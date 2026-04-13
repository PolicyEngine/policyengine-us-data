"""Typed build-result contracts for local H5 publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from .._contract_utils import jsonable_contract_value
from ..requests import AreaBuildRequest
from ..validation import (
    ValidationIssue,
    ValidationResult,
)

BuildStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class AreaBuildResult:
    """The result of building one requested H5 output."""

    request: AreaBuildRequest
    build_status: BuildStatus
    output_path: Path | None = None
    build_error: str | None = None
    validation: ValidationResult = field(
        default_factory=lambda: ValidationResult(status="not_run")
    )

    def __post_init__(self) -> None:
        if self.build_status == "completed":
            if self.output_path is None:
                raise ValueError("completed AreaBuildResult requires output_path")
            if self.build_error is not None:
                raise ValueError(
                    "completed AreaBuildResult must not include build_error"
                )
        else:
            if not self.build_error:
                raise ValueError("failed AreaBuildResult requires build_error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "build_status": self.build_status,
            "output_path": jsonable_contract_value(self.output_path),
            "build_error": self.build_error,
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AreaBuildResult":
        output_path = data.get("output_path")
        return cls(
            request=AreaBuildRequest.from_dict(data["request"]),
            build_status=data["build_status"],
            output_path=Path(output_path) if output_path is not None else None,
            build_error=data.get("build_error"),
            validation=ValidationResult.from_dict(
                data.get("validation", {"status": "not_run"})
            ),
        )


@dataclass(frozen=True)
class WorkerResult:
    """The aggregate outcome returned by one worker chunk execution."""

    completed: tuple[AreaBuildResult, ...]
    failed: tuple[AreaBuildResult, ...]
    worker_issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if any(item.build_status != "completed" for item in self.completed):
            raise ValueError(
                "all results in completed must have build_status='completed'"
            )
        if any(item.build_status != "failed" for item in self.failed):
            raise ValueError("all results in failed must have build_status='failed'")

    def all_results(self) -> tuple[AreaBuildResult, ...]:
        return self.completed + self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": [jsonable_contract_value(item) for item in self.completed],
            "failed": [jsonable_contract_value(item) for item in self.failed],
            "worker_issues": [
                jsonable_contract_value(item) for item in self.worker_issues
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkerResult":
        return cls(
            completed=tuple(
                AreaBuildResult.from_dict(item)
                for item in data.get("completed", ())
            ),
            failed=tuple(
                AreaBuildResult.from_dict(item)
                for item in data.get("failed", ())
            ),
            worker_issues=tuple(
                ValidationIssue.from_dict(item)
                for item in data.get("worker_issues", ())
            ),
        )
