"""Core value contracts for the local H5 refactor.

These contracts intentionally avoid any PolicyEngine, Modal, or H5 IO.
They define the shapes that later services will exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

AreaType = Literal["national", "state", "district", "city", "custom"]
BuildStatus = Literal["completed", "failed"]
ValidationStatus = Literal["not_run", "passed", "failed", "error"]
FilterOp = Literal["eq", "in"]


def _jsonable(value: Any) -> Any:
    """Convert common contract values into JSON-serializable primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class AreaFilter:
    geography_field: str
    op: FilterOp
    value: str | int | tuple[str | int, ...]

    def __post_init__(self) -> None:
        if not self.geography_field:
            raise ValueError("geography_field must be non-empty")
        if self.op == "in" and not isinstance(self.value, tuple):
            raise ValueError("AreaFilter value must be a tuple when op='in'")
        if self.op == "eq" and isinstance(self.value, tuple):
            raise ValueError("AreaFilter value must not be a tuple when op='eq'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "geography_field": self.geography_field,
            "op": self.op,
            "value": _jsonable(self.value),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AreaFilter":
        value = data["value"]
        if data["op"] == "in":
            value = tuple(value)
        return cls(
            geography_field=str(data["geography_field"]),
            op=data["op"],
            value=value,
        )


@dataclass(frozen=True)
class AreaBuildRequest:
    area_type: AreaType
    area_id: str
    display_name: str
    output_relative_path: str
    filters: tuple[AreaFilter, ...] = ()
    validation_geo_level: str | None = None
    validation_geographic_ids: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.area_id:
            raise ValueError("area_id must be non-empty")
        if not self.display_name:
            raise ValueError("display_name must be non-empty")
        if not self.output_relative_path:
            raise ValueError("output_relative_path must be non-empty")
        if self.validation_geographic_ids and self.validation_geo_level is None:
            raise ValueError(
                "validation_geo_level must be set when validation_geographic_ids "
                "are provided"
            )

    @classmethod
    def national(cls, area_id: str = "US") -> "AreaBuildRequest":
        return cls(
            area_type="national",
            area_id=area_id,
            display_name=area_id,
            output_relative_path="national/US.h5",
            validation_geo_level="national",
            validation_geographic_ids=(area_id,),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_type": self.area_type,
            "area_id": self.area_id,
            "display_name": self.display_name,
            "output_relative_path": self.output_relative_path,
            "filters": [_jsonable(item) for item in self.filters],
            "validation_geo_level": self.validation_geo_level,
            "validation_geographic_ids": list(self.validation_geographic_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AreaBuildRequest":
        return cls(
            area_type=data["area_type"],
            area_id=str(data["area_id"]),
            display_name=str(data["display_name"]),
            output_relative_path=str(data["output_relative_path"]),
            filters=tuple(
                AreaFilter.from_dict(item)
                for item in data.get("filters", ())
            ),
            validation_geo_level=data.get("validation_geo_level"),
            validation_geographic_ids=tuple(
                str(item) for item in data.get("validation_geographic_ids", ())
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class PublishingInputBundle:
    weights_path: Path
    source_dataset_path: Path
    target_db_path: Path | None
    calibration_package_path: Path | None
    run_config_path: Path | None
    run_id: str
    version: str
    n_clones: int | None
    seed: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.n_clones is not None and self.n_clones <= 0:
            raise ValueError("n_clones must be positive when provided")

    def required_paths(self) -> tuple[Path, ...]:
        required = [self.weights_path, self.source_dataset_path]
        if self.target_db_path is not None:
            required.append(self.target_db_path)
        if self.calibration_package_path is not None:
            required.append(self.calibration_package_path)
        return tuple(required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "source_dataset_path": str(self.source_dataset_path),
            "target_db_path": _jsonable(self.target_db_path),
            "calibration_package_path": _jsonable(self.calibration_package_path),
            "run_config_path": _jsonable(self.run_config_path),
            "run_id": self.run_id,
            "version": self.version,
            "n_clones": self.n_clones,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ValidationPolicy:
    """Validation controls for H5 worker execution.

    Only `enabled` is enforced today. The finer-grained failure and
    sub-check flags are intentionally present as forward-compatible
    contract fields, but they are not yet fully wired through the
    validator implementations.
    """

    enabled: bool = True
    fail_on_exception: bool = False
    fail_on_validation_failure: bool = False
    run_sanity_checks: bool = True
    run_target_validation: bool = True
    run_national_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fail_on_exception": self.fail_on_exception,
            "fail_on_validation_failure": self.fail_on_validation_failure,
            "run_sanity_checks": self.run_sanity_checks,
            "run_target_validation": self.run_target_validation,
            "run_national_validation": self.run_national_validation,
        }


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: Literal["warning", "error"]
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("code must be non-empty")
        if not self.message:
            raise ValueError("message must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "details": _jsonable(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationIssue":
        return cls(
            code=str(data["code"]),
            message=str(data["message"]),
            severity=data["severity"],
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    rows: tuple[Mapping[str, Any], ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rows": [_jsonable(item) for item in self.rows],
            "issues": [_jsonable(item) for item in self.issues],
            "summary": _jsonable(self.summary),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationResult":
        return cls(
            status=data["status"],
            rows=tuple(dict(item) for item in data.get("rows", ())),
            issues=tuple(
                ValidationIssue.from_dict(item)
                for item in data.get("issues", ())
            ),
            summary=dict(data.get("summary", {})),
        )


@dataclass(frozen=True)
class AreaBuildResult:
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
            "output_path": _jsonable(self.output_path),
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
    completed: tuple[AreaBuildResult, ...]
    failed: tuple[AreaBuildResult, ...]
    worker_issues: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        if any(item.build_status != "completed" for item in self.completed):
            raise ValueError("all results in completed must have build_status='completed'")
        if any(item.build_status != "failed" for item in self.failed):
            raise ValueError("all results in failed must have build_status='failed'")

    def all_results(self) -> tuple[AreaBuildResult, ...]:
        return self.completed + self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": [_jsonable(item) for item in self.completed],
            "failed": [_jsonable(item) for item in self.failed],
            "worker_issues": [_jsonable(item) for item in self.worker_issues],
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
