"""Typed validation contracts for local H5 publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .._contract_utils import jsonable_contract_value

ValidationStatus = Literal["not_run", "passed", "failed", "error"]


@dataclass(frozen=True)
class ValidationPolicy:
    """Validation controls for H5 worker execution."""

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
    """One warning or error emitted while validating an H5 output."""

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
            "details": jsonable_contract_value(self.details),
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
    """The structured validation outcome attached to a build result."""

    status: ValidationStatus
    rows: tuple[Mapping[str, Any], ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rows": [jsonable_contract_value(item) for item in self.rows],
            "issues": [jsonable_contract_value(item) for item in self.issues],
            "summary": jsonable_contract_value(self.summary),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationResult":
        return cls(
            status=data["status"],
            rows=tuple(dict(item) for item in data.get("rows", ())),
            issues=tuple(
                ValidationIssue.from_dict(item) for item in data.get("issues", ())
            ),
            summary=dict(data.get("summary", {})),
        )
