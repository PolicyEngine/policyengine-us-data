"""Validation check declarations used by the shared runner."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import ValidationFinding, ValidationReport
from policyengine_us_data.stage_contracts.stages import (
    is_canonical_stage_id,
    is_canonical_substage_id,
)

from .context import ValidationContext

ValidationCheckResult: TypeAlias = (
    ValidationFinding | ValidationReport | Iterable[ValidationFinding] | None
)
ValidationCheckCallable: TypeAlias = Callable[
    [ValidationContext], ValidationCheckResult
]

_CHECK_SEVERITIES = frozenset({"warning", "error"})


def _required_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _canonical_stage_id(value: str) -> str:
    stage_id = _required_string(value, "stage_id")
    if not is_canonical_stage_id(stage_id):
        raise ValueError(f"Invalid canonical stage_id: {stage_id!r}")
    return stage_id


def _canonical_substage_id(stage_id: str, value: str | None) -> str | None:
    if value is None:
        return None
    substage_id = _required_string(value, "substage_id")
    if not is_canonical_substage_id(stage_id, substage_id):
        raise ValueError(
            f"Invalid canonical substage_id {substage_id!r} for stage_id {stage_id!r}"
        )
    return substage_id


@pipeline_node(
    id="validation_core_check",
    label="ValidationCheck",
    node_type="library",
    description="Stable declaration for one executable validation check and its artifact dependencies.",
    status="current",
    stability="stable",
    pathways=["cross_stage_validation"],
    validation_commands=["uv run pytest tests/unit/test_validation_core.py"],
)
@dataclass(frozen=True, kw_only=True)
class ValidationCheck:
    """One executable validation check with stable identity and dependencies."""

    check_id: str
    stage_id: str
    substage_id: str | None = None
    description: str
    severity: Literal["warning", "error"] = "error"
    required_artifacts: tuple[str, ...] = ()
    run: ValidationCheckCallable

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "check_id", _required_string(self.check_id, "check_id")
        )
        stage_id = _canonical_stage_id(self.stage_id)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(
            self,
            "substage_id",
            _canonical_substage_id(stage_id, self.substage_id),
        )
        object.__setattr__(
            self,
            "description",
            _required_string(self.description, "description"),
        )
        if self.severity not in _CHECK_SEVERITIES:
            raise ValueError(f"Invalid validation check severity: {self.severity!r}")
        if isinstance(self.required_artifacts, str) or not isinstance(
            self.required_artifacts, tuple | list
        ):
            raise TypeError("required_artifacts must be a tuple or list of strings")
        required_artifacts = tuple(
            _required_string(artifact, "required_artifacts")
            for artifact in self.required_artifacts
        )
        object.__setattr__(self, "required_artifacts", required_artifacts)
        if not callable(self.run):
            raise TypeError("run must be callable")


@pipeline_node(
    id="validation_core_suite",
    label="ValidationSuite",
    node_type="library",
    description="Ordered cross-stage validation suite for a canonical stage or substage boundary.",
    status="current",
    stability="stable",
    pathways=["cross_stage_validation"],
    validation_commands=["uv run pytest tests/unit/test_validation_core.py"],
)
@dataclass(frozen=True, kw_only=True)
class ValidationSuite:
    """Ordered validation checks for one stage or substage boundary."""

    suite_id: str
    stage_id: str
    substage_id: str | None = None
    checks: tuple[ValidationCheck, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "suite_id", _required_string(self.suite_id, "suite_id")
        )
        stage_id = _canonical_stage_id(self.stage_id)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(
            self,
            "substage_id",
            _canonical_substage_id(stage_id, self.substage_id),
        )
        checks = tuple(self.checks)
        if not checks:
            raise ValueError("ValidationSuite must include at least one check")
        seen_check_ids: set[str] = set()
        for check in checks:
            if not isinstance(check, ValidationCheck):
                raise TypeError("checks must contain ValidationCheck instances")
            if check.check_id in seen_check_ids:
                raise ValueError(f"Duplicate validation check_id: {check.check_id!r}")
            seen_check_ids.add(check.check_id)
            if check.stage_id != self.stage_id:
                raise ValueError(
                    f"Check {check.check_id!r} stage_id {check.stage_id!r} "
                    f"does not match suite stage_id {self.stage_id!r}"
                )
            if self.substage_id is None and check.substage_id is not None:
                raise ValueError(
                    f"Check {check.check_id!r} is substage-scoped but suite "
                    "is stage-scoped"
                )
            if self.substage_id is not None and check.substage_id != self.substage_id:
                raise ValueError(
                    f"Check {check.check_id!r} substage_id {check.substage_id!r} "
                    f"does not match suite substage_id {self.substage_id!r}"
                )
        object.__setattr__(self, "checks", checks)
