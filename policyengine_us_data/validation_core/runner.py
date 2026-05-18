"""Shared validation runner that aggregates canonical validation reports."""

from __future__ import annotations

from collections.abc import Iterable
import traceback
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import (
    DiagnosticRef,
    ValidationFinding,
    ValidationReport,
)
from policyengine_us_data.utils.error_redaction import (
    DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    redacted_bounded_error_text,
)

from .checks import ValidationCheck, ValidationCheckResult, ValidationSuite
from .context import ValidationContext


@pipeline_node(
    id="validation_core_runner",
    label="ValidationRunner",
    node_type="library",
    description="Shared runner that executes validation suites and aggregates canonical ValidationReport output.",
    status="current",
    stability="stable",
    pathways=["cross_stage_validation"],
    artifacts_out=["ValidationReport"],
    validation_commands=["uv run pytest tests/unit/test_validation_core.py"],
)
class ValidationRunner:
    """Run validation suites and aggregate canonical stage-contract reports."""

    def run(
        self,
        suite: ValidationSuite,
        context: ValidationContext,
        *,
        skip_reason: str | None = None,
    ) -> ValidationReport:
        """Execute ``suite`` against ``context`` and return a canonical report."""

        self._validate_suite_context(suite, context)
        if skip_reason is not None:
            return ValidationReport(
                status="not_run",
                metadata=self._report_metadata(suite, context, skip_reason=skip_reason),
            )

        findings: list[ValidationFinding] = []
        diagnostics: list[DiagnosticRef] = []

        for check in suite.checks:
            missing_finding = self._missing_required_artifact_finding(check, context)
            if missing_finding is not None:
                findings.append(missing_finding)
                continue

            try:
                result = check.run(context)
                check_findings, check_diagnostics = self._normalize_result(
                    check,
                    context,
                    result,
                )
            except Exception as exc:
                message = redacted_bounded_error_text(
                    f"Validation check raised {exc.__class__.__name__}: {exc}",
                    max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
                )
                trace = redacted_bounded_error_text(
                    "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                )
                metadata = {
                    "failure_type": "exception",
                    "exception_type": exc.__class__.__name__,
                    "status": "fail",
                    "traceback": trace.text,
                    "traceback_truncated": trace.truncated,
                }
                if trace.truncated:
                    metadata["traceback_omitted_chars"] = trace.omitted_chars
                findings.append(
                    self._generated_finding(
                        check,
                        context,
                        message=message.text,
                        **metadata,
                    )
                )
                continue

            check_findings = self._validate_finding_ids(check, context, check_findings)
            findings.extend(check_findings)
            diagnostics.extend(check_diagnostics)

        return ValidationReport(
            status=self._status_from_findings(findings),
            findings=tuple(findings),
            diagnostics=tuple(diagnostics),
            metadata=self._report_metadata(suite, context),
        )

    def _validate_suite_context(
        self,
        suite: ValidationSuite,
        context: ValidationContext,
    ) -> None:
        if not isinstance(suite, ValidationSuite):
            raise TypeError("suite must be a ValidationSuite")
        if not isinstance(context, ValidationContext):
            raise TypeError("context must be a ValidationContext")
        if context.stage_id != suite.stage_id:
            raise ValueError(
                f"Context stage_id {context.stage_id!r} does not match "
                f"suite stage_id {suite.stage_id!r}"
            )
        if context.substage_id != suite.substage_id:
            raise ValueError(
                f"Context substage_id {context.substage_id!r} does not match "
                f"suite substage_id {suite.substage_id!r}"
            )

    def _missing_required_artifact_finding(
        self,
        check: ValidationCheck,
        context: ValidationContext,
    ) -> ValidationFinding | None:
        for logical_name in check.required_artifacts:
            try:
                context.resolver.require(logical_name)
            except KeyError:
                return ValidationFinding(
                    check_id=check.check_id,
                    status=self._generated_finding_status(check),
                    message=f"Missing required validation artifact: {logical_name}",
                    metric="required_artifact",
                    value=logical_name,
                    metadata=self._check_metadata(
                        check,
                        context,
                        failure_type="missing_required_artifact",
                    ),
                )
        return None

    def _normalize_result(
        self,
        check: ValidationCheck,
        context: ValidationContext,
        result: ValidationCheckResult,
    ) -> tuple[tuple[ValidationFinding, ...], tuple[DiagnosticRef, ...]]:
        if result is None:
            return (), ()
        if isinstance(result, ValidationFinding):
            return (result,), ()
        if isinstance(result, ValidationReport):
            return (
                self._findings_from_child_report(check, context, result),
                result.diagnostics,
            )
        if isinstance(result, Iterable):
            findings = tuple(result)
            for finding in findings:
                if not isinstance(finding, ValidationFinding):
                    return (
                        (
                            self._generated_finding(
                                check,
                                context,
                                message=(
                                    "Validation check iterable results must contain "
                                    "ValidationFinding instances"
                                ),
                                failure_type="invalid_check_result",
                                invalid_result_type=type(finding).__name__,
                                status="fail",
                            ),
                        ),
                        (),
                    )
            return findings, ()
        return (
            (
                self._generated_finding(
                    check,
                    context,
                    message=(
                        "Validation checks must return a ValidationFinding, "
                        "ValidationReport, iterable of ValidationFinding objects, "
                        "or None"
                    ),
                    failure_type="invalid_check_result",
                    invalid_result_type=type(result).__name__,
                    status="fail",
                ),
            ),
            (),
        )

    def _findings_from_child_report(
        self,
        check: ValidationCheck,
        context: ValidationContext,
        report: ValidationReport,
    ) -> tuple[ValidationFinding, ...]:
        findings = tuple(report.findings)
        statuses = {finding.status for finding in findings}
        if report.status == "fail" and "fail" not in statuses:
            return (
                *findings,
                self._generated_finding(
                    check,
                    context,
                    message="Validation check returned a fail report without a failing finding.",
                    failure_type="child_report_status",
                    child_report_status=report.status,
                    status="fail",
                ),
            )
        if report.status == "warn" and statuses.isdisjoint({"warn", "fail"}):
            return (
                *findings,
                self._generated_finding(
                    check,
                    context,
                    message="Validation check returned a warn report without a warning finding.",
                    failure_type="child_report_status",
                    child_report_status=report.status,
                    status="warn",
                ),
            )
        if report.status == "not_run":
            return (
                *findings,
                self._generated_finding(
                    check,
                    context,
                    message="Validation check returned a not_run report during suite execution.",
                    failure_type="child_report_status",
                    child_report_status=report.status,
                    status="fail",
                ),
            )
        return findings

    def _validate_finding_ids(
        self,
        check: ValidationCheck,
        context: ValidationContext,
        findings: tuple[ValidationFinding, ...],
    ) -> tuple[ValidationFinding, ...]:
        for finding in findings:
            if finding.check_id != check.check_id:
                return (
                    self._generated_finding(
                        check,
                        context,
                        message=(
                            f"Validation check returned finding for "
                            f"{finding.check_id!r} instead of {check.check_id!r}."
                        ),
                        failure_type="invalid_check_result",
                        invalid_finding_check_id=finding.check_id,
                        status="fail",
                    ),
                )
        return findings

    def _status_from_findings(
        self,
        findings: Iterable[ValidationFinding],
    ) -> str:
        statuses = tuple(finding.status for finding in findings)
        if any(status == "fail" for status in statuses):
            return "fail"
        if any(status == "warn" for status in statuses):
            return "warn"
        return "pass"

    def _generated_finding(
        self,
        check: ValidationCheck,
        context: ValidationContext,
        *,
        message: str,
        status: str | None = None,
        **metadata: Any,
    ) -> ValidationFinding:
        return ValidationFinding(
            check_id=check.check_id,
            status=status or self._generated_finding_status(check),
            message=message,
            metadata=self._check_metadata(check, context, **metadata),
        )

    def _generated_finding_status(self, check: ValidationCheck) -> str:
        return "warn" if check.severity == "warning" else "fail"

    def _report_metadata(
        self,
        suite: ValidationSuite,
        context: ValidationContext,
        **extra: Any,
    ) -> dict[str, Any]:
        metadata = {
            "suite_id": suite.suite_id,
            "stage_id": suite.stage_id,
            "substage_id": suite.substage_id,
            "run_id": context.run_id,
            "check_ids": [check.check_id for check in suite.checks],
            "context_metadata": context.metadata,
        }
        metadata.update(extra)
        return metadata

    def _check_metadata(
        self,
        check: ValidationCheck,
        context: ValidationContext,
        **extra: Any,
    ) -> dict[str, Any]:
        metadata = {
            "check_id": check.check_id,
            "stage_id": check.stage_id,
            "substage_id": check.substage_id,
            "run_id": context.run_id,
            "severity": check.severity,
            "context_metadata": context.metadata,
        }
        metadata.update(extra)
        return metadata
