"""Writers for canonical validation reports and findings."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import DiagnosticRef, ValidationReport

__all__ = [
    "ValidationFindingsJsonlOutputStrategy",
    "ValidationReportJsonOutputStrategy",
    "ValidationReportOutput",
    "ValidationReportOutputStrategy",
    "ValidationReportWriter",
    "ValidationResultWriter",
    "ValidationSummaryJsonOutputStrategy",
]


_DEFAULT_REPORT_FILENAME = "validation_report.json"
_DEFAULT_FINDINGS_FILENAME = "validation_findings.jsonl"


def _safe_child_path(output_dir: Path, filename: str, field_name: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{field_name} must be a non-empty filename")
    path = Path(filename)
    if path.is_absolute() or path.name != filename or ".." in path.parts:
        raise ValueError(f"{field_name} must be a plain filename under output_dir")
    return output_dir / path


@dataclass(frozen=True, kw_only=True)
class ValidationReportOutput:
    """One concrete text artifact produced from a validation report."""

    key: str
    filename: str
    content: str
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("output key must be a non-empty string")
        if not isinstance(self.content, str):
            raise TypeError("output content must be a string")
        if not isinstance(self.encoding, str) or not self.encoding.strip():
            raise ValueError("output encoding must be a non-empty string")


class ValidationReportOutputStrategy(Protocol):
    """Build one output artifact from a canonical validation report."""

    def build(self, report: ValidationReport) -> ValidationReportOutput:
        """Return the output artifact to write for ``report``."""


@dataclass(frozen=True, kw_only=True)
class ValidationReportJsonOutputStrategy:
    """Build the canonical validation report JSON artifact."""

    key: str = "report"
    filename: str = _DEFAULT_REPORT_FILENAME

    def build(self, report: ValidationReport) -> ValidationReportOutput:
        """Return formatted canonical ``ValidationReport`` JSON."""

        return ValidationReportOutput(
            key=self.key,
            filename=self.filename,
            content=json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        )


@dataclass(frozen=True, kw_only=True)
class ValidationFindingsJsonlOutputStrategy:
    """Build the JSONL artifact containing one validation finding per line."""

    key: str = "findings"
    filename: str = _DEFAULT_FINDINGS_FILENAME

    def build(self, report: ValidationReport) -> ValidationReportOutput:
        """Return JSONL content for each finding in ``report``."""

        return ValidationReportOutput(
            key=self.key,
            filename=self.filename,
            content="".join(
                json.dumps(finding.to_dict(), sort_keys=True) + "\n"
                for finding in report.findings
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ValidationSummaryJsonOutputStrategy:
    """Build a compact diagnostic summary JSON artifact."""

    key: str = "summary"
    filename: str

    def build(self, report: ValidationReport) -> ValidationReportOutput:
        """Return a diagnostic summary derived from ``report``."""

        summary = DiagnosticRef(
            name="validation_summary",
            kind="json",
            summary={
                "status": report.status,
                "finding_count": len(report.findings),
                "fail_count": sum(
                    1 for finding in report.findings if finding.status == "fail"
                ),
                "warn_count": sum(
                    1 for finding in report.findings if finding.status == "warn"
                ),
                "pass_count": sum(
                    1 for finding in report.findings if finding.status == "pass"
                ),
                "metadata": report.to_dict()["metadata"],
            },
            severity=_summary_severity(report),
        )
        return ValidationReportOutput(
            key=self.key,
            filename=self.filename,
            content=json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        )


@pipeline_node(
    id="validation_core_report_writer",
    label="ValidationReportWriter",
    node_type="library",
    description="Write canonical validation artifacts using configurable output strategies.",
    status="current",
    stability="stable",
    pathways=["cross_stage_validation"],
    artifacts_in=["ValidationReport"],
    artifacts_out=["validation_report.json", "validation_findings.jsonl"],
    validation_commands=["uv run pytest tests/unit/test_validation_core.py"],
)
@dataclass(frozen=True, kw_only=True)
class ValidationReportWriter:
    """Write validation report outputs generated by output strategies."""

    output_dir: Path
    strategies: tuple[ValidationReportOutputStrategy, ...] | None = None
    report_filename: str = _DEFAULT_REPORT_FILENAME
    findings_filename: str = _DEFAULT_FINDINGS_FILENAME
    summary_filename: str | None = None

    def write(self, report: ValidationReport) -> dict[str, Path]:
        """Write strategy-generated outputs for ``report`` and return paths by key."""

        if not isinstance(report, ValidationReport):
            raise TypeError("report must be a ValidationReport")

        output_dir = Path(self.output_dir)
        outputs = self._outputs(report)
        paths = self._resolve_output_paths(output_dir, outputs)

        output_dir.mkdir(parents=True, exist_ok=True)
        for output in outputs:
            paths[output.key].write_text(output.content, encoding=output.encoding)

        return paths

    def _outputs(self, report: ValidationReport) -> tuple[ValidationReportOutput, ...]:
        strategies = self._output_strategies()
        outputs = tuple(strategy.build(report) for strategy in strategies)
        for output in outputs:
            if not isinstance(output, ValidationReportOutput):
                raise TypeError(
                    "validation report output strategies must return "
                    "ValidationReportOutput instances"
                )
        return outputs

    def _output_strategies(self) -> tuple[ValidationReportOutputStrategy, ...]:
        if self.strategies is not None:
            strategies = tuple(self.strategies)
        else:
            strategies = _default_output_strategies(
                report_filename=self.report_filename,
                findings_filename=self.findings_filename,
                summary_filename=self.summary_filename,
            )
        if not strategies:
            raise ValueError("ValidationReportWriter must include output strategies")
        return strategies

    def _resolve_output_paths(
        self,
        output_dir: Path,
        outputs: Iterable[ValidationReportOutput],
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        seen_paths: set[Path] = set()
        for output in outputs:
            if output.key in paths:
                raise ValueError("Validation writer output keys must be distinct")
            path = _safe_child_path(
                output_dir,
                output.filename,
                f"{output.key} filename",
            )
            if path in seen_paths:
                raise ValueError("Validation writer filenames must be distinct")
            paths[output.key] = path
            seen_paths.add(path)
        return paths


ValidationResultWriter = ValidationReportWriter


def _default_output_strategies(
    *,
    report_filename: str,
    findings_filename: str,
    summary_filename: str | None,
) -> tuple[ValidationReportOutputStrategy, ...]:
    strategies: tuple[ValidationReportOutputStrategy, ...] = (
        ValidationReportJsonOutputStrategy(filename=report_filename),
        ValidationFindingsJsonlOutputStrategy(filename=findings_filename),
    )
    if summary_filename is not None:
        strategies = (
            *strategies,
            ValidationSummaryJsonOutputStrategy(filename=summary_filename),
        )
    return strategies


def _summary_severity(report: ValidationReport) -> str:
    if report.status == "fail":
        return "error"
    if report.status == "warn":
        return "warning"
    return "info"
