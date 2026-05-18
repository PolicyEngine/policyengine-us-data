"""Writers for canonical validation reports and findings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import DiagnosticRef, ValidationReport


def _safe_child_path(output_dir: Path, filename: str, field_name: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{field_name} must be a non-empty filename")
    path = Path(filename)
    if path.is_absolute() or path.name != filename or ".." in path.parts:
        raise ValueError(f"{field_name} must be a plain filename under output_dir")
    return output_dir / path


@pipeline_node(
    id="validation_core_result_writer",
    label="ValidationResultWriter",
    node_type="library",
    description="Write canonical ValidationReport JSON and JSONL finding artifacts to explicit local paths.",
    status="current",
    stability="stable",
    pathways=["cross_stage_validation"],
    artifacts_in=["ValidationReport"],
    artifacts_out=["validation_report.json", "validation_findings.jsonl"],
    validation_commands=["uv run pytest tests/unit/test_validation_core.py"],
)
@dataclass(frozen=True, kw_only=True)
class ValidationResultWriter:
    """Write validation reports to explicit local output paths."""

    output_dir: Path
    report_filename: str = "validation_report.json"
    findings_filename: str = "validation_findings.jsonl"
    summary_filename: str | None = None

    def write(self, report: ValidationReport) -> dict[str, Path]:
        """Write a report JSON file, JSONL findings, and optional summary JSON."""

        if not isinstance(report, ValidationReport):
            raise TypeError("report must be a ValidationReport")

        output_dir = Path(self.output_dir)

        report_path = _safe_child_path(
            output_dir, self.report_filename, "report_filename"
        )
        findings_path = _safe_child_path(
            output_dir,
            self.findings_filename,
            "findings_filename",
        )
        paths = {"report": report_path, "findings": findings_path}
        if self.summary_filename is not None:
            paths["summary"] = _safe_child_path(
                output_dir,
                self.summary_filename,
                "summary_filename",
            )
        if len(set(paths.values())) != len(paths):
            raise ValueError("Validation writer filenames must be distinct")

        output_dir.mkdir(parents=True, exist_ok=True)

        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        findings_path.write_text(
            "".join(
                json.dumps(finding.to_dict(), sort_keys=True) + "\n"
                for finding in report.findings
            ),
            encoding="utf-8",
        )

        if self.summary_filename is not None:
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
            paths["summary"].write_text(
                json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return paths


def _summary_severity(report: ValidationReport) -> str:
    if report.status == "fail":
        return "error"
    if report.status == "warn":
        return "warning"
    return "info"
