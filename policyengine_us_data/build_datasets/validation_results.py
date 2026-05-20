"""Durable Stage 1 validation result writing."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    DiagnosticRef,
    ValidationFinding,
    ValidationReport,
)
from policyengine_us_data.utils.step_manifest import sha256_file
from policyengine_us_data.validation_core import ValidationReportWriter


@dataclass(frozen=True, kw_only=True)
class Stage1ValidationSummary:
    """Result of writing Stage 1 validation outputs."""

    report: ValidationReport
    substage_reports: Mapping[str, ValidationReport]
    diagnostics: tuple[DiagnosticRef, ...]
    paths: Mapping[str, Path]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary payload."""

        return {
            "report": self.report.to_dict(),
            "substage_reports": {
                substage_id: report.to_dict()
                for substage_id, report in self.substage_reports.items()
            },
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "paths": {key: str(path) for key, path in self.paths.items()},
        }


@pipeline_node(
    id="stage_1_validation_result_writer",
    label="Stage 1 Validation Result Writer",
    node_type="library",
    description="Write Stage 1 validation reports, findings, metrics, and SQLite rows.",
    source_file="policyengine_us_data/build_datasets/validation_results.py",
    status="current",
    stability="stable",
    pathways=["data_build", "stage_contracts", "cross_stage_validation"],
    artifacts_in=["ValidationReport"],
    artifacts_out=[
        "validation/summary.json",
        "validation/findings.jsonl",
        "validation/metrics.jsonl",
        "validation/validation_results.sqlite",
    ],
    validation_commands=[
        "uv run pytest tests/unit/test_build_dataset_validation_results.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class Stage1ValidationResultWriter:
    """Write durable Stage 1 validation artifacts from canonical reports."""

    output_dir: Path

    def write(
        self,
        reports: Iterable[ValidationReport],
    ) -> Stage1ValidationSummary:
        """Write aggregate and per-substage validation outputs."""

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        substage_reports = _merge_reports_by_substage(tuple(reports))
        aggregate = _aggregate_report(substage_reports.values())

        paths: dict[str, Path] = {}
        for substage_id, report in substage_reports.items():
            writer = ValidationReportWriter(
                output_dir=output_dir,
                strategies=("report",),
                report_filename=f"{substage_id}.json",
            )
            paths[f"{substage_id}.report"] = writer.write(report)["report"]

        paths["summary"] = output_dir / "summary.json"
        paths["summary"].write_text(
            json.dumps(_summary_payload(aggregate, substage_reports), indent=2) + "\n",
            encoding="utf-8",
        )
        paths["findings"] = output_dir / "findings.jsonl"
        paths["findings"].write_text(
            "".join(
                json.dumps(finding.to_dict(), sort_keys=True) + "\n"
                for finding in aggregate.findings
            ),
            encoding="utf-8",
        )
        paths["metrics"] = output_dir / "metrics.jsonl"
        paths["metrics"].write_text(
            "".join(
                json.dumps(_metrics_payload(substage_id, report), sort_keys=True) + "\n"
                for substage_id, report in substage_reports.items()
            ),
            encoding="utf-8",
        )
        paths["sqlite"] = output_dir / "validation_results.sqlite"
        _write_sqlite(paths["sqlite"], substage_reports)

        diagnostics = tuple(
            _diagnostic_ref(name=key, kind=_diagnostic_kind(path), path=path)
            for key, path in paths.items()
        )
        aggregate = ValidationReport(
            status=aggregate.status,
            findings=aggregate.findings,
            diagnostics=diagnostics,
            metadata=aggregate.metadata,
        )
        return Stage1ValidationSummary(
            report=aggregate,
            substage_reports=substage_reports,
            diagnostics=diagnostics,
            paths=paths,
        )


def _merge_reports_by_substage(
    reports: tuple[ValidationReport, ...],
) -> dict[str, ValidationReport]:
    grouped: dict[str, list[ValidationReport]] = defaultdict(list)
    for report in reports:
        substage_id = report.metadata.get("substage_id")
        if isinstance(substage_id, str) and report.status != "not_run":
            grouped[substage_id].append(report)

    merged: dict[str, ValidationReport] = {}
    for substage_id in sorted(grouped):
        findings: list[ValidationFinding] = []
        diagnostics: list[DiagnosticRef] = []
        check_ids: list[str] = []
        for report in grouped[substage_id]:
            findings.extend(report.findings)
            diagnostics.extend(report.diagnostics)
            check_ids.extend(report.metadata.get("check_ids", ()))
        merged[substage_id] = ValidationReport(
            status=_status_from_findings(findings),
            findings=tuple(findings),
            diagnostics=tuple(diagnostics),
            metadata={
                "stage_id": "1_build_datasets",
                "substage_id": substage_id,
                "check_ids": sorted(set(check_ids)),
                "report_count": len(grouped[substage_id]),
            },
        )
    return merged


def _aggregate_report(reports: Iterable[ValidationReport]) -> ValidationReport:
    reports = tuple(reports)
    findings: list[ValidationFinding] = []
    diagnostics: list[DiagnosticRef] = []
    for report in reports:
        findings.extend(report.findings)
        diagnostics.extend(report.diagnostics)
    if not reports:
        return ValidationReport(
            status="not_run",
            metadata={"stage_id": "1_build_datasets", "report_count": 0},
        )
    return ValidationReport(
        status=_status_from_findings(findings),
        findings=tuple(findings),
        diagnostics=tuple(diagnostics),
        metadata={
            "stage_id": "1_build_datasets",
            "report_count": len(reports),
            "substage_ids": [
                report.metadata.get("substage_id")
                for report in reports
                if report.metadata.get("substage_id") is not None
            ],
        },
    )


def _status_from_findings(findings: Iterable[ValidationFinding]) -> str:
    statuses = tuple(finding.status for finding in findings)
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _summary_payload(
    aggregate: ValidationReport,
    substage_reports: Mapping[str, ValidationReport],
) -> dict[str, Any]:
    return {
        "status": aggregate.status,
        "finding_count": len(aggregate.findings),
        "fail_count": sum(
            1 for finding in aggregate.findings if finding.status == "fail"
        ),
        "warn_count": sum(
            1 for finding in aggregate.findings if finding.status == "warn"
        ),
        "pass_count": sum(
            1 for finding in aggregate.findings if finding.status == "pass"
        ),
        "substages": {
            substage_id: _metrics_payload(substage_id, report)
            for substage_id, report in substage_reports.items()
        },
    }


def _metrics_payload(substage_id: str, report: ValidationReport) -> dict[str, Any]:
    return {
        "substage_id": substage_id,
        "status": report.status,
        "finding_count": len(report.findings),
        "fail_count": sum(1 for finding in report.findings if finding.status == "fail"),
        "warn_count": sum(1 for finding in report.findings if finding.status == "warn"),
        "pass_count": sum(1 for finding in report.findings if finding.status == "pass"),
    }


def _write_sqlite(
    path: Path,
    substage_reports: Mapping[str, ValidationReport],
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                substage_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                finding_count INTEGER NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                substage_id TEXT NOT NULL,
                check_id TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                finding_json TEXT NOT NULL
            )
            """
        )
        connection.execute("DELETE FROM findings")
        connection.execute("DELETE FROM reports")
        for substage_id, report in substage_reports.items():
            connection.execute(
                """
                INSERT INTO reports
                    (substage_id, status, finding_count, report_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    substage_id,
                    report.status,
                    len(report.findings),
                    json.dumps(report.to_dict(), sort_keys=True),
                ),
            )
            for finding in report.findings:
                connection.execute(
                    """
                    INSERT INTO findings
                        (substage_id, check_id, status, message, finding_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        substage_id,
                        finding.check_id,
                        finding.status,
                        finding.message,
                        json.dumps(finding.to_dict(), sort_keys=True),
                    ),
                )


def _diagnostic_ref(*, name: str, kind: str, path: Path) -> DiagnosticRef:
    artifact = ArtifactRef(
        logical_name=f"stage_1_validation_{name}",
        uri=path.resolve().as_uri(),
        sha256=f"sha256:{sha256_file(path)}",
        size_bytes=path.stat().st_size,
        media_type=_media_type_for_path(path),
        metadata={"stage_id": "1_build_datasets", "artifact_family": "validation"},
    )
    return DiagnosticRef(
        name=f"stage_1_validation_{name}",
        kind=kind,
        artifact=artifact,
        severity="info",
    )


def _diagnostic_kind(path: Path) -> str:
    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix in {".sqlite", ".db"}:
        return "sqlite"
    return "json"


def _media_type_for_path(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    if path.suffix in {".sqlite", ".db"}:
        return "application/vnd.sqlite3"
    return "application/octet-stream"


__all__ = [
    "Stage1ValidationResultWriter",
    "Stage1ValidationSummary",
]
