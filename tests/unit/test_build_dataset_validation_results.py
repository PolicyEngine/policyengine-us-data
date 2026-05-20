import json
import sqlite3

from policyengine_us_data.build_datasets import Stage1ValidationResultWriter
from policyengine_us_data.stage_contracts import ValidationFinding, ValidationReport


def _report(*, substage_id: str, status: str = "fail") -> ValidationReport:
    findings = ()
    if status == "fail":
        findings = (
            ValidationFinding(
                check_id=f"stage_1.{substage_id}.artifact_contract",
                status="fail",
                message="missing artifact",
                metric="artifact_exists",
                value="missing",
            ),
        )
    return ValidationReport(
        status=status,
        findings=findings,
        metadata={
            "stage_id": "1_build_datasets",
            "substage_id": substage_id,
            "check_ids": [f"stage_1.{substage_id}.artifact_contract"],
        },
    )


def test_stage_1_validation_result_writer_writes_queryable_outputs(tmp_path):
    summary = Stage1ValidationResultWriter(output_dir=tmp_path).write(
        [
            _report(substage_id="1b_base_dataset_construction"),
            _report(substage_id="1c_extended_cps_puf_clone", status="pass"),
        ]
    )

    assert summary.report.status == "fail"
    assert (tmp_path / "1b_base_dataset_construction.json").exists()
    assert (tmp_path / "1c_extended_cps_puf_clone.json").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "fail"
    assert json.loads((tmp_path / "findings.jsonl").read_text())["metric"] == (
        "artifact_exists"
    )
    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert {row["substage_id"] for row in metrics} == {
        "1b_base_dataset_construction",
        "1c_extended_cps_puf_clone",
    }

    with sqlite3.connect(tmp_path / "validation_results.sqlite") as connection:
        rows = connection.execute(
            "SELECT substage_id, status FROM reports ORDER BY substage_id"
        ).fetchall()
    assert rows == [
        ("1b_base_dataset_construction", "fail"),
        ("1c_extended_cps_puf_clone", "pass"),
    ]
    assert {diagnostic.kind for diagnostic in summary.diagnostics} >= {
        "json",
        "jsonl",
        "sqlite",
    }


def test_stage_1_validation_result_writer_handles_no_reports(tmp_path):
    summary = Stage1ValidationResultWriter(output_dir=tmp_path).write([])

    assert summary.report.status == "not_run"
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "not_run"
    assert (tmp_path / "validation_results.sqlite").exists()
