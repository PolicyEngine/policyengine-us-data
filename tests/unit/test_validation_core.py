import json

import pytest

from policyengine_us_data.stage_contracts import (
    ArtifactRef,
    DiagnosticRef,
    ValidationFinding,
    ValidationReport,
)
from policyengine_us_data.validation_core import (
    ValidationFindingsJsonlOutputStrategy,
    ValidationReportJsonOutputStrategy,
    ValidationReportOutput,
    ValidationReportWriter,
    ValidationArtifactResolver,
    ValidationCheck,
    ValidationContext,
    ValidationRunner,
    ValidationSuite,
    ValidationSummaryJsonOutputStrategy,
)


def _artifact(logical_name: str = "calibration_package") -> ArtifactRef:
    return ArtifactRef(
        logical_name=logical_name,
        uri=f"file:///tmp/{logical_name}.json",
        sha256="sha256:artifact",
        media_type="application/json",
    )


def _resolver() -> ValidationArtifactResolver:
    artifact = _artifact()
    return ValidationArtifactResolver(artifacts={artifact.logical_name: artifact})


def _context() -> ValidationContext:
    return ValidationContext(
        run_id="run-123",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        resolver=_resolver(),
        metadata={"fixture": True},
    )


def _check(
    check_id: str,
    status: str = "pass",
    *,
    required_artifacts: tuple[str, ...] = ("calibration_package",),
    severity: str = "error",
) -> ValidationCheck:
    return ValidationCheck(
        check_id=check_id,
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description=f"{check_id} check",
        severity=severity,
        required_artifacts=required_artifacts,
        run=lambda _context: ValidationFinding(
            check_id=check_id,
            status=status,
            message=f"{check_id} {status}",
        ),
    )


def _suite(*checks: ValidationCheck) -> ValidationSuite:
    return ValidationSuite(
        suite_id="stage_2_validation",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        checks=checks,
    )


def test_artifact_resolver_returns_required_and_optional_artifacts():
    artifact = _artifact("weights")
    resolver = ValidationArtifactResolver(artifacts={"weights": artifact})

    assert resolver.require("weights") == artifact
    assert resolver.optional("weights") == artifact
    assert resolver.optional("missing") is None


def test_artifact_resolver_missing_required_artifact_raises_key_error():
    resolver = ValidationArtifactResolver(artifacts={})

    with pytest.raises(KeyError, match="Missing required validation artifact"):
        resolver.require("weights")


def test_artifact_resolver_rejects_mismatched_mapping_key_and_logical_name():
    with pytest.raises(ValueError, match="does not match resolver key"):
        ValidationArtifactResolver(artifacts={"alias": _artifact("weights")})


def test_validation_check_identity_validation_rejects_empty_check_id():
    with pytest.raises(ValueError, match="check_id"):
        ValidationCheck(
            check_id="",
            stage_id="2_build_calibration_package",
            description="empty check id",
            run=lambda _context: None,
        )


def test_validation_check_rejects_bare_string_required_artifacts():
    with pytest.raises(TypeError, match="required_artifacts"):
        ValidationCheck(
            check_id="bad_artifacts",
            stage_id="2_build_calibration_package",
            description="invalid artifact declaration",
            required_artifacts="weights",
            run=lambda _context: None,
        )


def test_validation_check_rejects_invalid_stage_id():
    with pytest.raises(ValueError, match="Invalid canonical stage_id"):
        ValidationCheck(
            check_id="bad_stage",
            stage_id="not_a_stage",
            description="invalid stage",
            run=lambda _context: None,
        )


def test_validation_context_rejects_invalid_substage_id_for_stage():
    with pytest.raises(ValueError, match="Invalid canonical substage_id"):
        ValidationContext(
            run_id="run-123",
            stage_id="2_build_calibration_package",
            substage_id="1a_raw_data_download",
            resolver=_resolver(),
        )


def test_validation_suite_rejects_empty_check_list():
    with pytest.raises(ValueError, match="at least one check"):
        ValidationSuite(
            suite_id="empty",
            stage_id="2_build_calibration_package",
        )


def test_validation_suite_rejects_duplicate_check_ids():
    with pytest.raises(ValueError, match="Duplicate validation check_id"):
        _suite(_check("duplicate"), _check("duplicate"))


def test_validation_suite_rejects_substage_check_in_stage_scoped_suite():
    with pytest.raises(ValueError, match="substage-scoped"):
        ValidationSuite(
            suite_id="stage_scoped",
            stage_id="2_build_calibration_package",
            checks=(_check("substage_check"),),
        )


def test_validation_context_deep_freezes_metadata():
    source_metadata = {"nested": {"items": ["a"]}}
    context = ValidationContext(
        run_id="run-123",
        stage_id="2_build_calibration_package",
        resolver=_resolver(),
        metadata=source_metadata,
    )

    source_metadata["nested"]["items"].append("changed")

    assert context.metadata["nested"]["items"] == ("a",)
    with pytest.raises(TypeError):
        context.metadata["nested"]["items"][0] = "blocked"
    with pytest.raises(TypeError):
        context.metadata["nested"]["new"] = "blocked"


def test_runner_converts_one_passing_check_to_pass_report():
    report = ValidationRunner().run(_suite(_check("artifact_exists")), _context())

    assert report.status == "pass"
    assert [finding.status for finding in report.findings] == ["pass"]
    assert report.metadata["suite_id"] == "stage_2_validation"
    assert report.metadata["run_id"] == "run-123"
    assert report.metadata["context_metadata"]["fixture"] is True


def test_runner_converts_warnings_to_warn_report():
    report = ValidationRunner().run(
        _suite(_check("target_warning", "warn")), _context()
    )

    assert report.status == "warn"
    assert report.findings[0].check_id == "target_warning"


def test_runner_converts_failing_findings_to_fail_report():
    report = ValidationRunner().run(_suite(_check("target_error", "fail")), _context())

    assert report.status == "fail"
    assert report.findings[0].status == "fail"


def test_runner_preserves_empty_child_report_fail_status_as_failure_finding():
    check = ValidationCheck(
        check_id="child_report_fail",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns an empty fail report",
        run=lambda _context: ValidationReport(status="fail"),
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "fail"
    assert report.findings[0].metadata["failure_type"] == "child_report_status"
    assert report.findings[0].metadata["child_report_status"] == "fail"


def test_runner_preserves_empty_child_report_warn_status_as_warning_finding():
    check = ValidationCheck(
        check_id="child_report_warn",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns an empty warn report",
        run=lambda _context: ValidationReport(status="warn"),
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "warn"
    assert report.findings[0].metadata["child_report_status"] == "warn"


def test_runner_converts_child_not_run_report_to_generated_status():
    check = ValidationCheck(
        check_id="child_report_not_run",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns an empty not_run report",
        severity="warning",
        run=lambda _context: ValidationReport(status="not_run"),
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "fail"
    assert report.findings[0].status == "fail"
    assert report.findings[0].metadata["child_report_status"] == "not_run"


def test_runner_converts_exceptions_to_failure_findings_with_check_metadata():
    def raise_error(_context: ValidationContext) -> ValidationFinding:
        raise RuntimeError("boom")

    check = ValidationCheck(
        check_id="raises",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="raises an exception",
        run=raise_error,
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "fail"
    assert report.findings[0].check_id == "raises"
    assert report.findings[0].metadata["failure_type"] == "exception"
    assert report.findings[0].metadata["exception_type"] == "RuntimeError"


def test_runner_converts_warning_check_exceptions_to_warning_findings():
    def raise_error(_context: ValidationContext) -> ValidationFinding:
        raise RuntimeError("boom")

    check = ValidationCheck(
        check_id="warning_raises",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="raises an exception",
        severity="warning",
        run=raise_error,
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "fail"
    assert report.findings[0].status == "fail"
    assert report.findings[0].metadata["severity"] == "warning"


def test_runner_redacts_and_bounds_exception_findings(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "secret-value")
    long_message = (
        "old exception "
        + ("x" * 30_000)
        + " newest secret-value API_TOKEN=secret-value"
    )

    def raise_error(_context: ValidationContext) -> ValidationFinding:
        raise RuntimeError(long_message)

    check = ValidationCheck(
        check_id="redacted_exception",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="raises a long exception with a secret",
        run=raise_error,
    )

    report = ValidationRunner().run(_suite(check), _context())
    finding = report.findings[0]

    assert report.status == "fail"
    assert "secret-value" not in finding.message
    assert "secret-value" not in finding.metadata["traceback"]
    assert "<redacted:API_TOKEN>" in finding.message
    assert "API_TOKEN=<redacted>" in finding.message
    assert len(finding.message) <= 2_000
    assert finding.metadata["traceback_truncated"] is True
    assert "old exception" not in finding.metadata["traceback"]
    assert "newest <redacted:API_TOKEN>" in finding.metadata["traceback"]


def test_runner_reports_missing_required_artifacts_as_failures():
    report = ValidationRunner().run(
        _suite(_check("requires_missing", required_artifacts=("missing",))),
        _context(),
    )

    assert report.status == "fail"
    assert report.findings[0].metadata["failure_type"] == "missing_required_artifact"
    assert report.findings[0].value == "missing"


def test_runner_reports_missing_required_artifacts_as_warnings_for_warning_checks():
    report = ValidationRunner().run(
        _suite(
            _check(
                "warning_requires_missing",
                required_artifacts=("missing",),
                severity="warning",
            )
        ),
        _context(),
    )

    assert report.status == "warn"
    assert report.findings[0].status == "warn"
    assert report.findings[0].metadata["severity"] == "warning"


def test_runner_converts_invalid_check_result_to_canonical_finding():
    check = ValidationCheck(
        check_id="invalid_result",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns the wrong type",
        run=lambda _context: object(),
    )

    report = ValidationRunner().run(_suite(check, _check("still_runs")), _context())

    assert report.status == "fail"
    assert report.findings[0].check_id == "invalid_result"
    assert report.findings[0].metadata["failure_type"] == "invalid_check_result"
    assert report.findings[1].check_id == "still_runs"


def test_runner_converts_invalid_iterable_result_to_canonical_finding():
    check = ValidationCheck(
        check_id="invalid_iterable",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns an invalid iterable",
        run=lambda _context: ("not a finding",),
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "fail"
    assert report.findings[0].metadata["failure_type"] == "invalid_check_result"
    assert report.findings[0].metadata["invalid_result_type"] == "str"


def test_runner_invalid_warning_check_result_still_fails():
    check = ValidationCheck(
        check_id="warning_invalid_result",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns the wrong type",
        severity="warning",
        run=lambda _context: object(),
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "fail"
    assert report.findings[0].status == "fail"
    assert report.findings[0].metadata["severity"] == "warning"


def test_runner_converts_generator_exceptions_to_canonical_finding():
    def bad_generator():
        yield ValidationFinding(
            check_id="generator_error",
            status="pass",
            message="before error",
        )
        raise RuntimeError("generator boom")

    check = ValidationCheck(
        check_id="generator_error",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns a generator that raises",
        run=lambda _context: bad_generator(),
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "fail"
    assert report.findings[0].metadata["failure_type"] == "exception"
    assert report.findings[0].metadata["exception_type"] == "RuntimeError"


def test_runner_converts_mismatched_returned_check_id_to_canonical_finding():
    check = ValidationCheck(
        check_id="executed_check",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns the wrong check id",
        run=lambda _context: ValidationFinding(
            check_id="other_check",
            status="pass",
            message="wrong identity",
        ),
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "fail"
    assert report.findings[0].check_id == "executed_check"
    assert report.findings[0].metadata["failure_type"] == "invalid_check_result"
    assert report.findings[0].metadata["invalid_finding_check_id"] == "other_check"


def test_runner_skip_reason_still_validates_suite_and_context_match():
    suite = ValidationSuite(
        suite_id="stage_3_validation",
        stage_id="3_fit_weights",
        checks=(
            ValidationCheck(
                check_id="stage_3_check",
                stage_id="3_fit_weights",
                description="stage 3 check",
                run=lambda _context: None,
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not match"):
        ValidationRunner().run(suite, _context(), skip_reason="reused")


def test_runner_rejects_stage_scoped_suite_with_substage_context():
    suite = ValidationSuite(
        suite_id="stage_2_stage_scoped",
        stage_id="2_build_calibration_package",
        checks=(
            ValidationCheck(
                check_id="stage_scoped_check",
                stage_id="2_build_calibration_package",
                description="stage-scoped check",
                run=lambda _context: None,
            ),
        ),
    )

    with pytest.raises(ValueError, match="Context substage_id"):
        ValidationRunner().run(suite, _context())


def test_runner_preserves_check_order():
    report = ValidationRunner().run(
        _suite(_check("first"), _check("second", "warn"), _check("third")),
        _context(),
    )

    assert [finding.check_id for finding in report.findings] == [
        "first",
        "second",
        "third",
    ]


def test_runner_collects_diagnostics_from_check_reports():
    diagnostic = DiagnosticRef(name="summary", kind="json", summary={"rows": 3})

    check = ValidationCheck(
        check_id="returns_report",
        stage_id="2_build_calibration_package",
        substage_id="2a_matrix_build_calibration_target_construction",
        description="returns an embedded report",
        run=lambda _context: ValidationReport(
            status="pass",
            findings=(
                ValidationFinding(
                    check_id="returns_report",
                    status="pass",
                    message="ok",
                ),
            ),
            diagnostics=(diagnostic,),
        ),
    )

    report = ValidationRunner().run(_suite(check), _context())

    assert report.status == "pass"
    assert report.diagnostics == (diagnostic,)


def test_result_writer_emits_report_json_and_jsonl_findings(tmp_path):
    report = ValidationRunner().run(
        _suite(_check("first"), _check("second", "warn")),
        _context(),
    )
    paths = ValidationReportWriter(
        output_dir=tmp_path,
        summary_filename="summary.json",
    ).write(report)

    restored = ValidationReport.from_dict(
        json.loads(paths["report"].read_text(encoding="utf-8"))
    )
    finding_rows = [
        json.loads(line)
        for line in paths["findings"].read_text(encoding="utf-8").splitlines()
    ]
    summary = DiagnosticRef.from_dict(
        json.loads(paths["summary"].read_text(encoding="utf-8"))
    )

    assert restored == report
    assert [row["check_id"] for row in finding_rows] == ["first", "second"]
    assert summary.severity == "warning"
    assert summary.summary["status"] == "warn"
    assert summary.summary["finding_count"] == 2


def test_validation_report_output_round_trips_through_stage_contract_schema(tmp_path):
    report = ValidationRunner().run(_suite(_check("artifact_exists")), _context())
    paths = ValidationReportWriter(output_dir=tmp_path).write(report)

    restored = ValidationReport.from_dict(
        json.loads(paths["report"].read_text(encoding="utf-8"))
    )

    assert restored == report


@pytest.mark.parametrize(
    "field,filename",
    [
        ("report_filename", "../validation_report.json"),
        ("findings_filename", "nested/validation_findings.jsonl"),
        ("summary_filename", "/tmp/validation_summary.json"),
    ],
)
def test_result_writer_rejects_filenames_outside_output_dir(tmp_path, field, filename):
    report = ValidationRunner().run(_suite(_check("artifact_exists")), _context())
    writer = ValidationReportWriter(output_dir=tmp_path, **{field: filename})

    with pytest.raises(ValueError, match="plain filename"):
        writer.write(report)


def test_result_writer_rejects_duplicate_output_filenames(tmp_path):
    report = ValidationRunner().run(_suite(_check("artifact_exists")), _context())
    writer = ValidationReportWriter(
        output_dir=tmp_path,
        summary_filename="validation_report.json",
    )

    with pytest.raises(ValueError, match="distinct"):
        writer.write(report)
    assert list(tmp_path.iterdir()) == []


class _StatusTextOutputStrategy:
    def build(self, report: ValidationReport) -> ValidationReportOutput:
        return ValidationReportOutput(
            key="status",
            filename="status.txt",
            content=f"{report.status}\n",
        )


def test_result_writer_accepts_custom_output_strategy(tmp_path):
    report = ValidationRunner().run(_suite(_check("artifact_exists")), _context())
    writer = ValidationReportWriter(
        output_dir=tmp_path,
        strategies=(
            ValidationReportJsonOutputStrategy(),
            _StatusTextOutputStrategy(),
        ),
    )

    paths = writer.write(report)

    assert set(paths) == {"report", "status"}
    assert paths["status"].read_text(encoding="utf-8") == "pass\n"
    assert paths["report"].name == "validation_report.json"


def test_result_writer_rejects_duplicate_output_keys(tmp_path):
    report = ValidationRunner().run(_suite(_check("artifact_exists")), _context())
    writer = ValidationReportWriter(
        output_dir=tmp_path,
        strategies=(
            ValidationReportJsonOutputStrategy(key="report"),
            ValidationFindingsJsonlOutputStrategy(key="report"),
        ),
    )

    with pytest.raises(ValueError, match="output keys"):
        writer.write(report)
    assert list(tmp_path.iterdir()) == []


def test_summary_strategy_can_be_composed_explicitly(tmp_path):
    report = ValidationRunner().run(
        _suite(_check("first"), _check("second", "warn")),
        _context(),
    )
    writer = ValidationReportWriter(
        output_dir=tmp_path,
        strategies=(
            ValidationReportJsonOutputStrategy(),
            ValidationFindingsJsonlOutputStrategy(),
            ValidationSummaryJsonOutputStrategy(filename="summary.json"),
        ),
    )

    paths = writer.write(report)
    summary = DiagnosticRef.from_dict(
        json.loads(paths["summary"].read_text(encoding="utf-8"))
    )

    assert summary.severity == "warning"
    assert summary.summary["finding_count"] == 2
