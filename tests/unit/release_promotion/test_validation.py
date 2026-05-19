from pathlib import Path
from typing import Any

from policyengine_us_data.release_promotion import (
    BASE_RELEASE_ARTIFACT_PATHS,
    ReleaseCandidateInputBundle,
    ReleaseCandidateValidationDependencies,
    ReleaseCandidateValidator,
    ReleasePromotionContext,
    VALIDATION_REPORT_POLICY_PRESENCE_ONLY,
    infer_release_artifact_spec,
)
from policyengine_us_data.stage_contracts import ValidationFinding, ValidationReport

_H5_RELEASE_PATHS = (
    "national/US.h5",
    "states/AL.h5",
    "districts/NC-01.h5",
    "cities/NYC.h5",
)
_FULL_RELEASE_PATHS = (*BASE_RELEASE_ARTIFACT_PATHS, *_H5_RELEASE_PATHS)


def _context() -> ReleasePromotionContext:
    return ReleasePromotionContext(
        run_id="run-123",
        candidate_version="1.73.0rc1",
        release_version="1.73.0",
        hf_repo_name="policyengine/policyengine-us-data",
        gcs_bucket_name="policyengine-us-data",
    )


def _bundle(
    *,
    paths: tuple[str, ...] = _FULL_RELEASE_PATHS,
    validation_report_paths: tuple[str, ...] = (
        "calibration/runs/run-123/diagnostics/validation_report.json",
    ),
) -> ReleaseCandidateInputBundle:
    return ReleaseCandidateInputBundle(
        context=_context(),
        artifacts=tuple(
            infer_release_artifact_spec(
                path,
                sha256=f"sha256:{path}",
                size_bytes=100,
            )
            for path in paths
        ),
        validation_report_paths=validation_report_paths,
    )


def _manifest_files(
    paths: tuple[str, ...] = _FULL_RELEASE_PATHS,
) -> tuple[tuple[Path, str], ...]:
    return tuple((Path(path), path) for path in paths)


class FakeReleaseCandidateValidationDependencies:
    def __init__(
        self,
        *,
        finalized_manifest: dict[str, Any] | None = None,
        finalized_error: Exception | None = None,
        marker_exists: bool = False,
        missing_staged_artifacts: tuple[str, ...] = (),
        missing_validation_reports: tuple[str, ...] = (),
        validation_reports: tuple[ValidationReport, ...] | None = None,
        validation_report_error: Exception | None = None,
        preflight_result: tuple[bool, list[str]] = (True, []),
    ) -> None:
        self.finalized_manifest = finalized_manifest
        self.finalized_error = finalized_error
        self.marker_exists = marker_exists
        self.missing_staged_artifacts = missing_staged_artifacts
        self.missing_validation_reports = missing_validation_reports
        self.validation_reports = (
            validation_reports
            if validation_reports is not None
            else (ValidationReport(status="pass"),)
        )
        self.validation_report_error = validation_report_error
        self.preflight_result = preflight_result
        self.calls: list[str] = []

    def as_dependencies(self) -> ReleaseCandidateValidationDependencies:
        return ReleaseCandidateValidationDependencies(
            get_matching_finalized_release_manifest=(
                self.get_matching_finalized_release_manifest
            ),
            list_missing_staged_artifacts=self.list_missing_staged_artifacts,
            list_missing_validation_reports=self.list_missing_validation_reports,
            load_validation_reports=self.load_validation_reports,
            preflight_release_manifest_publish=self.preflight_release_manifest_publish,
            release_completion_marker_exists=self.release_completion_marker_exists,
        )

    def get_matching_finalized_release_manifest(self, *args, **kwargs):
        self.calls.append("get_matching_finalized_release_manifest")
        if self.finalized_error is not None:
            raise self.finalized_error
        return self.finalized_manifest

    def list_missing_staged_artifacts(self, *args, **kwargs):
        self.calls.append("list_missing_staged_artifacts")
        return list(self.missing_staged_artifacts)

    def list_missing_validation_reports(self, *args, **kwargs):
        self.calls.append("list_missing_validation_reports")
        return list(self.missing_validation_reports)

    def load_validation_reports(self, *args, **kwargs):
        self.calls.append("load_validation_reports")
        if self.validation_report_error is not None:
            raise self.validation_report_error
        return self.validation_reports

    def preflight_release_manifest_publish(self, *args, **kwargs):
        self.calls.append("preflight_release_manifest_publish")
        return self.preflight_result

    def release_completion_marker_exists(self, *args, **kwargs):
        self.calls.append("release_completion_marker_exists")
        return self.marker_exists


def _validator(
    fake_deps: FakeReleaseCandidateValidationDependencies,
    **kwargs: Any,
) -> ReleaseCandidateValidator:
    return ReleaseCandidateValidator(
        dependencies=fake_deps.as_dependencies(),
        **kwargs,
    )


def _finding(report: ValidationReport, check_id: str) -> ValidationFinding:
    return next(finding for finding in report.findings if finding.check_id == check_id)


def test_release_candidate_validator_passes_complete_candidate() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies()

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    assert isinstance(report, ValidationReport)
    assert report.status == "pass"
    assert [finding.status for finding in report.findings] == ["pass"] * 6
    assert report.metadata["suite_id"] == "release_candidate_validation"
    assert report.metadata["substage_id"] == "5a_validate_outputs"
    assert fake_deps.calls == [
        "get_matching_finalized_release_manifest",
        "list_missing_staged_artifacts",
        "list_missing_validation_reports",
        "load_validation_reports",
        "preflight_release_manifest_publish",
    ]


def test_release_candidate_validator_reports_missing_required_families() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies()

    report = _validator(fake_deps).validate(
        _bundle(
            paths=(
                *BASE_RELEASE_ARTIFACT_PATHS,
                "national/US.h5",
                "states/AL.h5",
                "districts/NC-01.h5",
            )
        ),
        files_with_paths=_manifest_files(
            (
                *BASE_RELEASE_ARTIFACT_PATHS,
                "national/US.h5",
                "states/AL.h5",
                "districts/NC-01.h5",
            )
        ),
    )

    finding = _finding(report, "release_candidate_required_artifact_families")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == ("city_h5",)


def test_release_candidate_validator_reports_missing_base_artifacts() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies()

    report = _validator(fake_deps).validate(
        _bundle(paths=_H5_RELEASE_PATHS),
        files_with_paths=_manifest_files(_H5_RELEASE_PATHS),
    )

    finding = _finding(report, "release_candidate_required_base_artifacts")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == BASE_RELEASE_ARTIFACT_PATHS


def test_release_candidate_validator_reports_missing_staged_artifacts() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies(
        missing_staged_artifacts=("staging/1.73.0rc1-run-123/states/AL.h5",),
    )

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_staged_artifacts_present")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == ("staging/1.73.0rc1-run-123/states/AL.h5",)


def test_release_candidate_validator_reports_missing_validation_reports() -> None:
    missing_report = "calibration/runs/run-123/diagnostics/validation_report.json"
    fake_deps = FakeReleaseCandidateValidationDependencies(
        missing_validation_reports=(missing_report,),
    )

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_validation_reports_present")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == (missing_report,)
    assert "load_validation_reports" not in fake_deps.calls


def test_release_candidate_validator_requires_validation_report_paths() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies()

    report = _validator(fake_deps).validate(
        _bundle(validation_report_paths=()),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_validation_reports_present")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == ()
    assert "list_missing_validation_reports" not in fake_deps.calls
    assert "load_validation_reports" not in fake_deps.calls


def test_release_candidate_validator_rejects_failing_validation_report() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies(
        validation_reports=(
            ValidationReport(
                status="fail",
                findings=(
                    ValidationFinding(
                        check_id="stage4_check",
                        status="fail",
                        message="Stage 4 output validation failed.",
                    ),
                ),
            ),
        ),
    )

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_validation_reports_present")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == ("fail",)
    assert "load_validation_reports" in fake_deps.calls


def test_release_candidate_validator_reports_validation_report_load_errors() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies(
        validation_report_error=RuntimeError("report unavailable"),
    )

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_validation_reports_present")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == "RuntimeError"


def test_release_candidate_validator_allows_presence_only_report_policy() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies(
        validation_reports=(ValidationReport(status="fail"),),
    )

    report = _validator(
        fake_deps,
        validation_report_policy=VALIDATION_REPORT_POLICY_PRESENCE_ONLY,
    ).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_validation_reports_present")
    assert report.status == "pass"
    assert finding.status == "pass"
    assert finding.metadata["validation_report_policy"] == "presence_only"
    assert "load_validation_reports" not in fake_deps.calls


def test_release_candidate_validator_rejects_unknown_report_policy() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies()

    try:
        _validator(fake_deps, validation_report_policy="skip_everything")
    except ValueError as exc:
        assert "validation_report_policy" in str(exc)
    else:
        raise AssertionError("Expected invalid validation_report_policy to fail")


def test_release_candidate_validator_reports_incomplete_local_area_prefixes() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies(
        preflight_result=(False, ["districts/"]),
    )

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_release_manifest_preflight")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == ("districts/",)


def test_release_candidate_validator_requires_manifest_files_for_preflight() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies()

    report = _validator(fake_deps).validate(_bundle())

    finalized_finding = _finding(
        report,
        "release_candidate_finalized_release_state",
    )
    preflight_finding = _finding(
        report,
        "release_candidate_release_manifest_preflight",
    )
    assert report.status == "fail"
    assert finalized_finding.status == "fail"
    assert preflight_finding.status == "fail"
    assert "get_matching_finalized_release_manifest" not in fake_deps.calls


def test_release_candidate_validator_accepts_finalized_release_with_marker() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies(
        finalized_manifest={"artifacts": {"national": {"path": "national/US.h5"}}},
        marker_exists=True,
    )

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    assert report.status == "pass"
    assert _finding(report, "release_candidate_finalized_release_state").value is True
    assert "release_completion_marker_exists" in fake_deps.calls
    assert "list_missing_staged_artifacts" not in fake_deps.calls
    assert "list_missing_validation_reports" not in fake_deps.calls
    assert "preflight_release_manifest_publish" not in fake_deps.calls


def test_release_candidate_validator_rejects_finalized_release_without_marker() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies(
        finalized_manifest={"artifacts": {"national": {"path": "national/US.h5"}}},
        marker_exists=False,
    )

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_finalized_release_state")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value is False
    assert "list_missing_staged_artifacts" not in fake_deps.calls
    assert "preflight_release_manifest_publish" not in fake_deps.calls


def test_release_candidate_validator_reports_finalized_lookup_errors() -> None:
    fake_deps = FakeReleaseCandidateValidationDependencies(
        finalized_error=RuntimeError("finalized manifest unavailable"),
    )

    report = _validator(fake_deps).validate(
        _bundle(),
        files_with_paths=_manifest_files(),
    )

    finding = _finding(report, "release_candidate_finalized_release_state")
    assert report.status == "fail"
    assert finding.status == "fail"
    assert finding.value == "RuntimeError"
    assert "list_missing_staged_artifacts" not in fake_deps.calls
    assert "preflight_release_manifest_publish" not in fake_deps.calls
