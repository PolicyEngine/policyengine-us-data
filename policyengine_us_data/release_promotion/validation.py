"""Canonical validation-report adapters for Stage 5 release candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, TypeAlias

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.stage_contracts import ValidationFinding, ValidationReport
from policyengine_us_data.stage_contracts.stages import (
    STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
)
from policyengine_us_data.utils.error_redaction import (
    DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    redacted_bounded_error_text,
)
from policyengine_us_data.validation_core import (
    ValidationArtifactResolver,
    ValidationCheck,
    ValidationContext,
    ValidationRunner,
    ValidationSuite,
)

from .artifacts import BASE_RELEASE_ARTIFACT_PATHS
from .candidate import ReleaseCandidateInputBundle

ManifestFile: TypeAlias = tuple[Path | str, str]

RELEASE_VALIDATION_SUBSTAGE_ID = "5a_validate_outputs"
VALIDATION_REPORT_POLICY_REQUIRE_PASSING = "require_passing"
VALIDATION_REPORT_POLICY_PRESENCE_ONLY = "presence_only"
VALIDATION_REPORT_POLICIES = frozenset(
    {
        VALIDATION_REPORT_POLICY_REQUIRE_PASSING,
        VALIDATION_REPORT_POLICY_PRESENCE_ONLY,
    }
)
DEFAULT_REQUIRED_RELEASE_ARTIFACT_FAMILIES = (
    "national_h5",
    "state_h5",
    "district_h5",
    "city_h5",
)


@dataclass(frozen=True, kw_only=True)
class ReleaseCandidateValidationDependencies:
    """Side-effecting release checks used by ``ReleaseCandidateValidator``."""

    get_matching_finalized_release_manifest: Callable[..., Mapping[str, Any] | None]
    list_missing_staged_artifacts: Callable[..., Sequence[str]]
    list_missing_validation_reports: Callable[..., Sequence[str]]
    load_validation_reports: Callable[..., Sequence[ValidationReport]]
    preflight_release_manifest_publish: Callable[..., tuple[bool, Sequence[str]]]
    release_completion_marker_exists: Callable[..., bool]


@dataclass(frozen=True, kw_only=True)
class _FinalizedReleaseState:
    manifest: Mapping[str, Any] | None = None
    error: Exception | None = None
    checked: bool = False


def default_release_candidate_validation_dependencies() -> (
    ReleaseCandidateValidationDependencies
):
    """Return production adapters for Stage 5 candidate validation checks."""

    from policyengine_us_data.utils import data_upload

    return ReleaseCandidateValidationDependencies(
        get_matching_finalized_release_manifest=(
            data_upload.get_matching_finalized_release_manifest
        ),
        list_missing_staged_artifacts=data_upload.list_missing_staged_artifacts,
        list_missing_validation_reports=_list_missing_validation_reports_on_hf,
        load_validation_reports=_load_validation_reports_from_hf,
        preflight_release_manifest_publish=(
            data_upload.preflight_release_manifest_publish
        ),
        release_completion_marker_exists=data_upload.release_completion_marker_exists_on_hf,
    )


@pipeline_node(
    id="release_candidate_shape_report",
    label="Release Candidate Shape Report",
    node_type="validation",
    description="Adapt Stage 5 release-candidate shape checks into the canonical validation report schema.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_candidate.py"
    ],
)
def build_release_candidate_shape_report(
    bundle: ReleaseCandidateInputBundle,
) -> ValidationReport:
    """Describe candidate-bundle shape using the shared validation schema."""

    families = Counter(artifact.artifact_family for artifact in bundle.artifacts)
    return ValidationReport(
        status="pass",
        findings=(
            ValidationFinding(
                check_id="release_candidate_identity_declared",
                status="pass",
                message="Release candidate declares one canonical run and release identity.",
                metadata={
                    "run_id": bundle.context.run_id,
                    "candidate_version": bundle.context.candidate_version,
                    "release_version": bundle.context.release_version,
                    "hf_staging_prefix": bundle.context.hf_staging_prefix,
                    "release_candidate_fingerprint": (
                        bundle.release_candidate_fingerprint
                    ),
                },
            ),
            ValidationFinding(
                check_id="release_candidate_artifacts_declared",
                status="pass",
                message="Release candidate declares typed release artifacts.",
                metric="artifact_count",
                value=len(bundle.artifacts),
                threshold=1,
                metadata={
                    "artifact_families": dict(sorted(families.items())),
                    "required_artifacts": sum(
                        1 for artifact in bundle.artifacts if artifact.required
                    ),
                },
            ),
        ),
        metadata={
            "stage_id": STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
            "substage_id": RELEASE_VALIDATION_SUBSTAGE_ID,
            "run_id": bundle.context.run_id,
            "release_candidate_fingerprint": bundle.release_candidate_fingerprint,
            "validation_kind": "candidate_shape",
        },
    )


@pipeline_node(
    id="release_candidate_validator",
    label="ReleaseCandidateValidator",
    node_type="validation",
    description="Stage 5 validation-core adapter for release candidates before promotion side effects.",
    status="transitional",
    stability="moving",
    pathways=["5_validate_and_promote_release"],
    artifacts_in=["ReleaseCandidateInputBundle", "staged release artifacts"],
    artifacts_out=["ValidationReport"],
    validation_commands=[
        "uv run pytest tests/unit/release_promotion/test_validation.py"
    ],
)
@dataclass(frozen=True, kw_only=True)
class ReleaseCandidateValidator:
    """Validate a Stage 5 release candidate before public release writes."""

    dependencies: ReleaseCandidateValidationDependencies = field(
        default_factory=default_release_candidate_validation_dependencies,
    )
    required_artifact_families: tuple[str, ...] = (
        DEFAULT_REQUIRED_RELEASE_ARTIFACT_FAMILIES
    )
    required_base_artifact_paths: tuple[str, ...] = BASE_RELEASE_ARTIFACT_PATHS
    validation_report_policy: str = VALIDATION_REPORT_POLICY_REQUIRE_PASSING
    runner: ValidationRunner = field(default_factory=ValidationRunner)

    def __post_init__(self) -> None:
        if self.validation_report_policy not in VALIDATION_REPORT_POLICIES:
            raise ValueError(
                "validation_report_policy must be one of: "
                + ", ".join(sorted(VALIDATION_REPORT_POLICIES))
            )
        object.__setattr__(
            self,
            "required_artifact_families",
            tuple(self.required_artifact_families),
        )
        object.__setattr__(
            self,
            "required_base_artifact_paths",
            tuple(self.required_base_artifact_paths),
        )

    def validate(
        self,
        bundle: ReleaseCandidateInputBundle,
        *,
        files_with_paths: Sequence[ManifestFile] = (),
    ) -> ValidationReport:
        """Run Stage 5 candidate checks and return a canonical report."""

        if not isinstance(bundle, ReleaseCandidateInputBundle):
            raise TypeError("bundle must be a ReleaseCandidateInputBundle")
        files = tuple(files_with_paths)
        finalized_state = self._matching_finalized_release(bundle, files)
        suite = self._validation_suite(
            bundle=bundle,
            files_with_paths=files,
            finalized_state=finalized_state,
        )
        return self.runner.run(suite, _validation_context(bundle, files))

    def _matching_finalized_release(
        self,
        bundle: ReleaseCandidateInputBundle,
        files_with_paths: Sequence[ManifestFile],
    ) -> _FinalizedReleaseState:
        if not files_with_paths:
            return _FinalizedReleaseState()
        try:
            manifest = self.dependencies.get_matching_finalized_release_manifest(
                files_with_paths=list(files_with_paths),
                version=bundle.context.release_version,
                hf_repo_name=bundle.context.hf_repo_name,
                hf_repo_type=bundle.context.hf_repo_type,
                model_package_name="policyengine-us",
            )
        except Exception as exc:
            return _FinalizedReleaseState(error=exc, checked=True)
        return _FinalizedReleaseState(manifest=manifest, checked=True)

    def _validation_suite(
        self,
        *,
        bundle: ReleaseCandidateInputBundle,
        files_with_paths: Sequence[ManifestFile],
        finalized_state: _FinalizedReleaseState,
    ) -> ValidationSuite:
        return ValidationSuite(
            suite_id="release_candidate_validation",
            stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
            substage_id=RELEASE_VALIDATION_SUBSTAGE_ID,
            checks=(
                ValidationCheck(
                    check_id="release_candidate_required_artifact_families",
                    stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
                    substage_id=RELEASE_VALIDATION_SUBSTAGE_ID,
                    description="Required release artifact families are present.",
                    run=lambda context: self._check_required_artifact_families(
                        bundle,
                    ),
                ),
                ValidationCheck(
                    check_id="release_candidate_finalized_release_state",
                    stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
                    substage_id=RELEASE_VALIDATION_SUBSTAGE_ID,
                    description="Already-finalized releases have a completion marker.",
                    run=lambda context: self._check_finalized_release_state(
                        bundle,
                        finalized_state,
                    ),
                ),
                ValidationCheck(
                    check_id="release_candidate_required_base_artifacts",
                    stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
                    substage_id=RELEASE_VALIDATION_SUBSTAGE_ID,
                    description="Required base dataset artifacts are present.",
                    run=lambda context: self._check_required_base_artifacts(
                        bundle,
                    ),
                ),
                ValidationCheck(
                    check_id="release_candidate_staged_artifacts_present",
                    stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
                    substage_id=RELEASE_VALIDATION_SUBSTAGE_ID,
                    description="All candidate artifacts exist under the staging prefix.",
                    run=lambda context: self._check_staged_artifacts_present(
                        bundle,
                        finalized_state,
                    ),
                ),
                ValidationCheck(
                    check_id="release_candidate_validation_reports_present",
                    stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
                    substage_id=RELEASE_VALIDATION_SUBSTAGE_ID,
                    description="Run-scoped validation reports exist before release completion.",
                    run=lambda context: self._check_validation_reports_present(
                        bundle,
                        finalized_state,
                    ),
                ),
                ValidationCheck(
                    check_id="release_candidate_release_manifest_preflight",
                    stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
                    substage_id=RELEASE_VALIDATION_SUBSTAGE_ID,
                    description="Release manifest preflight can finalize local-area artifacts.",
                    run=lambda context: self._check_release_manifest_preflight(
                        bundle,
                        files_with_paths,
                        finalized_state,
                    ),
                ),
            ),
        )

    def _check_required_artifact_families(
        self,
        bundle: ReleaseCandidateInputBundle,
    ) -> ValidationFinding:
        check_id = "release_candidate_required_artifact_families"
        family_counts = Counter(
            artifact.artifact_family
            for artifact in bundle.artifacts
            if artifact.required
        )
        missing_families = sorted(
            family
            for family in self.required_artifact_families
            if family_counts.get(family, 0) < 1
        )
        if missing_families:
            return _finding(
                check_id,
                "fail",
                "Release candidate is missing required artifact families.",
                metric="missing_required_artifact_families",
                value=missing_families,
                threshold=list(self.required_artifact_families),
                artifact_family_counts=dict(sorted(family_counts.items())),
            )
        return _finding(
            check_id,
            "pass",
            "Release candidate includes required artifact families.",
            metric="required_artifact_family_count",
            value=len(self.required_artifact_families),
            artifact_family_counts=dict(sorted(family_counts.items())),
        )

    def _check_required_base_artifacts(
        self,
        bundle: ReleaseCandidateInputBundle,
    ) -> ValidationFinding:
        check_id = "release_candidate_required_base_artifacts"
        required_paths = tuple(self.required_base_artifact_paths)
        release_paths = set(_release_paths(bundle))
        missing_paths = tuple(
            path for path in required_paths if path not in release_paths
        )
        if missing_paths:
            return _finding(
                check_id,
                "fail",
                "Release candidate is missing required base dataset artifacts.",
                metric="missing_required_base_artifacts",
                value=missing_paths,
                threshold=list(required_paths),
            )
        return _finding(
            check_id,
            "pass",
            "Release candidate includes required base dataset artifacts.",
            metric="required_base_artifact_count",
            value=len(required_paths),
        )

    def _check_finalized_release_state(
        self,
        bundle: ReleaseCandidateInputBundle,
        finalized_state: _FinalizedReleaseState,
    ) -> ValidationFinding:
        check_id = "release_candidate_finalized_release_state"
        if not finalized_state.checked:
            return _finding(
                check_id,
                "fail",
                "Finalized-release comparison requires local files with repo paths.",
                metric="manifest_files",
                value=0,
            )
        if finalized_state.error is not None:
            redacted_error = redacted_bounded_error_text(
                str(finalized_state.error),
                max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
            )
            return _finding(
                check_id,
                "fail",
                "Could not compare the candidate against finalized releases.",
                metric="finalized_release_lookup",
                value=finalized_state.error.__class__.__name__,
                exception_type=finalized_state.error.__class__.__name__,
                exception_message=redacted_error.text,
                exception_message_truncated=redacted_error.truncated,
            )
        if finalized_state.manifest is None:
            return _finding(
                check_id,
                "pass",
                "Release is not already finalized with a matching manifest.",
                metric="already_finalized",
                value=False,
            )
        marker_exists = self.dependencies.release_completion_marker_exists(
            version=bundle.context.release_version,
            hf_repo_name=bundle.context.hf_repo_name,
            hf_repo_type=bundle.context.hf_repo_type,
        )
        if not marker_exists:
            return _finding(
                check_id,
                "fail",
                "Matching finalized release is missing its completion marker.",
                metric="release_completion_marker_exists",
                value=False,
                already_finalized=True,
            )
        return _finding(
            check_id,
            "pass",
            "Matching finalized release has a completion marker.",
            metric="release_completion_marker_exists",
            value=True,
            already_finalized=True,
        )

    def _check_staged_artifacts_present(
        self,
        bundle: ReleaseCandidateInputBundle,
        finalized_state: _FinalizedReleaseState,
    ) -> ValidationFinding:
        check_id = "release_candidate_staged_artifacts_present"
        if _skip_side_effect_checks(finalized_state):
            return _skipped_for_finalized_state(check_id, finalized_state)
        missing_paths = sorted(
            self.dependencies.list_missing_staged_artifacts(
                _release_paths(bundle),
                candidate_version=bundle.context.candidate_version,
                hf_repo_name=bundle.context.hf_repo_name,
                hf_repo_type=bundle.context.hf_repo_type,
                run_id=bundle.context.run_id,
            )
        )
        if missing_paths:
            return _finding(
                check_id,
                "fail",
                "Release candidate is missing staged artifacts.",
                metric="missing_staged_artifacts",
                value=missing_paths,
            )
        return _finding(
            check_id,
            "pass",
            "All release candidate artifacts are present in staging.",
            metric="missing_staged_artifacts",
            value=[],
        )

    def _check_validation_reports_present(
        self,
        bundle: ReleaseCandidateInputBundle,
        finalized_state: _FinalizedReleaseState,
    ) -> ValidationFinding:
        check_id = "release_candidate_validation_reports_present"
        if _skip_side_effect_checks(finalized_state):
            return _skipped_for_finalized_state(check_id, finalized_state)
        if not bundle.validation_report_paths:
            return _finding(
                check_id,
                "fail",
                "Release candidate does not include validation report paths.",
                metric="validation_report_paths",
                value=[],
            )
        missing_paths = sorted(
            self.dependencies.list_missing_validation_reports(
                bundle.validation_report_paths,
                context=bundle.context,
            )
        )
        if missing_paths:
            return _finding(
                check_id,
                "fail",
                "Release candidate is missing validation reports.",
                metric="missing_validation_reports",
                value=missing_paths,
            )
        if self.validation_report_policy == VALIDATION_REPORT_POLICY_PRESENCE_ONLY:
            return _finding(
                check_id,
                "pass",
                "Release candidate validation reports are present.",
                metric="validation_report_paths",
                value=list(bundle.validation_report_paths),
                validation_report_policy=self.validation_report_policy,
            )
        canonical_report_paths = _canonical_validation_report_paths(
            bundle.validation_report_paths,
        )
        if not canonical_report_paths:
            return _finding(
                check_id,
                "pass",
                "Release candidate validation reports are present; no canonical report JSON was declared.",
                metric="validation_report_paths",
                value=list(bundle.validation_report_paths),
                validation_report_policy=self.validation_report_policy,
                canonical_validation_report_paths=[],
            )
        try:
            reports = tuple(
                self.dependencies.load_validation_reports(
                    canonical_report_paths,
                    context=bundle.context,
                )
            )
        except Exception as exc:
            redacted_error = redacted_bounded_error_text(
                str(exc),
                max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
            )
            return _finding(
                check_id,
                "fail",
                "Could not load canonical validation report JSON.",
                metric="validation_report_load_error",
                value=exc.__class__.__name__,
                validation_report_policy=self.validation_report_policy,
                canonical_validation_report_paths=list(canonical_report_paths),
                exception_type=exc.__class__.__name__,
                exception_message=redacted_error.text,
                exception_message_truncated=redacted_error.truncated,
            )
        invalid_report_types = tuple(
            type(report).__name__
            for report in reports
            if not isinstance(report, ValidationReport)
        )
        if invalid_report_types:
            return _finding(
                check_id,
                "fail",
                "Canonical validation report loader returned invalid report objects.",
                metric="invalid_validation_report_type",
                value=invalid_report_types,
                validation_report_policy=self.validation_report_policy,
                canonical_validation_report_paths=list(canonical_report_paths),
            )
        if len(reports) != len(canonical_report_paths):
            return _finding(
                check_id,
                "fail",
                "Canonical validation report loader returned an unexpected report count.",
                metric="validation_report_count",
                value=len(reports),
                threshold=len(canonical_report_paths),
                validation_report_policy=self.validation_report_policy,
                canonical_validation_report_paths=list(canonical_report_paths),
            )
        unacceptable_statuses = tuple(
            report.status for report in reports if report.status not in {"pass", "warn"}
        )
        if unacceptable_statuses:
            return _finding(
                check_id,
                "fail",
                "Canonical validation reports include blocking statuses.",
                metric="validation_report_statuses",
                value=unacceptable_statuses,
                threshold=["pass", "warn"],
                validation_report_policy=self.validation_report_policy,
                canonical_validation_report_paths=list(canonical_report_paths),
                validation_report_status_counts=dict(
                    sorted(Counter(report.status for report in reports).items())
                ),
            )
        return _finding(
            check_id,
            "pass",
            "Release candidate validation reports are present and acceptable.",
            metric="validation_report_statuses",
            value=[report.status for report in reports],
            validation_report_policy=self.validation_report_policy,
            canonical_validation_report_paths=list(canonical_report_paths),
        )

    def _check_release_manifest_preflight(
        self,
        bundle: ReleaseCandidateInputBundle,
        files_with_paths: Sequence[ManifestFile],
        finalized_state: _FinalizedReleaseState,
    ) -> ValidationFinding:
        check_id = "release_candidate_release_manifest_preflight"
        if _skip_side_effect_checks(finalized_state):
            return _skipped_for_finalized_state(check_id, finalized_state)
        if not files_with_paths:
            return _finding(
                check_id,
                "fail",
                "Release manifest preflight requires local files with repo paths.",
                metric="manifest_files",
                value=0,
            )
        should_finalize, missing_prefixes = (
            self.dependencies.preflight_release_manifest_publish(
                list(files_with_paths),
                version=bundle.context.release_version,
                new_repo_paths=_release_paths(bundle),
                hf_repo_name=bundle.context.hf_repo_name,
                hf_repo_type=bundle.context.hf_repo_type,
            )
        )
        if not should_finalize:
            return _finding(
                check_id,
                "fail",
                "Release manifest preflight cannot finalize local-area artifacts.",
                metric="missing_local_area_prefixes",
                value=sorted(missing_prefixes),
            )
        return _finding(
            check_id,
            "pass",
            "Release manifest preflight can finalize the candidate.",
            metric="missing_local_area_prefixes",
            value=[],
        )


def _validation_context(
    bundle: ReleaseCandidateInputBundle,
    files_with_paths: Sequence[ManifestFile],
) -> ValidationContext:
    return ValidationContext(
        run_id=bundle.context.run_id,
        stage_id=STAGE_5_VALIDATE_AND_PROMOTE_RELEASE,
        substage_id=RELEASE_VALIDATION_SUBSTAGE_ID,
        resolver=ValidationArtifactResolver(
            artifacts={
                artifact.logical_name: artifact.to_artifact_ref(
                    uri_prefix=bundle.context.hf_staging_prefix,
                )
                for artifact in bundle.artifacts
            },
        ),
        metadata={
            "candidate_version": bundle.context.candidate_version,
            "release_version": bundle.context.release_version,
            "hf_repo_name": bundle.context.hf_repo_name,
            "hf_repo_type": bundle.context.hf_repo_type,
            "release_candidate_fingerprint": bundle.release_candidate_fingerprint,
            "manifest_file_count": len(files_with_paths),
        },
    )


def _release_paths(bundle: ReleaseCandidateInputBundle) -> tuple[str, ...]:
    return tuple(artifact.relative_path for artifact in bundle.artifacts)


def _canonical_validation_report_paths(
    validation_report_paths: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        path
        for path in validation_report_paths
        if path.rsplit("/", 1)[-1] == "validation_report.json"
    )


def _finding(
    check_id: str,
    status: str,
    message: str,
    *,
    metric: str | None = None,
    value: Any | None = None,
    threshold: Any | None = None,
    **metadata: Any,
) -> ValidationFinding:
    return ValidationFinding(
        check_id=check_id,
        status=status,
        message=message,
        metric=metric,
        value=value,
        threshold=threshold,
        metadata=metadata,
    )


def _skip_side_effect_checks(finalized_state: _FinalizedReleaseState) -> bool:
    return finalized_state.error is not None or finalized_state.manifest is not None


def _skipped_for_finalized_state(
    check_id: str,
    finalized_state: _FinalizedReleaseState,
) -> ValidationFinding:
    if finalized_state.error is not None:
        return _finding(
            check_id,
            "pass",
            "Check skipped because finalized-release comparison failed.",
            metric="finalized_release_lookup",
            value="failed",
        )
    return _finding(
        check_id,
        "pass",
        "Check skipped because the matching release is already finalized.",
        metric="already_finalized",
        value=True,
    )


def _list_missing_validation_reports_on_hf(
    validation_report_paths: Sequence[str],
    *,
    context,
) -> list[str]:
    import os

    from huggingface_hub import HfApi

    if not validation_report_paths:
        return [f"calibration/runs/{context.run_id}/diagnostics/<validation report>"]
    token = os.environ.get("HUGGING_FACE_TOKEN")
    repo_files = set(
        HfApi().list_repo_files(
            repo_id=context.hf_repo_name,
            repo_type=context.hf_repo_type,
            token=token,
        )
    )
    return sorted(path for path in validation_report_paths if path not in repo_files)


def _load_validation_reports_from_hf(
    validation_report_paths: Sequence[str],
    *,
    context,
) -> tuple[ValidationReport, ...]:
    import os

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HUGGING_FACE_TOKEN")
    reports = []
    for path in validation_report_paths:
        local_path = hf_hub_download(
            repo_id=context.hf_repo_name,
            filename=path,
            repo_type=context.hf_repo_type,
            token=token,
        )
        with open(local_path, encoding="utf-8") as report_file:
            reports.append(ValidationReport.from_dict(json.load(report_file)))
    return tuple(reports)
