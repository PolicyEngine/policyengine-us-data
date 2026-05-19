"""Typed result schemas for deployed pipeline discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

from policyengine_us_data.utils.error_redaction import (
    DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    redacted_bounded_error_text,
)


class PipelineDiscoveryFiltersPayload(TypedDict):
    """Serialized discovery filters."""

    status: str
    branch: str
    include_unreachable: bool


class PipelineProgressSummaryPayload(TypedDict):
    """Serialized manifest progress for one run."""

    expected_manifests: int
    present_manifests: int
    missing_manifests: int


class LatestManifestSummaryPayload(TypedDict):
    """Serialized latest manifest summary."""

    step_id: str | None
    stage_id: str | None
    substage_id: str | None
    title: str | None
    status: str | None
    started_at: str | None
    completed_at: str | None
    duration_s: float | int | None
    reuse_decision: str


class PipelineLookupErrorPayload(TypedDict):
    """Serialized lookup or pipeline error summary."""

    error_type: str
    message: str
    traceback_available: bool
    stage_id: NotRequired[str]
    substage_id: NotRequired[str]
    surface: NotRequired[str]
    message_truncated: NotRequired[bool]
    record_path: NotRequired[str]
    latest_path: NotRequired[str]


class DeployedPipelineRunPayload(TypedDict):
    """Serialized status summary for one deployed publication pipeline app."""

    modal_app_id: str
    modal_app_name: str
    modal_app_state: str
    modal_task_count: int
    modal_app_created_at: str | None
    modal_app_stopped_at: str | None
    run_id: str
    status_lookup: str
    status: str
    message: str
    branch: str | None
    sha: str | None
    candidate_version: str | None
    release_version: str | None
    started_at: str | None
    updated_at: str | None
    completed_at: str | None
    modal_environment: str | None
    hf_staging_prefix: str | None
    github_run_url: str | None
    latest_manifest: LatestManifestSummaryPayload | None
    progress: PipelineProgressSummaryPayload | None
    error: PipelineLookupErrorPayload | None


class DeployedPipelineRunsPayloadDict(TypedDict):
    """Serialized cross-app pipeline run index."""

    schema_version: str
    source: str
    modal_environment: str
    discovered_count: int
    queried_count: int
    count: int
    limit: int
    filters: PipelineDiscoveryFiltersPayload
    runs: list[DeployedPipelineRunPayload]


class ModalAppRecord(TypedDict):
    """Normalized Modal app-list record consumed by discovery core."""

    app_id: str
    name: str
    description: str
    state: str
    tasks: int
    created_at: float | str | None
    stopped_at: float | str | None


@dataclass(frozen=True)
class PipelineDiscoveryFilters:
    """Filters applied to the deployed pipeline run index."""

    status: str = ""
    branch: str = ""
    include_unreachable: bool = True

    def to_dict(self) -> PipelineDiscoveryFiltersPayload:
        return {
            "status": self.status,
            "branch": self.branch,
            "include_unreachable": self.include_unreachable,
        }


@dataclass(frozen=True)
class PipelineProgressSummary:
    """Manifest progress for one discovered pipeline run."""

    expected_manifests: int
    present_manifests: int
    missing_manifests: int

    def to_dict(self) -> PipelineProgressSummaryPayload:
        return {
            "expected_manifests": self.expected_manifests,
            "present_manifests": self.present_manifests,
            "missing_manifests": self.missing_manifests,
        }


@dataclass(frozen=True)
class LatestManifestSummary:
    """Summary of the latest stage or substage manifest for one run."""

    step_id: str | None
    stage_id: str | None
    substage_id: str | None
    title: str | None
    status: str | None
    started_at: str | None
    completed_at: str | None
    duration_s: float | int | None
    reuse_decision: str

    def to_dict(self) -> LatestManifestSummaryPayload:
        return {
            "step_id": self.step_id,
            "stage_id": self.stage_id,
            "substage_id": self.substage_id,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "reuse_decision": self.reuse_decision,
        }


@dataclass(frozen=True)
class PipelineLookupErrorSummary:
    """Bounded error surfaced while querying one discovered app."""

    error_type: str
    message: str
    stage_id: str | None = None
    substage_id: str | None = None
    surface: str | None = None
    message_truncated: bool | None = None
    record_path: str | None = None
    latest_path: str | None = None
    traceback_available: bool = False

    @classmethod
    def from_exception(cls, exc: BaseException) -> PipelineLookupErrorSummary:
        """Build a bounded lookup-error summary from an exception."""

        message = redacted_bounded_error_text(
            f"{type(exc).__name__}: {exc}",
            max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
        ).text
        return cls(
            error_type=type(exc).__name__,
            message=message,
        )

    def to_dict(self) -> PipelineLookupErrorPayload:
        payload: PipelineLookupErrorPayload = {
            "error_type": self.error_type,
            "message": self.message,
            "traceback_available": self.traceback_available,
        }
        optional = {
            "stage_id": self.stage_id,
            "substage_id": self.substage_id,
            "surface": self.surface,
            "message_truncated": self.message_truncated,
            "record_path": self.record_path,
            "latest_path": self.latest_path,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return payload


@dataclass(frozen=True)
class DeployedPipelineRunSummary:
    """Status summary for one deployed publication pipeline app."""

    run_id: str
    status_lookup: str
    status: str
    message: str
    modal_app_id: str
    modal_app_name: str
    modal_app_state: str
    modal_task_count: int
    modal_app_created_at: str | None = None
    modal_app_stopped_at: str | None = None
    branch: str | None = None
    sha: str | None = None
    candidate_version: str | None = None
    release_version: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    modal_environment: str | None = None
    hf_staging_prefix: str | None = None
    github_run_url: str | None = None
    latest_manifest: LatestManifestSummary | None = None
    progress: PipelineProgressSummary | None = None
    error: PipelineLookupErrorSummary | None = None

    def to_dict(self) -> DeployedPipelineRunPayload:
        return {
            "modal_app_id": self.modal_app_id,
            "modal_app_name": self.modal_app_name,
            "modal_app_state": self.modal_app_state,
            "modal_task_count": self.modal_task_count,
            "modal_app_created_at": self.modal_app_created_at,
            "modal_app_stopped_at": self.modal_app_stopped_at,
            "run_id": self.run_id,
            "status_lookup": self.status_lookup,
            "status": self.status,
            "message": self.message,
            "branch": self.branch,
            "sha": self.sha,
            "candidate_version": self.candidate_version,
            "release_version": self.release_version,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "modal_environment": self.modal_environment,
            "hf_staging_prefix": self.hf_staging_prefix,
            "github_run_url": self.github_run_url,
            "latest_manifest": (
                self.latest_manifest.to_dict() if self.latest_manifest else None
            ),
            "progress": self.progress.to_dict() if self.progress else None,
            "error": self.error.to_dict() if self.error else None,
        }


@dataclass(frozen=True)
class DeployedPipelineRunsPayload:
    """Typed cross-app pipeline run index discovered from Modal app names."""

    schema_version: str
    source: str
    modal_environment: str
    discovered_count: int
    queried_count: int
    count: int
    limit: int
    filters: PipelineDiscoveryFilters
    runs: tuple[DeployedPipelineRunSummary, ...]

    def to_dict(self) -> DeployedPipelineRunsPayloadDict:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "modal_environment": self.modal_environment,
            "discovered_count": self.discovered_count,
            "queried_count": self.queried_count,
            "count": self.count,
            "limit": self.limit,
            "filters": self.filters.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
        }
