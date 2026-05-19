"""Pure helpers for discovering deployed US data pipeline apps."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping

from policyengine_us_data.utils.error_redaction import (
    DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    redacted_bounded_error_text,
)


PIPELINE_DISCOVERY_SCHEMA_VERSION = "1"
PIPELINE_DISCOVERY_SOURCE = "modal_app_names"
DEFAULT_DISCOVERY_LIMIT = 25
MAX_DISCOVERY_LIMIT = 100
DEFAULT_DISCOVERY_WORKERS = 8
PUBLICATION_APP_PREFIXES = ("us-data-", "policyengine-us-data-pub-")
RUN_ID_RE = re.compile(r"(usdata-gha\d+-a\d+)")
DEPLOYED_STATES = {"deployed", "app_state_deployed"}

JSONDict = dict[str, object]
RawRecord = Mapping[str, object]


@dataclass(frozen=True)
class PipelineAppCandidate:
    """A deployed Modal app that appears to represent one publication run."""

    app_id: str
    app_name: str
    run_id: str
    state: str
    task_count: int
    created_at: str | None = None
    stopped_at: str | None = None


@dataclass(frozen=True)
class PipelineDiscoveryFilters:
    """Filters applied to the deployed pipeline run index."""

    status: str = ""
    branch: str = ""
    include_unreachable: bool = True

    def to_dict(self) -> JSONDict:
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

    def to_dict(self) -> JSONDict:
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

    def to_dict(self) -> JSONDict:
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

    def to_dict(self) -> JSONDict:
        payload: JSONDict = {
            "error_type": self.error_type,
            "message": self.message,
            "traceback_available": self.traceback_available,
        }
        optional: JSONDict = {
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

    @classmethod
    def unreachable(
        cls,
        candidate: PipelineAppCandidate,
        exc: BaseException,
    ) -> DeployedPipelineRunSummary:
        message = redacted_bounded_error_text(
            f"{type(exc).__name__}: {exc}",
            max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
        ).text
        return cls(
            run_id=candidate.run_id,
            status_lookup="unreachable",
            status="unreachable",
            message=message,
            modal_app_id=candidate.app_id,
            modal_app_name=candidate.app_name,
            modal_app_state=candidate.state,
            modal_task_count=candidate.task_count,
            modal_app_created_at=candidate.created_at,
            modal_app_stopped_at=candidate.stopped_at,
            progress=PipelineProgressSummary(
                expected_manifests=0,
                present_manifests=0,
                missing_manifests=0,
            ),
            error=PipelineLookupErrorSummary(
                error_type=type(exc).__name__,
                message=message,
            ),
        )

    def to_dict(self) -> JSONDict:
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

    def to_dict(self) -> JSONDict:
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


StatusLookup = Callable[[PipelineAppCandidate], RawRecord]


def derive_run_id_from_app_name(app_name: str) -> str:
    """Extract the canonical US data run ID from a Modal app name."""

    match = RUN_ID_RE.search(app_name or "")
    return match.group(1) if match else ""


def is_publication_pipeline_app_name(app_name: str) -> bool:
    """Return whether an app name belongs to a publication pipeline run."""

    if not app_name.startswith(PUBLICATION_APP_PREFIXES):
        return False
    return bool(derive_run_id_from_app_name(app_name))


def _bounded_limit(limit: int | str | None) -> int:
    try:
        parsed = int(limit if limit is not None else DEFAULT_DISCOVERY_LIMIT)
    except (TypeError, ValueError):
        parsed = DEFAULT_DISCOVERY_LIMIT
    return max(0, min(parsed, MAX_DISCOVERY_LIMIT))


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _text_value(value: object) -> str:
    return "" if value is None else str(value)


def _string_or_none(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _number_or_none(value: object) -> float | int | None:
    return value if isinstance(value, (float, int)) else None


def _timestamp_value(value: object) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    return str(value)


def _record_value(record: RawRecord, *keys: str) -> object:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _mapping_value(record: RawRecord, key: str) -> RawRecord:
    value = record.get(key)
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(record: RawRecord, key: str) -> list[RawRecord]:
    value = record.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def pipeline_app_candidates(
    app_records: Iterable[RawRecord],
) -> list[PipelineAppCandidate]:
    """Return deployed publication-pipeline app candidates from Modal records."""

    candidates: list[PipelineAppCandidate] = []
    for record in app_records:
        app_name = _text_value(
            _record_value(record, "name", "description", "Description")
        )
        state = _text_value(_record_value(record, "state", "State")).lower()
        if state and state not in DEPLOYED_STATES:
            continue
        if not is_publication_pipeline_app_name(app_name):
            continue
        run_id = derive_run_id_from_app_name(app_name)
        candidates.append(
            PipelineAppCandidate(
                app_id=_text_value(_record_value(record, "app_id", "App ID")),
                app_name=app_name,
                run_id=run_id,
                state=state or "unknown",
                task_count=_int_value(_record_value(record, "tasks", "Tasks")),
                created_at=_timestamp_value(
                    _record_value(record, "created_at", "Created at")
                ),
                stopped_at=_timestamp_value(
                    _record_value(record, "stopped_at", "Stopped at")
                ),
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.created_at or "",
            candidate.app_name,
        ),
        reverse=True,
    )
    return candidates


def _latest_manifest_summary(
    stage_manifests: list[RawRecord],
) -> LatestManifestSummary | None:
    if not stage_manifests:
        return None
    item = stage_manifests[-1]
    manifest = _mapping_value(item, "manifest")
    return LatestManifestSummary(
        step_id=_string_or_none(item.get("step_id")),
        stage_id=_string_or_none(item.get("stage_id")),
        substage_id=_string_or_none(item.get("substage_id")),
        title=_string_or_none(item.get("title")),
        status=_string_or_none(item.get("status")),
        started_at=_string_or_none(manifest.get("started_at")),
        completed_at=_string_or_none(manifest.get("completed_at")),
        duration_s=_number_or_none(manifest.get("duration_s")),
        reuse_decision=_text_value(manifest.get("reuse_decision")) or "not_applicable",
    )


def _index_error_summary(error: RawRecord) -> PipelineLookupErrorSummary | None:
    if not error:
        return None
    return PipelineLookupErrorSummary(
        stage_id=_string_or_none(error.get("stage_id")),
        substage_id=_string_or_none(error.get("substage_id")),
        surface=_string_or_none(error.get("surface")),
        error_type=_text_value(error.get("error_type")) or "Error",
        message=_text_value(error.get("message")),
        message_truncated=(
            error.get("message_truncated")
            if isinstance(error.get("message_truncated"), bool)
            else None
        ),
        record_path=_string_or_none(error.get("record_path")),
        latest_path=_string_or_none(error.get("latest_path")),
        traceback_available=(
            error.get("traceback_available")
            if isinstance(error.get("traceback_available"), bool)
            else False
        ),
    )


def _status_summary(
    candidate: PipelineAppCandidate,
    payload: RawRecord,
) -> DeployedPipelineRunSummary:
    run_manifest = _mapping_value(payload, "run_manifest")
    stage_manifests = _list_of_mappings(payload, "stage_manifests")
    missing = payload.get("missing_expected_manifest_ids")
    missing_count = len(missing) if isinstance(missing, list) else 0
    known_step_ids = run_manifest.get("known_step_ids")
    expected_count = max(
        len(known_step_ids) if isinstance(known_step_ids, list) else 0,
        len(stage_manifests) + missing_count,
    )
    status = _text_value(payload.get("status")) or "unknown"
    status_lookup = "not_found" if status == "not_found" else "ok"
    return DeployedPipelineRunSummary(
        run_id=_text_value(payload.get("run_id")) or candidate.run_id,
        status_lookup=status_lookup,
        status=status,
        message=_text_value(payload.get("message")),
        modal_app_id=candidate.app_id,
        modal_app_name=candidate.app_name,
        modal_app_state=candidate.state,
        modal_task_count=candidate.task_count,
        modal_app_created_at=candidate.created_at,
        modal_app_stopped_at=candidate.stopped_at,
        branch=_string_or_none(run_manifest.get("branch")),
        sha=_string_or_none(run_manifest.get("sha")),
        candidate_version=_string_or_none(run_manifest.get("candidate_version")),
        release_version=_string_or_none(run_manifest.get("release_version")),
        started_at=_string_or_none(run_manifest.get("started_at")),
        updated_at=_string_or_none(payload.get("updated_at")),
        completed_at=_string_or_none(run_manifest.get("completed_at")),
        modal_environment=_string_or_none(payload.get("modal_environment"))
        or _string_or_none(run_manifest.get("modal_environment")),
        hf_staging_prefix=_string_or_none(run_manifest.get("hf_staging_prefix")),
        github_run_url=_string_or_none(
            _mapping_value(run_manifest, "run_context").get("github_run_url")
        ),
        latest_manifest=_latest_manifest_summary(stage_manifests),
        progress=PipelineProgressSummary(
            expected_manifests=expected_count,
            present_manifests=len(stage_manifests),
            missing_manifests=missing_count,
        ),
        error=_index_error_summary(_mapping_value(payload, "error")),
    )


def _passes_filters(
    item: DeployedPipelineRunSummary,
    filters: PipelineDiscoveryFilters,
) -> bool:
    if not filters.include_unreachable and item.status_lookup != "ok":
        return False
    if filters.status and item.status != filters.status:
        return False
    if filters.branch and item.branch != filters.branch:
        return False
    return True


def _lookup_items(
    candidates: list[PipelineAppCandidate],
    status_lookup: StatusLookup,
    *,
    max_workers: int,
) -> list[DeployedPipelineRunSummary]:
    if max_workers <= 1 or len(candidates) <= 1:
        items = []
        for candidate in candidates:
            try:
                items.append(_status_summary(candidate, status_lookup(candidate)))
            except Exception as exc:
                items.append(DeployedPipelineRunSummary.unreachable(candidate, exc))
        return items

    items: list[DeployedPipelineRunSummary] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(status_lookup, candidate): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                items.append(_status_summary(candidate, future.result()))
            except Exception as exc:
                items.append(DeployedPipelineRunSummary.unreachable(candidate, exc))
    items.sort(
        key=lambda item: (
            item.updated_at or item.modal_app_created_at or "",
            item.run_id,
        ),
        reverse=True,
    )
    return items


def build_deployed_pipeline_runs_payload(
    app_records: Iterable[RawRecord],
    status_lookup: StatusLookup,
    *,
    limit: int | str | None = DEFAULT_DISCOVERY_LIMIT,
    status: str = "",
    branch: str = "",
    include_unreachable: bool = True,
    modal_environment: str = "main",
    max_workers: int = DEFAULT_DISCOVERY_WORKERS,
) -> DeployedPipelineRunsPayload:
    """Build a typed cross-app pipeline run index from Modal app names."""

    bounded_limit = _bounded_limit(limit)
    filters = PipelineDiscoveryFilters(
        status=status or "",
        branch=branch or "",
        include_unreachable=bool(include_unreachable),
    )
    candidates = pipeline_app_candidates(app_records)
    needs_filter_window = (
        bool(filters.status or filters.branch) or not filters.include_unreachable
    )
    selected_limit = MAX_DISCOVERY_LIMIT if needs_filter_window else bounded_limit
    selected = candidates[:selected_limit]
    items = _lookup_items(selected, status_lookup, max_workers=max_workers)
    runs = tuple(item for item in items if _passes_filters(item, filters))[
        :bounded_limit
    ]
    return DeployedPipelineRunsPayload(
        schema_version=PIPELINE_DISCOVERY_SCHEMA_VERSION,
        source=PIPELINE_DISCOVERY_SOURCE,
        modal_environment=modal_environment,
        discovered_count=len(candidates),
        queried_count=len(selected),
        count=len(runs),
        limit=bounded_limit,
        filters=filters,
        runs=runs,
    )
