"""Pure helpers for discovering deployed US data pipeline apps."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

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


StatusLookup = Callable[[PipelineAppCandidate], dict[str, Any]]


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


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _text_value(value: Any) -> str:
    return "" if value is None else str(value)


def _timestamp_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    return str(value)


def _record_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def pipeline_app_candidates(
    app_records: Iterable[dict[str, Any]],
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


def _modal_fields(candidate: PipelineAppCandidate) -> dict[str, Any]:
    return {
        "modal_app_id": candidate.app_id,
        "modal_app_name": candidate.app_name,
        "modal_app_state": candidate.state,
        "modal_task_count": candidate.task_count,
        "modal_app_created_at": candidate.created_at,
        "modal_app_stopped_at": candidate.stopped_at,
    }


def _latest_manifest_payload(
    stage_manifests: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not stage_manifests:
        return None
    item = stage_manifests[-1]
    manifest = item.get("manifest") or {}
    return {
        "step_id": item.get("step_id"),
        "stage_id": item.get("stage_id"),
        "substage_id": item.get("substage_id"),
        "title": item.get("title"),
        "status": item.get("status"),
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
        "duration_s": manifest.get("duration_s"),
        "reuse_decision": manifest.get("reuse_decision", "not_applicable"),
    }


def _index_error_payload(error: dict[str, Any] | None) -> dict[str, Any] | None:
    if error is None:
        return None
    allowed = (
        "stage_id",
        "substage_id",
        "surface",
        "error_type",
        "message",
        "message_truncated",
        "record_path",
        "latest_path",
        "traceback_available",
    )
    return {key: error[key] for key in allowed if key in error}


def _status_item(
    candidate: PipelineAppCandidate,
    payload: dict[str, Any],
) -> dict[str, Any]:
    run_manifest = payload.get("run_manifest") or {}
    stage_manifests = payload.get("stage_manifests") or []
    missing = payload.get("missing_expected_manifest_ids") or []
    known_step_ids = run_manifest.get("known_step_ids") or []
    expected_count = max(len(known_step_ids), len(stage_manifests) + len(missing))
    status_lookup = "not_found" if payload.get("status") == "not_found" else "ok"
    return {
        **_modal_fields(candidate),
        "run_id": payload.get("run_id") or candidate.run_id,
        "status_lookup": status_lookup,
        "status": payload.get("status", "unknown"),
        "message": payload.get("message", ""),
        "branch": run_manifest.get("branch"),
        "sha": run_manifest.get("sha"),
        "candidate_version": run_manifest.get("candidate_version"),
        "release_version": run_manifest.get("release_version"),
        "started_at": run_manifest.get("started_at"),
        "updated_at": payload.get("updated_at"),
        "completed_at": run_manifest.get("completed_at"),
        "modal_environment": (
            payload.get("modal_environment") or run_manifest.get("modal_environment")
        ),
        "hf_staging_prefix": run_manifest.get("hf_staging_prefix"),
        "github_run_url": (run_manifest.get("run_context") or {}).get("github_run_url"),
        "latest_manifest": _latest_manifest_payload(stage_manifests),
        "progress": {
            "expected_manifests": expected_count,
            "present_manifests": len(stage_manifests),
            "missing_manifests": len(missing),
        },
        "error": _index_error_payload(payload.get("error")),
    }


def _lookup_error_item(
    candidate: PipelineAppCandidate,
    exc: BaseException,
) -> dict[str, Any]:
    message = redacted_bounded_error_text(
        f"{type(exc).__name__}: {exc}",
        max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    ).text
    return {
        **_modal_fields(candidate),
        "run_id": candidate.run_id,
        "status_lookup": "unreachable",
        "status": "unreachable",
        "message": message,
        "branch": None,
        "sha": None,
        "candidate_version": None,
        "release_version": None,
        "started_at": None,
        "updated_at": None,
        "completed_at": None,
        "modal_environment": None,
        "hf_staging_prefix": None,
        "github_run_url": None,
        "latest_manifest": None,
        "progress": {
            "expected_manifests": 0,
            "present_manifests": 0,
            "missing_manifests": 0,
        },
        "error": {
            "error_type": type(exc).__name__,
            "message": message,
            "traceback_available": False,
        },
    }


def _passes_filters(
    item: dict[str, Any],
    *,
    status: str,
    branch: str,
    include_unreachable: bool,
) -> bool:
    if not include_unreachable and item.get("status_lookup") != "ok":
        return False
    if status and item.get("status") != status:
        return False
    if branch and item.get("branch") != branch:
        return False
    return True


def _lookup_items(
    candidates: list[PipelineAppCandidate],
    status_lookup: StatusLookup,
    *,
    max_workers: int,
) -> list[dict[str, Any]]:
    if max_workers <= 1 or len(candidates) <= 1:
        items = []
        for candidate in candidates:
            try:
                items.append(_status_item(candidate, status_lookup(candidate)))
            except Exception as exc:
                items.append(_lookup_error_item(candidate, exc))
        return items

    items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(status_lookup, candidate): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                items.append(_status_item(candidate, future.result()))
            except Exception as exc:
                items.append(_lookup_error_item(candidate, exc))
    items.sort(
        key=lambda item: (
            item.get("updated_at") or item.get("modal_app_created_at") or "",
            item.get("run_id") or "",
        ),
        reverse=True,
    )
    return items


def build_deployed_pipeline_runs_payload(
    app_records: Iterable[dict[str, Any]],
    status_lookup: StatusLookup,
    *,
    limit: int | str | None = DEFAULT_DISCOVERY_LIMIT,
    status: str = "",
    branch: str = "",
    include_unreachable: bool = True,
    modal_environment: str = "main",
    max_workers: int = DEFAULT_DISCOVERY_WORKERS,
) -> dict[str, Any]:
    """Build a cross-app pipeline run index from Modal app names."""

    bounded_limit = _bounded_limit(limit)
    filters = {
        "status": status or "",
        "branch": branch or "",
        "include_unreachable": bool(include_unreachable),
    }
    candidates = pipeline_app_candidates(app_records)
    needs_filter_window = (
        bool(filters["status"] or filters["branch"])
        or not filters["include_unreachable"]
    )
    selected_limit = MAX_DISCOVERY_LIMIT if needs_filter_window else bounded_limit
    selected = candidates[:selected_limit]
    items = _lookup_items(selected, status_lookup, max_workers=max_workers)
    runs = [
        item
        for item in items
        if _passes_filters(
            item,
            status=filters["status"],
            branch=filters["branch"],
            include_unreachable=filters["include_unreachable"],
        )
    ][:bounded_limit]
    return {
        "schema_version": PIPELINE_DISCOVERY_SCHEMA_VERSION,
        "source": PIPELINE_DISCOVERY_SOURCE,
        "modal_environment": modal_environment,
        "discovered_count": len(candidates),
        "queried_count": len(selected),
        "count": len(runs),
        "limit": bounded_limit,
        "filters": filters,
        "runs": runs,
    }
