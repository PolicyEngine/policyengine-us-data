"""Structured status payloads for Modal pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from policyengine_us_data.utils.step_manifest import (
    read_run_manifest,
    read_step_manifest,
    run_manifest_path,
    step_manifest_dir,
)
from policyengine_us_data.build_datasets import (
    empty_stage_1_status_snapshot,
    read_stage_1_status_snapshot,
)
from policyengine_us_data.utils.error_redaction import (
    DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    bound_error_text,
    redacted_bounded_error_text,
    redact_error_text,
)

from modal_app.step_manifests import state as pipeline_state
from modal_app.step_manifests.errors import (
    PipelineErrorRecord,
    read_latest_pipeline_error,
    stage_ids_for_manifest,
)
from modal_app.step_manifests.specs import (
    BUILD_DATASETS,
    RUN_MANIFEST_STEP_IDS,
    step_title,
)

PIPELINE_STATUS_SCHEMA_VERSION = "1"
DEFAULT_RUNS_LIMIT = 25
MAX_RUNS_LIMIT = 100


def _run_dir(run_id: str, runs_dir: str | Path | None = None) -> Path:
    return (
        Path(runs_dir) if runs_dir is not None else Path(pipeline_state.RUNS_DIR)
    ) / run_id


def _step_sort_key(step_id: str, expected_ids: list[str]) -> tuple[int, str]:
    try:
        return expected_ids.index(step_id), step_id
    except ValueError:
        return len(expected_ids), step_id


def _error_payload(
    error_record: PipelineErrorRecord | None,
) -> dict[str, Any] | None:
    if error_record is None:
        return None
    return error_record.to_status_dict()


def _latest_error_payload(run_dir: Path) -> dict[str, Any] | None:
    try:
        return _error_payload(read_latest_pipeline_error(run_dir))
    except Exception as exc:
        message = redacted_bounded_error_text(
            f"{type(exc).__name__}: {exc}",
            max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
        ).text
        return {
            "source": "latest_error.json",
            "surface": "error_record_read",
            "stage_id": None,
            "substage_id": None,
            "error_type": type(exc).__name__,
            "message": message,
            "traceback_available": False,
        }


def _run_manifest_error_payload(error_text: str | None) -> dict[str, Any] | None:
    if not error_text:
        return None
    redacted = redact_error_text(error_text)
    bounded = bound_error_text(redacted)
    first_line = redacted.splitlines()[0] if redacted else "Pipeline error"
    error_type = "RuntimeError"
    message = first_line
    if ":" in first_line:
        maybe_type, maybe_message = first_line.split(":", 1)
        if maybe_type.strip():
            error_type = maybe_type.strip()
            message = maybe_message.strip() or first_line
    bounded_message = bound_error_text(
        message,
        max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    )
    payload = {
        "source": "run_manifest.error",
        "surface": "run_manifest",
        "stage_id": None,
        "substage_id": None,
        "error_type": error_type,
        "message": bounded_message.text,
        "message_truncated": bounded_message.truncated,
        "traceback": bounded.text,
        "traceback_available": bool(bounded.text),
        "traceback_truncated": bounded.truncated,
    }
    if bounded_message.truncated:
        payload["message_omitted_chars"] = bounded_message.omitted_chars
    if bounded.truncated:
        payload["traceback_omitted_chars"] = bounded.omitted_chars
    return payload


def _message(
    *,
    status: str,
    stage_manifests: list[dict[str, Any]],
    error: dict[str, Any] | None,
    stage_1_status: dict[str, Any] | None = None,
) -> str:
    if error:
        location = (
            error.get("substage_id") or error.get("stage_id") or error.get("surface")
        )
        detail = error.get("message") or error.get("error_type") or "unknown error"
        if status == "failed":
            return f"Pipeline failed in {location}: {detail}"
        return f"Pipeline has a recorded error in {location}: {detail}"
    if status == "not_found":
        return "Pipeline run not found."
    if stage_manifests:
        latest = stage_manifests[-1]
        current_stage_1 = (stage_1_status or {}).get("current") or {}
        if (
            latest["step_id"] == BUILD_DATASETS.id
            and latest["status"] == "running"
            and current_stage_1
        ):
            substep_id = current_stage_1.get("substep_id")
            title = current_stage_1.get("title") or substep_id
            substep_status = current_stage_1.get("status", "unknown")
            return (
                f"Pipeline {status}; current Stage 1 substep "
                f"{substep_id} ({title}) is {substep_status}."
            )
        return (
            f"Pipeline {status}; latest manifest "
            f"{latest['substage_id'] or latest['stage_id']} is {latest['status']}."
        )
    return f"Pipeline {status}."


def _sanitize_error_value(value: Any) -> Any:
    if isinstance(value, str):
        return redacted_bounded_error_text(
            value,
            max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
        ).text
    if isinstance(value, dict):
        return {key: _sanitize_error_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_error_value(item) for item in value]
    return value


def _run_manifest_payload(run_manifest) -> dict[str, Any]:
    payload = run_manifest.to_dict()
    if payload.get("error"):
        payload["error"] = redacted_bounded_error_text(
            payload["error"],
            max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
        ).text
    return payload


def _manifest_payload(manifest) -> dict[str, Any]:
    stage_id, substage_id = stage_ids_for_manifest(manifest)
    manifest_payload = manifest.to_dict()
    if manifest_payload.get("error"):
        manifest_payload["error"] = _sanitize_error_value(manifest_payload["error"])
    return {
        "stage_id": stage_id,
        "substage_id": substage_id,
        "step_id": manifest.step_id,
        "title": step_title(manifest.step_id),
        "status": manifest.status,
        "manifest": manifest_payload,
    }


def _bounded_limit(limit: int | str | None) -> int:
    try:
        parsed = int(limit if limit is not None else DEFAULT_RUNS_LIMIT)
    except (TypeError, ValueError):
        parsed = DEFAULT_RUNS_LIMIT
    return max(0, min(parsed, MAX_RUNS_LIMIT))


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


def _latest_manifest_payload(
    stage_manifests: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not stage_manifests:
        return None
    item = stage_manifests[-1]
    manifest = item["manifest"]
    return {
        "step_id": item["step_id"],
        "stage_id": item["stage_id"],
        "substage_id": item["substage_id"],
        "title": item["title"],
        "status": item["status"],
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
        "duration_s": manifest.get("duration_s"),
        "reuse_decision": manifest.get("reuse_decision", "not_applicable"),
    }


def _stage_1_status_payload(run_dir: Path) -> dict[str, Any]:
    snapshot = read_stage_1_status_snapshot(run_dir)
    return _sanitize_error_value(snapshot.to_dict())


def _run_index_item(
    run_id: str,
    *,
    runs_dir: str | Path | None = None,
) -> dict[str, Any]:
    payload = build_pipeline_status_payload(run_id, runs_dir=runs_dir)
    run_manifest = payload.get("run_manifest") or {}
    stage_manifests = payload.get("stage_manifests") or []
    missing = payload.get("missing_expected_manifest_ids") or []
    expected = list(run_manifest.get("known_step_ids") or RUN_MANIFEST_STEP_IDS)
    return {
        "run_id": payload["run_id"],
        "status": payload["status"],
        "message": payload["message"],
        "branch": run_manifest.get("branch"),
        "sha": run_manifest.get("sha"),
        "candidate_version": run_manifest.get("candidate_version"),
        "release_version": run_manifest.get("release_version"),
        "started_at": run_manifest.get("started_at"),
        "updated_at": payload.get("updated_at"),
        "completed_at": run_manifest.get("completed_at"),
        "modal_app_name": payload.get("modal_app_name"),
        "modal_environment": payload.get("modal_environment"),
        "hf_staging_prefix": run_manifest.get("hf_staging_prefix"),
        "github_run_url": (run_manifest.get("run_context") or {}).get("github_run_url"),
        "latest_manifest": _latest_manifest_payload(stage_manifests),
        "stage_1_current": (payload.get("stage_1_status") or {}).get("current"),
        "progress": {
            "expected_manifests": len(expected),
            "present_manifests": len(stage_manifests),
            "missing_manifests": len(missing),
        },
        "error": _index_error_payload(payload.get("error")),
    }


def _unreadable_run_index_item(run_id: str, exc: BaseException) -> dict[str, Any]:
    message = redacted_bounded_error_text(
        f"{type(exc).__name__}: {exc}",
        max_chars=DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    ).text
    return {
        "run_id": run_id,
        "status": "unreadable",
        "message": message,
        "branch": None,
        "sha": None,
        "candidate_version": None,
        "release_version": None,
        "started_at": None,
        "updated_at": None,
        "completed_at": None,
        "modal_app_name": None,
        "modal_environment": None,
        "hf_staging_prefix": None,
        "github_run_url": None,
        "latest_manifest": None,
        "stage_1_current": None,
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


def _run_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    return (
        str(item.get("updated_at") or item.get("started_at") or ""),
        str(item.get("run_id") or ""),
    )


def build_pipeline_runs_payload(
    *,
    limit: int | str | None = DEFAULT_RUNS_LIMIT,
    status: str = "",
    branch: str = "",
    runs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable index of recent pipeline runs."""

    bounded_limit = _bounded_limit(limit)
    root = Path(runs_dir) if runs_dir is not None else Path(pipeline_state.RUNS_DIR)
    filters = {"status": status or "", "branch": branch or ""}
    if not root.exists():
        return {
            "schema_version": PIPELINE_STATUS_SCHEMA_VERSION,
            "count": 0,
            "limit": bounded_limit,
            "filters": filters,
            "runs": [],
        }

    items = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = run_manifest_path(entry)
        if not manifest_path.exists():
            continue
        try:
            item = _run_index_item(entry.name, runs_dir=root)
        except Exception as exc:
            item = _unreadable_run_index_item(entry.name, exc)
        if filters["status"] and item.get("status") != filters["status"]:
            continue
        if filters["branch"] and item.get("branch") != filters["branch"]:
            continue
        items.append(item)

    items.sort(key=_run_sort_key, reverse=True)
    runs = items[:bounded_limit]
    return {
        "schema_version": PIPELINE_STATUS_SCHEMA_VERSION,
        "count": len(runs),
        "limit": bounded_limit,
        "filters": filters,
        "runs": runs,
    }


def build_pipeline_status_payload(
    run_id: str,
    *,
    runs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable status payload for a pipeline run."""

    if not run_id:
        return {
            "schema_version": PIPELINE_STATUS_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "invalid_request",
            "message": "run_id is required.",
            "run_manifest": None,
            "stage_manifests": [],
            "missing_expected_manifest_ids": [],
            "error": None,
            "stage_1_status": empty_stage_1_status_snapshot().to_dict(),
        }

    run_dir = _run_dir(run_id, runs_dir)
    manifest_path = run_manifest_path(run_dir)
    if not manifest_path.exists():
        return {
            "schema_version": PIPELINE_STATUS_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "not_found",
            "message": f"Pipeline run {run_id} not found.",
            "run_manifest": None,
            "stage_manifests": [],
            "missing_expected_manifest_ids": list(RUN_MANIFEST_STEP_IDS),
            "error": None,
            "stage_1_status": empty_stage_1_status_snapshot().to_dict(),
        }

    run_manifest = read_run_manifest(manifest_path)
    expected_ids = list(run_manifest.known_step_ids or RUN_MANIFEST_STEP_IDS)
    manifests = []
    steps_dir = step_manifest_dir(run_dir)
    if steps_dir.exists():
        manifests = [
            read_step_manifest(path) for path in sorted(steps_dir.glob("*.json"))
        ]
    manifests.sort(key=lambda manifest: _step_sort_key(manifest.step_id, expected_ids))
    stage_manifests = [_manifest_payload(manifest) for manifest in manifests]
    present_ids = {manifest.step_id for manifest in manifests}
    missing_expected = [
        step_id for step_id in expected_ids if step_id not in present_ids
    ]
    error = _latest_error_payload(run_dir) or _run_manifest_error_payload(
        run_manifest.error
    )
    status = run_manifest.status
    stage_1_status = _stage_1_status_payload(run_dir)
    return {
        "schema_version": PIPELINE_STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "message": _message(
            status=status,
            stage_manifests=stage_manifests,
            error=error,
            stage_1_status=stage_1_status,
        ),
        "run_manifest": _run_manifest_payload(run_manifest),
        "stage_manifests": stage_manifests,
        "missing_expected_manifest_ids": missing_expected,
        "error": error,
        "stage_1_status": stage_1_status,
        "updated_at": run_manifest.updated_at,
        "modal_app_name": run_manifest.modal_app_name,
        "modal_environment": run_manifest.modal_environment,
        "pipeline_volume_name": run_manifest.run_context.get("pipeline_volume_name"),
    }
