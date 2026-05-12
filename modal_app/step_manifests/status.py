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
from policyengine_us_data.utils.error_redaction import (
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
from modal_app.step_manifests.specs import RUN_MANIFEST_STEP_IDS, step_title

PIPELINE_STATUS_SCHEMA_VERSION = "1"


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
    return error_record.to_dict()


def _legacy_error_payload(error_text: str | None) -> dict[str, Any] | None:
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
    payload = {
        "source": "run_manifest.error",
        "surface": "run_manifest",
        "stage_id": None,
        "substage_id": None,
        "error_type": error_type,
        "message": message,
        "traceback": bounded.text,
        "traceback_available": bool(bounded.text),
        "traceback_truncated": bounded.truncated,
    }
    if bounded.truncated:
        payload["traceback_omitted_chars"] = bounded.omitted_chars
    return payload


def _message(
    *,
    status: str,
    stage_manifests: list[dict[str, Any]],
    error: dict[str, Any] | None,
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
        return (
            f"Pipeline {status}; latest manifest "
            f"{latest['substage_id'] or latest['stage_id']} is {latest['status']}."
        )
    return f"Pipeline {status}."


def _sanitize_error_value(value: Any) -> Any:
    if isinstance(value, str):
        return redacted_bounded_error_text(value).text
    if isinstance(value, dict):
        return {key: _sanitize_error_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_error_value(item) for item in value]
    return value


def _run_manifest_payload(run_manifest) -> dict[str, Any]:
    payload = run_manifest.to_dict()
    if payload.get("error"):
        payload["error"] = redacted_bounded_error_text(payload["error"]).text
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
    error = _error_payload(
        read_latest_pipeline_error(run_dir),
    ) or _legacy_error_payload(run_manifest.error)
    status = run_manifest.status
    return {
        "schema_version": PIPELINE_STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "message": _message(
            status=status,
            stage_manifests=stage_manifests,
            error=error,
        ),
        "run_manifest": _run_manifest_payload(run_manifest),
        "stage_manifests": stage_manifests,
        "missing_expected_manifest_ids": missing_expected,
        "error": error,
        "updated_at": run_manifest.updated_at,
        "modal_app_name": run_manifest.modal_app_name,
        "modal_environment": run_manifest.modal_environment,
        "pipeline_volume_name": run_manifest.run_context.get("pipeline_volume_name"),
    }
