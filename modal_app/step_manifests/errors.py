"""Durable error records for Modal pipeline runs."""

from __future__ import annotations

import re
import traceback as traceback_module
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from policyengine_us_data.utils.canonical_json import (
    canonical_json_dumps,
    canonical_json_loads,
)
from policyengine_us_data.utils.error_redaction import (
    DEFAULT_ERROR_MESSAGE_MAX_CHARS,
    DEFAULT_ERROR_TEXT_MAX_CHARS,
    bound_error_text,
    redact_error_text,
)
from policyengine_us_data.utils.step_manifest import ArtifactReference, StepManifest
from policyengine_us_data.stage_contracts.stages import (
    CANONICAL_STAGE_IDS,
    SUBSTAGE_IDS_BY_STAGE_ID,
    is_canonical_stage_id,
    is_canonical_substage_id,
)

from modal_app.step_manifests import state as pipeline_state
from modal_app.step_manifests.specs import parent_step_id

PIPELINE_ERROR_RECORD_SCHEMA_VERSION = "1"
ERRORS_DIR_NAME = "errors"
LATEST_ERROR_FILENAME = "latest_error.json"


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def stage_ids_for_manifest(
    manifest: StepManifest | None,
) -> tuple[str | None, str | None]:
    """Return public stage/substage IDs for an internal step manifest."""

    if manifest is None:
        return None, None
    if is_canonical_stage_id(manifest.step_id):
        return manifest.step_id, None
    explicit_parent = manifest.parent_step_id or parent_step_id(manifest.step_id)
    inferred_parent = (
        explicit_parent
        if explicit_parent
        and is_canonical_substage_id(explicit_parent, manifest.step_id)
        else _canonical_stage_for_substage(manifest.step_id)
    )
    if inferred_parent and is_canonical_substage_id(inferred_parent, manifest.step_id):
        return inferred_parent, manifest.step_id
    return None, None


def _canonical_stage_for_substage(substage_id: str) -> str | None:
    for stage_id, substage_ids in SUBSTAGE_IDS_BY_STAGE_ID.items():
        if substage_id in substage_ids:
            return stage_id
    return None


def _validate_explicit_stage_ids(
    *,
    stage_id: str | None,
    substage_id: str | None,
) -> None:
    if stage_id is None:
        if substage_id is not None:
            raise ValueError("substage_id cannot be set without stage_id")
        return
    if stage_id not in CANONICAL_STAGE_IDS:
        raise ValueError(f"Invalid canonical stage_id: {stage_id!r}")
    if substage_id is not None and not is_canonical_substage_id(stage_id, substage_id):
        raise ValueError(
            f"substage_id {substage_id!r} does not belong to stage_id {stage_id!r}"
        )


@dataclass(frozen=True)
class PipelineErrorRecord:
    """Durable traceback record for one pipeline failure surface."""

    run_id: str
    stage_id: str | None
    substage_id: str | None
    surface: str
    occurred_at: str
    error_type: str
    message: str
    traceback: str
    branch: str | None = None
    sha: str | None = None
    version: str | None = None
    modal_app_name: str | None = None
    modal_environment: str | None = None
    modal_call_id: str | None = None
    record_path: str | None = None
    latest_path: str | None = None
    traceback_format: str = "python"
    schema_version: str = PIPELINE_ERROR_RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    def to_status_dict(
        self,
        *,
        message_max_chars: int | None = DEFAULT_ERROR_MESSAGE_MAX_CHARS,
        traceback_max_chars: int | None = DEFAULT_ERROR_TEXT_MAX_CHARS,
    ) -> dict[str, Any]:
        payload = self.to_dict()
        bounded_message = bound_error_text(
            self.message,
            max_chars=message_max_chars,
        )
        bounded_traceback = bound_error_text(
            self.traceback,
            max_chars=traceback_max_chars,
        )
        payload["message"] = bounded_message.text
        payload["message_truncated"] = bounded_message.truncated
        if bounded_message.truncated:
            payload["message_omitted_chars"] = bounded_message.omitted_chars
        payload["traceback"] = bounded_traceback.text
        payload["traceback_available"] = bool(self.traceback)
        payload["traceback_truncated"] = bounded_traceback.truncated
        if bounded_traceback.truncated:
            payload["traceback_omitted_chars"] = bounded_traceback.omitted_chars
        return _drop_none(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineErrorRecord":
        stage_id = data.get("stage_id")
        substage_id = data.get("substage_id")
        resolved_stage_id = str(stage_id) if stage_id is not None else None
        resolved_substage_id = str(substage_id) if substage_id is not None else None
        _validate_explicit_stage_ids(
            stage_id=resolved_stage_id,
            substage_id=resolved_substage_id,
        )
        return cls(
            schema_version=str(
                data.get("schema_version", PIPELINE_ERROR_RECORD_SCHEMA_VERSION)
            ),
            run_id=str(data["run_id"]),
            stage_id=resolved_stage_id,
            substage_id=resolved_substage_id,
            surface=str(data.get("surface", "pipeline")),
            occurred_at=str(data["occurred_at"]),
            error_type=str(data["error_type"]),
            message=str(data["message"]),
            traceback=str(data.get("traceback", "")),
            branch=data.get("branch"),
            sha=data.get("sha"),
            version=data.get("version"),
            modal_app_name=data.get("modal_app_name"),
            modal_environment=data.get("modal_environment"),
            modal_call_id=data.get("modal_call_id"),
            record_path=data.get("record_path"),
            latest_path=data.get("latest_path"),
            traceback_format=str(data.get("traceback_format", "python")),
        )


@dataclass(frozen=True)
class PipelineErrorWriteResult:
    """File references written for one durable pipeline error record."""

    record: PipelineErrorRecord
    record_ref: ArtifactReference
    latest_ref: ArtifactReference


def build_pipeline_error_record(
    exc: BaseException,
    *,
    run_id: str,
    manifest: StepManifest | None = None,
    meta: Any | None = None,
    stage_id: str | None = None,
    substage_id: str | None = None,
    surface: str = "pipeline",
    traceback_text: str | None = None,
    occurred_at: str | None = None,
    env: Mapping[str, str] | None = None,
) -> PipelineErrorRecord:
    """Build a redacted error record from an exception and run context."""

    inferred_stage_id, inferred_substage_id = stage_ids_for_manifest(manifest)
    resolved_stage_id = stage_id or inferred_stage_id
    resolved_substage_id = (
        substage_id if substage_id is not None else inferred_substage_id
    )
    _validate_explicit_stage_ids(
        stage_id=resolved_stage_id,
        substage_id=resolved_substage_id,
    )
    trace = traceback_text
    if trace is None:
        trace = "".join(
            traceback_module.format_exception(type(exc), exc, exc.__traceback__)
        )
    return PipelineErrorRecord(
        run_id=run_id,
        stage_id=resolved_stage_id,
        substage_id=resolved_substage_id,
        surface=surface,
        occurred_at=occurred_at or datetime.now(timezone.utc).isoformat(),
        error_type=type(exc).__name__,
        message=redact_error_text(str(exc), env=env),
        traceback=redact_error_text(trace, env=env),
        branch=getattr(meta, "branch", None) or getattr(manifest, "branch", None),
        sha=getattr(meta, "sha", None) or getattr(manifest, "sha", None),
        version=getattr(meta, "version", None) or getattr(manifest, "version", None),
        modal_app_name=getattr(meta, "modal_app_name", None)
        or getattr(manifest, "modal_app_name", None),
        modal_environment=getattr(meta, "modal_environment", None)
        or getattr(manifest, "modal_environment", None),
        modal_call_id=getattr(manifest, "modal_call_id", None),
    )


def error_records_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / ERRORS_DIR_NAME


def latest_error_record_path(run_dir: str | Path) -> Path:
    return error_records_dir(run_dir) / LATEST_ERROR_FILENAME


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe or "pipeline"


def _timestamp_for_filename(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _volume_root_for_run_dir(run_dir: Path) -> Path:
    try:
        return run_dir.parent.parent
    except IndexError:
        return Path(".")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_pipeline_error_record(
    record: PipelineErrorRecord,
    *,
    run_dir: str | Path,
    volume_root: str | Path | None = None,
) -> PipelineErrorWriteResult:
    """Write timestamped and latest error records for a run."""

    resolved_run_dir = Path(run_dir)
    root = (
        Path(volume_root)
        if volume_root is not None
        else _volume_root_for_run_dir(resolved_run_dir)
    )
    errors_dir = error_records_dir(resolved_run_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)

    stage_label = record.substage_id or record.stage_id or record.surface
    filename = f"{_timestamp_for_filename(record.occurred_at)}-{_safe_filename(stage_label)}.json"
    record_path = errors_dir / filename
    latest_path = errors_dir / LATEST_ERROR_FILENAME
    record_with_paths = replace(
        record,
        record_path=_relative_path(record_path, root),
        latest_path=_relative_path(latest_path, root),
    )
    payload = canonical_json_dumps(record_with_paths.to_dict())
    record_path.write_text(payload)
    latest_path.write_text(payload)
    return PipelineErrorWriteResult(
        record=record_with_paths,
        record_ref=ArtifactReference.from_path(
            record_path,
            role="error",
            base_dir=root,
            media_type="application/json",
        ),
        latest_ref=ArtifactReference.from_path(
            latest_path,
            role="error",
            base_dir=root,
            media_type="application/json",
        ),
    )


def read_latest_pipeline_error(
    run_dir: str | Path,
) -> PipelineErrorRecord | None:
    path = latest_error_record_path(run_dir)
    if not path.exists():
        return None
    return PipelineErrorRecord.from_dict(canonical_json_loads(path.read_text()))


def clear_latest_pipeline_error(
    run_dir: str | Path,
    *,
    vol: Any | None = None,
    strict: bool = False,
) -> bool:
    """Clear the mutable latest-error pointer after a successful retry."""

    try:
        path = latest_error_record_path(run_dir)
        if path.exists():
            path.unlink()
            if vol is not None:
                vol.commit()
        return True
    except Exception:
        if strict:
            raise
        return False


def record_pipeline_error(
    exc: BaseException,
    *,
    run_id: str,
    manifest: StepManifest | None = None,
    meta: Any | None = None,
    surface: str = "pipeline",
    traceback_text: str | None = None,
    vol: Any | None = None,
) -> PipelineErrorWriteResult:
    """Persist an exception to the run error ledger and optionally commit volume."""

    record = build_pipeline_error_record(
        exc,
        run_id=run_id,
        manifest=manifest,
        meta=meta,
        surface=surface,
        traceback_text=traceback_text,
    )
    result = write_pipeline_error_record(
        record,
        run_dir=pipeline_state.run_dir(run_id),
        volume_root=pipeline_state.PIPELINE_MOUNT,
    )
    if vol is not None:
        vol.commit()
    return result
