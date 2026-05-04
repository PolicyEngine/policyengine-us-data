"""Persistence helpers for Modal pipeline step manifests."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from policyengine_us_data.utils.step_manifest import (
    ArtifactReference,
    ReuseMeasurement,
    RunManifest,
    StepManifest,
    evaluate_step_reuse,
    read_step_manifest,
    run_manifest_path,
    step_manifest_path,
    utc_now,
    write_run_manifest,
    write_step_manifest,
)

from modal_app.step_manifests.state import RUN_STEP_IDS, RunMetadata, run_dir


def build_run_manifest(meta: RunMetadata) -> RunManifest:
    """Build the run-scoped execution ledger from compatibility metadata."""
    return RunManifest(
        run_id=meta.run_id,
        branch=meta.branch,
        sha=meta.sha,
        version=meta.version,
        status=meta.status,
        started_at=meta.start_time,
        run_context=meta.run_context,
        modal_app_name=meta.modal_app_name,
        modal_environment=meta.modal_environment,
        hf_staging_prefix=meta.hf_staging_prefix,
        updated_at=utc_now(),
        completed_at=utc_now()
        if meta.status in {"completed", "failed", "promoted"}
        else None,
        known_step_ids=RUN_STEP_IDS,
        resume_history=meta.resume_history,
        error=meta.error,
    )


def write_run_manifest_for_meta(meta: RunMetadata) -> None:
    """Write the canonical run manifest for a pipeline run."""
    write_run_manifest(
        run_manifest_path(run_dir(meta.run_id)),
        build_run_manifest(meta),
    )


def write_run_meta(meta: RunMetadata, vol: Any) -> None:
    """Write compatibility metadata and the canonical run manifest."""
    destination = run_dir(meta.run_id)
    destination.mkdir(parents=True, exist_ok=True)
    meta_path = destination / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta.to_dict(), f, indent=2)
    write_run_manifest_for_meta(meta)
    vol.commit()


def read_run_meta(run_id: str, vol: Any) -> RunMetadata:
    """Read run metadata from the pipeline volume."""
    vol.reload()
    meta_path = run_dir(run_id) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found for run {run_id} at {meta_path}")
    with open(meta_path) as f:
        return RunMetadata.from_dict(json.load(f))


def _next_step_attempt(run_id: str, step_id: str) -> int:
    path = step_manifest_path(run_dir(run_id), step_id)
    if not path.exists():
        return 1
    try:
        return read_step_manifest(path).attempt + 1
    except Exception:
        return 1


def start_step_manifest(
    meta: RunMetadata,
    step_id: str,
    *,
    parameters: dict | None = None,
    input_identities: dict | None = None,
    scope: str | None = None,
    modal_call_id: str | None = None,
    vol: Any | None = None,
) -> StepManifest:
    manifest = StepManifest(
        run_id=meta.run_id,
        step_id=step_id,
        scope=scope,
        status="running",
        attempt=_next_step_attempt(meta.run_id, step_id),
        started_at=utc_now(),
        branch=meta.branch,
        sha=meta.sha,
        version=meta.version,
        modal_app_name=meta.modal_app_name,
        modal_environment=meta.modal_environment,
        hf_staging_prefix=meta.hf_staging_prefix,
        modal_call_id=modal_call_id,
        parameters=parameters or {},
        input_identities=input_identities or {},
    )
    write_step_manifest(step_manifest_path(run_dir(meta.run_id), step_id), manifest)
    if vol is not None:
        vol.commit()
    return manifest


def complete_step_manifest(
    manifest: StepManifest,
    *,
    outputs: list[ArtifactReference] | None = None,
    diagnostics: list[ArtifactReference] | None = None,
    reuse_decision: str = "computed",
    reuse_reason: str | None = None,
    reuse_measurement: ReuseMeasurement | None = None,
    status: str = "completed",
    vol: Any | None = None,
) -> StepManifest:
    completed = manifest.complete(
        status=status,
        outputs=outputs,
        diagnostics=diagnostics,
        reuse_decision=reuse_decision,
        reuse_reason=reuse_reason,
        reuse_measurement=reuse_measurement,
    )
    write_step_manifest(
        step_manifest_path(run_dir(completed.run_id), completed.step_id),
        completed,
    )
    if vol is not None:
        vol.commit()
    return completed


def fail_step_manifest(
    manifest: StepManifest | None,
    exc: BaseException,
    vol: Any,
) -> None:
    if manifest is None:
        return
    failed = manifest.fail(exc)
    write_step_manifest(
        step_manifest_path(run_dir(failed.run_id), failed.step_id), failed
    )
    vol.commit()


def mark_step_reused(
    meta: RunMetadata,
    step_id: str,
    decision,
    *,
    vol: Any,
    legacy_step: str | None = None,
) -> StepManifest:
    previous = decision.manifest
    if previous is None:
        raise RuntimeError(f"Cannot reuse {step_id}: missing prior manifest")
    reused = StepManifest(
        run_id=previous.run_id,
        step_id=previous.step_id,
        scope=previous.scope,
        status="reused",
        attempt=previous.attempt + 1,
        started_at=utc_now(),
        completed_at=utc_now(),
        duration_s=0.0,
        branch=meta.branch,
        sha=meta.sha,
        version=meta.version,
        modal_app_name=meta.modal_app_name or previous.modal_app_name,
        modal_environment=meta.modal_environment or previous.modal_environment,
        hf_staging_prefix=meta.hf_staging_prefix or previous.hf_staging_prefix,
        modal_app_id=previous.modal_app_id,
        modal_call_id=previous.modal_call_id,
        parameters=previous.parameters,
        input_identities=previous.input_identities,
        outputs=previous.outputs,
        diagnostics=previous.diagnostics,
        reuse_decision="reused",
        reuse_reason=decision.reason,
        reuse_measurement=ReuseMeasurement(
            expected_outputs=len(previous.outputs),
            valid_reused_outputs=len(previous.outputs),
            recomputed_outputs=0,
            invalid_outputs=0,
        ),
    )
    write_step_manifest(step_manifest_path(run_dir(meta.run_id), step_id), reused)
    meta.step_timings[legacy_step or step_id] = {
        "start": reused.started_at,
        "end": reused.completed_at,
        "duration_s": 0.0,
        "status": "completed",
        "reuse_decision": "reused",
        "reuse_reason": decision.reason,
    }
    write_run_meta(meta, vol)
    return reused


def step_reusable(
    meta: RunMetadata,
    step_id: str,
    *,
    expected_input_identities: dict | None = None,
    expected_parameters: dict | None = None,
) -> object:
    return evaluate_step_reuse(
        step_manifest_path(run_dir(meta.run_id), step_id),
        expected_input_identities=expected_input_identities,
        expected_parameters=expected_parameters,
    )


def record_step(
    meta: RunMetadata,
    step: str,
    start: float,
    vol: Any,
    status: str = "completed",
    *,
    step_id: str | None = None,
    step_manifest: StepManifest | None = None,
    parameters: dict | None = None,
    input_identities: dict | None = None,
    outputs: list[ArtifactReference] | None = None,
    diagnostics: list[ArtifactReference] | None = None,
    reuse_decision: str = "computed",
    reuse_reason: str | None = None,
    reuse_measurement: ReuseMeasurement | None = None,
) -> None:
    """Record step timing/status and complete the step manifest."""
    meta.step_timings[step] = {
        "start": datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
        "end": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - start, 1),
        "status": status,
    }
    canonical_step_id = step_id or step
    manifest = step_manifest or StepManifest(
        run_id=meta.run_id,
        step_id=canonical_step_id,
        status="running",
        attempt=_next_step_attempt(meta.run_id, canonical_step_id),
        started_at=datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
        branch=meta.branch,
        sha=meta.sha,
        version=meta.version,
        modal_app_name=meta.modal_app_name,
        modal_environment=meta.modal_environment,
        hf_staging_prefix=meta.hf_staging_prefix,
        parameters=parameters or {},
        input_identities=input_identities or {},
    )
    complete_step_manifest(
        manifest,
        outputs=outputs or [],
        diagnostics=diagnostics or [],
        reuse_decision=reuse_decision,
        reuse_reason=reuse_reason,
        reuse_measurement=reuse_measurement,
        status=status,
    )
    write_run_meta(meta, vol)
