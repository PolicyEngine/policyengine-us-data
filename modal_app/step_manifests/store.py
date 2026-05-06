"""Persistence helpers for Modal pipeline step manifests."""

from __future__ import annotations

from typing import Any

from policyengine_us_data.utils.step_manifest import (
    ArtifactReference,
    ReuseMeasurement,
    RunManifest,
    StepManifest,
    evaluate_step_reuse,
    read_run_manifest,
    read_step_manifest,
    run_manifest_path,
    step_manifest_path,
    utc_now,
    write_run_manifest,
    write_step_manifest,
)

from modal_app.step_manifests.specs import (
    RUN_MANIFEST_STEP_IDS,
    PipelineStepRef,
    parent_step_id,
    step_id,
)
from modal_app.step_manifests.state import RunMetadata, run_dir


def build_run_manifest(meta: RunMetadata) -> RunManifest:
    """Build the run-scoped execution ledger from pipeline metadata."""
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
        known_step_ids=list(RUN_MANIFEST_STEP_IDS),
        resume_history=meta.resume_history,
        error=meta.error,
    )


def run_manifest_to_metadata(manifest: RunManifest) -> RunMetadata:
    """Build the in-memory pipeline state from the canonical run manifest."""
    return RunMetadata(
        run_id=manifest.run_id,
        branch=manifest.branch,
        sha=manifest.sha,
        version=manifest.version,
        start_time=manifest.started_at,
        status=manifest.status,
        error=manifest.error,
        resume_history=manifest.resume_history,
        run_context=manifest.run_context,
        modal_app_name=manifest.modal_app_name,
        modal_environment=manifest.modal_environment,
        hf_staging_prefix=manifest.hf_staging_prefix,
    )


def write_run_manifest_for_meta(meta: RunMetadata) -> None:
    """Write the canonical run manifest for a pipeline run."""
    write_run_manifest(
        run_manifest_path(run_dir(meta.run_id)),
        build_run_manifest(meta),
    )


def write_run_meta(meta: RunMetadata, vol: Any) -> None:
    """Write the canonical run manifest for this run."""
    destination = run_dir(meta.run_id)
    destination.mkdir(parents=True, exist_ok=True)
    write_run_manifest_for_meta(meta)
    vol.commit()


def read_run_meta(run_id: str, vol: Any) -> RunMetadata:
    """Read run state from the canonical run manifest."""
    vol.reload()
    manifest_path = run_manifest_path(run_dir(run_id))
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No run manifest found for run {run_id} at {manifest_path}"
        )
    return run_manifest_to_metadata(read_run_manifest(manifest_path))


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
    step: PipelineStepRef,
    *,
    parameters: dict | None = None,
    input_identities: dict | None = None,
    scope: str | None = None,
    modal_call_id: str | None = None,
    vol: Any | None = None,
) -> StepManifest:
    manifest_step_id = step_id(step)
    manifest = StepManifest(
        run_id=meta.run_id,
        step_id=manifest_step_id,
        parent_step_id=parent_step_id(step),
        scope=scope,
        status="running",
        attempt=_next_step_attempt(meta.run_id, manifest_step_id),
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
    write_step_manifest(
        step_manifest_path(run_dir(meta.run_id), manifest_step_id), manifest
    )
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
    step: PipelineStepRef,
    decision,
    *,
    vol: Any,
) -> StepManifest:
    manifest_step_id = step_id(step)
    previous = decision.manifest
    if previous is None:
        raise RuntimeError(f"Cannot reuse {manifest_step_id}: missing prior manifest")
    reused = StepManifest(
        run_id=meta.run_id,
        step_id=manifest_step_id,
        parent_step_id=parent_step_id(step) or previous.parent_step_id,
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
    write_step_manifest(
        step_manifest_path(run_dir(meta.run_id), manifest_step_id), reused
    )
    write_run_meta(meta, vol)
    return reused


def step_reusable(
    meta: RunMetadata,
    step: PipelineStepRef,
    *,
    expected_input_identities: dict | None = None,
    expected_parameters: dict | None = None,
) -> object:
    manifest_step_id = step_id(step)
    return evaluate_step_reuse(
        step_manifest_path(run_dir(meta.run_id), manifest_step_id),
        expected_input_identities=expected_input_identities,
        expected_parameters=expected_parameters,
    )
