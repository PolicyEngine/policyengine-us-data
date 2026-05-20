from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Optional

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as image  # noqa: E402
from policyengine_us_data.__version__ import __version__ as DATA_PACKAGE_VERSION  # noqa: E402
from policyengine_us_data.build_datasets import (  # noqa: E402
    CheckpointStore,
    CommandRunner,
    DatasetCommand,
    DatasetCommandError,
    DatasetCommandResult,
    DatasetBuildContext,
    DatasetBuildOutputContractBuilder,
    PipelineArtifactStager,
    Stage1Coordinator,
    Stage1IdentityMaterial,
    Stage1RerunPlanner,
    Stage1ReuseDecision,
    Stage1ReuseManifestRecord,
    Stage1StatusRecorder,
    stage_1_artifact_specs,
    stage_1_script_outputs,
    stage_1_substep_id_for_script,
    stage_1_substep_title,
    write_stage_1_diagnostics,
)
from policyengine_us_data.pipeline_metadata import pipeline_node  # noqa: E402
from policyengine_us_data.pipeline_schema import PipelineNode  # noqa: E402
from policyengine_us_data.stage_contracts import (  # noqa: E402
    StageContract,
)
from policyengine_us_data.utils.run_context import (  # noqa: E402
    CANDIDATE_VERSION_ENV,
    DATA_PACKAGE_VERSION_ENV,
    RUN_ID_ENV,
    resolve_run_id,
)

app = modal.App(
    os.environ.get("US_DATA_DATA_BUILD_APP_NAME")
    or os.environ.get("US_DATA_MODAL_APP_NAME")
    or "policyengine-us-data"
)

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

# Create persistent volume for checkpoints
checkpoint_volume = modal.Volume.from_name(
    os.environ.get("US_DATA_CHECKPOINT_VOLUME_NAME", "data-build-checkpoints"),
    create_if_missing=True,
)

# Shared pipeline volume for inter-step artifact transport
pipeline_volume = modal.Volume.from_name(
    os.environ.get("US_DATA_PIPELINE_VOLUME_NAME", "pipeline-artifacts"),
    create_if_missing=True,
    version=2,
)
PIPELINE_MOUNT = "/pipeline"

VOLUME_MOUNT = "/checkpoints"
_volume_lock = threading.Lock()


@dataclass
class CheckpointStats:
    expected_outputs: int = 0
    valid_reused_outputs: int = 0
    recomputed_outputs: int = 0
    invalid_outputs: int = 0
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def record(
        self,
        *,
        expected_outputs: int,
        valid_reused_outputs: int = 0,
        recomputed_outputs: int = 0,
        invalid_outputs: int = 0,
    ) -> None:
        with self._lock:
            self.expected_outputs += expected_outputs
            self.valid_reused_outputs += valid_reused_outputs
            self.recomputed_outputs += recomputed_outputs
            self.invalid_outputs += invalid_outputs

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "expected_outputs": self.expected_outputs,
                "valid_reused_outputs": self.valid_reused_outputs,
                "recomputed_outputs": self.recomputed_outputs,
                "invalid_outputs": self.invalid_outputs,
            }


# Script to output file mapping for checkpointing.
# Values can be a single file path (str) or a list of file paths.
SCRIPT_OUTPUTS = stage_1_script_outputs()

CPS_BUILD_SCRIPT = "policyengine_us_data/datasets/cps/cps.py"
PUF_BUILD_SCRIPT = "policyengine_us_data/datasets/puf/puf.py"

# Post-build validation modules to run individually for checkpoint tracking.
VALIDATION_MODULES = [
    "validation/stage_1/",
]


def _python_cmd(*args: str) -> list[str]:
    """Build a command that uses the current interpreter."""
    return [sys.executable, *args]


def _utc_timestamp(value: datetime | None = None) -> str:
    """Render a UTC timestamp for persisted pipeline metadata."""
    value = value or datetime.now(timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _dataset_build_env(*, run_id: str, version: str) -> dict[str, str]:
    """Return the child-process environment for one dataset-build run."""

    env = os.environ.copy()
    env[RUN_ID_ENV] = run_id
    env[CANDIDATE_VERSION_ENV] = version
    env[DATA_PACKAGE_VERSION_ENV] = version
    return env


def setup_gcp_credentials():
    """Write GCP credentials JSON to a temp file for google.auth.default()."""
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_path = "/tmp/gcp-credentials.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        return creds_path
    return None


@functools.cache
def get_current_commit() -> str:
    """Get the current git commit SHA (cached per process).

    Checks BUILD_COMMIT_SHA env var first (set at image build time
    from the local .git), then falls back to git and finally a hash
    of pyproject.toml.
    """
    env_sha = os.environ.get("BUILD_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        import hashlib

        version_file = Path("/root/policyengine-us-data/pyproject.toml")
        if version_file.exists():
            content = version_file.read_bytes()
            return hashlib.sha256(content).hexdigest()[:12]
        return "unknown"


def get_checkpoint_path(branch: str, output_file: str) -> Path:
    """Get the checkpoint path for an output file, scoped by branch and commit."""
    return _checkpoint_store(branch).checkpoint_path(output_file)


def is_checkpointed(branch: str, output_file: str) -> bool:
    """Check if output file exists in checkpoint volume and is valid."""
    return _checkpoint_store(branch).decision_for(output_file).action == "reuse"


def restore_from_checkpoint(branch: str, output_file: str) -> bool:
    """Restore output file from checkpoint volume if it exists."""
    restored = _checkpoint_store(branch).restore_output(output_file)
    if restored:
        print(f"Restored from checkpoint: {output_file}")
    return restored


def save_checkpoint(
    branch: str,
    output_file: str,
    volume: modal.Volume,
) -> None:
    """Save output file to checkpoint volume."""
    saved = _checkpoint_store(branch, volume).save_output(output_file)
    if saved:
        print(f"Checkpointed: {output_file}")


def cleanup_checkpoints(branch: str, volume: modal.Volume) -> None:
    """Delete checkpoints for this branch after successful completion."""
    cleaned = _checkpoint_store(branch, volume).cleanup_branch()
    if cleaned:
        print(f"Cleaned up checkpoints for branch: {branch}")


def _checkpoint_store(
    branch: str,
    volume: modal.Volume | None = None,
) -> CheckpointStore:
    def commit_volume() -> None:
        if volume is not None:
            with _volume_lock:
                volume.commit()

    return CheckpointStore(
        root=Path(VOLUME_MOUNT),
        branch=branch,
        commit_sha=get_current_commit(),
        commit=commit_volume if volume is not None else None,
    )


def run_script_logged(
    cmd: list,
    log_file: IO,
    env: dict,
    check: bool = True,
    command_results: list[DatasetCommandResult] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, streaming output to both stdout and a log file."""
    command = DatasetCommand(
        name=" ".join(cmd),
        argv=tuple(cmd),
        kind="side_effect",
        metadata={"command": cmd},
    )
    try:
        result = CommandRunner().run(
            command,
            env=env,
            log_file=log_file,
            check=check,
        )
    except DatasetCommandError as exc:
        if command_results is not None:
            command_results.append(exc.result)
        raise
    if command_results is not None:
        command_results.append(result)
    return subprocess.CompletedProcess(cmd, result.returncode)


def run_script(
    script_path: str,
    args: Optional[list] = None,
    env: Optional[dict] = None,
    log_file: IO = None,
    command_results: list[DatasetCommandResult] | None = None,
) -> str:
    """Run a script with the current interpreter and return its path.

    Args:
        script_path: Path to the Python script to run.
        args: Optional list of command-line arguments.
        env: Optional environment variables dict.

    Returns:
        The script_path that was executed.

    Raises:
        DatasetCommandError: If the script fails.
    """
    command = DatasetCommand.from_script(
        script_path,
        args=tuple(args or ()),
        python_executable=sys.executable,
    )
    run_env = env or os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    print(f"Starting {script_path}...")
    if log_file:
        log_file.write(f"\n{'=' * 60}\nStarting {script_path}...\n{'=' * 60}\n")
        log_file.flush()
        try:
            result = CommandRunner().run(command, env=run_env, log_file=log_file)
        except DatasetCommandError as exc:
            if command_results is not None:
                command_results.append(exc.result)
            raise
    else:
        try:
            result = CommandRunner().run(command, env=run_env)
        except DatasetCommandError as exc:
            if command_results is not None:
                command_results.append(exc.result)
            raise
    if command_results is not None:
        command_results.append(result)
    print(f"Completed {script_path}")
    return script_path


def validate_and_maybe_upload_datasets(
    *,
    upload: bool,
    skip_enhanced_cps: bool,
    env: dict,
    require_small_enhanced_cps: bool = True,
    stage_only: bool = False,
    run_id: str = "",
    version: str = DATA_PACKAGE_VERSION,
) -> None:
    validation_args = ["--validate-only"]
    if skip_enhanced_cps:
        validation_args.append("--no-require-enhanced-cps")
    elif not require_small_enhanced_cps:
        validation_args.append("--no-require-small-enhanced-cps")

    print("=== Validating built datasets ===")
    run_script(
        "policyengine_us_data/storage/upload_completed_datasets.py",
        args=validation_args,
        env=env,
    )

    if upload:
        upload_args = []
        if skip_enhanced_cps:
            upload_args.append("--no-require-enhanced-cps")
        elif not require_small_enhanced_cps:
            upload_args.append("--no-require-small-enhanced-cps")
        if stage_only:
            upload_args.append("--stage-only")
        if run_id:
            upload_args.append(f"--run-id={run_id}")
        if version:
            upload_args.append(f"--version={version}")
        run_script(
            "policyengine_us_data/storage/upload_completed_datasets.py",
            args=upload_args,
            env=env,
        )


def run_script_with_checkpoint(
    script_path: str,
    output_files: str | list[str],
    branch: str,
    volume: modal.Volume,
    args: Optional[list] = None,
    env: Optional[dict] = None,
    log_file: IO = None,
    checkpoint_stats: CheckpointStats | None = None,
    command_results: list[DatasetCommandResult] | None = None,
    checkpoint_store: CheckpointStore | None = None,
    reuse_decision: Stage1ReuseDecision | None = None,
    identity_material: Stage1IdentityMaterial | None = None,
) -> str:
    """Run script if output not checkpointed, then checkpoint result.

    Args:
        script_path: Path to the Python script to run.
        output_files: Path(s) to output file(s) produced by the script.
            Can be a single string or a list of strings.
        branch: Git branch name for checkpoint scoping.
        volume: Modal volume for checkpointing.
        args: Optional list of command-line arguments.
        env: Optional environment variables dict.

    Returns:
        The script_path that was executed.
    """
    # Normalize to list
    if isinstance(output_files, str):
        output_files = [output_files]
    expected_count = len(output_files)
    checkpoint_store = checkpoint_store or _checkpoint_store(branch, volume)
    identity_material = identity_material or _script_identity_material(
        script_path=script_path,
        output_files=output_files,
        branch=branch,
    )
    reuse_decision = reuse_decision or _reuse_decision_for_material(
        identity_material,
        checkpoint_store=checkpoint_store,
        run_id=(env or {}).get(RUN_ID_ENV),
    )
    if reuse_decision.action == "blocked":
        raise RuntimeError(
            "Stage 1 checkpoint reuse is blocked for "
            f"{script_path}: {reuse_decision.reason}"
        )

    # Check if ALL outputs are checkpointed
    checkpoint_decisions = (
        checkpoint_store.decisions_for(output_files)
        if reuse_decision.action == "reuse"
        else ()
    )
    all_checkpointed = reuse_decision.action == "reuse" and all(
        decision.action == "reuse" for decision in checkpoint_decisions
    )

    if all_checkpointed:
        # Restore all files from checkpoint
        checkpoint_store.restore_all_outputs(output_files)
        for output_file in output_files:
            print(f"Restored from checkpoint: {output_file}")
        print(f"Skipping {script_path} (restored from checkpoint)")
        if checkpoint_stats is not None:
            checkpoint_stats.record(
                expected_outputs=expected_count,
                valid_reused_outputs=expected_count,
            )
        _record_reuse_manifest(
            checkpoint_store=checkpoint_store,
            identity_material=identity_material,
            reuse_decision=reuse_decision,
            checkpoint_decisions=checkpoint_decisions,
        )
        return script_path

    missing_or_invalid = sum(
        1 for decision in checkpoint_decisions if decision.action != "reuse"
    )

    # Run the script
    run_script(
        script_path,
        args=args,
        env=env,
        log_file=log_file,
        command_results=command_results,
    )

    # Checkpoint all outputs
    for output_file in output_files:
        saved = checkpoint_store.save_output(output_file)
        if saved:
            print(f"Checkpointed: {output_file}")
    saved_decisions = checkpoint_store.decisions_for(output_files)
    _record_reuse_manifest(
        checkpoint_store=checkpoint_store,
        identity_material=identity_material,
        reuse_decision=reuse_decision,
        checkpoint_decisions=saved_decisions,
    )
    if checkpoint_stats is not None:
        checkpoint_stats.record(
            expected_outputs=expected_count,
            recomputed_outputs=expected_count,
            invalid_outputs=missing_or_invalid,
        )

    return script_path


def _output_paths(output_files: str | list[str]) -> tuple[Path, ...]:
    paths = output_files if isinstance(output_files, list) else [output_files]
    return tuple(Path(path) for path in paths)


def _stage_base_artifact_paths(artifacts_dir: Path) -> tuple[Path, ...]:
    paths = [
        artifacts_dir / spec.filename
        for spec in stage_1_artifact_specs()
        if spec.substage_id == "1g_stage_base_datasets"
    ]
    paths.append(artifacts_dir / "dataset_build_output.json")
    return tuple(paths)


def _stage_1_status_metadata(coordinator: Stage1Coordinator) -> dict[str, Any]:
    substep_results = [
        result.to_dict() for result in getattr(coordinator, "results", ())
    ]
    status_events = [
        event.to_dict() for event in getattr(coordinator, "status_events", ())
    ]
    error_records = [
        error.to_dict() for error in getattr(coordinator, "error_records", ())
    ]
    return {
        "substep_results": substep_results,
        "status_events": status_events,
        "error_records": error_records,
        "reuse_reasoning": _reuse_reasoning(substep_results),
    }


def _reuse_reasoning(
    substep_results: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "substep_id": str(result.get("substep_id", "")),
            "title": result.get("title"),
            "status": result.get("status"),
            "outcome_reason": _reuse_outcome_reason(result),
            "reuse_decision": result.get("reuse_decision"),
            "identity_decisions": _identity_reuse_decisions(
                result.get("reuse_decision")
            ),
            "checkpoint_summary": _checkpoint_summary(
                result.get("checkpoint_decisions", ())
            ),
        }
        for result in substep_results
        if result.get("reuse_decision") or result.get("checkpoint_decisions")
    ]


def _reuse_outcome_reason(result: Mapping[str, Any]) -> str:
    status = result.get("status")
    reuse_decision = result.get("reuse_decision")
    action = _reuse_action(reuse_decision)
    semantic_reason = _reuse_reason(reuse_decision)
    reason_counts = _reuse_reason_counts(reuse_decision)
    checkpoint_summary = _checkpoint_summary(result.get("checkpoint_decisions", ()))

    if status == "reused":
        return (
            "Prior semantic identity matched and every expected checkpoint "
            "output was present and non-empty."
        )
    if action == "blocked":
        return f"Semantic reuse blocked: {semantic_reason or 'unspecified'}."
    if checkpoint_summary["missing"] or checkpoint_summary["empty"]:
        return (
            "Recomputed because at least one expected checkpoint output was "
            "missing or empty."
        )
    if semantic_reason == "no_previous_identity":
        return (
            "Recomputed because no prior semantic identity manifest record "
            "existed for this Stage 1 execution unit."
        )
    if semantic_reason == "identity_mismatch":
        return (
            "Recomputed because persisted semantic identity did not match the "
            "current Stage 1 execution-unit identity."
        )
    if reason_counts:
        return (
            "Recomputed with per-identity semantic reuse reasons: "
            f"{_format_reason_counts(reason_counts)}."
        )
    if action == "recompute":
        return f"Recomputed because semantic reuse returned {semantic_reason}."
    return "Recomputed because no reusable checkpoint decision was available."


def _reuse_action(reuse_decision: object) -> str | None:
    if not isinstance(reuse_decision, Mapping):
        return None
    if isinstance(reuse_decision.get("action"), str):
        return str(reuse_decision["action"])
    nested = reuse_decision.get("decisions")
    if isinstance(nested, list):
        actions = {
            item.get("action")
            for item in nested
            if isinstance(item, Mapping) and isinstance(item.get("action"), str)
        }
        if len(actions) == 1:
            return str(next(iter(actions)))
        if "blocked" in actions:
            return "blocked"
        if "recompute" in actions:
            return "recompute"
    return None


def _reuse_reason(reuse_decision: object) -> str | None:
    if not isinstance(reuse_decision, Mapping):
        return None
    if isinstance(reuse_decision.get("reason"), str):
        return str(reuse_decision["reason"])
    nested = reuse_decision.get("decisions")
    if isinstance(nested, list):
        reasons = [
            str(item["reason"])
            for item in nested
            if isinstance(item, Mapping) and isinstance(item.get("reason"), str)
        ]
        if reasons:
            return ", ".join(sorted(set(reasons)))
    return None


def _identity_reuse_decisions(reuse_decision: object) -> list[dict[str, Any]]:
    decisions = _reuse_decision_records(reuse_decision)
    identity_fields = (
        "substep_id",
        "identity_key",
        "action",
        "reason",
        "identity_fingerprint",
        "artifact_namespace",
        "run_id",
        "rerun_id",
    )
    return [
        {
            field: _metadata_scalar(decision[field])
            for field in identity_fields
            if field in decision
        }
        for decision in decisions
    ]


def _reuse_reason_counts(reuse_decision: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in _reuse_decision_records(reuse_decision):
        reason = decision.get("reason")
        if isinstance(reason, str) and reason:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _reuse_decision_records(reuse_decision: object) -> list[Mapping[str, Any]]:
    if not isinstance(reuse_decision, Mapping):
        return []
    nested = reuse_decision.get("decisions")
    if isinstance(nested, list):
        return [item for item in nested if isinstance(item, Mapping)]
    return [reuse_decision]


def _format_reason_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{reason}={counts[reason]}" for reason in sorted(counts))


def _metadata_scalar(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _checkpoint_summary(checkpoint_decisions: object) -> dict[str, int]:
    summary = {"total": 0, "reusable": 0, "missing": 0, "empty": 0, "other": 0}
    if not isinstance(checkpoint_decisions, (list, tuple)):
        return summary
    for decision in checkpoint_decisions:
        if not isinstance(decision, Mapping):
            continue
        summary["total"] += 1
        action = decision.get("action")
        reason = decision.get("reason")
        if action == "reuse":
            summary["reusable"] += 1
        elif reason == "missing":
            summary["missing"] += 1
        elif reason == "empty":
            summary["empty"] += 1
        else:
            summary["other"] += 1
    return summary


def _identity_material_for_substep(
    *,
    substep_id: str,
    identity_key: str,
    inputs: Mapping[str, Any],
    output_files: list[str],
    branch: str,
    artifact_specs: list[Mapping[str, Any]],
) -> Stage1IdentityMaterial:
    return Stage1IdentityMaterial(
        substep_id=substep_id,
        identity_key=identity_key,
        inputs=inputs,
        parameters={"branch": branch, "outputs": output_files},
        artifact_specs=artifact_specs,
        code_sha=get_current_commit(),
        schema_version="stage-1-rerun-v1",
        upstream_contract_fingerprints=(),
        randomness={"checkpoint_scope": "branch_commit"},
    )


def _stage_1_identity_key(
    *,
    substep_id: str,
    execution_id: str,
    output_files: Sequence[str],
) -> str:
    output_names = ",".join(
        sorted(Path(output_file).name for output_file in output_files)
    )
    return f"{substep_id}:{execution_id}:{output_names}"


def _reuse_decision_for_material(
    material: Stage1IdentityMaterial,
    *,
    checkpoint_store: CheckpointStore,
    run_id: str | None = None,
    rerun_id: str | None = None,
) -> Stage1ReuseDecision:
    manifest = checkpoint_store.load_reuse_manifest()
    planner = Stage1RerunPlanner(previous_identities=manifest.previous_identities())
    return planner.decide(
        material,
        run_id=run_id or os.environ.get(RUN_ID_ENV, "unknown"),
        rerun_id=rerun_id,
    )


def _artifact_spec_identity(
    *,
    script_path: str | None = None,
    output_files: list[str] | None = None,
) -> list[Mapping[str, Any]]:
    output_names = {Path(output_file).name for output_file in output_files or ()}
    return [
        {
            "filename": spec.filename,
            "logical_name": spec.logical_name,
            "storage_path": spec.storage_path,
            "substage_id": spec.substage_id,
        }
        for spec in stage_1_artifact_specs()
        if (
            (script_path is not None and spec.script_path == script_path)
            or spec.filename in output_names
        )
    ]


def _raw_data_command_names() -> tuple[str, ...]:
    return (
        "policyengine_us_data/storage/download_prerequisites.py",
        "make database",
    )


def _raw_data_outputs() -> tuple[str, ...]:
    return ("policyengine_us_data/storage/calibration/policy_data.db",)


def _raw_data_identity_material(*, branch: str) -> Stage1IdentityMaterial:
    output_files = list(_raw_data_outputs())
    return _identity_material_for_substep(
        substep_id="1a_raw_data_download",
        identity_key=_stage_1_identity_key(
            substep_id="1a_raw_data_download",
            execution_id="command_group:raw_data_download",
            output_files=output_files,
        ),
        inputs={"commands": list(_raw_data_command_names())},
        output_files=output_files,
        branch=branch,
        artifact_specs=_artifact_spec_identity(output_files=output_files),
    )


def _raw_data_reuse_decision(
    *,
    branch: str,
    checkpoint_store: CheckpointStore,
    run_id: str | None = None,
    rerun_id: str | None = None,
) -> Stage1ReuseDecision:
    material = _raw_data_identity_material(branch=branch)
    return _reuse_decision_for_material(
        material,
        checkpoint_store=checkpoint_store,
        run_id=run_id,
        rerun_id=rerun_id,
    )


def run_command_group_with_checkpoint(
    *,
    substep_id: str,
    output_files: tuple[str, ...],
    branch: str,
    volume: modal.Volume,
    action: Callable[[], None],
    checkpoint_stats: CheckpointStats | None = None,
    checkpoint_store: CheckpointStore | None = None,
    reuse_decision: Stage1ReuseDecision | None = None,
    identity_material: Stage1IdentityMaterial | None = None,
) -> str:
    """Run a non-script command group behind the checkpoint/reuse gate."""

    expected_count = len(output_files)
    checkpoint_store = checkpoint_store or _checkpoint_store(branch, volume)
    identity_material = identity_material or _identity_material_for_substep(
        substep_id=substep_id,
        identity_key=_stage_1_identity_key(
            substep_id=substep_id,
            execution_id="command_group",
            output_files=output_files,
        ),
        inputs={},
        output_files=list(output_files),
        branch=branch,
        artifact_specs=_artifact_spec_identity(output_files=list(output_files)),
    )
    reuse_decision = reuse_decision or _reuse_decision_for_material(
        identity_material,
        checkpoint_store=checkpoint_store,
    )
    if reuse_decision.action == "blocked":
        raise RuntimeError(
            "Stage 1 checkpoint reuse is blocked for "
            f"{substep_id}: {reuse_decision.reason}"
        )

    checkpoint_decisions = (
        checkpoint_store.decisions_for(output_files)
        if reuse_decision.action == "reuse"
        else ()
    )
    all_checkpointed = reuse_decision.action == "reuse" and all(
        decision.action == "reuse" for decision in checkpoint_decisions
    )
    if all_checkpointed:
        checkpoint_store.restore_all_outputs(output_files)
        for output_file in output_files:
            print(f"Restored from checkpoint: {output_file}")
        print(f"Skipping {substep_id} (restored from checkpoint)")
        if checkpoint_stats is not None:
            checkpoint_stats.record(
                expected_outputs=expected_count,
                valid_reused_outputs=expected_count,
            )
        _record_reuse_manifest(
            checkpoint_store=checkpoint_store,
            identity_material=identity_material,
            reuse_decision=reuse_decision,
            checkpoint_decisions=checkpoint_decisions,
        )
        return substep_id

    missing_or_invalid = sum(
        1 for decision in checkpoint_decisions if decision.action != "reuse"
    )
    action()
    for output_file in output_files:
        saved = checkpoint_store.save_output(output_file)
        if saved:
            print(f"Checkpointed: {output_file}")
    saved_decisions = checkpoint_store.decisions_for(output_files)
    _record_reuse_manifest(
        checkpoint_store=checkpoint_store,
        identity_material=identity_material,
        reuse_decision=reuse_decision,
        checkpoint_decisions=saved_decisions,
    )
    if checkpoint_stats is not None:
        checkpoint_stats.record(
            expected_outputs=expected_count,
            recomputed_outputs=expected_count,
            invalid_outputs=missing_or_invalid,
        )
    return substep_id


def _record_reuse_manifest(
    *,
    checkpoint_store: CheckpointStore,
    identity_material: Stage1IdentityMaterial,
    reuse_decision: Stage1ReuseDecision,
    checkpoint_decisions: tuple,
) -> None:
    if not checkpoint_decisions or any(
        decision.action != "reuse" for decision in checkpoint_decisions
    ):
        return
    checkpoint_store.record_reuse_manifest(
        Stage1ReuseManifestRecord(
            substep_id=identity_material.substep_id,
            identity_key=identity_material.identity_key,
            identity_fingerprint=identity_material.fingerprint(),
            identity_material=identity_material.to_dict(),
            reuse_decision=reuse_decision.to_dict(),
            checkpoint_summary=_checkpoint_summary(
                tuple(decision.to_dict() for decision in checkpoint_decisions)
            ),
        )
    )


def _script_identity_material(
    *,
    script_path: str,
    output_files: list[str],
    branch: str,
) -> Stage1IdentityMaterial:
    substep_id = stage_1_substep_id_for_script(script_path)
    return _identity_material_for_substep(
        substep_id=substep_id,
        identity_key=_stage_1_identity_key(
            substep_id=substep_id,
            execution_id=f"script:{script_path}",
            output_files=output_files,
        ),
        inputs={"script_path": script_path},
        output_files=output_files,
        branch=branch,
        artifact_specs=_artifact_spec_identity(
            script_path=script_path,
            output_files=output_files,
        ),
    )


def _run_checkpointed_substep(
    *,
    coordinator: Stage1Coordinator | None,
    script_path: str,
    output_files: str | list[str],
    branch: str,
    volume: modal.Volume,
    env: dict,
    log_file: IO = None,
    checkpoint_stats: CheckpointStats | None = None,
) -> str:
    command_results: list[DatasetCommandResult] = []
    output_list = output_files if isinstance(output_files, list) else [output_files]
    if coordinator is None:
        return run_script_with_checkpoint(
            script_path,
            output_files,
            branch,
            volume,
            env=env,
            log_file=log_file,
            checkpoint_stats=checkpoint_stats,
            command_results=command_results,
        )

    checkpoint_store = _checkpoint_store(branch, volume)
    identity_material = _script_identity_material(
        script_path=script_path,
        output_files=output_list,
        branch=branch,
    )
    reuse_decision = _reuse_decision_for_material(
        identity_material,
        checkpoint_store=checkpoint_store,
        run_id=env.get(RUN_ID_ENV),
    )
    checkpoint_decisions = (
        checkpoint_store.decisions_for(output_list)
        if reuse_decision.action == "reuse"
        else ()
    )

    def action() -> str:
        return run_script_with_checkpoint(
            script_path,
            output_files,
            branch,
            volume,
            env=env,
            log_file=log_file,
            checkpoint_stats=checkpoint_stats,
            command_results=command_results,
            checkpoint_store=checkpoint_store,
            reuse_decision=reuse_decision,
            identity_material=identity_material,
        )

    substep_id = stage_1_substep_id_for_script(script_path)
    return coordinator.run_substep(
        substep_id,
        stage_1_substep_title(substep_id),
        action,
        command_names=(script_path,),
        command_results=command_results,
        artifact_paths=_output_paths(output_files),
        reuse_decision=reuse_decision.to_dict(),
        checkpoint_decisions=tuple(
            decision.to_dict() for decision in checkpoint_decisions
        ),
        aggregate=True,
    )


@pipeline_node(
    PipelineNode(
        id="cps_puf_build_phase",
        label="CPS Then PUF Build Phase",
        node_type="entrypoint",
        description="Build CPS before PUF to avoid shared raw-cache and fixture races.",
        source_file="modal_app/data_build.py",
        status="current",
        stability="moving",
        pathways=["data_build"],
        validation_commands=["uv run pytest tests/unit/test_modal_data_build.py"],
    )
)
def run_cps_then_puf_phase(
    branch: str,
    volume: modal.Volume,
    *,
    env: dict,
    log_file: IO = None,
    checkpoint_stats: CheckpointStats | None = None,
    coordinator: Stage1Coordinator | None = None,
) -> None:
    """Build CPS before PUF because PUF pension imputation loads CPS_2024."""
    for script in (CPS_BUILD_SCRIPT, PUF_BUILD_SCRIPT):
        _run_checkpointed_substep(
            coordinator=coordinator,
            script_path=script,
            output_files=SCRIPT_OUTPUTS[script],
            branch=branch,
            volume=volume,
            env=env,
            log_file=log_file,
            checkpoint_stats=checkpoint_stats,
        )


def run_tests_with_checkpoints(
    branch: str,
    volume: modal.Volume,
    env: dict,
) -> None:
    """Run post-build validators module-by-module, checkpointing progress.

    Args:
        branch: Git branch name for checkpoint scoping.
        volume: Modal volume for checkpointing.
        env: Environment variables dict.

    Raises:
        RuntimeError: If any validation module fails.
    """
    commit = get_current_commit()
    checkpoint_dir = Path(VOLUME_MOUNT) / branch / commit / "tests"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for module in VALIDATION_MODULES:
        # Use stem for files, or last component for directories
        module_path = Path(module)
        if module_path.suffix:
            module_name = module_path.stem
        else:
            module_name = module_path.name.rstrip("/")

        marker_file = checkpoint_dir / f"{module_name}.passed"

        if marker_file.exists():
            print(f"Skipping {module} (already passed)")
            continue

        print(f"Running validation: {module}")
        result = subprocess.run(
            _python_cmd("-u", "-m", "pytest", module, "-v"),
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Validation failed: {module}")

        # Mark as passed
        marker_file.touch()
        volume.commit()
        print(f"Checkpointed: {module} passed")


def write_dataset_build_contract(
    *,
    artifacts_dir: Path,
    run_id: str,
    code_sha: str,
    checkpoint_stats: Mapping[str, int],
    started_at: str | None,
    completed_at: str,
    duration_s: float | None,
    upload_requested: bool,
    stage_only: bool,
    skip_enhanced_cps: bool,
    skip_stage_5: bool = False,
    package_version: str = DATA_PACKAGE_VERSION,
    branch: str = "unknown",
    diagnostics: tuple = (),
    stage_1_status_metadata: Mapping[str, Any] | None = None,
) -> StageContract:
    """Write the Stage 1 semantic handoff contract next to copied artifacts."""
    context = DatasetBuildContext(
        run_id=run_id,
        branch=branch,
        code_sha=code_sha,
        package_version=package_version,
        artifacts_dir=artifacts_dir,
    )
    return DatasetBuildOutputContractBuilder(context=context).write(
        checkpoint_stats=checkpoint_stats,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        upload_requested=upload_requested,
        stage_only=stage_only,
        skip_enhanced_cps=skip_enhanced_cps,
        skip_stage_5=skip_stage_5,
        diagnostics=diagnostics,
        stage_1_status_metadata=stage_1_status_metadata,
    )


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: checkpoint_volume,
        PIPELINE_MOUNT: pipeline_volume,
    },
    memory=32768,
    cpu=8.0,
    timeout=28800,  # 8 hours
    nonpreemptible=True,
)
@pipeline_node(
    PipelineNode(
        id="build_datasets",
        label="Build Datasets On Modal",
        node_type="entrypoint",
        description="Build base datasets, source-imputed artifacts, and optional uploads inside the Modal runtime.",
        source_file="modal_app/data_build.py",
        status="current",
        stability="moving",
        pathways=["data_build", "orchestration"],
        artifacts_out=[
            "dataset_build_output.json",
            "dataset_inventory.json",
            "source_dataset_schema_summary.json",
            "target_database_schema_summary.json",
            "source_imputed_stratified_extended_cps_2024.h5",
            "source_imputed_stratified_extended_cps.h5",
            "policy_data.db",
        ],
        validation_commands=["uv run pytest tests/unit/test_modal_data_build.py"],
    )
)
def build_datasets(
    upload: bool = False,
    branch: str = "main",
    sequential: bool = False,
    clear_checkpoints: bool = False,
    skip_tests: bool = False,
    skip_enhanced_cps: bool = False,
    skip_stage_5: bool = False,
    stage_only: bool = False,
    run_id: str = "",
    version: str = DATA_PACKAGE_VERSION,
):
    """Build all datasets with preemption-resilient checkpointing.

    Args:
        upload: Whether to upload completed datasets.
        branch: Git branch to build from.
        sequential: Use sequential (non-parallel) execution.
        clear_checkpoints: Clear existing checkpoints before starting.
        skip_tests: Skip running the test suite (useful for calibration runs).
        skip_enhanced_cps: Skip enhanced_cps.py and small_enhanced_cps.py
            (useful for calibration runs that only need source_imputed H5).
        skip_stage_5: Skip source-imputed CPS and small enhanced CPS after
            enhanced_cps_2024.h5 is built.
        stage_only: Upload to HF staging only, without promoting a release.
        version: policyengine-us-data package version used for staging and
            dataset-build contracts.
    """
    setup_gcp_credentials()
    checkpoint_stats = CheckpointStats()
    run_id = run_id or resolve_run_id()
    if not run_id:
        raise RuntimeError(
            "run_id is required. Production data builds must receive the "
            "GitHub-created run ID via --run-id or US_DATA_RUN_ID."
        )
    version = version or DATA_PACKAGE_VERSION
    env = _dataset_build_env(run_id=run_id, version=version)

    # Reload volume to see latest checkpoints
    checkpoint_volume.reload()

    if clear_checkpoints:
        branch_dir = Path(VOLUME_MOUNT) / branch
        if branch_dir.exists():
            shutil.rmtree(branch_dir)
            checkpoint_volume.commit()
        print(f"Cleared checkpoints for branch: {branch}")

    os.chdir("/root/policyengine-us-data")

    # Clean stale checkpoints from other commits
    for removed_checkpoint in _checkpoint_store(
        branch,
        checkpoint_volume,
    ).cleanup_other_commits():
        print(f"Removed stale checkpoint dir: {removed_checkpoint.name[:12]}")

    # Open persistent build log with provenance header
    commit = get_current_commit()
    log_path = Path("build_log.txt")
    log_file = open(log_path, "w")
    started_at_dt = datetime.now(timezone.utc)
    started = _utc_timestamp(started_at_dt)
    log_file.write(
        f"{'=' * 40}\n"
        f" Data Build Log\n"
        f" Branch:  {branch}\n"
        f" Commit:  {commit[:8]}\n"
        f" Started (UTC): {started}\n"
        f"{'=' * 40}\n"
    )
    log_file.flush()
    coordinator = Stage1Coordinator()
    if run_id:
        coordinator.status_recorder = Stage1StatusRecorder(
            Path(PIPELINE_MOUNT) / "runs" / run_id,
            commit_callback=pipeline_volume.commit,
        )
    recorded_skips: set[tuple[str, str]] = set()

    def record_skipped_script(script: str, reason: str) -> None:
        substep_id = stage_1_substep_id_for_script(script)
        if reason == "--skip-stage-5" and substep_id != "1f_source_imputation":
            return
        key = (substep_id, reason)
        if key in recorded_skips:
            return
        recorded_skips.add(key)
        coordinator.run_substep(
            substep_id,
            stage_1_substep_title(substep_id),
            lambda: None,
            command_names=(script,),
            skip=True,
            skip_reason=reason,
            aggregate=True,
        )

    raw_data_command_results: list[DatasetCommandResult] = []
    raw_data_outputs = _raw_data_outputs()
    raw_data_checkpoint_store = _checkpoint_store(branch, checkpoint_volume)
    raw_data_identity_material = _raw_data_identity_material(branch=branch)
    raw_data_reuse_decision = _raw_data_reuse_decision(
        branch=branch,
        checkpoint_store=raw_data_checkpoint_store,
        run_id=run_id,
    )
    raw_data_checkpoint_decisions = (
        raw_data_checkpoint_store.decisions_for(raw_data_outputs)
        if raw_data_reuse_decision.action == "reuse"
        else ()
    )

    def run_raw_data_download() -> None:
        def build_raw_data() -> None:
            run_script(
                "policyengine_us_data/storage/download_prerequisites.py",
                env=env,
                log_file=log_file,
                command_results=raw_data_command_results,
            )
            env["PYTHONUNBUFFERED"] = "1"
            log_file.write(f"\n{'=' * 60}\nStarting make database...\n{'=' * 60}\n")
            log_file.flush()
            run_script_logged(
                ["make", "database"],
                log_file,
                env,
                command_results=raw_data_command_results,
            )

        run_command_group_with_checkpoint(
            substep_id="1a_raw_data_download",
            output_files=raw_data_outputs,
            branch=branch,
            volume=checkpoint_volume,
            action=build_raw_data,
            checkpoint_stats=checkpoint_stats,
            checkpoint_store=raw_data_checkpoint_store,
            reuse_decision=raw_data_reuse_decision,
            identity_material=raw_data_identity_material,
        )

    try:
        coordinator.run_substep(
            "1a_raw_data_download",
            stage_1_substep_title("1a_raw_data_download"),
            run_raw_data_download,
            command_names=_raw_data_command_names(),
            command_results=raw_data_command_results,
            artifact_paths=raw_data_outputs,
            reuse_decision=raw_data_reuse_decision.to_dict(),
            checkpoint_decisions=tuple(
                decision.to_dict() for decision in raw_data_checkpoint_decisions
            ),
            aggregate=True,
        )

        if sequential:
            for script, output in SCRIPT_OUTPUTS.items():
                if skip_stage_5 and script in (
                    "policyengine_us_data/calibration/create_source_imputed_cps.py",
                    "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
                ):
                    print(f"Skipping {script} (--skip-stage-5)")
                    record_skipped_script(script, "--skip-stage-5")
                    continue
                if skip_enhanced_cps and script in (
                    "policyengine_us_data/datasets/cps/enhanced_cps.py",
                    "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
                ):
                    print(f"Skipping {script} (--skip-enhanced-cps)")
                    record_skipped_script(script, "--skip-enhanced-cps")
                    continue
                _run_checkpointed_substep(
                    coordinator=coordinator,
                    script_path=script,
                    output_files=output,
                    branch=branch,
                    volume=checkpoint_volume,
                    env=env,
                    log_file=log_file,
                    checkpoint_stats=checkpoint_stats,
                )
        else:
            # Parallel execution based on dependency groups with checkpointing
            # GROUP 1: Independent scripts - run in parallel
            print("=== Phase 1: Building independent datasets (parallel) ===")
            group1 = [
                (
                    "policyengine_us_data/utils/uprating.py",
                    SCRIPT_OUTPUTS["policyengine_us_data/utils/uprating.py"],
                ),
                (
                    "policyengine_us_data/datasets/acs/acs.py",
                    SCRIPT_OUTPUTS["policyengine_us_data/datasets/acs/acs.py"],
                ),
                (
                    "policyengine_us_data/datasets/puf/irs_puf.py",
                    SCRIPT_OUTPUTS["policyengine_us_data/datasets/puf/irs_puf.py"],
                ),
            ]
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        _run_checkpointed_substep,
                        coordinator=coordinator,
                        script_path=script,
                        output_files=output,
                        branch=branch,
                        volume=checkpoint_volume,
                        env=env,
                        log_file=log_file,
                        checkpoint_stats=checkpoint_stats,
                    ): script
                    for script, output in group1
                }
                for future in as_completed(futures):
                    future.result()  # Raises if script failed

            # GROUP 2: Depends on Group 1 - run sequentially.
            # puf.py pension imputation can instantiate CPS_2024, so it must
            # not run while cps.py is writing cps_2024.h5.
            print("=== Phase 2: Building CPS then PUF (sequential) ===")
            run_cps_then_puf_phase(
                branch,
                checkpoint_volume,
                env=env,
                log_file=log_file,
                checkpoint_stats=checkpoint_stats,
                coordinator=coordinator,
            )

            # SEQUENTIAL: Extended CPS (needs both cps and puf)
            print("=== Phase 3: Building extended CPS ===")
            _run_checkpointed_substep(
                coordinator=coordinator,
                script_path="policyengine_us_data/datasets/cps/extended_cps.py",
                output_files=SCRIPT_OUTPUTS[
                    "policyengine_us_data/datasets/cps/extended_cps.py"
                ],
                branch=branch,
                volume=checkpoint_volume,
                env=env,
                log_file=log_file,
                checkpoint_stats=checkpoint_stats,
            )

            # GROUP 3: After extended_cps - run in parallel
            # enhanced_cps and stratified_cps both depend on extended_cps
            print("=== Phase 4: Building enhanced and stratified CPS (parallel) ===")
            phase4_futures = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                if not skip_enhanced_cps:
                    phase4_futures.append(
                        executor.submit(
                            _run_checkpointed_substep,
                            coordinator=coordinator,
                            script_path=(
                                "policyengine_us_data/datasets/cps/enhanced_cps.py"
                            ),
                            output_files=SCRIPT_OUTPUTS[
                                "policyengine_us_data/datasets/cps/enhanced_cps.py"
                            ],
                            branch=branch,
                            volume=checkpoint_volume,
                            env=env,
                            log_file=log_file,
                            checkpoint_stats=checkpoint_stats,
                        )
                    )
                else:
                    print("Skipping enhanced_cps.py (--skip-enhanced-cps)")
                    record_skipped_script(
                        "policyengine_us_data/datasets/cps/enhanced_cps.py",
                        "--skip-enhanced-cps",
                    )
                phase4_futures.append(
                    executor.submit(
                        _run_checkpointed_substep,
                        coordinator=coordinator,
                        script_path=(
                            "policyengine_us_data/calibration/create_stratified_cps.py"
                        ),
                        output_files=SCRIPT_OUTPUTS[
                            "policyengine_us_data/calibration/create_stratified_cps.py"
                        ],
                        branch=branch,
                        volume=checkpoint_volume,
                        env=env,
                        log_file=log_file,
                        checkpoint_stats=checkpoint_stats,
                    )
                )
                for future in as_completed(phase4_futures):
                    future.result()

            # GROUP 4: After Phase 4 - run in parallel
            # create_source_imputed_cps needs stratified_cps
            # small_enhanced_cps needs enhanced_cps
            if skip_stage_5:
                print("Skipping Phase 5 (--skip-stage-5)")
                record_skipped_script(
                    "policyengine_us_data/calibration/create_source_imputed_cps.py",
                    "--skip-stage-5",
                )
            else:
                print(
                    "=== Phase 5: Building source imputed CPS "
                    "and small enhanced CPS (parallel) ==="
                )
                phase5_futures = []
                with ThreadPoolExecutor(max_workers=2) as executor:
                    phase5_futures.append(
                        executor.submit(
                            _run_checkpointed_substep,
                            coordinator=coordinator,
                            script_path=(
                                "policyengine_us_data/calibration/"
                                "create_source_imputed_cps.py"
                            ),
                            output_files=SCRIPT_OUTPUTS[
                                "policyengine_us_data/calibration/"
                                "create_source_imputed_cps.py"
                            ],
                            branch=branch,
                            volume=checkpoint_volume,
                            env=env,
                            log_file=log_file,
                            checkpoint_stats=checkpoint_stats,
                        )
                    )
                    if not skip_enhanced_cps:
                        phase5_futures.append(
                            executor.submit(
                                _run_checkpointed_substep,
                                coordinator=coordinator,
                                script_path=(
                                    "policyengine_us_data/datasets/cps/"
                                    "small_enhanced_cps.py"
                                ),
                                output_files=SCRIPT_OUTPUTS[
                                    "policyengine_us_data/datasets/cps/"
                                    "small_enhanced_cps.py"
                                ],
                                branch=branch,
                                volume=checkpoint_volume,
                                env=env,
                                log_file=log_file,
                                checkpoint_stats=checkpoint_stats,
                            )
                        )
                    else:
                        print("Skipping small_enhanced_cps.py (--skip-enhanced-cps)")
                        record_skipped_script(
                            "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
                            "--skip-enhanced-cps",
                        )
                    for future in as_completed(phase5_futures):
                        future.result()
    finally:
        coordinator.finalize_results()

    # Copy pipeline artifacts to shared volume before tests so that a test
    # failure does not block downstream calibration steps.
    artifacts_dir = Path(PIPELINE_MOUNT) / "artifacts"
    if run_id:
        artifacts_dir = artifacts_dir / run_id
    build_context = DatasetBuildContext(
        run_id=run_id,
        branch=branch,
        code_sha=commit,
        package_version=version,
        artifacts_dir=artifacts_dir,
    )

    def run_stage_base_handoff() -> None:
        log_file.flush()
        save_checkpoint(branch, str(log_path), checkpoint_volume)
        print("Copying pipeline artifacts to shared volume...")
        stager = PipelineArtifactStager(context=build_context)
        staged_paths = stager.stage_declared_artifacts(
            skip_enhanced_cps=skip_enhanced_cps,
            skip_stage_5=skip_stage_5,
        )
        for staged_path in staged_paths:
            print(
                f"  Copied {staged_path.name} "
                f"({staged_path.stat().st_size / 1024 / 1024:.1f} MB)"
            )
        checkpoint_snapshot = checkpoint_stats.snapshot()
        stager.write_checkpoint_stats(checkpoint_snapshot)
        log_file.close()
        completed_at_dt = datetime.now(timezone.utc)
        diagnostics = write_stage_1_diagnostics(
            context=build_context,
            skip_enhanced_cps=skip_enhanced_cps,
            skip_stage_5=skip_stage_5,
        )
        write_dataset_build_contract(
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            code_sha=commit,
            checkpoint_stats=checkpoint_snapshot,
            started_at=started,
            completed_at=_utc_timestamp(completed_at_dt),
            duration_s=(completed_at_dt - started_at_dt).total_seconds(),
            upload_requested=upload,
            stage_only=stage_only,
            skip_enhanced_cps=skip_enhanced_cps,
            skip_stage_5=skip_stage_5,
            package_version=version,
            branch=branch,
            diagnostics=diagnostics,
            stage_1_status_metadata=_stage_1_status_metadata(coordinator),
        )
        pipeline_volume.commit()
        print("Pipeline artifacts committed to shared volume")

    coordinator.run_substep(
        "1g_stage_base_datasets",
        stage_1_substep_title("1g_stage_base_datasets"),
        run_stage_base_handoff,
        command_names=("stage_base_datasets",),
        artifact_paths=_stage_base_artifact_paths(artifacts_dir),
    )

    # Run post-build validators with checkpointing.
    if skip_tests:
        print("Skipping tests (--skip-tests)")
    else:
        print("=== Running post-build validation with checkpointing ===")
        run_tests_with_checkpoints(branch, checkpoint_volume, env)

    validate_and_maybe_upload_datasets(
        upload=upload,
        skip_enhanced_cps=skip_enhanced_cps,
        require_small_enhanced_cps=not skip_stage_5,
        env=env,
        stage_only=stage_only,
        run_id=run_id,
        version=version,
    )

    # Clean up checkpoints after successful completion
    cleanup_checkpoints(branch, checkpoint_volume)

    return "Data build completed successfully"


@app.local_entrypoint()
def main(
    upload: bool = False,
    branch: str = "main",
    sequential: bool = False,
    clear_checkpoints: bool = False,
    skip_tests: bool = False,
    skip_enhanced_cps: bool = False,
    skip_stage_5: bool = False,
    stage_only: bool = False,
    run_id: str = "",
    version: str = DATA_PACKAGE_VERSION,
):
    run_id = run_id or resolve_run_id()
    if not run_id:
        raise RuntimeError(
            "run_id is required. Pass --run-id or run inside GitHub Actions."
        )
    result = build_datasets.remote(
        upload=upload,
        branch=branch,
        sequential=sequential,
        clear_checkpoints=clear_checkpoints,
        skip_tests=skip_tests,
        skip_enhanced_cps=skip_enhanced_cps,
        skip_stage_5=skip_stage_5,
        stage_only=stage_only,
        run_id=run_id,
        version=version,
    )
    print(result)
