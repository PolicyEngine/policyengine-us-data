from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping
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
    DatasetBuildContext,
    DatasetBuildOutputContractBuilder,
    PipelineArtifactStager,
    Stage1Coordinator,
    Stage1IdentityMaterial,
    Stage1RerunPlanner,
    Stage1ReuseDecision,
    Stage1ValidationResultWriter,
    Stage1ValidationRunner,
    ValidationTargetCatalog,
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
    ValidationReport,
)
from policyengine_us_data.utils.run_context import (  # noqa: E402
    CANDIDATE_VERSION_ENV,
    DATA_PACKAGE_VERSION_ENV,
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


def _utc_timestamp(value: datetime | None = None) -> str:
    """Render a UTC timestamp for persisted pipeline metadata."""
    value = value or datetime.now(timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
) -> subprocess.CompletedProcess:
    """Run a command, streaming output to both stdout and a log file."""
    command = DatasetCommand(
        name=" ".join(cmd),
        argv=tuple(cmd),
        kind="side_effect",
        metadata={"command": cmd},
    )
    result = CommandRunner().run(
        command,
        env=env,
        log_file=log_file,
        check=check,
    )
    return subprocess.CompletedProcess(cmd, result.returncode)


def run_script(
    script_path: str,
    args: Optional[list] = None,
    env: Optional[dict] = None,
    log_file: IO = None,
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
        CommandRunner().run(command, env=run_env, log_file=log_file)
    else:
        CommandRunner().run(command, env=run_env)
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
    checkpoint_store: CheckpointStore | None = None,
    reuse_decision: Stage1ReuseDecision | None = None,
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
    reuse_decision = reuse_decision or _compat_reuse_decision(
        script_path=script_path,
        output_files=output_files,
        branch=branch,
    )
    if reuse_decision.action == "blocked":
        raise RuntimeError(
            "Stage 1 checkpoint reuse is blocked for "
            f"{script_path}: {reuse_decision.reason}"
        )

    # Check if ALL outputs are checkpointed
    checkpoint_decisions = checkpoint_store.decisions_for(output_files)
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
        return script_path

    missing_or_invalid = sum(
        1 for decision in checkpoint_decisions if decision.action != "reuse"
    )

    # Run the script
    run_script(script_path, args=args, env=env, log_file=log_file)

    # Checkpoint all outputs
    for output_file in output_files:
        saved = checkpoint_store.save_output(output_file)
        if saved:
            print(f"Checkpointed: {output_file}")
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


def _compat_reuse_decision(
    *,
    script_path: str,
    output_files: list[str],
    branch: str,
    run_id: str | None = None,
    rerun_id: str | None = None,
) -> Stage1ReuseDecision:
    """Return the current compatible semantic reuse decision for a script."""

    substep_id = stage_1_substep_id_for_script(script_path)
    material = Stage1IdentityMaterial(
        substep_id=substep_id,
        inputs={"script_path": script_path},
        parameters={"branch": branch, "outputs": output_files},
        artifact_specs=[
            {
                "filename": spec.filename,
                "logical_name": spec.logical_name,
                "storage_path": spec.storage_path,
                "substage_id": spec.substage_id,
            }
            for spec in stage_1_artifact_specs()
            if spec.script_path == script_path
        ],
        code_sha=get_current_commit(),
        schema_version="stage-1-rerun-v1",
        upstream_contract_fingerprints=(),
        randomness={"checkpoint_scope": "branch_commit"},
    )
    planner = Stage1RerunPlanner(
        previous_identities={substep_id: material.fingerprint()}
    )
    return planner.decide(
        material,
        run_id=run_id or os.environ.get("US_DATA_RUN_ID", "unknown"),
        rerun_id=rerun_id,
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
        )

    checkpoint_store = _checkpoint_store(branch, volume)
    reuse_decision = _compat_reuse_decision(
        script_path=script_path,
        output_files=output_list,
        branch=branch,
    )
    checkpoint_decisions = checkpoint_store.decisions_for(output_list)

    def action() -> str:
        return run_script_with_checkpoint(
            script_path,
            output_files,
            branch,
            volume,
            env=env,
            log_file=log_file,
            checkpoint_stats=checkpoint_stats,
            checkpoint_store=checkpoint_store,
            reuse_decision=reuse_decision,
        )

    substep_id = stage_1_substep_id_for_script(script_path)
    return coordinator.run_substep(
        substep_id,
        stage_1_substep_title(substep_id),
        action,
        command_names=(script_path,),
        artifact_paths=_output_paths(output_files),
        reuse_decision=reuse_decision.to_dict(),
        checkpoint_decisions=tuple(
            decision.to_dict() for decision in checkpoint_decisions
        ),
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
    validation: ValidationReport | None = None,
    substage_validation: Mapping[str, ValidationReport] | None = None,
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
        validation=validation,
        substage_validation=substage_validation,
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
        artifacts_out=["source_imputed_*.h5", "policy_data.db"],
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
    os.environ["US_DATA_RUN_ID"] = run_id
    version = version or DATA_PACKAGE_VERSION
    os.environ[CANDIDATE_VERSION_ENV] = version
    os.environ[DATA_PACKAGE_VERSION_ENV] = version

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

    env = os.environ.copy()

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
    validation_runner = None
    if not skip_tests:
        validation_runner = Stage1ValidationRunner(
            run_id=run_id,
            catalog=ValidationTargetCatalog.from_stage_1_specs(
                skip_enhanced_cps=skip_enhanced_cps,
                skip_stage_5=skip_stage_5,
            ),
            metadata={"branch": branch, "code_sha": commit},
        )
    coordinator = Stage1Coordinator(validation_runner=validation_runner)
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
        )

    def run_raw_data_download() -> None:
        run_script(
            "policyengine_us_data/storage/download_prerequisites.py",
            env=env,
            log_file=log_file,
        )
        env["PYTHONUNBUFFERED"] = "1"
        log_file.write(f"\n{'=' * 60}\nStarting make database...\n{'=' * 60}\n")
        log_file.flush()
        run_script_logged(["make", "database"], log_file, env)
        # Checkpoint policy_data.db immediately after build so it survives
        # test failures and can be restored on retries.
        save_checkpoint(
            branch,
            "policyengine_us_data/storage/calibration/policy_data.db",
            checkpoint_volume,
        )

    coordinator.run_substep(
        "1a_raw_data_download",
        stage_1_substep_title("1a_raw_data_download"),
        run_raw_data_download,
        command_names=(
            "policyengine_us_data/storage/download_prerequisites.py",
            "make database",
        ),
        artifact_paths=("policyengine_us_data/storage/calibration/policy_data.db",),
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
                        script_path="policyengine_us_data/datasets/cps/enhanced_cps.py",
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
                            "policyengine_us_data/calibration/create_source_imputed_cps.py"
                        ),
                        output_files=SCRIPT_OUTPUTS[
                            "policyengine_us_data/calibration/create_source_imputed_cps.py"
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
                                "policyengine_us_data/datasets/cps/small_enhanced_cps.py"
                            ),
                            output_files=SCRIPT_OUTPUTS[
                                "policyengine_us_data/datasets/cps/small_enhanced_cps.py"
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

    # Checkpoint the build log so it survives preemption
    log_file.flush()
    save_checkpoint(branch, str(log_path), checkpoint_volume)

    # Copy pipeline artifacts to shared volume before tests so that a test
    # failure does not block downstream calibration steps.
    print("Copying pipeline artifacts to shared volume...")
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
    validation_report: ValidationReport | None = None
    validation_diagnostics: tuple = ()
    substage_validation: Mapping[str, ValidationReport] = {}
    if skip_tests:
        print("Skipping Stage 1 validation (--skip-tests)")
        validation_report = ValidationReport(
            status="not_run",
            metadata={
                "stage_id": "1_build_datasets",
                "run_id": run_id,
                "skip_reason": "--skip-tests",
            },
        )
    else:
        print("Writing Stage 1 validation artifacts...")
        validation_reports = [
            ValidationReport.from_dict(result.validation_report)
            for result in coordinator.results
            if result.validation_report is not None
        ]
        validation_summary = Stage1ValidationResultWriter(
            output_dir=Path(PIPELINE_MOUNT) / "runs" / run_id / "validation"
        ).write(validation_reports)
        validation_report = validation_summary.report
        validation_diagnostics = validation_summary.diagnostics
        substage_validation = validation_summary.substage_reports
    stage_1_status_metadata = {
        "substep_results": [result.to_dict() for result in coordinator.results],
        "status_events": [event.to_dict() for event in coordinator.status_events],
        "error_records": [error.to_dict() for error in coordinator.error_records],
    }
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
        diagnostics=(*diagnostics, *validation_diagnostics),
        validation=validation_report,
        substage_validation=substage_validation,
        stage_1_status_metadata=stage_1_status_metadata,
    )
    pipeline_volume.commit()
    print("Pipeline artifacts committed to shared volume")

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
