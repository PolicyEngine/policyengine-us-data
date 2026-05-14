import functools
import json
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
from policyengine_us_data.pipeline_metadata import pipeline_node  # noqa: E402
from policyengine_us_data.pipeline_schema import PipelineNode  # noqa: E402
from policyengine_us_data.stage_contracts import (  # noqa: E402
    DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
    StageContract,
    build_dataset_build_output_contract,
    write_contract,
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


# Script to output file mapping for checkpointing
# Values can be a single file path (str) or a list of file paths
SCRIPT_OUTPUTS = {
    "policyengine_us_data/utils/uprating.py": (
        "policyengine_us_data/storage/uprating_factors.csv"
    ),
    "policyengine_us_data/datasets/acs/acs.py": (
        "policyengine_us_data/storage/acs_2022.h5"
    ),
    "policyengine_us_data/datasets/puf/irs_puf.py": (
        "policyengine_us_data/storage/irs_puf_2015.h5"
    ),
    "policyengine_us_data/datasets/cps/cps.py": (
        "policyengine_us_data/storage/cps_2024.h5"
    ),
    "policyengine_us_data/datasets/puf/puf.py": (
        "policyengine_us_data/storage/puf_2024.h5"
    ),
    "policyengine_us_data/datasets/cps/extended_cps.py": (
        "policyengine_us_data/storage/extended_cps_2024.h5"
    ),
    # enhanced_cps.py produces both the dataset and calibration log
    "policyengine_us_data/datasets/cps/enhanced_cps.py": [
        "policyengine_us_data/storage/enhanced_cps_2024.h5",
        "policyengine_us_data/storage/enhanced_cps_2024.clone_diagnostics.json",
        "calibration_log.csv",
    ],
    "policyengine_us_data/calibration/create_stratified_cps.py": (
        "policyengine_us_data/storage/stratified_extended_cps_2024.h5"
    ),
    "policyengine_us_data/calibration/create_source_imputed_cps.py": (
        "policyengine_us_data/storage/source_imputed_stratified_extended_cps_2024.h5"
    ),
    "policyengine_us_data/datasets/cps/small_enhanced_cps.py": (
        "policyengine_us_data/storage/small_enhanced_cps_2024.h5"
    ),
}

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
    commit = get_current_commit()
    return Path(VOLUME_MOUNT) / branch / commit / Path(output_file).name


def is_checkpointed(branch: str, output_file: str) -> bool:
    """Check if output file exists in checkpoint volume and is valid."""
    checkpoint_path = get_checkpoint_path(branch, output_file)
    if checkpoint_path.exists():
        # Verify file is not empty/corrupted
        if checkpoint_path.stat().st_size > 0:
            return True
    return False


def restore_from_checkpoint(branch: str, output_file: str) -> bool:
    """Restore output file from checkpoint volume if it exists."""
    checkpoint_path = get_checkpoint_path(branch, output_file)
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        local_path = Path(output_file)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, local_path)
        print(f"Restored from checkpoint: {output_file}")
        return True
    return False


def save_checkpoint(
    branch: str,
    output_file: str,
    volume: modal.Volume,
) -> None:
    """Save output file to checkpoint volume."""
    local_path = Path(output_file)
    if local_path.exists() and local_path.stat().st_size > 0:
        checkpoint_path = get_checkpoint_path(branch, output_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, checkpoint_path)
        with _volume_lock:
            volume.commit()
        print(f"Checkpointed: {output_file}")


def cleanup_checkpoints(branch: str, volume: modal.Volume) -> None:
    """Delete checkpoints for this branch after successful completion."""
    branch_dir = Path(VOLUME_MOUNT) / branch
    if branch_dir.exists():
        shutil.rmtree(branch_dir)
        volume.commit()
        print(f"Cleaned up checkpoints for branch: {branch}")


def run_script_logged(
    cmd: list,
    log_file: IO,
    env: dict,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command, streaming output to both stdout and a log file."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
    proc.wait()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return subprocess.CompletedProcess(cmd, proc.returncode)


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
        subprocess.CalledProcessError: If the script fails.
    """
    script = Path(script_path)
    if (
        script.suffix == ".py"
        and script.parts
        and script.parts[0] in {"policyengine_us_data", "modal_app"}
    ):
        cmd = _python_cmd("-u", "-m", ".".join(script.with_suffix("").parts))
    else:
        cmd = _python_cmd("-u", script_path)
    if args:
        cmd.extend(args)
    run_env = env or os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    print(f"Starting {script_path}...")
    if log_file:
        log_file.write(f"\n{'=' * 60}\nStarting {script_path}...\n{'=' * 60}\n")
        log_file.flush()
        run_script_logged(cmd, log_file, run_env)
    else:
        subprocess.run(cmd, check=True, env=run_env)
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

    # Check if ALL outputs are checkpointed
    all_checkpointed = all(is_checkpointed(branch, f) for f in output_files)

    if all_checkpointed:
        # Restore all files from checkpoint
        for output_file in output_files:
            restore_from_checkpoint(branch, output_file)
        print(f"Skipping {script_path} (restored from checkpoint)")
        if checkpoint_stats is not None:
            checkpoint_stats.record(
                expected_outputs=expected_count,
                valid_reused_outputs=expected_count,
            )
        return script_path

    missing_or_invalid = sum(
        1 for output_file in output_files if not is_checkpointed(branch, output_file)
    )

    # Run the script
    run_script(script_path, args=args, env=env, log_file=log_file)

    # Checkpoint all outputs
    for output_file in output_files:
        save_checkpoint(branch, output_file, volume)
    if checkpoint_stats is not None:
        checkpoint_stats.record(
            expected_outputs=expected_count,
            recomputed_outputs=expected_count,
            invalid_outputs=missing_or_invalid,
        )

    return script_path


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
) -> None:
    """Build CPS before PUF because PUF pension imputation loads CPS_2024."""
    for script in (CPS_BUILD_SCRIPT, PUF_BUILD_SCRIPT):
        run_script_with_checkpoint(
            script,
            SCRIPT_OUTPUTS[script],
            branch,
            volume,
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
) -> StageContract:
    """Write the Stage 1 semantic handoff contract next to copied artifacts."""
    contract = build_dataset_build_output_contract(
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        code_sha=code_sha,
        package_version=package_version,
        checkpoint_stats=checkpoint_stats,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        upload_requested=upload_requested,
        stage_only=stage_only,
        skip_enhanced_cps=skip_enhanced_cps,
        skip_stage_5=skip_stage_5,
    )
    write_contract(
        contract,
        artifacts_dir / DATASET_BUILD_OUTPUT_CONTRACT_FILENAME,
    )
    return contract


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
    branch_dir = Path(VOLUME_MOUNT) / branch
    if branch_dir.exists():
        current_commit = get_current_commit()
        for entry in branch_dir.iterdir():
            if entry.is_dir() and entry.name != current_commit:
                shutil.rmtree(entry)
                print(f"Removed stale checkpoint dir: {entry.name[:12]}")
        checkpoint_volume.commit()

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

    # Download prerequisites
    run_script(
        "policyengine_us_data/storage/download_prerequisites.py",
        env=env,
        log_file=log_file,
    )
    # Build policy_data.db from source
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

    if sequential:
        for script, output in SCRIPT_OUTPUTS.items():
            if skip_stage_5 and script in (
                "policyengine_us_data/calibration/create_source_imputed_cps.py",
                "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
            ):
                print(f"Skipping {script} (--skip-stage-5)")
                continue
            if skip_enhanced_cps and script in (
                "policyengine_us_data/datasets/cps/enhanced_cps.py",
                "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
            ):
                print(f"Skipping {script} (--skip-enhanced-cps)")
                continue
            run_script_with_checkpoint(
                script,
                output,
                branch,
                checkpoint_volume,
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
                    run_script_with_checkpoint,
                    script,
                    output,
                    branch,
                    checkpoint_volume,
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
        )

        # SEQUENTIAL: Extended CPS (needs both cps and puf)
        print("=== Phase 3: Building extended CPS ===")
        run_script_with_checkpoint(
            "policyengine_us_data/datasets/cps/extended_cps.py",
            SCRIPT_OUTPUTS["policyengine_us_data/datasets/cps/extended_cps.py"],
            branch,
            checkpoint_volume,
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
                        run_script_with_checkpoint,
                        "policyengine_us_data/datasets/cps/enhanced_cps.py",
                        SCRIPT_OUTPUTS[
                            "policyengine_us_data/datasets/cps/enhanced_cps.py"
                        ],
                        branch,
                        checkpoint_volume,
                        env=env,
                        log_file=log_file,
                        checkpoint_stats=checkpoint_stats,
                    )
                )
            else:
                print("Skipping enhanced_cps.py (--skip-enhanced-cps)")
            phase4_futures.append(
                executor.submit(
                    run_script_with_checkpoint,
                    "policyengine_us_data/calibration/create_stratified_cps.py",
                    SCRIPT_OUTPUTS[
                        "policyengine_us_data/calibration/create_stratified_cps.py"
                    ],
                    branch,
                    checkpoint_volume,
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
        else:
            print(
                "=== Phase 5: Building source imputed CPS "
                "and small enhanced CPS (parallel) ==="
            )
            phase5_futures = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                phase5_futures.append(
                    executor.submit(
                        run_script_with_checkpoint,
                        "policyengine_us_data/calibration/create_source_imputed_cps.py",
                        SCRIPT_OUTPUTS[
                            "policyengine_us_data/calibration/create_source_imputed_cps.py"
                        ],
                        branch,
                        checkpoint_volume,
                        env=env,
                        log_file=log_file,
                        checkpoint_stats=checkpoint_stats,
                    )
                )
                if not skip_enhanced_cps:
                    phase5_futures.append(
                        executor.submit(
                            run_script_with_checkpoint,
                            "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
                            SCRIPT_OUTPUTS[
                                "policyengine_us_data/datasets/cps/small_enhanced_cps.py"
                            ],
                            branch,
                            checkpoint_volume,
                            env=env,
                            log_file=log_file,
                            checkpoint_stats=checkpoint_stats,
                        )
                    )
                else:
                    print("Skipping small_enhanced_cps.py (--skip-enhanced-cps)")
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
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Copy all intermediate H5 datasets for lineage tracing
    for output in SCRIPT_OUTPUTS.values():
        paths = output if isinstance(output, list) else [output]
        for p in paths:
            src = Path(p)
            if src.suffix == ".h5" and src.exists():
                shutil.copy2(src, artifacts_dir / src.name)
                print(
                    f"  Copied {src.name} ({src.stat().st_size / 1024 / 1024:.1f} MB)"
                )

    # Yearless alias for pipeline consumers (remote_calibration_runner, local_area)
    si = artifacts_dir / "source_imputed_stratified_extended_cps_2024.h5"
    if si.exists():
        shutil.copy2(si, artifacts_dir / "source_imputed_stratified_extended_cps.h5")

    shutil.copy2(
        "policyengine_us_data/storage/calibration/policy_data.db",
        artifacts_dir / "policy_data.db",
    )
    cal_weights = Path("policyengine_us_data/storage/calibration_weights.npy")
    if cal_weights.exists():
        shutil.copy2(
            cal_weights,
            artifacts_dir / "calibration_weights.npy",
        )
        print("  Copied calibration_weights.npy")
    shutil.copy2(log_path, artifacts_dir / "build_log.txt")
    checkpoint_snapshot = checkpoint_stats.snapshot()
    with open(artifacts_dir / "data_build_checkpoint_stats.json", "w") as f:
        json.dump(checkpoint_snapshot, f, indent=2, sort_keys=True)
    log_file.close()
    completed_at_dt = datetime.now(timezone.utc)
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
    )
    pipeline_volume.commit()
    print("Pipeline artifacts committed to shared volume")

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
