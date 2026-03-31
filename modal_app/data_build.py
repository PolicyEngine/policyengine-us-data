import functools
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as image

app = modal.App(os.environ.get("MODAL_APP_NAME", "policyengine-us-data"))

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

# Create persistent volume for checkpoints
checkpoint_volume = modal.Volume.from_name(
    "data-build-checkpoints",
    create_if_missing=True,
)

# Shared pipeline volume for inter-step artifact transport
pipeline_volume = modal.Volume.from_name(
    "pipeline-artifacts",
    create_if_missing=True,
)
PIPELINE_MOUNT = "/pipeline"

VOLUME_MOUNT = "/checkpoints"
_volume_lock = threading.Lock()

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

# Test modules to run individually for checkpoint tracking
TEST_MODULES = [
    "policyengine_us_data/tests/unit/",
    "policyengine_us_data/tests/integration/",
]

# Short names for --script mode (maps to SCRIPT_OUTPUTS keys)
SCRIPT_SHORT_NAMES = {
    "download_prerequisites": "policyengine_us_data/storage/download_private_prerequisites.py",
    "uprating": "policyengine_us_data/utils/uprating.py",
    "acs": "policyengine_us_data/datasets/acs/acs.py",
    "irs_puf": "policyengine_us_data/datasets/puf/irs_puf.py",
    "cps": "policyengine_us_data/datasets/cps/cps.py",
    "puf": "policyengine_us_data/datasets/puf/puf.py",
    "extended_cps": "policyengine_us_data/datasets/cps/extended_cps.py",
    "enhanced_cps": "policyengine_us_data/datasets/cps/enhanced_cps.py",
    "stratified_cps": "policyengine_us_data/calibration/create_stratified_cps.py",
    "source_imputed_cps": "policyengine_us_data/calibration/create_source_imputed_cps.py",
    "small_enhanced_cps": "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
}

# Files downloaded by download_private_prerequisites.py that must be
# checkpointed so subsequent --script calls in separate containers
# can access them.
PREREQUISITE_FILES = [
    "policyengine_us_data/storage/puf_2015.csv",
    "policyengine_us_data/storage/demographics_2015.csv",
    "policyengine_us_data/storage/soi.csv",
    "policyengine_us_data/storage/np2023_d5_mid.csv",
    "policyengine_us_data/storage/calibration/policy_data.db",
]

# Integration tests to run after each script build.
# Scripts not listed here have no associated tests.
SCRIPT_TESTS = {
    "acs": ["policyengine_us_data/tests/integration/test_acs.py"],
    "cps": ["policyengine_us_data/tests/integration/test_cps.py"],
    "extended_cps": ["policyengine_us_data/tests/integration/test_extended_cps.py"],
    "enhanced_cps": [
        "policyengine_us_data/tests/integration/test_enhanced_cps.py",
        "policyengine_us_data/tests/integration/test_sparse_enhanced_cps.py",
        "policyengine_us_data/tests/integration/test_sipp_assets.py",
    ],
    "source_imputed_cps": [
        "policyengine_us_data/tests/integration/test_source_imputed_cps_masking.py",
        "policyengine_us_data/tests/integration/test_source_imputed_cps_consistency.py",
    ],
    "small_enhanced_cps": [
        "policyengine_us_data/tests/integration/test_small_enhanced_cps.py",
    ],
}


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


def _get_storage_folder() -> Path:
    """Resolve the installed package's STORAGE_FOLDER path.

    This is where Dataset classes (CPS_2024, etc.) look for H5 files.
    In an editable install it matches the source tree; in a regular
    install it's inside .venv/lib/.../site-packages/.
    """
    try:
        from policyengine_us_data.storage import STORAGE_FOLDER

        return Path(STORAGE_FOLDER)
    except ImportError:
        # Fallback if package not importable (shouldn't happen in
        # the Modal image, but safe for local dev)
        return Path("policyengine_us_data/storage")


def get_checkpoint_path(branch: str, output_file: str) -> Path:
    """Get the checkpoint path for an output file, scoped by branch and commit.

    Preserves the relative path structure to avoid filename collisions
    (e.g., calibration/policy_data.db stays distinct from policy_data.db).
    """
    commit = get_current_commit()
    # Use the relative path as-is (not just filename) to avoid collisions
    return Path(VOLUME_MOUNT) / branch / commit / output_file


def is_checkpointed(branch: str, output_file: str) -> bool:
    """Check if output file exists in checkpoint volume and is valid."""
    checkpoint_path = get_checkpoint_path(branch, output_file)
    if checkpoint_path.exists():
        # Verify file is not empty/corrupted
        if checkpoint_path.stat().st_size > 0:
            return True
    return False


def _resolve_local_path(output_file: str) -> Path:
    """Resolve where a checkpointed file should be restored to.

    Maps the relative source-tree path to the installed package's
    STORAGE_FOLDER so that Dataset classes can find the files.
    """
    output_path = Path(output_file)
    storage_folder = _get_storage_folder()

    # Files under policyengine_us_data/storage/ get mapped to
    # the installed package's STORAGE_FOLDER
    storage_prefix = Path("policyengine_us_data/storage")
    try:
        relative = output_path.relative_to(storage_prefix)
        return storage_folder / relative
    except ValueError:
        # Not under storage/ — use the path as-is (relative to cwd)
        return output_path


def restore_from_checkpoint(branch: str, output_file: str) -> bool:
    """Restore output file from checkpoint volume to STORAGE_FOLDER.

    Writes to the installed package's storage directory so that
    Dataset classes (which use STORAGE_FOLDER) can find the files.
    """
    checkpoint_path = get_checkpoint_path(branch, output_file)
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        local_path = _resolve_local_path(output_file)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, local_path)
        # Also restore to the source-tree relative path so that
        # scripts run via subprocess (which use cwd-relative paths)
        # can find the file.
        source_path = Path(output_file)
        if source_path != local_path:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, source_path)
        print(f"Restored from checkpoint: {output_file}")
        return True
    return False


def save_checkpoint(
    branch: str,
    output_file: str,
    volume: modal.Volume,
) -> None:
    """Save output file to checkpoint volume.

    Checks both the installed package path and the source-tree
    relative path to find the file.
    """
    local_path = _resolve_local_path(output_file)
    source_path = Path(output_file)
    # Try installed path first, fall back to source-tree path
    actual_path = None
    if local_path.exists() and local_path.stat().st_size > 0:
        actual_path = local_path
    elif source_path.exists() and source_path.stat().st_size > 0:
        actual_path = source_path

    if actual_path:
        checkpoint_path = get_checkpoint_path(branch, output_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actual_path, checkpoint_path)
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
    """Run a script with uv and return its path for logging.

    Args:
        script_path: Path to the Python script to run.
        args: Optional list of command-line arguments.
        env: Optional environment variables dict.

    Returns:
        The script_path that was executed.

    Raises:
        subprocess.CalledProcessError: If the script fails.
    """
    cmd = ["uv", "run", "python", "-u", script_path]
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


def run_script_with_checkpoint(
    script_path: str,
    output_files: str | list[str],
    branch: str,
    volume: modal.Volume,
    args: Optional[list] = None,
    env: Optional[dict] = None,
    log_file: IO = None,
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

    # Check if ALL outputs are checkpointed
    all_checkpointed = all(is_checkpointed(branch, f) for f in output_files)

    if all_checkpointed:
        # Restore all files from checkpoint
        for output_file in output_files:
            restore_from_checkpoint(branch, output_file)
        print(f"Skipping {script_path} (restored from checkpoint)")
        return script_path

    # Run the script
    run_script(script_path, args=args, env=env, log_file=log_file)

    # Checkpoint all outputs
    for output_file in output_files:
        save_checkpoint(branch, output_file, volume)

    return script_path


def run_tests_with_checkpoints(
    branch: str,
    volume: modal.Volume,
    env: dict,
) -> None:
    """Run tests module-by-module, checkpointing progress.

    Args:
        branch: Git branch name for checkpoint scoping.
        volume: Modal volume for checkpointing.
        env: Environment variables dict.

    Raises:
        RuntimeError: If any test module fails.
    """
    commit = get_current_commit()
    checkpoint_dir = Path(VOLUME_MOUNT) / branch / commit / "tests"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for module in TEST_MODULES:
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

        print(f"Running tests: {module}")
        result = subprocess.run(
            ["uv", "run", "python", "-u", "-m", "pytest", module, "-v"],
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Tests failed: {module}")

        # Mark as passed
        marker_file.touch()
        volume.commit()
        print(f"Checkpointed: {module} passed")


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
def build_datasets(
    upload: bool = False,
    branch: str = "main",
    sequential: bool = False,
    clear_checkpoints: bool = False,
    skip_tests: bool = False,
    skip_enhanced_cps: bool = False,
    run_id: str = "",
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
    """
    setup_gcp_credentials()

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
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    log_file.write(
        f"{'=' * 40}\n"
        f" Data Build Log\n"
        f" Branch:  {branch}\n"
        f" Commit:  {commit[:8]}\n"
        f" Started: {started}\n"
        f"{'=' * 40}\n"
    )
    log_file.flush()

    # Download prerequisites
    run_script(
        "policyengine_us_data/storage/download_private_prerequisites.py",
        env=env,
        log_file=log_file,
    )
    # Checkpoint policy_data.db immediately after download so it survives
    # test failures and can be restored on retries.
    save_checkpoint(
        branch,
        "policyengine_us_data/storage/calibration/policy_data.db",
        checkpoint_volume,
    )

    if sequential:
        for script, output in SCRIPT_OUTPUTS.items():
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
                ): script
                for script, output in group1
            }
            for future in as_completed(futures):
                future.result()  # Raises if script failed

        # GROUP 2: Depends on Group 1 - run in parallel
        # cps.py needs acs, puf.py needs irs_puf + uprating
        print("=== Phase 2: Building CPS and PUF (parallel) ===")
        group2 = [
            (
                "policyengine_us_data/datasets/cps/cps.py",
                SCRIPT_OUTPUTS["policyengine_us_data/datasets/cps/cps.py"],
            ),
            (
                "policyengine_us_data/datasets/puf/puf.py",
                SCRIPT_OUTPUTS["policyengine_us_data/datasets/puf/puf.py"],
            ),
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    run_script_with_checkpoint,
                    script,
                    output,
                    branch,
                    checkpoint_volume,
                    env=env,
                    log_file=log_file,
                ): script
                for script, output in group2
            }
            for future in as_completed(futures):
                future.result()

        # SEQUENTIAL: Extended CPS (needs both cps and puf)
        print("=== Phase 3: Building extended CPS ===")
        run_script_with_checkpoint(
            "policyengine_us_data/datasets/cps/extended_cps.py",
            SCRIPT_OUTPUTS["policyengine_us_data/datasets/cps/extended_cps.py"],
            branch,
            checkpoint_volume,
            env=env,
            log_file=log_file,
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
                )
            )
            for future in as_completed(phase4_futures):
                future.result()

        # GROUP 4: After Phase 4 - run in parallel
        # create_source_imputed_cps needs stratified_cps
        # small_enhanced_cps needs enhanced_cps
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
    log_file.close()
    pipeline_volume.commit()
    print("Pipeline artifacts committed to shared volume")

    # Run tests with checkpointing
    if skip_tests:
        print("Skipping tests (--skip-tests)")
    else:
        print("=== Running tests with checkpointing ===")
        run_tests_with_checkpoints(branch, checkpoint_volume, env)

    # Upload if requested (HF publication only)
    if upload:
        upload_args = []
        if skip_enhanced_cps:
            upload_args.append("--no-require-enhanced-cps")
        run_script(
            "policyengine_us_data/storage/upload_completed_datasets.py",
            args=upload_args,
            env=env,
        )

    # Clean up checkpoints after successful completion
    cleanup_checkpoints(branch, checkpoint_volume)

    return "Data build completed successfully"


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: checkpoint_volume,
        PIPELINE_MOUNT: pipeline_volume,
    },
    memory=32768,
    cpu=8.0,
    timeout=14400,
    nonpreemptible=True,
)
def run_single_script(
    script_name: str,
    branch: str = "main",
    run_tests: bool = False,
) -> str:
    """Run a single dataset build script with checkpointing.

    Optionally runs associated integration tests after the build,
    inside the same container where the data was just created.

    Args:
        script_name: Short name (e.g. 'cps') or full path to the script.
        branch: Git branch for checkpoint scoping.
        run_tests: If True, run integration tests for this dataset
            after building.

    Returns:
        Status message.

    Raises:
        subprocess.CalledProcessError: If the build or tests fail.
    """
    setup_gcp_credentials()
    os.chdir("/root/policyengine-us-data")

    # Reload volume to see writes from prior --script containers
    checkpoint_volume.reload()

    # Resolve short name to full path
    script_path = SCRIPT_SHORT_NAMES.get(script_name, script_name)

    # Handle download_prerequisites specially (no SCRIPT_OUTPUTS entry)
    if script_name == "download_prerequisites":
        run_script(script_path)
        # Checkpoint prerequisite files so subsequent containers can
        # restore them.
        for prereq in PREREQUISITE_FILES:
            save_checkpoint(branch, prereq, checkpoint_volume)
        return f"Completed {script_name}"

    output_files = SCRIPT_OUTPUTS.get(script_path)
    if output_files is None:
        raise ValueError(
            f"Unknown script: {script_name}. "
            f"Valid names: {', '.join(SCRIPT_SHORT_NAMES.keys())}"
        )

    # Restore prerequisite files from checkpoint volume
    for prereq in PREREQUISITE_FILES:
        restore_from_checkpoint(branch, prereq)

    # Restore any existing checkpoints for dependencies
    for dep_path, dep_outputs in SCRIPT_OUTPUTS.items():
        if dep_path == script_path:
            continue
        if isinstance(dep_outputs, str):
            dep_outputs = [dep_outputs]
        for dep_output in dep_outputs:
            restore_from_checkpoint(branch, dep_output)

    run_script_with_checkpoint(
        script_path,
        output_files,
        branch,
        checkpoint_volume,
    )

    # Run associated integration tests inside this container
    if run_tests:
        test_paths = SCRIPT_TESTS.get(script_name, [])
        if test_paths:
            print(f"\n=== Running integration tests for {script_name} ===")
            cmd = ["uv", "run", "python", "-m", "pytest", "-v", "--tb=short"]
            cmd.extend(test_paths)
            subprocess.run(cmd, check=True, env=os.environ.copy())
            print(f"=== Tests passed for {script_name} ===")
        else:
            print(f"No integration tests defined for {script_name}")

    return f"Completed {script_name}"


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={
        VOLUME_MOUNT: checkpoint_volume,
        PIPELINE_MOUNT: pipeline_volume,
    },
    memory=32768,
    cpu=8.0,
    timeout=3600,
    nonpreemptible=True,
)
def run_integration_test(
    test_path: str,
    branch: str = "main",
) -> str:
    """Run integration tests inside Modal where built data exists.

    Restores all checkpointed artifacts (prerequisites + datasets),
    then runs pytest on the given test path.

    Args:
        test_path: Path to a test file or directory.
        branch: Git branch for checkpoint scoping.

    Returns:
        Status message.

    Raises:
        subprocess.CalledProcessError: If tests fail.
    """
    setup_gcp_credentials()
    os.chdir("/root/policyengine-us-data")

    # Reload volume to see writes from prior containers
    checkpoint_volume.reload()

    # Restore all prerequisites and dataset outputs
    for prereq in PREREQUISITE_FILES:
        restore_from_checkpoint(branch, prereq)
    for dep_path, dep_outputs in SCRIPT_OUTPUTS.items():
        if isinstance(dep_outputs, str):
            dep_outputs = [dep_outputs]
        for dep_output in dep_outputs:
            restore_from_checkpoint(branch, dep_output)

    print(f"\n=== Running integration test: {test_path} ===")
    cmd = ["uv", "run", "python", "-m", "pytest", test_path, "-v", "--tb=short"]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return f"Tests passed: {test_path}"


@app.local_entrypoint()
def main(
    upload: bool = False,
    branch: str = "main",
    sequential: bool = False,
    clear_checkpoints: bool = False,
    skip_tests: bool = False,
    skip_enhanced_cps: bool = False,
    script: str = "",
    run_tests: bool = False,
    test: str = "",
):
    if test:
        result = run_integration_test.remote(
            test_path=test,
            branch=branch,
        )
        print(result)
    elif script:
        result = run_single_script.remote(
            script_name=script,
            branch=branch,
            run_tests=run_tests,
        )
        print(result)
    else:
        result = build_datasets.remote(
            upload=upload,
            branch=branch,
            sequential=sequential,
            clear_checkpoints=clear_checkpoints,
            skip_tests=skip_tests,
            skip_enhanced_cps=skip_enhanced_cps,
        )
        print(result)
