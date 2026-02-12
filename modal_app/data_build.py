import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import modal

app = modal.App("policyengine-us-data")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

# Create persistent volume for checkpoints
checkpoint_volume = modal.Volume.from_name(
    "data-build-checkpoints",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("uv")
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"
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
    "policyengine_us_data/datasets/cps/"
    "local_area_calibration/create_stratified_cps.py": (
        "policyengine_us_data/storage/stratified_extended_cps_2024.h5"
    ),
    "policyengine_us_data/datasets/cps/small_enhanced_cps.py": (
        "policyengine_us_data/storage/small_enhanced_cps_2024.h5"
    ),
}

# Test modules to run individually for checkpoint tracking
TEST_MODULES = [
    "policyengine_us_data/tests/test_import.py",
    "policyengine_us_data/tests/test_database.py",
    "policyengine_us_data/tests/test_pandas3_compatibility.py",
    "policyengine_us_data/tests/test_datasets/",
    "policyengine_us_data/tests/test_local_area_calibration/",
]


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


def get_checkpoint_path(branch: str, output_file: str) -> Path:
    """Get the checkpoint path for an output file, scoped by branch."""
    return Path(VOLUME_MOUNT) / branch / Path(output_file).name


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


def run_script(
    script_path: str,
    args: Optional[list] = None,
    env: Optional[dict] = None,
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
    cmd = ["uv", "run", "python", script_path]
    if args:
        cmd.extend(args)
    print(f"Starting {script_path}...")
    subprocess.run(cmd, check=True, env=env or os.environ.copy())
    print(f"Completed {script_path}")
    return script_path


def run_script_with_checkpoint(
    script_path: str,
    output_files: str | list[str],
    branch: str,
    volume: modal.Volume,
    args: Optional[list] = None,
    env: Optional[dict] = None,
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
    run_script(script_path, args=args, env=env)

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
    checkpoint_dir = Path(VOLUME_MOUNT) / branch / "tests"
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
            ["uv", "run", "pytest", module, "-v"],
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
    volumes={VOLUME_MOUNT: checkpoint_volume},
    memory=32768,
    cpu=8.0,
    timeout=14400,
)
def build_datasets(
    upload: bool = False,
    branch: str = "main",
    sequential: bool = False,
    clear_checkpoints: bool = False,
):
    """Build all datasets with preemption-resilient checkpointing.

    Args:
        upload: Whether to upload completed datasets.
        branch: Git branch to build from.
        sequential: Use sequential (non-parallel) execution.
        clear_checkpoints: Clear existing checkpoints before starting.
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

    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")
    # Use uv sync to install exact versions from uv.lock
    subprocess.run(["uv", "sync", "--locked"], check=True)

    env = os.environ.copy()

    # Download prerequisites
    run_script(
        "policyengine_us_data/storage/download_private_prerequisites.py",
        env=env,
    )

    if sequential:
        for script, output in SCRIPT_OUTPUTS.items():
            run_script_with_checkpoint(
                script,
                output,
                branch,
                checkpoint_volume,
                env=env,
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
                ): script
                for script, output in group2
            }
            for future in as_completed(futures):
                future.result()

        # SEQUENTIAL: Extended CPS (needs both cps and puf)
        print("=== Phase 3: Building extended CPS ===")
        run_script_with_checkpoint(
            "policyengine_us_data/datasets/cps/extended_cps.py",
            SCRIPT_OUTPUTS[
                "policyengine_us_data/datasets/cps/extended_cps.py"
            ],
            branch,
            checkpoint_volume,
            env=env,
        )

        # GROUP 3: After extended_cps - run in parallel
        # enhanced_cps and stratified_cps both depend on extended_cps
        print(
            "=== Phase 4: Building enhanced and stratified CPS (parallel)"
            " ==="
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_script_with_checkpoint,
                    "policyengine_us_data/datasets/cps/enhanced_cps.py",
                    SCRIPT_OUTPUTS[
                        "policyengine_us_data/datasets/cps/enhanced_cps.py"
                    ],
                    branch,
                    checkpoint_volume,
                    env=env,
                ),
                executor.submit(
                    run_script_with_checkpoint,
                    "policyengine_us_data/datasets/cps/"
                    "local_area_calibration/create_stratified_cps.py",
                    SCRIPT_OUTPUTS[
                        "policyengine_us_data/datasets/cps/"
                        "local_area_calibration/create_stratified_cps.py"
                    ],
                    branch,
                    checkpoint_volume,
                    env=env,
                ),
            ]
            for future in as_completed(futures):
                future.result()

        # SEQUENTIAL: Small enhanced CPS (needs enhanced_cps)
        print("=== Phase 5: Building small enhanced CPS ===")
        run_script_with_checkpoint(
            "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
            SCRIPT_OUTPUTS[
                "policyengine_us_data/datasets/cps/small_enhanced_cps.py"
            ],
            branch,
            checkpoint_volume,
            env=env,
        )

    # Run tests with checkpointing
    print("=== Running tests with checkpointing ===")
    run_tests_with_checkpoints(branch, checkpoint_volume, env)

    # Upload if requested
    if upload:
        run_script(
            "policyengine_us_data/storage/upload_completed_datasets.py",
            env=env,
        )

    # Clean up checkpoints after successful completion
    cleanup_checkpoints(branch, checkpoint_volume)

    return "Data build and tests completed successfully"


@app.local_entrypoint()
def main(
    upload: bool = False,
    branch: str = "main",
    sequential: bool = False,
    clear_checkpoints: bool = False,
):
    result = build_datasets.remote(
        upload=upload,
        branch=branch,
        sequential=sequential,
        clear_checkpoints=clear_checkpoints,
    )
    print(result)
