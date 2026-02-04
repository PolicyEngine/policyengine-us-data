import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import modal

app = modal.App("policyengine-us-data")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("uv")
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"


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


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    memory=32768,
    cpu=8.0,
    timeout=14400,
)
def build_datasets(
    upload: bool = False,
    branch: str = "main",
    sequential: bool = False,
):
    setup_gcp_credentials()

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
        # Original sequential execution for backward compatibility
        scripts = [
            "policyengine_us_data/utils/uprating.py",
            "policyengine_us_data/datasets/acs/acs.py",
            "policyengine_us_data/datasets/cps/cps.py",
            "policyengine_us_data/datasets/puf/irs_puf.py",
            "policyengine_us_data/datasets/puf/puf.py",
            "policyengine_us_data/datasets/cps/extended_cps.py",
            "policyengine_us_data/datasets/cps/enhanced_cps.py",
            "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
        ]
        for script in scripts:
            run_script(script, env=env)

        # Build stratified CPS
        run_script(
            "policyengine_us_data/datasets/cps/"
            "local_area_calibration/create_stratified_cps.py",
            args=["10500"],
            env=env,
        )
    else:
        # Parallel execution based on dependency groups
        # GROUP 1: Independent scripts - run in parallel
        print("=== Phase 1: Building independent datasets (parallel) ===")
        group1 = [
            "policyengine_us_data/utils/uprating.py",
            "policyengine_us_data/datasets/acs/acs.py",
            "policyengine_us_data/datasets/puf/irs_puf.py",
        ]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_script, s, env=env): s for s in group1
            }
            for future in as_completed(futures):
                future.result()  # Raises if script failed

        # GROUP 2: Depends on Group 1 - run in parallel
        # cps.py needs acs, puf.py needs irs_puf + uprating
        print("=== Phase 2: Building CPS and PUF (parallel) ===")
        group2 = [
            "policyengine_us_data/datasets/cps/cps.py",
            "policyengine_us_data/datasets/puf/puf.py",
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_script, s, env=env): s for s in group2
            }
            for future in as_completed(futures):
                future.result()

        # SEQUENTIAL: Extended CPS (needs both cps and puf)
        print("=== Phase 3: Building extended CPS ===")
        run_script(
            "policyengine_us_data/datasets/cps/extended_cps.py",
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
                    run_script,
                    "policyengine_us_data/datasets/cps/enhanced_cps.py",
                    env=env,
                ),
                executor.submit(
                    run_script,
                    "policyengine_us_data/datasets/cps/"
                    "local_area_calibration/create_stratified_cps.py",
                    args=["10500"],
                    env=env,
                ),
            ]
            for future in as_completed(futures):
                future.result()

        # SEQUENTIAL: Small enhanced CPS (needs enhanced_cps)
        print("=== Phase 5: Building small enhanced CPS ===")
        run_script(
            "policyengine_us_data/datasets/cps/small_enhanced_cps.py",
            env=env,
        )

    # Run local area calibration tests
    print("Running local area calibration tests...")
    subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "policyengine_us_data/tests/test_local_area_calibration/",
            "-v",
        ],
        check=True,
        env=env,
    )

    # Run main test suite
    print("Running main test suite...")
    subprocess.run(["uv", "run", "pytest"], check=True, env=env)

    # Upload if requested
    if upload:
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "policyengine_us_data/storage/upload_completed_datasets.py",
            ],
            check=True,
            env=env,
        )

    return "Data build and tests completed successfully"


@app.local_entrypoint()
def main(
    upload: bool = False,
    branch: str = "main",
    sequential: bool = False,
):
    result = build_datasets.remote(
        upload=upload,
        branch=branch,
        sequential=sequential,
    )
    print(result)
