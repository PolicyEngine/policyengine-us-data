"""Modal wrapper to run QRF subsample benchmark on a 32 GB container.

Clones the repo, builds PUF and CPS datasets, then runs
validation/benchmark_qrf_subsample.py. Uses the same datasets
as the production pipeline (PUF_2024 + CPS_2024_Full).

Usage:
    modal run modal_app/benchmark_runner.py \
        --branch maria/qrf_investigation \
        --sizes 20000,40000,60000,80000,100000
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

app = modal.App("policyengine-benchmark-qrf")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("uv")
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"


def setup_gcp_credentials():
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_path = "/tmp/gcp-credentials.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


def run_script(script_path: str, env: Optional[dict] = None):
    cmd = ["uv", "run", "python", script_path]
    print(f"Running {script_path}...")
    subprocess.run(cmd, check=True, env=env or os.environ.copy())
    print(f"Completed {script_path}")


def build_datasets(env: dict):
    """Build PUF and CPS datasets from scratch.

    Runs the same prerequisite scripts as the production
    data build pipeline. The generated h5 files land in
    STORAGE_FOLDER (resolved from the installed package).
    """
    storage = "policyengine_us_data/storage"
    scripts = [
        f"{storage}/download_private_prerequisites.py",
        "policyengine_us_data/utils/uprating.py",
        "policyengine_us_data/datasets/acs/acs.py",
        "policyengine_us_data/datasets/puf/irs_puf.py",
        "policyengine_us_data/datasets/cps/cps.py",
        "policyengine_us_data/datasets/puf/puf.py",
    ]
    for script in scripts:
        run_script(script, env=env)


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    memory=32768,
    cpu=8.0,
    timeout=14400,
)
def run_benchmark(
    branch: str = "main",
    sizes: Optional[list[int]] = None,
):
    """Run QRF subsample benchmark on Modal.

    Args:
        branch: Git branch to clone and run from.
        sizes: Subsample sizes to benchmark.

    Returns:
        CSV contents of the benchmark results.
    """
    if sizes is None:
        sizes = [20_000, 40_000, 60_000, 80_000, 100_000]

    setup_gcp_credentials()
    env = os.environ.copy()

    os.chdir("/root")
    subprocess.run(
        ["git", "clone", "-b", branch, REPO_URL],
        check=True,
    )
    os.chdir("policyengine-us-data")
    subprocess.run(["uv", "sync", "--locked"], check=True)

    # Build prerequisite datasets
    build_datasets(env)

    # Run the benchmark (defaults to PUF_2024 + CPS_2024_Full,
    # the same datasets the production pipeline uses)
    output_csv = "validation/outputs/subsample_benchmark.csv"
    cmd = [
        "uv",
        "run",
        "python",
        "validation/benchmark_qrf_subsample.py",
        "--sizes",
        *[str(s) for s in sizes],
        "--output",
        output_csv,
        "--verbose",
    ]
    print(f"Running benchmark: sizes={sizes}")
    subprocess.run(cmd, check=True, env=env)

    # Read and return results
    results = Path(output_csv).read_text()
    print("\n=== BENCHMARK RESULTS CSV ===")
    print(results)
    return results


@app.local_entrypoint()
def main(
    branch: str = "main",
    sizes: str = "20000,40000,60000,80000,100000",
):
    size_list = [int(s.strip()) for s in sizes.split(",")]
    print(
        f"Launching benchmark on Modal: " f"branch={branch}, sizes={size_list}"
    )
    results = run_benchmark.remote(branch=branch, sizes=size_list)
    print("\n=== RESULTS ===")
    print(results)
