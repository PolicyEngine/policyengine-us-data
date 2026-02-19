"""Modal wrapper to run QRF subsample benchmark on a 32 GB container.

Clones the repo, builds PUF and CPS datasets, then runs
validation/benchmark_qrf_subsample.py. Uses the same datasets
as the production pipeline (PUF_2024 + CPS_2024_Full).

Results are saved to a Modal volume for retrieval after the run.

Usage:
    modal run --detach modal_app/benchmark_runner.py \
        --branch maria/qrf_investigation \
        --sizes 20000,40000,60000,80000,100000

Retrieve results:
    modal volume ls benchmark-qrf-results
    modal volume get benchmark-qrf-results subsample_benchmark.csv
    modal volume get benchmark-qrf-results summary.txt
"""

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import modal

app = modal.App("policyengine-benchmark-qrf")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

results_volume = modal.Volume.from_name(
    "benchmark-qrf-results",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("uv")
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"
RESULTS_MOUNT = "/results"


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
    volumes={RESULTS_MOUNT: results_volume},
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

    # Read results
    csv_text = Path(output_csv).read_text()
    print("\n=== BENCHMARK RESULTS CSV ===")
    print(csv_text)

    # Save results to volume for retrieval
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RESULTS_MOUNT) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(output_csv, run_dir / "subsample_benchmark.csv")

    # Also save to volume root (latest results, easy to grab)
    shutil.copy2(output_csv, Path(RESULTS_MOUNT) / "subsample_benchmark.csv")

    # Generate and save a text summary
    summary_lines = [
        f"QRF Subsample Benchmark Results",
        f"Branch: {branch}",
        f"Sizes: {sizes}",
        f"Timestamp: {timestamp}",
        f"",
    ]
    summary_lines.append(csv_text)
    summary_text = "\n".join(summary_lines)

    (run_dir / "summary.txt").write_text(summary_text)
    (Path(RESULTS_MOUNT) / "summary.txt").write_text(summary_text)

    results_volume.commit()
    print(f"\nResults saved to volume 'benchmark-qrf-results'")
    print(f"  Run dir: run_{timestamp}/")
    print(f"Retrieve with:")
    print(
        f"  modal volume get benchmark-qrf-results " f"subsample_benchmark.csv"
    )
    print(f"  modal volume get benchmark-qrf-results summary.txt")

    return csv_text


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
