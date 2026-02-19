"""Modal wrapper to run QRF subsample benchmark on a 32 GB container.

Clones the repo, builds (or restores) PUF and CPS datasets, then
runs validation/benchmark_qrf_subsample.py with the requested sizes.

Usage:
    modal run modal_app/benchmark_runner.py \
        --branch maria/qrf_investigation \
        --sizes 20000 40000 60000 80000 100000
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import modal

app = modal.App("policyengine-benchmark-qrf")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

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

STORAGE = "policyengine_us_data/storage"
PUF_H5 = f"{STORAGE}/puf_2024.h5"
CPS_H5 = f"{STORAGE}/cps_2024.h5"

# Scripts needed to build PUF and CPS from scratch, in order.
BUILD_PREREQS = [
    f"{STORAGE}/download_private_prerequisites.py",
    "policyengine_us_data/utils/uprating.py",
    "policyengine_us_data/datasets/acs/acs.py",
    "policyengine_us_data/datasets/puf/irs_puf.py",
    "policyengine_us_data/datasets/cps/cps.py",
    "policyengine_us_data/datasets/puf/puf.py",
]

# Map build scripts to their output files for checkpointing.
SCRIPT_OUTPUTS = {
    "policyengine_us_data/utils/uprating.py": (
        f"{STORAGE}/uprating_factors.csv"
    ),
    "policyengine_us_data/datasets/acs/acs.py": (f"{STORAGE}/acs_2022.h5"),
    "policyengine_us_data/datasets/puf/irs_puf.py": (
        f"{STORAGE}/irs_puf_2015.h5"
    ),
    "policyengine_us_data/datasets/cps/cps.py": CPS_H5,
    "policyengine_us_data/datasets/puf/puf.py": PUF_H5,
}


def setup_gcp_credentials():
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_path = "/tmp/gcp-credentials.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


def restore_checkpoint(branch: str, output_file: str) -> bool:
    """Restore a file from the checkpoint volume if available."""
    cp = Path(VOLUME_MOUNT) / branch / Path(output_file).name
    if cp.exists() and cp.stat().st_size > 0:
        local = Path(output_file)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cp, local)
        print(f"Restored from checkpoint: {output_file}")
        return True
    return False


def run_script(script_path: str, env: Optional[dict] = None):
    cmd = ["uv", "run", "python", script_path]
    print(f"Running {script_path}...")
    subprocess.run(cmd, check=True, env=env or os.environ.copy())
    print(f"Completed {script_path}")


def ensure_datasets(branch: str, env: dict):
    """Build or restore PUF and CPS datasets."""
    checkpoint_volume.reload()

    # Check if both final datasets are already checkpointed
    puf_ok = restore_checkpoint(branch, PUF_H5)
    cps_ok = restore_checkpoint(branch, CPS_H5)
    if puf_ok and cps_ok:
        print("Both datasets restored from checkpoints.")
        return

    # Need to build -- restore any intermediate checkpoints
    # then run missing build steps
    for script in BUILD_PREREQS:
        output = SCRIPT_OUTPUTS.get(script)
        if output and restore_checkpoint(branch, output):
            continue
        run_script(script, env=env)


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={VOLUME_MOUNT: checkpoint_volume},
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
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")
    subprocess.run(["uv", "sync", "--locked"], check=True)

    ensure_datasets(branch, env)

    # Resolve absolute paths to the h5 files
    puf_path = str(Path.cwd() / PUF_H5)
    cps_path = str(Path.cwd() / CPS_H5)

    if not Path(puf_path).exists():
        raise FileNotFoundError(f"PUF dataset not found: {puf_path}")
    if not Path(cps_path).exists():
        raise FileNotFoundError(f"CPS dataset not found: {cps_path}")

    # Run the benchmark
    sizes_str = " ".join(str(s) for s in sizes)
    output_csv = "validation/outputs/subsample_benchmark.csv"

    cmd = [
        "uv",
        "run",
        "python",
        "validation/benchmark_qrf_subsample.py",
        "--puf-dataset",
        puf_path,
        "--cps-dataset",
        cps_path,
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
