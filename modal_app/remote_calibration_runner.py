import os
import subprocess
import modal

app = modal.App("policyengine-us-data-fit-weights")

hf_secret = modal.Secret.from_name("huggingface-token")
calibration_vol = modal.Volume.from_name(
    "calibration-data", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("uv")
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"


def _run_streaming(cmd, env=None, label=""):
    """Run a subprocess, streaming output line-by-line.

    Returns (returncode, captured_stdout_lines).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if label:
            print(f"[{label}] {line}", flush=True)
        else:
            print(line, flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, lines


def _clone_and_install(branch: str):
    """Clone the repo and install dependencies."""
    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")
    subprocess.run(["uv", "sync", "--extra", "l0"], check=True)


def _append_hyperparams(
    cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq=None
):
    """Append optional hyperparameter flags to a command list."""
    if beta is not None:
        cmd.extend(["--beta", str(beta)])
    if lambda_l0 is not None:
        cmd.extend(["--lambda-l0", str(lambda_l0)])
    if lambda_l2 is not None:
        cmd.extend(["--lambda-l2", str(lambda_l2)])
    if learning_rate is not None:
        cmd.extend(["--learning-rate", str(learning_rate)])
    if log_freq is not None:
        cmd.extend(["--log-freq", str(log_freq)])


def _collect_outputs(cal_lines):
    """Extract weights and log bytes from calibration output lines."""
    output_path = None
    log_path = None
    cal_log_path = None
    config_path = None
    blocks_path = None
    for line in cal_lines:
        if "OUTPUT_PATH:" in line:
            output_path = line.split("OUTPUT_PATH:")[1].strip()
        elif "CONFIG_PATH:" in line:
            config_path = line.split("CONFIG_PATH:")[1].strip()
        elif "CAL_LOG_PATH:" in line:
            cal_log_path = line.split("CAL_LOG_PATH:")[1].strip()
        elif "BLOCKS_PATH:" in line:
            blocks_path = line.split("BLOCKS_PATH:")[1].strip()
        elif "LOG_PATH:" in line:
            log_path = line.split("LOG_PATH:")[1].strip()

    with open(output_path, "rb") as f:
        weights_bytes = f.read()

    log_bytes = None
    if log_path:
        with open(log_path, "rb") as f:
            log_bytes = f.read()

    cal_log_bytes = None
    if cal_log_path:
        with open(cal_log_path, "rb") as f:
            cal_log_bytes = f.read()

    config_bytes = None
    if config_path:
        with open(config_path, "rb") as f:
            config_bytes = f.read()

    blocks_bytes = None
    if blocks_path and os.path.exists(blocks_path):
        with open(blocks_path, "rb") as f:
            blocks_bytes = f.read()

    return {
        "weights": weights_bytes,
        "log": log_bytes,
        "cal_log": cal_log_bytes,
        "config": config_bytes,
        "blocks": blocks_bytes,
    }


def _upload_logs_to_hf(log_files: dict):
    """Upload calibration log files to HuggingFace.

    Args:
        log_files: dict mapping HF path suffixes to local file paths,
            e.g. {"calibration_log.csv": "calibration_log.csv"}
    """
    from huggingface_hub import HfApi, CommitOperationAdd

    token = os.environ.get("HUGGING_FACE_TOKEN")
    repo = "policyengine/policyengine-us-data"

    api = HfApi()
    operations = []
    for hf_name, local_path in log_files.items():
        if not os.path.exists(local_path):
            print(f"Skipping {local_path} (not found)", flush=True)
            continue
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"calibration/logs/{hf_name}",
                path_or_fileobj=local_path,
            )
        )

    if not operations:
        print("No log files to upload.", flush=True)
        return

    api.create_commit(
        token=token,
        repo_id=repo,
        operations=operations,
        repo_type="model",
        commit_message=(f"Upload {len(operations)} calibration log file(s)"),
    )
    uploaded = [op.path_in_repo for op in operations]
    print(f"Uploaded to HuggingFace: {uploaded}", flush=True)


def _upload_calibration_artifact(local_path: str, hf_name: str):
    """Upload a calibration artifact to calibration/ on HuggingFace."""
    from huggingface_hub import HfApi, CommitOperationAdd

    if not os.path.exists(local_path):
        print(f"Skipping {local_path} (not found)", flush=True)
        return

    token = os.environ.get("HUGGING_FACE_TOKEN")
    repo = "policyengine/policyengine-us-data"
    api = HfApi()
    api.create_commit(
        token=token,
        repo_id=repo,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"calibration/{hf_name}",
                path_or_fileobj=local_path,
            )
        ],
        repo_type="model",
        commit_message=f"Upload calibration artifact: {hf_name}",
    )
    print(
        f"Uploaded {local_path} to calibration/{hf_name}",
        flush=True,
    )


def _fit_weights_impl(
    branch: str,
    epochs: int,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 1,
) -> dict:
    """Full pipeline: download data, build matrix, fit weights."""
    _clone_and_install(branch)

    print("Downloading calibration inputs from HuggingFace...", flush=True)
    dl_rc, dl_lines = _run_streaming(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from policyengine_us_data.utils.huggingface import "
            "download_calibration_inputs; "
            "paths = download_calibration_inputs('/root/calibration_data'); "
            "print(f\"DB: {paths['database']}\"); "
            "print(f\"DATASET: {paths['dataset']}\")",
        ],
        env=os.environ.copy(),
        label="download",
    )
    if dl_rc != 0:
        raise RuntimeError(f"Download failed with code {dl_rc}")

    db_path = dataset_path = None
    for line in dl_lines:
        if "DB:" in line:
            db_path = line.split("DB:")[1].strip()
        elif "DATASET:" in line:
            dataset_path = line.split("DATASET:")[1].strip()

    script_path = "policyengine_us_data/calibration/unified_calibration.py"
    cmd = [
        "uv",
        "run",
        "python",
        script_path,
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--db-path",
        db_path,
        "--dataset",
        dataset_path,
    ]
    if target_config:
        cmd.extend(["--target-config", target_config])
    if skip_county:
        cmd.append("--skip-county")
    if workers > 1:
        cmd.extend(["--workers", str(workers)])
    _append_hyperparams(
        cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq
    )

    cal_rc, cal_lines = _run_streaming(
        cmd,
        env=os.environ.copy(),
        label="calibrate",
    )
    if cal_rc != 0:
        raise RuntimeError(f"Script failed with code {cal_rc}")

    return _collect_outputs(cal_lines)


def _fit_from_package_impl(
    branch: str,
    epochs: int,
    package_bytes: bytes = None,
    volume_package_path: str = None,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
) -> dict:
    """Fit weights from a pre-built calibration package."""
    _clone_and_install(branch)

    pkg_path = "/root/calibration_package.pkl"
    if volume_package_path:
        import shutil

        shutil.copy(volume_package_path, pkg_path)
        size = os.path.getsize(pkg_path)
        print(
            f"Copied package from volume ({size:,} bytes) to {pkg_path}",
            flush=True,
        )
    elif package_bytes:
        with open(pkg_path, "wb") as f:
            f.write(package_bytes)
        print(
            f"Wrote calibration package ({len(package_bytes)} bytes) "
            f"to {pkg_path}",
            flush=True,
        )
    else:
        raise ValueError(
            "Either package_bytes or volume_package_path required"
        )

    script_path = "policyengine_us_data/calibration/unified_calibration.py"
    cmd = [
        "uv",
        "run",
        "python",
        script_path,
        "--device",
        "cuda",
        "--epochs",
        str(epochs),
        "--package-path",
        pkg_path,
    ]
    if target_config:
        cmd.extend(["--target-config", target_config])
    _append_hyperparams(
        cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq
    )

    print(f"Running command: {' '.join(cmd)}", flush=True)

    cal_rc, cal_lines = _run_streaming(
        cmd,
        env=os.environ.copy(),
        label="calibrate",
    )
    if cal_rc != 0:
        raise RuntimeError(f"Script failed with code {cal_rc}")

    return _collect_outputs(cal_lines)


# --- Full pipeline GPU functions ---


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=4.0,
    gpu="T4",
    timeout=14400,
)
def fit_weights_t4(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 1,
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=4.0,
    gpu="A10",
    timeout=14400,
)
def fit_weights_a10(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 1,
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=4.0,
    gpu="A100-40GB",
    timeout=14400,
)
def fit_weights_a100_40(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 1,
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=4.0,
    gpu="A100-80GB",
    timeout=14400,
)
def fit_weights_a100_80(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 1,
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=4.0,
    gpu="H100",
    timeout=14400,
)
def fit_weights_h100(
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    skip_county: bool = True,
    workers: int = 1,
) -> dict:
    return _fit_weights_impl(
        branch,
        epochs,
        target_config,
        beta,
        lambda_l0,
        lambda_l2,
        learning_rate,
        log_freq,
        skip_county=skip_county,
        workers=workers,
    )


GPU_FUNCTIONS = {
    "T4": fit_weights_t4,
    "A10": fit_weights_a10,
    "A100-40GB": fit_weights_a100_40,
    "A100-80GB": fit_weights_a100_80,
    "H100": fit_weights_h100,
}


# --- Package-path GPU functions ---


@app.function(
    image=image,
    memory=32768,
    cpu=4.0,
    gpu="T4",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_t4(
    package_bytes: bytes = None,
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        package_bytes=package_bytes,
        volume_package_path=volume_package_path,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


@app.function(
    image=image,
    memory=32768,
    cpu=4.0,
    gpu="A10",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_a10(
    package_bytes: bytes = None,
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        package_bytes=package_bytes,
        volume_package_path=volume_package_path,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


@app.function(
    image=image,
    memory=32768,
    cpu=4.0,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_a100_40(
    package_bytes: bytes = None,
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        package_bytes=package_bytes,
        volume_package_path=volume_package_path,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


@app.function(
    image=image,
    memory=32768,
    cpu=4.0,
    gpu="A100-80GB",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_a100_80(
    package_bytes: bytes = None,
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        package_bytes=package_bytes,
        volume_package_path=volume_package_path,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


@app.function(
    image=image,
    memory=32768,
    cpu=4.0,
    gpu="H100",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_h100(
    package_bytes: bytes = None,
    branch: str = "main",
    epochs: int = 200,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    volume_package_path: str = None,
) -> dict:
    return _fit_from_package_impl(
        branch,
        epochs,
        package_bytes=package_bytes,
        volume_package_path=volume_package_path,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
        log_freq=log_freq,
    )


PACKAGE_GPU_FUNCTIONS = {
    "T4": fit_from_package_t4,
    "A10": fit_from_package_a10,
    "A100-40GB": fit_from_package_a100_40,
    "A100-80GB": fit_from_package_a100_80,
    "H100": fit_from_package_h100,
}


VOLUME_MOUNT = "/calibration-data"


@app.local_entrypoint()
def main(
    branch: str = "main",
    epochs: int = 200,
    gpu: str = "T4",
    output: str = "calibration_weights.npy",
    log_output: str = "unified_diagnostics.csv",
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
    package_path: str = None,
    package_volume: bool = False,
    county_level: bool = False,
    workers: int = 1,
    upload_logs: bool = False,
):
    if gpu not in GPU_FUNCTIONS:
        raise ValueError(
            f"Unknown GPU: {gpu}. "
            f"Choose from: {list(GPU_FUNCTIONS.keys())}"
        )

    if package_volume:
        vol_path = f"{VOLUME_MOUNT}/calibration_package.pkl"
        print(
            f"Using package from Modal volume at {vol_path}",
            flush=True,
        )
        func = PACKAGE_GPU_FUNCTIONS[gpu]
        result = func.remote(
            branch=branch,
            epochs=epochs,
            target_config=target_config,
            beta=beta,
            lambda_l0=lambda_l0,
            lambda_l2=lambda_l2,
            learning_rate=learning_rate,
            log_freq=log_freq,
            volume_package_path=vol_path,
        )
    elif package_path:
        print(f"Reading package from {package_path}...", flush=True)
        with open(package_path, "rb") as f:
            package_bytes = f.read()
        print(
            f"Uploading package ({len(package_bytes)} bytes) "
            f"to {gpu} on Modal...",
            flush=True,
        )
        func = PACKAGE_GPU_FUNCTIONS[gpu]
        result = func.remote(
            package_bytes=package_bytes,
            branch=branch,
            epochs=epochs,
            target_config=target_config,
            beta=beta,
            lambda_l0=lambda_l0,
            lambda_l2=lambda_l2,
            learning_rate=learning_rate,
            log_freq=log_freq,
        )
    else:
        print(
            f"Running full pipeline with GPU: {gpu}, "
            f"epochs: {epochs}, branch: {branch}",
            flush=True,
        )
        func = GPU_FUNCTIONS[gpu]
        result = func.remote(
            branch=branch,
            epochs=epochs,
            target_config=target_config,
            beta=beta,
            lambda_l0=lambda_l0,
            lambda_l2=lambda_l2,
            learning_rate=learning_rate,
            log_freq=log_freq,
            skip_county=not county_level,
            workers=workers,
        )

    with open(output, "wb") as f:
        f.write(result["weights"])
    print(f"Weights saved to: {output}")

    if result["log"]:
        with open(log_output, "wb") as f:
            f.write(result["log"])
        print(f"Diagnostics log saved to: {log_output}")

    cal_log_output = "calibration_log.csv"
    if result.get("cal_log"):
        with open(cal_log_output, "wb") as f:
            f.write(result["cal_log"])
        print(f"Calibration log saved to: {cal_log_output}")

    config_output = "unified_run_config.json"
    if result.get("config"):
        with open(config_output, "wb") as f:
            f.write(result["config"])
        print(f"Run config saved to: {config_output}")

    blocks_output = "stacked_blocks.npy"
    if result.get("blocks"):
        with open(blocks_output, "wb") as f:
            f.write(result["blocks"])
        print(f"Stacked blocks saved to: {blocks_output}")

    if upload_logs:
        log_files = {
            "calibration_log.csv": cal_log_output,
            "unified_diagnostics.csv": log_output,
            "unified_run_config.json": config_output,
        }
        _upload_logs_to_hf(log_files)

        if result.get("blocks"):
            _upload_calibration_artifact(blocks_output, "stacked_blocks.npy")
