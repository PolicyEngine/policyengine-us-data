import os
import subprocess
import modal

app = modal.App("policyengine-us-data-fit-weights")

hf_secret = modal.Secret.from_name("huggingface-token")
calibration_vol = modal.Volume.from_name("calibration-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("uv")
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"
VOLUME_MOUNT = "/calibration-data"


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


def _append_hyperparams(cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq=None):
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
    for line in cal_lines:
        if "OUTPUT_PATH:" in line:
            output_path = line.split("OUTPUT_PATH:")[1].strip()
        elif "CONFIG_PATH:" in line:
            config_path = line.split("CONFIG_PATH:")[1].strip()
        elif "CAL_LOG_PATH:" in line:
            cal_log_path = line.split("CAL_LOG_PATH:")[1].strip()
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

    return {
        "weights": weights_bytes,
        "log": log_bytes,
        "cal_log": cal_log_bytes,
        "config": config_bytes,
    }


def _trigger_repository_dispatch(event_type: str = "calibration-updated"):
    """Fire a repository_dispatch event on GitHub."""
    import json
    import urllib.request

    token = os.environ.get(
        "GITHUB_TOKEN",
        os.environ.get("POLICYENGINE_US_DATA_GITHUB_TOKEN"),
    )
    if not token:
        print(
            "WARNING: No GITHUB_TOKEN or "
            "POLICYENGINE_US_DATA_GITHUB_TOKEN found. "
            "Skipping repository_dispatch.",
            flush=True,
        )
        return False

    url = "https://api.github.com/repos/PolicyEngine/policyengine-us-data/dispatches"
    payload = json.dumps({"event_type": event_type}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    print(
        f"Triggered repository_dispatch '{event_type}' (HTTP {resp.status})",
        flush=True,
    )
    return True


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
    workers: int = 8,
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
    if not skip_county:
        cmd.append("--county-level")
    if workers > 1:
        cmd.extend(["--workers", str(workers)])
    _append_hyperparams(cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq)

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
    volume_package_path: str = None,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
    log_freq: int = None,
) -> dict:
    """Fit weights from a pre-built calibration package."""
    if not volume_package_path:
        raise ValueError("volume_package_path is required")

    _clone_and_install(branch)

    pkg_path = "/root/calibration_package.pkl"
    import shutil

    shutil.copy(volume_package_path, pkg_path)
    size = os.path.getsize(pkg_path)
    print(
        f"Copied package from volume ({size:,} bytes) to {pkg_path}",
        flush=True,
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
    _append_hyperparams(cmd, beta, lambda_l0, lambda_l2, learning_rate, log_freq)

    print(f"Running command: {' '.join(cmd)}", flush=True)

    cal_rc, cal_lines = _run_streaming(
        cmd,
        env=os.environ.copy(),
        label="calibrate",
    )
    if cal_rc != 0:
        raise RuntimeError(f"Script failed with code {cal_rc}")

    return _collect_outputs(cal_lines)


def _print_provenance_from_meta(meta: dict, current_branch: str = None) -> None:
    """Print provenance info and warn on branch mismatch."""
    built = meta.get("created_at", "unknown")
    branch = meta.get("git_branch", "unknown")
    commit = meta.get("git_commit")
    commit_short = commit[:8] if commit else "unknown"
    dirty = " (DIRTY)" if meta.get("git_dirty") else ""
    version = meta.get("package_version", "unknown")
    print("--- Package Provenance ---", flush=True)
    print(f"  Built:   {built}", flush=True)
    print(
        f"  Branch:  {branch} @ {commit_short}{dirty}",
        flush=True,
    )
    print(f"  Version: {version}", flush=True)
    print("--------------------------", flush=True)
    if current_branch and branch != "unknown" and branch != current_branch:
        print(
            f"WARNING: Package built on branch "
            f"'{branch}', but fitting with "
            f"--branch {current_branch}",
            flush=True,
        )


def _write_package_sidecar(pkg_path: str) -> None:
    """Extract metadata from a pickle package and write a JSON sidecar."""
    import json
    import pickle

    sidecar_path = pkg_path.replace(".pkl", "_meta.json")
    try:
        with open(pkg_path, "rb") as f:
            package = pickle.load(f)
        meta = package.get("metadata", {})
        del package
        with open(sidecar_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(
            f"Sidecar metadata written to {sidecar_path}",
            flush=True,
        )
    except Exception as e:
        print(
            f"WARNING: Failed to write sidecar: {e}",
            flush=True,
        )


def _build_package_impl(
    branch: str,
    target_config: str = None,
    skip_county: bool = True,
    workers: int = 8,
) -> str:
    """Download data, build X matrix, save package to volume."""
    _clone_and_install(branch)

    print(
        "Downloading calibration inputs from HuggingFace...",
        flush=True,
    )
    dl_rc, dl_lines = _run_streaming(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from policyengine_us_data.utils.huggingface import "
            "download_calibration_inputs; "
            "paths = download_calibration_inputs("
            "'/root/calibration_data'); "
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

    pkg_path = f"{VOLUME_MOUNT}/calibration_package.pkl"
    script_path = "policyengine_us_data/calibration/unified_calibration.py"
    cmd = [
        "uv",
        "run",
        "python",
        script_path,
        "--device",
        "cpu",
        "--epochs",
        "0",
        "--db-path",
        db_path,
        "--dataset",
        dataset_path,
        "--build-only",
        "--package-output",
        pkg_path,
    ]
    if target_config:
        cmd.extend(["--target-config", target_config])
    if not skip_county:
        cmd.append("--county-level")
    if workers > 1:
        cmd.extend(["--workers", str(workers)])

    build_rc, build_lines = _run_streaming(
        cmd,
        env=os.environ.copy(),
        label="build",
    )
    if build_rc != 0:
        raise RuntimeError(f"Package build failed with code {build_rc}")

    _write_package_sidecar(pkg_path)

    size = os.path.getsize(pkg_path)
    print(
        f"Package saved to volume at {pkg_path} ({size:,} bytes)",
        flush=True,
    )
    calibration_vol.commit()
    return pkg_path


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=65536,
    cpu=8.0,
    timeout=50400,
    volumes={VOLUME_MOUNT: calibration_vol},
)
def build_package_remote(
    branch: str = "main",
    target_config: str = None,
    skip_county: bool = True,
    workers: int = 8,
) -> str:
    return _build_package_impl(
        branch,
        target_config=target_config,
        skip_county=skip_county,
        workers=workers,
    )


@app.function(
    image=image,
    timeout=30,
    volumes={VOLUME_MOUNT: calibration_vol},
)
def check_volume_package() -> dict:
    """Check if a calibration package exists on the volume.

    Reads the lightweight JSON sidecar for provenance fields.
    Falls back to size/mtime if sidecar is missing.
    """
    import datetime
    import json

    pkg_path = f"{VOLUME_MOUNT}/calibration_package.pkl"
    sidecar_path = f"{VOLUME_MOUNT}/calibration_package_meta.json"
    if not os.path.exists(pkg_path):
        return {"exists": False}

    stat = os.stat(pkg_path)
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc)
    info = {
        "exists": True,
        "size": stat.st_size,
        "modified": mtime.strftime("%Y-%m-%d %H:%M UTC"),
    }
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path) as f:
                meta = json.load(f)
            for key in (
                "git_branch",
                "git_commit",
                "git_dirty",
                "package_version",
                "created_at",
                "dataset_sha256",
                "db_sha256",
            ):
                if key in meta:
                    info[key] = meta[key]
        except Exception:
            pass
    return info


# --- Full pipeline GPU functions ---


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=8.0,
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
    workers: int = 8,
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
    cpu=8.0,
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
    workers: int = 8,
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
    cpu=8.0,
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
    workers: int = 8,
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
    cpu=8.0,
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
    workers: int = 8,
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
    cpu=8.0,
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
    workers: int = 8,
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
    cpu=8.0,
    gpu="T4",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_t4(
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
    cpu=8.0,
    gpu="A10",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_a10(
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
    cpu=8.0,
    gpu="A100-40GB",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_a100_40(
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
    cpu=8.0,
    gpu="A100-80GB",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_a100_80(
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
    cpu=8.0,
    gpu="H100",
    timeout=14400,
    volumes={"/calibration-data": calibration_vol},
)
def fit_from_package_h100(
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
    full_pipeline: bool = False,
    county_level: bool = False,
    workers: int = 8,
    push_results: bool = False,
    trigger_publish: bool = False,
    national: bool = False,
):
    prefix = "national_" if national else ""
    if national:
        if lambda_l0 is None:
            lambda_l0 = 1e-4
        output = f"{prefix}{output}"
        log_output = f"{prefix}{log_output}"

    if gpu not in GPU_FUNCTIONS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Choose from: {list(GPU_FUNCTIONS.keys())}"
        )

    if package_path:
        vol_path = f"{VOLUME_MOUNT}/calibration_package.pkl"
        print(f"Reading package from {package_path}...", flush=True)
        import json as _json
        import pickle as _pkl

        with open(package_path, "rb") as f:
            package_bytes = f.read()
        size = len(package_bytes)
        # Extract metadata for sidecar
        pkg_meta = _pkl.loads(package_bytes).get("metadata", {})
        sidecar_bytes = _json.dumps(pkg_meta, indent=2).encode()
        print(
            f"Uploading package ({size:,} bytes) to Modal volume...",
            flush=True,
        )
        with calibration_vol.batch_upload(force=True) as batch:
            from io import BytesIO

            batch.put(
                BytesIO(package_bytes),
                "calibration_package.pkl",
            )
            batch.put(
                BytesIO(sidecar_bytes),
                "calibration_package_meta.json",
            )
        calibration_vol.commit()
        del package_bytes
        print("Upload complete.", flush=True)
        _print_provenance_from_meta(pkg_meta, branch)
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
    elif full_pipeline:
        print(
            "========================================",
            flush=True,
        )
        print(
            "Mode: full pipeline (download, build matrix, fit)",
            flush=True,
        )
        print(
            f"GPU: {gpu} | Epochs: {epochs} | Branch: {branch}",
            flush=True,
        )
        print(
            "========================================",
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
    else:
        vol_path = f"{VOLUME_MOUNT}/calibration_package.pkl"
        vol_info = check_volume_package.remote()
        if not vol_info["exists"]:
            raise SystemExit(
                "\nNo calibration package found on Modal volume.\n"
                "Run 'make build-matrices' first, or use "
                "--full-pipeline to build from scratch.\n"
            )
        if vol_info.get("created_at") or vol_info.get("git_branch"):
            _print_provenance_from_meta(vol_info, branch)
        mode_label = (
            "national calibration" if national else "fitting from pre-built package"
        )
        print(
            "========================================",
            flush=True,
        )
        print(f"Mode: {mode_label}", flush=True)
        print(
            f"GPU: {gpu} | Epochs: {epochs} | Branch: {branch}",
            flush=True,
        )
        if push_results:
            print(
                "After fitting, will upload to HuggingFace:",
                flush=True,
            )
            print(
                f"  - calibration/{prefix}calibration_weights.npy",
                flush=True,
            )
            print(
                f"  - calibration/logs/{prefix}* (diagnostics, "
                "config, calibration log)",
                flush=True,
            )
        print(
            "========================================",
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

    with open(output, "wb") as f:
        f.write(result["weights"])
    print(f"Weights saved to: {output}")

    if result["log"]:
        with open(log_output, "wb") as f:
            f.write(result["log"])
        print(f"Diagnostics log saved to: {log_output}")

    cal_log_output = f"{prefix}calibration_log.csv"
    if result.get("cal_log"):
        with open(cal_log_output, "wb") as f:
            f.write(result["cal_log"])
        print(f"Calibration log saved to: {cal_log_output}")

    config_output = f"{prefix}unified_run_config.json"
    if result.get("config"):
        with open(config_output, "wb") as f:
            f.write(result["config"])
        print(f"Run config saved to: {config_output}")

    if push_results:
        from policyengine_us_data.utils.huggingface import (
            upload_calibration_artifacts,
        )

        upload_calibration_artifacts(
            weights_path=output,
            log_dir=".",
            prefix=prefix,
        )

    if trigger_publish:
        _trigger_repository_dispatch()


@app.local_entrypoint()
def build_package(
    branch: str = "main",
    target_config: str = None,
    county_level: bool = False,
    workers: int = 8,
):
    """Build the calibration package (X matrix) on CPU and save
    to Modal volume. Then run main() to fit."""
    print(
        "========================================",
        flush=True,
    )
    print(
        f"Mode: building calibration package (CPU only)",
        flush=True,
    )
    print(f"Branch: {branch}", flush=True)
    print(
        "This builds the X matrix and saves it to a Modal volume.",
        flush=True,
    )
    print(
        "No GPU is used. Timeout: 14 hours.",
        flush=True,
    )
    print(
        "========================================",
        flush=True,
    )
    vol_path = build_package_remote.remote(
        branch=branch,
        target_config=target_config,
        skip_county=not county_level,
        workers=workers,
    )
    print(
        f"Package built and saved to Modal volume at {vol_path}",
        flush=True,
    )
    print(
        "\nTo fit weights, run:\n"
        "  modal run modal_app/remote_calibration_runner.py"
        "::main \\\n"
        f"    --branch {branch} --gpu <GPU> "
        "--epochs <N> --push-results",
        flush=True,
    )
