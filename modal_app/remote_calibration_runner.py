import os
import subprocess
import modal

app = modal.App("policyengine-us-data-fit-weights")

hf_secret = modal.Secret.from_name("huggingface-token")

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


def _fit_weights_impl(
    branch: str,
    epochs: int,
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
) -> dict:
    """Shared implementation for weight fitting."""
    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")

    subprocess.run(["uv", "sync", "--extra", "l0"], check=True)

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
    if beta is not None:
        cmd.extend(["--beta", str(beta)])
    if lambda_l0 is not None:
        cmd.extend(["--lambda-l0", str(lambda_l0)])
    if lambda_l2 is not None:
        cmd.extend(["--lambda-l2", str(lambda_l2)])
    if learning_rate is not None:
        cmd.extend(["--learning-rate", str(learning_rate)])

    cal_rc, cal_lines = _run_streaming(
        cmd,
        env=os.environ.copy(),
        label="calibrate",
    )
    if cal_rc != 0:
        raise RuntimeError(f"Script failed with code {cal_rc}")

    output_path = None
    log_path = None
    for line in cal_lines:
        if "OUTPUT_PATH:" in line:
            output_path = line.split("OUTPUT_PATH:")[1].strip()
        elif "LOG_PATH:" in line:
            log_path = line.split("LOG_PATH:")[1].strip()

    with open(output_path, "rb") as f:
        weights_bytes = f.read()

    log_bytes = None
    if log_path:
        with open(log_path, "rb") as f:
            log_bytes = f.read()

    return {"weights": weights_bytes, "log": log_bytes}


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
) -> dict:
    return _fit_weights_impl(
        branch, epochs, target_config, beta, lambda_l0, lambda_l2,
        learning_rate,
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
) -> dict:
    return _fit_weights_impl(
        branch, epochs, target_config, beta, lambda_l0, lambda_l2,
        learning_rate,
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
) -> dict:
    return _fit_weights_impl(
        branch, epochs, target_config, beta, lambda_l0, lambda_l2,
        learning_rate,
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
) -> dict:
    return _fit_weights_impl(
        branch, epochs, target_config, beta, lambda_l0, lambda_l2,
        learning_rate,
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
) -> dict:
    return _fit_weights_impl(
        branch, epochs, target_config, beta, lambda_l0, lambda_l2,
        learning_rate,
    )


GPU_FUNCTIONS = {
    "T4": fit_weights_t4,
    "A10": fit_weights_a10,
    "A100-40GB": fit_weights_a100_40,
    "A100-80GB": fit_weights_a100_80,
    "H100": fit_weights_h100,
}


@app.local_entrypoint()
def main(
    branch: str = "main",
    epochs: int = 200,
    gpu: str = "T4",
    output: str = "calibration_weights.npy",
    log_output: str = "calibration_log.csv",
    target_config: str = None,
    beta: float = None,
    lambda_l0: float = None,
    lambda_l2: float = None,
    learning_rate: float = None,
):
    if gpu not in GPU_FUNCTIONS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Choose from: {list(GPU_FUNCTIONS.keys())}"
        )

    print(f"Running with GPU: {gpu}, epochs: {epochs}, branch: {branch}")
    func = GPU_FUNCTIONS[gpu]
    result = func.remote(
        branch=branch,
        epochs=epochs,
        target_config=target_config,
        beta=beta,
        lambda_l0=lambda_l0,
        lambda_l2=lambda_l2,
        learning_rate=learning_rate,
    )

    with open(output, "wb") as f:
        f.write(result["weights"])
    print(f"Weights saved to: {output}")

    if result["log"]:
        with open(log_output, "wb") as f:
            f.write(result["log"])
        print(f"Calibration log saved to: {log_output}")
