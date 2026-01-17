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


@app.function(
    image=image,
    secrets=[hf_secret],
    memory=32768,
    cpu=4.0,
    gpu="T4",
    timeout=14400,
)
def fit_weights(branch: str = "main", epochs: int = 200) -> bytes:
    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")

    subprocess.run(["uv", "sync", "--extra", "l0"], check=True)

    script_path = (
        "policyengine_us_data/datasets/cps/"
        "local_area_calibration/fit_calibration_weights.py"
    )
    result = subprocess.run(
        [
            "uv", "run", "python", script_path,
            "--device", "cuda",
            "--epochs", str(epochs),
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed with code {result.returncode}")

    output_line = [
        line for line in result.stdout.split('\n') if 'OUTPUT_PATH:' in line
    ][0]
    output_path = output_line.split('OUTPUT_PATH:')[1].strip()

    with open(output_path, 'rb') as f:
        return f.read()


@app.local_entrypoint()
def main(
    branch: str = "main",
    epochs: int = 200,
    output: str = "calibration_weights.npy"
):
    weights_bytes = fit_weights.remote(branch=branch, epochs=epochs)
    with open(output, 'wb') as f:
        f.write(weights_bytes)
    print(f"Weights saved to: {output}")
