import os
import subprocess
import modal

app = modal.App("policyengine-us-data-local-area")

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


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    memory=8192,
    cpu=4.0,
    timeout=86400,  # 24h: processes 50 states + 435 districts with checkpointing
)
def publish_all_local_areas(branch: str = "main"):
    setup_gcp_credentials()

    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")
    # Use uv sync to install exact versions from uv.lock
    subprocess.run(["uv", "sync", "--locked"], check=True)

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "policyengine_us_data/datasets/cps/local_area_calibration/publish_local_area.py",
        ],
        check=True,
        env=os.environ.copy(),
    )

    return "Local area publishing completed successfully"


@app.local_entrypoint()
def main(branch: str = "main"):
    result = publish_all_local_areas.remote(branch=branch)
    print(result)
