import os
import subprocess
import modal

app = modal.App("policyengine-us-data-local-area")

hf_secret = modal.Secret.from_name("huggingface-token")
gcp_secret = modal.Secret.from_name("gcp-credentials")

data_volume = modal.Volume.from_name(
    "policyengine-data", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        "policyengine-us>=1.353.0",
        "policyengine-core>=3.19.0",
        "pandas>=2.3.1",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "microdf_python>=1.0.0",
        "microimpute>=1.1.4",
        "google-cloud-storage>=2.0.0",
        "google-auth>=2.0.0",
        "scipy>=1.15.3",
        "statsmodels>=0.14.5",
        "openpyxl>=3.1.5",
        "tables>=3.10.2",
        "torch>=2.7.1",
        "us>=2.0.0",
        "sqlalchemy>=2.0.41",
        "sqlmodel>=0.0.24",
        "xlrd>=2.0.2",
        "huggingface_hub",
    )
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
    volumes={"/data": data_volume},
    memory=8192,
    cpu=4.0,
    timeout=86400,
)
def publish_all_local_areas(branch: str = "main"):
    setup_gcp_credentials()

    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")
    subprocess.run(["pip", "install", "-e", "."], check=True)

    subprocess.run(
        [
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
