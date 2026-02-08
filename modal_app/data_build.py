import os
import subprocess
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
):
    setup_gcp_credentials()

    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")
    # Use uv sync to install exact versions from uv.lock
    subprocess.run(["uv", "sync", "--locked"], check=True)

    env = os.environ.copy()

    # Download prerequisites
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "policyengine_us_data/storage/download_private_prerequisites.py",
        ],
        check=True,
        env=env,
    )

    # Build main datasets
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
        print(f"Running {script}...")
        subprocess.run(["uv", "run", "python", script], check=True, env=env)

    # Build stratified CPS for local area calibration
    print("Running create_stratified_cps.py...")
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "policyengine_us_data/datasets/cps/local_area_calibration/create_stratified_cps.py",
            "10500",
        ],
        check=True,
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
):
    result = build_datasets.remote(
        upload=upload,
        branch=branch,
    )
    print(result)
