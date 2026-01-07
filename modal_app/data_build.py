import os
import subprocess
import modal

app = modal.App("policyengine-us-data")

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
        "pytest",
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
    memory=32768,
    cpu=8.0,
    timeout=14400,
)
def build_datasets(
    upload: bool = False,
    branch: str = "main",
    test_lite: bool = False,
):
    setup_gcp_credentials()

    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")
    subprocess.run(["pip", "install", "-e", ".[dev]"], check=True)

    env = os.environ.copy()
    if test_lite:
        env["TEST_LITE"] = "true"

    # Download prerequisites
    subprocess.run(
        [
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
        subprocess.run(["python", script], check=True, env=env)

    os.rename(
        "policyengine_us_data/storage/enhanced_cps_2024.h5",
        "policyengine_us_data/storage/dense_enhanced_cps_2024.h5",
    )
    subprocess.run(
        [
            "cp",
            "policyengine_us_data/storage/sparse_enhanced_cps_2024.h5",
            "policyengine_us_data/storage/enhanced_cps_2024.h5",
        ],
        check=True,
    )

    # Build local area calibration datasets
    print("Building local area calibration datasets...")
    local_area_env = env.copy()
    local_area_env["LOCAL_AREA_CALIBRATION"] = "true"

    subprocess.run(
        ["python", "policyengine_us_data/datasets/cps/cps.py"],
        check=True,
        env=local_area_env,
    )
    subprocess.run(
        ["python", "policyengine_us_data/datasets/puf/puf.py"],
        check=True,
        env=local_area_env,
    )
    subprocess.run(
        ["python", "policyengine_us_data/datasets/cps/extended_cps.py"],
        check=True,
        env=local_area_env,
    )
    subprocess.run(
        [
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
            "pytest",
            "policyengine_us_data/tests/test_local_area_calibration/",
            "-v",
        ],
        check=True,
        env=env,
    )

    # Run main test suite
    print("Running main test suite...")
    subprocess.run(["pytest"], check=True, env=env)

    # Upload if requested
    if upload:
        subprocess.run(
            [
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
    test_lite: bool = False,
):
    result = build_datasets.remote(
        upload=upload,
        branch=branch,
        test_lite=test_lite,
    )
    print(result)
