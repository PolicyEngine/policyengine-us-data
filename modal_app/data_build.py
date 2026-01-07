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
    )
)

REPO_URL = "https://github.com/PolicyEngine/policyengine-us-data.git"


@app.function(
    image=image,
    secrets=[hf_secret, gcp_secret],
    volumes={"/data": data_volume},
    memory=32768,
    cpu=8.0,
    timeout=7200,
)
def build_datasets(
    upload: bool = False,
    branch: str = "main",
    test_lite: bool = False,
):
    os.chdir("/root")
    subprocess.run(["git", "clone", "-b", branch, REPO_URL], check=True)
    os.chdir("policyengine-us-data")
    subprocess.run(["pip", "install", "-e", "."], check=True)

    env = os.environ.copy()
    if test_lite:
        env["TEST_LITE"] = "true"

    subprocess.run(
        [
            "python",
            "policyengine_us_data/storage/download_private_prerequisites.py",
        ],
        check=True,
        env=env,
    )

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

    if upload:
        subprocess.run(
            [
                "python",
                "policyengine_us_data/storage/upload_completed_datasets.py",
            ],
            check=True,
            env=env,
        )

    return "Data build completed successfully"


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
