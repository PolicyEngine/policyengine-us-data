from policyengine_us_data.utils.huggingface import download
from pathlib import Path
import os

FOLDER = Path(__file__).parent

# Skip downloads if no token is available (e.g., in fork PRs)
if not os.environ.get("HUGGING_FACE_TOKEN"):
    print(
        "Warning: HUGGING_FACE_TOKEN not set, skipping private data downloads"
    )
    print(
        "This is expected for PRs from forks. Tests requiring this data may fail."
    )
else:
    download(
        repo="policyengine/irs-soi-puf",
        repo_filename="puf_2015.csv",
        local_folder=FOLDER,
        version=None,
    )
    download(
        repo="policyengine/irs-soi-puf",
        repo_filename="demographics_2015.csv",
        local_folder=FOLDER,
        version=None,
    )
    download(
        repo="policyengine/irs-soi-puf",
        repo_filename="soi.csv",
        local_folder=FOLDER,
        version=None,
    )
    download(
        repo="policyengine/irs-soi-puf",
        repo_filename="np2023_d5_mid.csv",
        local_folder=FOLDER,
        version=None,
    )
