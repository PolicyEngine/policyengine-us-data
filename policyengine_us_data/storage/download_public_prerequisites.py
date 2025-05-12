from policyengine_us_data.utils.github import download
from pathlib import Path
from policyengine_us_data.storage import STORAGE_FOLDER
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="PolicyEngine/policyengine-us-data",
    filename="pu2023.csv",
    repo_type="model",
    local_dir=STORAGE_FOLDER,
)

download(
    "PolicyEngine",
    "policyengine-us-data",
    "release",
    "soi.csv",
    STORAGE_FOLDER / "soi.csv",
)
download(
    "PolicyEngine",
    "policyengine-us-data",
    "release",
    "np2023_d5_mid.csv",
    STORAGE_FOLDER / "np2023_d5_mid.csv",
)
