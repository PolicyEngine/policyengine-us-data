from policyengine_us_data.utils.github import download
from pathlib import Path
from policyengine_us_data.storage import STORAGE_FOLDER

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
