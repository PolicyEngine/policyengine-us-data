from policyengine_us_data.utils.github import download
from pathlib import Path

FOLDER = Path(__file__).parent

download(
    "PolicyEngine",
    "policyengine-us-data",
    "release",
    "soi.csv",
    FOLDER / "soi.csv",
)
download(
    "PolicyEngine",
    "policyengine-us-data",
    "release",
    "np2023_d5_mid.csv",
    FOLDER / "np2023_d5_mid.csv",
)
