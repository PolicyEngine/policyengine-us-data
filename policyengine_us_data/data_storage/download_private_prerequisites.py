from policyengine_us_data.utils.github import download
from pathlib import Path

FOLDER = Path(__file__).parent

download(
    "PolicyEngine",
    "irs-soi-puf",
    "release",
    "puf_2015.csv",
    FOLDER / "puf_2015.csv",
)
download(
    "PolicyEngine",
    "irs-soi-puf",
    "release",
    "demographics_2015.csv",
    FOLDER / "demographics_2015.csv",
)
