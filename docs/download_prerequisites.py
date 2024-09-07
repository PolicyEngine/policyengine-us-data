from policyengine_us_data.utils.github import download
from policyengine_us_data.data_storage import STORAGE_FOLDER

PREREQUISITES = [
    {
        "repo": "policyengine-us-data",
        "file_name": "enhanced_cps_2024.h5",
    },
    {
        "repo": "policyengine-us-data",
        "file_name": "cps_2024.h5",
    },
    {
        "repo": "irs-soi-puf",
        "file_name": "puf_2024.h5",
    },
    {
        "repo": "policyengine-us-data",
        "file_name": "soi.csv",
    },
    {
        "repo": "policyengine-us-data",
        "file_name": "np2023_d5_mid.csv",
    },
    {
        "repo": "policyengine-us-data",
        "file_name": "soi.csv",
    },
]


def download_data():
    for prerequisite in PREREQUISITES:
        if not (STORAGE_FOLDER / prerequisite["file_name"]).exists():
            download(
                "PolicyEngine",
                prerequisite["repo"],
                "release",
                prerequisite["file_name"],
                STORAGE_FOLDER / prerequisite["file_name"],
            )
