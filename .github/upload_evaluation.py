from policyengine_us_data.evaluation.summary import main
from policyengine_us_data.utils.github import *
from policyengine_us_data.data_storage import STORAGE_FOLDER

if __name__ == "__main__":
    upload(
        "policyengine",
        "policyengine-us-data",
        "release",
        "evaluation.csv",
        STORAGE_FOLDER / "evaluation.csv",
    )
