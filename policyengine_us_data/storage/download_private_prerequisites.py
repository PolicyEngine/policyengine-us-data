from policyengine_us_data.utils.huggingface import download
from pathlib import Path

FOLDER = Path(__file__).parent

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
