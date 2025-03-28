import pandas as pd
import requests
from io import StringIO, BytesIO
from pathlib import Path
from policyengine_us_data.utils.huggingface import upload as upload_to_hf

LOCAL_FOLDER = Path(__file__).parent


def generate_county_fips_2020_dataset():
    """
    One-off script to generate a dataset of county FIPS codes used in the 2020 Census.

    Running this file will create the dataset and save it locally as county_fips.csv.gz.
    """
    # More information about this data at https://www.census.gov/library/reference/code-lists/ansi.html#cou

    # Dataset contains the following columns:
    # STATE - 2-digit state postal code (e.g., "AL")
    # STATEFP - State FIPS code (01 for AL)
    # COUNTYFP - Three-digit county portion of FIPS (001 for Autauga County, AL, if STATEFP is 01)
    # COUNTYNAME - County name

    COUNTY_FIPS_2020_URL = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"

    # Download the base tab-delimited data file
    response = requests.get(COUNTY_FIPS_2020_URL)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download county FIPS codes: {response.status_code}"
        )
    response.encoding = "utf-8"

    county_fips_raw = StringIO(response.text)

    county_fips = pd.read_csv(
        county_fips_raw,
        delimiter="|",
        usecols=["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"],
        dtype={
            "STATE": str,
            "STATEFP": str,
            "COUNTYFP": str,
            "COUNTYNAME": str,
        },
        encoding="utf-8",
    )

    county_fips = county_fips.rename(
        columns={
            "STATE": "state",
            "STATEFP": "state_fips_segment",
            "COUNTYFP": "county_fips_segment",
            "COUNTYNAME": "county_name",
        }
    )

    # Create composite county FIPS code, then drop segment columns;
    # note that the FIPS code is a 5-char str of digits
    county_fips["county_fips"] = (
        county_fips["state_fips_segment"] + county_fips["county_fips_segment"]
    )
    county_fips.drop(
        columns=["state_fips_segment", "county_fips_segment"], inplace=True
    )

    # Create buffer to save CSV
    csv_buffer = BytesIO()

    # Save CSV into buffer object and reset pointer
    county_fips.to_csv(csv_buffer, index=False, compression="gzip", encoding="utf-8")
    csv_buffer.seek(0)

    # Upload to Hugging Face
    upload_to_hf(
        local_file_path=csv_buffer,
        repo="policyengine/policyengine-us-data",
        repo_file_path="county_fips_2020.csv.gz",
    )


if __name__ == "__main__":
    generate_county_fips_2020_dataset()
