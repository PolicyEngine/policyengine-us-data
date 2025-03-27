from pathlib import Path
import pandas as pd
import os

ZIP_CODE_DATASET_PATH = (
    Path(__file__).parent.parent / "geography" / "zip_codes.csv.gz"
)

COUNTY_FIPS_DATASET_PATH = (
    Path(__file__).parent.parent / "geography" / "county_fips.csv.gz"
)

# Avoid circular import error when -us-data is initialized
if os.path.exists(ZIP_CODE_DATASET_PATH):
    ZIP_CODE_DATASET = pd.read_csv(ZIP_CODE_DATASET_PATH, compression="gzip")
else:
    ZIP_CODE_DATASET = None
