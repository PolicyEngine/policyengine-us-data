from pathlib import Path
import pandas as pd

ZIP_CODE_DATASET_PATH = (
    Path(__file__).parent.parent / "geography" / "zip_codes.csv.gz"
)

COUNTY_FIPS_DATASET_PATH = (
    Path(__file__).parent.parent / "geography" / "county_fips.csv.gz"
)

ZIP_CODE_DATASET = pd.read_csv(ZIP_CODE_DATASET_PATH, compression="gzip")
COUNTY_FIPS_DATASET = pd.read_csv(COUNTY_FIPS_DATASET_PATH, compression="gzip")
