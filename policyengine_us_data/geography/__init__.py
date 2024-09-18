from pathlib import Path
import pandas as pd

ZIP_CODE_DATASET_PATH = (
    Path(__file__).parent.parent / "geography" / "zip_codes.csv.gz"
)

ZIP_CODE_DATASET = pd.read_csv(ZIP_CODE_DATASET_PATH, compression="gzip")
