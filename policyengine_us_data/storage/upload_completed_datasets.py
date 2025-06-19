from policyengine_us_data.datasets import (
    EnhancedCPS_2024,
    Pooled_3_Year_CPS_2023,
    CPS_2023,
)
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.data_upload import upload_data_files
from google.cloud import storage
import google.auth


def upload_datasets():
    dataset_files = [
        EnhancedCPS_2024.file_path,
        Pooled_3_Year_CPS_2023.file_path,
        CPS_2023.file_path,
        STORAGE_FOLDER / "small_enhanced_cps_2024.h5",
    ]

    for file_path in dataset_files:
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")

    upload_data_files(
        files=dataset_files,
        hf_repo_name="policyengine/policyengine-us-data",
        hf_repo_type="model",
        gcs_bucket_name="policyengine-us-data",
    )


if __name__ == "__main__":
    upload_datasets()
