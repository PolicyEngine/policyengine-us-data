from policyengine_us_data.datasets import (
    EnhancedCPS_2024,
)
from policyengine_us_data.datasets.cps.cps import CPS_2024
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.data_upload import upload_data_files
from google.cloud import storage
import google.auth


def upload_datasets():
    dataset_files = [
        EnhancedCPS_2024.file_path,
        CPS_2024.file_path,
        STORAGE_FOLDER / "small_enhanced_cps_2024.h5",
        STORAGE_FOLDER / "calibration" / "policy_data.db",
    ]

    # Filter to only existing files
    existing_files = []
    for file_path in dataset_files:
        if file_path.exists():
            existing_files.append(file_path)
            print(f"✓ Found: {file_path}")
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    if not existing_files:
        raise ValueError("No dataset files found to upload!")

    print(f"\nUploading {len(existing_files)} files...")
    upload_data_files(
        files=existing_files,
        hf_repo_name="policyengine/policyengine-us-data",
        hf_repo_type="model",
        gcs_bucket_name="policyengine-us-data",
    )


if __name__ == "__main__":
    upload_datasets()
