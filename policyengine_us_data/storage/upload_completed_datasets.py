from policyengine_us_data.datasets import (
    EnhancedCPS_2024,
    Pooled_3_Year_CPS_2023,
    CPS_2023,
)
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.huggingface import upload
from google.cloud import storage

 
def upload_datasets():
    storage_client = storage.Client()
    bucket = storage_client.bucket("policyengine-us-data")

    datasets_to_upload = [
        EnhancedCPS_2024,
        Pooled_3_Year_CPS_2023,
        CPS_2023,
    ]

    for dataset in datasets_to_upload:
        dataset = dataset()
        if not dataset.exists:
            raise ValueError(
                f"Dataset {dataset.name} does not exist at {dataset.file_path}."
            )

        upload(
            dataset.file_path,
            "policyengine/policyengine-us-data",
            dataset.file_path.name,
        )

        blob = dataset.file_path.name
        blob = bucket.blob(blob)
        blob.upload_from_filename(dataset.file_path)
        print(
            f"Uploaded {dataset.file_path.name} to GCS bucket policyengine-us-data."
        )


if __name__ == "__main__":
    upload_datasets()
