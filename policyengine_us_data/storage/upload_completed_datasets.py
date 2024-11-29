from policyengine_us_data.datasets import (
    EnhancedCPS_2024,
    Pooled_3_Year_CPS_2023,
    CPS_2023,
)
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.huggingface import upload


def upload_datasets():
    for dataset in [EnhancedCPS_2024, Pooled_3_Year_CPS_2023, CPS_2023]:
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


if __name__ == "__main__":
    upload_datasets()
