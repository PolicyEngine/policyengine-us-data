from typing import List
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.errors import RevisionNotFoundError
from google.cloud import storage
from pathlib import Path
from importlib import metadata
import google.auth
import logging
import os


def upload_data_files(
    files: List[str],
    gcs_bucket_name: str = "policyengine-us-data",
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    version: str = None,
):
    if version is None:
        version = metadata.version("policyengine-us-data")

    upload_files_to_hf(
        files=files,
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )

    upload_files_to_gcs(
        files=files,
        version=version,
        gcs_bucket_name=gcs_bucket_name,
    )


def upload_files_to_hf(
    files: List[str],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
):
    """
    Upload files to Hugging Face repository and tag the commit with the version.
    """
    api = HfApi()
    hf_operations = []

    token = os.environ.get(
        "HUGGING_FACE_TOKEN",
    )
    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")
        hf_operations.append(
            CommitOperationAdd(
                path_in_repo=file_path.name,
                path_or_fileobj=str(file_path),
            )
        )
    commit_info = api.create_commit(
        token=token,
        repo_id=hf_repo_name,
        operations=hf_operations,
        repo_type=hf_repo_type,
        commit_message=f"Upload data files for version {version}",
    )
    logging.info(f"Uploaded files to Hugging Face repository {hf_repo_name}.")

    # Tag commit with version
    try:
        api.create_tag(
            token=token,
            repo_id=hf_repo_name,
            tag=version,
            revision=commit_info.oid,
            repo_type=hf_repo_type,
        )
        logging.info(
            f"Tagged commit with {version} in Hugging Face repository {hf_repo_name}."
        )
    except Exception as e:
        if "Tag reference exists already" in str(e) or "409" in str(e):
            logging.warning(
                f"Tag {version} already exists in {hf_repo_name}. Skipping tag creation."
            )
        else:
            raise


def upload_files_to_gcs(
    files: List[str],
    version: str,
    gcs_bucket_name: str = "policyengine-us-data",
):
    """
    Upload files to Google Cloud Storage and set metadata with the version.
    """
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(
        credentials=credentials, project=project_id
    )
    bucket = storage_client.bucket(gcs_bucket_name)

    for file_path in files:
        file_path = Path(file_path)
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(file_path)
        logging.info(
            f"Uploaded {file_path.name} to GCS bucket {gcs_bucket_name}."
        )

        # Set metadata
        blob.metadata = {"version": version}
        blob.patch()
        logging.info(
            f"Set metadata for {file_path.name} in GCS bucket {gcs_bucket_name}."
        )


def upload_local_area_file(
    file_path: str,
    subdirectory: str,
    gcs_bucket_name: str = "policyengine-us-data",
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    version: str = None,
):
    """
    Upload a single local area H5 file to a subdirectory (states/ or districts/).

    Uploads to both GCS and Hugging Face with the file placed in the specified
    subdirectory.
    """
    if version is None:
        version = metadata.version("policyengine-us-data")

    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist.")

    # Upload to GCS with subdirectory
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(
        credentials=credentials, project=project_id
    )
    bucket = storage_client.bucket(gcs_bucket_name)

    blob_name = f"{subdirectory}/{file_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    blob.metadata = {"version": version}
    blob.patch()
    logging.info(f"Uploaded {blob_name} to GCS bucket {gcs_bucket_name}.")

    # Upload to Hugging Face with subdirectory
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=f"{subdirectory}/{file_path.name}",
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Upload {subdirectory}/{file_path.name} for version {version}",
    )
    logging.info(
        f"Uploaded {subdirectory}/{file_path.name} to Hugging Face {hf_repo_name}."
    )
