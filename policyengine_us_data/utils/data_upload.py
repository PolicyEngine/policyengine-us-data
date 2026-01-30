from typing import List, Dict, Optional, Tuple
from huggingface_hub import (
    HfApi,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)
from huggingface_hub.errors import RevisionNotFoundError
from google.cloud import storage
from pathlib import Path
from importlib import metadata
import google.auth
import httpx
import json
import logging
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

DEFAULT_HF_TIMEOUT = 300
MAX_RETRIES = 5
RETRY_BASE_WAIT = 30


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
    skip_hf: bool = False,
):
    """
    Upload a single local area H5 file to a subdirectory (states/ or districts/).

    Uploads to both GCS and Hugging Face with the file placed in the specified
    subdirectory.

    Args:
        skip_hf: If True, skip HuggingFace upload (for batched uploads later)
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

    if skip_hf:
        return

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


def upload_local_area_batch_to_hf(
    files_with_subdirs: List[tuple],
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    version: str = None,
):
    """
    Upload multiple local area files to HuggingFace in a single commit.

    Args:
        files_with_subdirs: List of (file_path, subdirectory) tuples
        hf_repo_name: HuggingFace repository name
        hf_repo_type: Repository type
        version: Version string for commit message
    """
    if version is None:
        version = metadata.version("policyengine-us-data")

    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()

    operations = []
    for file_path, subdirectory in files_with_subdirs:
        file_path = Path(file_path)
        if not file_path.exists():
            logging.warning(f"File {file_path} does not exist, skipping.")
            continue
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{subdirectory}/{file_path.name}",
                path_or_fileobj=str(file_path),
            )
        )

    if not operations:
        logging.warning("No files to upload to HuggingFace.")
        return

    api.create_commit(
        token=token,
        repo_id=hf_repo_name,
        operations=operations,
        repo_type=hf_repo_type,
        commit_message=f"Upload {len(operations)} local area files for version {version}",
    )
    logging.info(
        f"Uploaded {len(operations)} files to Hugging Face {hf_repo_name} in single commit."
    )


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=RETRY_BASE_WAIT, min=30, max=300),
    retry=retry_if_exception_type(
        (
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
            ConnectionError,
        )
    ),
    before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING),
)
def hf_create_commit_with_retry(
    api: HfApi,
    operations: List[CommitOperationAdd],
    repo_id: str,
    repo_type: str,
    token: str,
    commit_message: str,
):
    """
    Create HuggingFace commit with retry logic for timeout errors.

    Uses exponential backoff: 30s, 60s, 120s, 240s, 300s (capped)
    """
    return api.create_commit(
        token=token,
        repo_id=repo_id,
        operations=operations,
        repo_type=repo_type,
        commit_message=commit_message,
    )


def upload_to_staging_hf(
    files_with_paths: List[Tuple[Path, str]],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    batch_size: int = 50,
) -> int:
    """
    Upload files to staging/ paths in HuggingFace.

    Args:
        files_with_paths: List of (local_path, relative_path) tuples
            relative_path is like "states/AL.h5"
        version: Version string for commit message
        hf_repo_name: HuggingFace repository name
        hf_repo_type: Repository type
        batch_size: Number of files per commit batch

    Returns:
        Number of files uploaded
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()

    total_uploaded = 0
    for i in range(0, len(files_with_paths), batch_size):
        batch = files_with_paths[i : i + batch_size]
        operations = []
        for local_path, rel_path in batch:
            local_path = Path(local_path)
            if not local_path.exists():
                logging.warning(f"File {local_path} does not exist, skipping.")
                continue
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"staging/{rel_path}",
                    path_or_fileobj=str(local_path),
                )
            )

        if not operations:
            continue

        hf_create_commit_with_retry(
            api=api,
            operations=operations,
            repo_id=hf_repo_name,
            repo_type=hf_repo_type,
            token=token,
            commit_message=f"Upload batch {i // batch_size + 1} to staging for version {version}",
        )
        total_uploaded += len(operations)
        logging.info(
            f"Uploaded batch {i // batch_size + 1}: {len(operations)} files to staging/"
        )

    logging.info(
        f"Total: uploaded {total_uploaded} files to staging/ in HuggingFace"
    )
    return total_uploaded


def promote_staging_to_production_hf(
    files: List[str],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
) -> int:
    """
    Atomically promote files from staging/ to production paths.

    This creates a single commit that copies each file from staging/{path}
    to {path}, effectively replacing the production files atomically.

    Args:
        files: List of relative paths (e.g., "states/AL.h5")
        version: Version string for commit message
        hf_repo_name: HuggingFace repository
        hf_repo_type: Repository type

    Returns:
        Number of files promoted
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()

    operations = []
    for rel_path in files:
        staging_path = f"staging/{rel_path}"
        operations.append(
            CommitOperationCopy(
                src_path_in_repo=staging_path,
                path_in_repo=rel_path,
            )
        )

    if not operations:
        logging.warning("No files to promote.")
        return 0

    hf_create_commit_with_retry(
        api=api,
        operations=operations,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Promote {len(files)} files from staging to production for version {version}",
    )

    logging.info(
        f"Promoted {len(files)} files from staging/ to production in one commit"
    )
    return len(files)


def cleanup_staging_hf(
    files: List[str],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
) -> int:
    """
    Clean up staging folder after successful promotion.

    Args:
        files: List of relative paths (e.g., "states/AL.h5")
        version: Version string for commit message
        hf_repo_name: HuggingFace repository
        hf_repo_type: Repository type

    Returns:
        Number of files deleted
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()

    operations = []
    for rel_path in files:
        staging_path = f"staging/{rel_path}"
        operations.append(CommitOperationDelete(path_in_repo=staging_path))

    if not operations:
        return 0

    hf_create_commit_with_retry(
        api=api,
        operations=operations,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Clean up staging after version {version} promotion",
    )

    logging.info(f"Cleaned up {len(files)} files from staging/")
    return len(files)
