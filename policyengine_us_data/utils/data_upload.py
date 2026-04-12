from io import BytesIO
from typing import List, Dict, Optional, Tuple
from huggingface_hub import (
    HfApi,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    hf_hub_download,
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

from policyengine_us_data.utils.release_manifest import (
    build_release_manifest,
    serialize_release_manifest,
)

DEFAULT_HF_TIMEOUT = 300
MAX_RETRIES = 5
RETRY_BASE_WAIT = 30
RELEASE_MANIFEST_PATH = "release_manifest.json"


def _get_model_package_version(
    package_name: str = "policyengine-us",
) -> Optional[str]:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        logging.warning(
            "Could not determine installed version for %s while building release manifest.",
            package_name,
        )
        return None


def _get_model_package_build_metadata(
    package_name: str = "policyengine-us",
) -> Dict[str, Optional[str]]:
    metadata_payload: Dict[str, Optional[str]] = {
        "version": _get_model_package_version(package_name),
        "git_sha": None,
        "data_build_fingerprint": None,
    }
    module_name = package_name.replace("-", "_")
    try:
        build_metadata_module = __import__(
            f"{module_name}.build_metadata",
            fromlist=["get_data_build_metadata"],
        )
        get_data_build_metadata = getattr(
            build_metadata_module, "get_data_build_metadata", None
        )
        if callable(get_data_build_metadata):
            package_metadata = get_data_build_metadata()
            metadata_payload["version"] = (
                package_metadata.get("version") or metadata_payload["version"]
            )
            metadata_payload["git_sha"] = package_metadata.get("git_sha")
            metadata_payload["data_build_fingerprint"] = package_metadata.get(
                "data_build_fingerprint"
            )
    except Exception:
        logging.warning(
            "Could not load build metadata from %s while building release manifest.",
            package_name,
            exc_info=True,
        )
    return metadata_payload


def load_release_manifest_from_hf(
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
) -> Optional[Dict]:
    token = os.environ.get("HUGGING_FACE_TOKEN")
    candidate_paths = [
        f"releases/{version}/{RELEASE_MANIFEST_PATH}",
        RELEASE_MANIFEST_PATH,
    ]

    for path_in_repo in candidate_paths:
        try:
            manifest_path = hf_hub_download(
                repo_id=hf_repo_name,
                filename=path_in_repo,
                repo_type=hf_repo_type,
                token=token,
            )
        except RevisionNotFoundError:
            raise
        except Exception:
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        data_package = manifest.get("data_package", {})
        if data_package.get("version") == version:
            return manifest

    return None


def create_release_manifest_commit_operations(
    files_with_repo_paths: List[Tuple[Path, str]],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    model_package_name: str = "policyengine-us",
    model_package_version: Optional[str] = None,
    model_package_git_sha: Optional[str] = None,
    model_package_data_build_fingerprint: Optional[str] = None,
    existing_manifest: Optional[Dict] = None,
) -> Tuple[Dict, List[CommitOperationAdd]]:
    manifest = build_release_manifest(
        files_with_repo_paths=files_with_repo_paths,
        version=version,
        repo_id=hf_repo_name,
        model_package_name=model_package_name,
        model_package_version=model_package_version,
        model_package_git_sha=model_package_git_sha,
        model_package_data_build_fingerprint=model_package_data_build_fingerprint,
        existing_manifest=existing_manifest,
    )
    manifest_payload = serialize_release_manifest(manifest)

    operations = [
        CommitOperationAdd(
            path_in_repo=RELEASE_MANIFEST_PATH,
            path_or_fileobj=BytesIO(manifest_payload),
        ),
        CommitOperationAdd(
            path_in_repo=f"releases/{version}/{RELEASE_MANIFEST_PATH}",
            path_or_fileobj=BytesIO(manifest_payload),
        ),
    ]
    return manifest, operations


def create_release_tag(
    version: str,
    revision: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    token: Optional[str] = None,
    api: Optional[HfApi] = None,
) -> None:
    api = api or HfApi()
    token = token or os.environ.get("HUGGING_FACE_TOKEN")
    try:
        api.create_tag(
            token=token,
            repo_id=hf_repo_name,
            tag=version,
            revision=revision,
            repo_type=hf_repo_type,
        )
        logging.info(
            "Tagged revision %s with %s in Hugging Face repository %s.",
            revision,
            version,
            hf_repo_name,
        )
    except Exception as e:
        if "Tag reference exists already" in str(e) or "409" in str(e):
            logging.warning(
                "Tag %s already exists in %s. Skipping tag creation.",
                version,
                hf_repo_name,
            )
        else:
            raise


def upload_data_files(
    files: List[str],
    gcs_bucket_name: str = "policyengine-us-data",
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    version: str = None,
    create_tag: bool = False,
):
    if version is None:
        version = metadata.version("policyengine-us-data")

    upload_files_to_hf(
        files=files,
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
        create_tag=create_tag,
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
    create_tag: bool = False,
):
    """
    Upload files to Hugging Face repository and tag the commit with the version.
    """
    api = HfApi()
    hf_operations = []
    files_with_repo_paths = []

    token = os.environ.get(
        "HUGGING_FACE_TOKEN",
    )
    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist.")
        repo_path = file_path.name
        files_with_repo_paths.append((file_path, repo_path))
        hf_operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=str(file_path),
            )
        )

    existing_manifest = load_release_manifest_from_hf(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    model_build_metadata = _get_model_package_build_metadata()
    _, manifest_operations = create_release_manifest_commit_operations(
        files_with_repo_paths=files_with_repo_paths,
        version=version,
        hf_repo_name=hf_repo_name,
        model_package_version=model_build_metadata["version"],
        model_package_git_sha=model_build_metadata["git_sha"],
        model_package_data_build_fingerprint=model_build_metadata[
            "data_build_fingerprint"
        ],
        existing_manifest=existing_manifest,
    )
    hf_operations.extend(manifest_operations)

    commit_info = api.create_commit(
        token=token,
        repo_id=hf_repo_name,
        operations=hf_operations,
        repo_type=hf_repo_type,
        commit_message=f"Upload data files for version {version}",
    )
    logging.info(f"Uploaded files to Hugging Face repository {hf_repo_name}.")

    if create_tag:
        create_release_tag(
            version=version,
            revision=commit_info.oid,
            hf_repo_name=hf_repo_name,
            hf_repo_type=hf_repo_type,
            token=token,
            api=api,
        )


def upload_files_to_gcs(
    files: List[str],
    version: str,
    gcs_bucket_name: str = "policyengine-us-data",
):
    """
    Upload files to Google Cloud Storage and set metadata with the version.
    """
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket = storage_client.bucket(gcs_bucket_name)

    for file_path in files:
        file_path = Path(file_path)
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(file_path)
        logging.info(f"Uploaded {file_path.name} to GCS bucket {gcs_bucket_name}.")

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
    Upload a single local area H5 file to a subdirectory.

    Supports states/, districts/, cities/, and national/.
    Uploads to both GCS and Hugging Face.

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
    storage_client = storage.Client(credentials=credentials, project=project_id)
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


def publish_release_manifest_to_hf(
    files_with_paths: List[Tuple[Path, str]],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    model_package_name: str = "policyengine-us",
    model_package_version: Optional[str] = None,
    create_tag: bool = False,
) -> Dict:
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    model_build_metadata = _get_model_package_build_metadata(model_package_name)
    existing_manifest = load_release_manifest_from_hf(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    manifest, operations = create_release_manifest_commit_operations(
        files_with_repo_paths=[
            (Path(path), repo_path) for path, repo_path in files_with_paths
        ],
        version=version,
        hf_repo_name=hf_repo_name,
        model_package_name=model_package_name,
        model_package_version=model_package_version or model_build_metadata["version"],
        model_package_git_sha=model_build_metadata["git_sha"],
        model_package_data_build_fingerprint=model_build_metadata[
            "data_build_fingerprint"
        ],
        existing_manifest=existing_manifest,
    )
    commit_info = hf_create_commit_with_retry(
        api=api,
        operations=operations,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Update release manifest for version {version}",
    )
    if create_tag:
        create_release_tag(
            version=version,
            revision=commit_info.oid,
            hf_repo_name=hf_repo_name,
            hf_repo_type=hf_repo_type,
            token=token,
            api=api,
        )
    logging.info(
        "Published release manifest for %s with %d tracked artifacts.",
        version,
        len(manifest["artifacts"]),
    )
    return manifest


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

    logging.info(f"Total: uploaded {total_uploaded} files to staging/ in HuggingFace")
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

    Raises:
        RuntimeError: If the commit was a no-op (HEAD unchanged)
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

    head_before = api.repo_info(
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
    ).sha

    result = hf_create_commit_with_retry(
        api=api,
        operations=operations,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Promote {len(files)} files from staging to production for version {version}",
    )

    if result.oid == head_before:
        raise RuntimeError(
            f"Promote commit was a no-op: HEAD stayed at {head_before}. "
            f"Staging files may be identical to production."
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

    Raises:
        RuntimeError: If the cleanup commit was a no-op (HEAD unchanged)
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()

    operations = []
    for rel_path in files:
        staging_path = f"staging/{rel_path}"
        operations.append(CommitOperationDelete(path_in_repo=staging_path))

    if not operations:
        return 0

    head_before = api.repo_info(
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
    ).sha

    result = hf_create_commit_with_retry(
        api=api,
        operations=operations,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Clean up staging after version {version} promotion",
    )

    if result.oid == head_before:
        raise RuntimeError(
            f"Cleanup commit was a no-op: HEAD stayed at {head_before}. "
            f"Staging files may not exist."
        )

    logging.info(f"Cleaned up {len(files)} files from staging/")
    return len(files)


def upload_from_hf_staging_to_gcs(
    rel_paths: List[str],
    version: str,
    gcs_bucket_name: str = "policyengine-us-data",
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
) -> int:
    """Download files from HF staging/ and upload to GCS production paths.

    Args:
        rel_paths: Relative paths like "states/AL.h5", "districts/NC-01.h5"
        version: Version string for GCS metadata
        gcs_bucket_name: GCS bucket name
        hf_repo_name: HuggingFace repository name
        hf_repo_type: Repository type

    Returns:
        Number of files uploaded
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")

    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket = storage_client.bucket(gcs_bucket_name)

    uploaded = 0
    for rel_path in rel_paths:
        staging_filename = f"staging/{rel_path}"
        local_path = hf_hub_download(
            repo_id=hf_repo_name,
            filename=staging_filename,
            repo_type=hf_repo_type,
            token=token,
        )

        blob = bucket.blob(rel_path)
        blob.upload_from_filename(local_path)
        blob.metadata = {"version": version}
        blob.patch()
        uploaded += 1
        logging.info(f"Uploaded {rel_path} to GCS (sourced from HF staging)")

    logging.info(f"Total: uploaded {uploaded} files from HF staging to GCS")
    return uploaded
