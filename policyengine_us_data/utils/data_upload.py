from io import BytesIO
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from huggingface_hub import (
    HfApi,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    hf_hub_download,
)
from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError
from google.cloud import storage
from pathlib import Path
from importlib import metadata
import google.auth
import httpx
import json
import logging
import os
import subprocess

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from policyengine_us_data.__version__ import __version__ as DATA_PACKAGE_VERSION
from policyengine_us_data.utils.release_manifest import (
    build_release_manifest,
    serialize_release_manifest,
)
from policyengine_us_data.utils.release_completion import (
    VERSION_MANIFEST_PATH,
    build_release_completion_marker,
    release_completion_marker_path,
    serialize_release_completion_marker,
    validate_release_completion_marker,
)
from policyengine_us_data.utils.release_promotion import (
    FullReleasePromotionConfig,
    FullReleasePromotionDependencies,
    promote_full_release,
)
from policyengine_us_data.utils.run_context import (
    RunContext,
    resolve_run_id,
)
from policyengine_us_data.utils.trace_tro import (
    TRACE_TRO_FILENAME,
    build_trace_tro_from_release_manifest,
    serialize_trace_tro,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_TIMEOUT = 300
MAX_RETRIES = 5
RETRY_BASE_WAIT = 30
RELEASE_MANIFEST_PATH = "release_manifest.json"
LOCAL_AREA_FINALIZE_REQUIRED_PREFIXES = (
    "national/",
    "states/",
    "districts/",
    "cities/",
)
LOCAL_AREA_FINALIZE_REQUIRED_COUNTS = {
    "national/": 1,
    "states/": 51,
    "districts/": 435,
    "cities/": 1,
}
VALIDATION_REPORT_FILENAMES = (
    "validation_summary.json",
    "validation_results.csv",
    "national_validation.txt",
)


def _resolve_staging_run_id(run_id: str = "") -> str:
    return run_id or resolve_run_id()


def _run_context_for_release() -> dict | None:
    run_id = resolve_run_id()
    if not run_id:
        return None
    return RunContext.from_env(run_id=run_id).to_dict()


def _apply_run_context_for_release(
    run_id: str,
    run_context: Optional[Dict] = None,
) -> dict | None:
    if not run_id and not run_context:
        return None
    context = RunContext.from_mapping(run_context, run_id=run_id)
    for key, value in context.export_env().items():
        os.environ[key] = value
    return context.to_dict()


def _pipeline_run_id_for_manifest(
    pipeline_run_id: str = "",
    run_context: Optional[Mapping[str, Any]] = None,
) -> str | None:
    if pipeline_run_id:
        return pipeline_run_id
    if run_context:
        run_id = run_context.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return None


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


def _get_data_package_git_sha() -> Optional[str]:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_core_package_runtime_metadata(
    package_name: str = "policyengine-core",
) -> Optional[Dict[str, Any]]:
    module_name = package_name.replace("-", "_")
    try:
        runtime_metadata_module = __import__(
            module_name,
            fromlist=["get_runtime_metadata"],
        )
        get_runtime_metadata = getattr(
            runtime_metadata_module,
            "get_runtime_metadata",
            None,
        )
        if callable(get_runtime_metadata):
            runtime_metadata = get_runtime_metadata()
            if isinstance(runtime_metadata, Mapping):
                return dict(runtime_metadata)
    except Exception:
        logging.warning(
            "Could not load runtime metadata from %s.",
            package_name,
            exc_info=True,
        )

    version = _get_model_package_version(package_name)
    if version is None:
        return None
    return {
        "name": package_name,
        "version": version,
    }


def _get_model_package_build_metadata(
    package_name: str = "policyengine-us",
) -> Dict[str, Any]:
    metadata_payload: Dict[str, Any] = {
        "name": package_name,
        "version": _get_model_package_version(package_name),
        "git_sha": None,
        "data_build_fingerprint": None,
        "core": _get_core_package_runtime_metadata(),
    }
    module_name = package_name.replace("-", "_")
    try:
        build_metadata_module = __import__(
            f"{module_name}.build_metadata",
            fromlist=["get_runtime_metadata", "get_data_build_metadata"],
        )
        get_runtime_metadata = getattr(
            build_metadata_module,
            "get_runtime_metadata",
            None,
        )
        get_data_build_metadata = getattr(
            build_metadata_module, "get_data_build_metadata", None
        )
        metadata_getter = (
            get_runtime_metadata
            if callable(get_runtime_metadata)
            else get_data_build_metadata
        )
        if callable(metadata_getter):
            package_metadata = metadata_getter()
            if not isinstance(package_metadata, Mapping):
                return metadata_payload
            metadata_payload["name"] = package_metadata.get(
                "name", metadata_payload["name"]
            )
            metadata_payload["version"] = (
                package_metadata.get("version") or metadata_payload["version"]
            )
            metadata_payload["git_sha"] = package_metadata.get("git_sha")
            metadata_payload["data_build_fingerprint"] = package_metadata.get(
                "data_build_fingerprint"
            )
            metadata_payload["core"] = package_metadata.get(
                "core"
            ) or metadata_payload.get("core")
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
    revision: Optional[str] = None,
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
                revision=revision,
            )
        except RevisionNotFoundError:
            return None
        except EntryNotFoundError:
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        data_package = manifest.get("data_package", {})
        if data_package.get("version") == version:
            return manifest

    return None


def assert_release_not_finalized(
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
) -> None:
    if (
        load_release_manifest_from_hf(
            version=version,
            hf_repo_name=hf_repo_name,
            hf_repo_type=hf_repo_type,
            revision=version,
        )
        is not None
    ):
        raise RuntimeError(
            f"Release {version} is already finalized on {hf_repo_name}. "
            "Refusing to mutate release manifest state after the tag exists."
        )


def get_repo_head_revision(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[str]:
    repo_info = api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    return getattr(repo_info, "sha", None)


def _collect_manifest_repo_paths(manifest: Optional[Dict]) -> set[str]:
    if not manifest:
        return set()
    return {
        artifact["path"]
        for artifact in manifest.get("artifacts", {}).values()
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
    }


def missing_release_prefixes(
    existing_manifest: Optional[Dict],
    new_repo_paths: Sequence[str],
    required_prefixes: Sequence[str] = LOCAL_AREA_FINALIZE_REQUIRED_PREFIXES,
    required_counts: Optional[Dict[str, int]] = None,
) -> list[str]:
    required_counts = required_counts or LOCAL_AREA_FINALIZE_REQUIRED_COUNTS
    combined_paths = _collect_manifest_repo_paths(existing_manifest) | set(
        new_repo_paths
    )
    prefix_counts = {prefix: 0 for prefix in required_prefixes}
    for path in combined_paths:
        for prefix in required_prefixes:
            if path.startswith(prefix):
                prefix_counts[prefix] += 1
                break

    return [
        prefix
        for prefix in required_prefixes
        if prefix_counts[prefix] < required_counts.get(prefix, 1)
    ]


def should_finalize_local_area_release(
    version: str,
    new_repo_paths: Sequence[str],
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
) -> tuple[bool, list[str]]:
    existing_manifest = load_release_manifest_from_hf(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    missing_prefixes = missing_release_prefixes(
        existing_manifest=existing_manifest,
        new_repo_paths=new_repo_paths,
    )
    return not missing_prefixes, missing_prefixes


def preflight_release_manifest_publish(
    files_with_paths: Sequence[Tuple[Path | str, str]],
    version: str,
    new_repo_paths: Sequence[str],
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    model_package_name: str = "policyengine-us",
    model_package_version: Optional[str] = None,
    pipeline_run_id: str = "",
    run_context: Optional[Dict] = None,
) -> tuple[bool, list[str]]:
    should_finalize, missing_prefixes = should_finalize_local_area_release(
        version=version,
        new_repo_paths=new_repo_paths,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    assert_release_not_finalized(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    existing_manifest = load_release_manifest_from_hf(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
    resolved_run_context = run_context or _run_context_for_release()
    model_build_metadata = _get_model_package_build_metadata(model_package_name)
    create_release_manifest_commit_operations(
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
        core_package_metadata=model_build_metadata.get("core"),
        run_context=resolved_run_context,
        pipeline_run_id=_pipeline_run_id_for_manifest(
            pipeline_run_id, resolved_run_context
        ),
        data_package_git_sha=_get_data_package_git_sha(),
        existing_manifest=existing_manifest,
    )
    return should_finalize, missing_prefixes


def create_release_manifest_commit_operations(
    files_with_repo_paths: List[Tuple[Path, str]],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    model_package_name: str = "policyengine-us",
    model_package_version: Optional[str] = None,
    model_package_git_sha: Optional[str] = None,
    model_package_data_build_fingerprint: Optional[str] = None,
    run_context: Optional[Dict] = None,
    core_package_metadata: Optional[Mapping[str, Any]] = None,
    pipeline_run_id: Optional[str] = None,
    data_package_git_sha: Optional[str] = None,
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
        run_context=run_context,
        core_package_metadata=core_package_metadata,
        pipeline_run_id=pipeline_run_id,
        data_package_git_sha=data_package_git_sha,
        existing_manifest=existing_manifest,
    )
    operations = create_release_manifest_operations_from_manifest(
        manifest,
        version=version,
    )
    return manifest, operations


def create_release_manifest_operations_from_manifest(
    manifest: Mapping[str, Any],
    *,
    version: str,
    include_root_paths: bool = True,
) -> List[CommitOperationAdd]:
    """Create HF commit operations for an already-built release manifest."""
    manifest_payload = serialize_release_manifest(manifest)
    trace_tro_payload = serialize_trace_tro(
        build_trace_tro_from_release_manifest(manifest)
    )

    operations = []
    if include_root_paths:
        operations.append(
            CommitOperationAdd(
                path_in_repo=RELEASE_MANIFEST_PATH,
                path_or_fileobj=BytesIO(manifest_payload),
            )
        )
    operations.append(
        CommitOperationAdd(
            path_in_repo=f"releases/{version}/{RELEASE_MANIFEST_PATH}",
            path_or_fileobj=BytesIO(manifest_payload),
        )
    )
    if include_root_paths:
        operations.append(
            CommitOperationAdd(
                path_in_repo=TRACE_TRO_FILENAME,
                path_or_fileobj=BytesIO(trace_tro_payload),
            )
        )
    operations.append(
        CommitOperationAdd(
            path_in_repo=f"releases/{version}/{TRACE_TRO_FILENAME}",
            path_or_fileobj=BytesIO(trace_tro_payload),
        )
    )
    return operations


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
            exist_ok=False,
        )
        logging.info(
            "Tagged revision %s with %s in Hugging Face repository %s.",
            revision,
            version,
            hf_repo_name,
        )
    except Exception as e:
        if "Tag reference exists already" in str(e) or "409" in str(e):
            tagged_revision = getattr(
                api.repo_info(
                    repo_id=hf_repo_name,
                    repo_type=hf_repo_type,
                    revision=version,
                    token=token,
                ),
                "sha",
                None,
            )
            if tagged_revision == revision:
                logging.info(
                    "Tag %s already exists in %s and already points to %s.",
                    version,
                    hf_repo_name,
                    revision,
                )
                return
            raise RuntimeError(
                f"Tag {version} already exists in {hf_repo_name} at "
                f"{tagged_revision}; refusing to treat {revision} as finalized."
            ) from e
        raise


def get_matching_finalized_release_manifest(
    files_with_paths: List[Tuple[Path, str]],
    version: str,
    hf_repo_name: str,
    hf_repo_type: str,
    model_package_name: str,
    model_package_version: Optional[str] = None,
    pipeline_run_id: str = "",
    run_context: Optional[Dict] = None,
) -> Optional[Dict]:
    finalized_manifest = load_release_manifest_from_hf(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
        revision=version,
    )
    if finalized_manifest is None:
        return None

    model_build_metadata = _get_model_package_build_metadata(model_package_name)
    finalized_build = finalized_manifest.get("build")
    finalized_build = finalized_build if isinstance(finalized_build, dict) else {}
    finalized_build_metadata = finalized_build.get("metadata")
    finalized_build_metadata = (
        finalized_build_metadata if isinstance(finalized_build_metadata, dict) else {}
    )
    finalized_run_context = finalized_build_metadata.get(
        "run_context"
    ) or finalized_build.get("run")
    finalized_run_context = (
        finalized_run_context if isinstance(finalized_run_context, Mapping) else None
    )
    finalized_core_metadata = finalized_build.get("built_with_core_package")
    finalized_core_metadata = (
        finalized_core_metadata
        if isinstance(finalized_core_metadata, Mapping)
        else None
    )
    candidate_manifest, _ = create_release_manifest_commit_operations(
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
        core_package_metadata=finalized_core_metadata,
        run_context=finalized_run_context,
        pipeline_run_id=finalized_build_metadata.get("pipeline_run_id"),
        data_package_git_sha=finalized_build_metadata.get("data_package_git_sha"),
        existing_manifest=finalized_manifest,
    )
    candidate_build = candidate_manifest.setdefault("build", {})
    for field in ("build_id", "built_at"):
        if field in finalized_build:
            candidate_build[field] = finalized_build[field]

    comparable_finalized_manifest = deepcopy(finalized_manifest)
    legacy_created_at = comparable_finalized_manifest.pop("created_at", None)
    if legacy_created_at is not None and "built_at" not in finalized_build:
        candidate_build["built_at"] = legacy_created_at
    if legacy_created_at is not None:
        comparable_finalized_manifest.setdefault("build", {}).setdefault(
            "built_at", legacy_created_at
        )
    comparable_build = comparable_finalized_manifest.get("build")
    if isinstance(comparable_build, dict):
        legacy_run = comparable_build.pop("run", None)
        if legacy_run:
            comparable_build.setdefault("metadata", {}).setdefault(
                "run_context", legacy_run
            )
    comparable_finalized_manifest.setdefault("compatible_core_packages", [])
    if candidate_manifest != comparable_finalized_manifest:
        raise RuntimeError(
            f"Release {version} is already finalized on {hf_repo_name}. "
            "Refusing to mutate the tagged release manifest."
        )
    return finalized_manifest


def upload_data_files(
    files: List[str],
    gcs_bucket_name: str = "policyengine-us-data",
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    version: str = None,
    create_tag: bool = False,
):
    if version is None:
        version = DATA_PACKAGE_VERSION

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
    assert_release_not_finalized(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
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
    run_context = _run_context_for_release()
    _, manifest_operations = create_release_manifest_commit_operations(
        files_with_repo_paths=files_with_repo_paths,
        version=version,
        hf_repo_name=hf_repo_name,
        model_package_version=model_build_metadata["version"],
        model_package_git_sha=model_build_metadata["git_sha"],
        model_package_data_build_fingerprint=model_build_metadata[
            "data_build_fingerprint"
        ],
        run_context=run_context,
        core_package_metadata=model_build_metadata.get("core"),
        pipeline_run_id=_pipeline_run_id_for_manifest(run_context=run_context),
        data_package_git_sha=_get_data_package_git_sha(),
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
    return commit_info.oid


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
        version = DATA_PACKAGE_VERSION

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
        version = DATA_PACKAGE_VERSION

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
    pipeline_run_id: str = "",
    run_context: Optional[Dict] = None,
) -> Dict:
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    resolved_run_context = run_context or _run_context_for_release()
    finalized_manifest = get_matching_finalized_release_manifest(
        files_with_paths=files_with_paths,
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
        model_package_name=model_package_name,
        model_package_version=model_package_version,
        pipeline_run_id=pipeline_run_id,
        run_context=resolved_run_context,
    )
    if finalized_manifest is not None:
        return finalized_manifest

    assert_release_not_finalized(
        version=version,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
    )
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
        run_context=resolved_run_context,
        core_package_metadata=model_build_metadata.get("core"),
        pipeline_run_id=_pipeline_run_id_for_manifest(
            pipeline_run_id, resolved_run_context
        ),
        data_package_git_sha=_get_data_package_git_sha(),
        existing_manifest=existing_manifest,
    )
    parent_commit = get_repo_head_revision(
        api=api,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
    )
    commit_info = hf_create_commit_with_retry(
        api=api,
        operations=operations,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Update release manifest for version {version}",
        parent_commit=parent_commit,
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
    parent_commit: Optional[str] = None,
    revision: Optional[str] = None,
    create_pr: Optional[bool] = None,
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
        parent_commit=parent_commit,
        revision=revision,
        create_pr=create_pr,
    )


def upload_to_staging_hf(
    files_with_paths: List[Tuple[Path, str]],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    batch_size: int = 50,
    run_id: str = "",
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
        run_id: Optional per-run scope. When set, files land under
            ``staging/{run_id}/{rel_path}`` so concurrent runs do not
            collide; otherwise they land under ``staging/{rel_path}``.

    Returns:
        Number of files uploaded
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    run_id = _resolve_staging_run_id(run_id)
    staging_prefix = _staging_prefix(run_id)
    context_payload = None
    if run_id:
        context_payload = RunContext.from_env(run_id=run_id).to_dict()
        context_payload["hf_staging_prefix"] = staging_prefix

    total_uploaded = 0
    for i in range(0, len(files_with_paths), batch_size):
        batch = files_with_paths[i : i + batch_size]
        operations = []
        if i == 0 and context_payload is not None:
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{staging_prefix}/_run_context.json",
                    path_or_fileobj=BytesIO(
                        (
                            json.dumps(context_payload, indent=2, sort_keys=True) + "\n"
                        ).encode("utf-8")
                    ),
                )
            )
        for local_path, rel_path in batch:
            local_path = Path(local_path)
            if not local_path.exists():
                logging.warning(f"File {local_path} does not exist, skipping.")
                continue
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{staging_prefix}/{rel_path}",
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
            commit_message=(
                f"Upload batch {i // batch_size + 1} to staging "
                f"for version {version}" + (f" ({run_id})" if run_id else "")
            ),
        )
        uploaded_files = len(operations) - (
            1 if i == 0 and context_payload is not None else 0
        )
        total_uploaded += uploaded_files
        logging.info(
            f"Uploaded batch {i // batch_size + 1}: "
            f"{uploaded_files} files to {staging_prefix}/"
        )

    logging.info(f"Total: uploaded {total_uploaded} files to staging/ in HuggingFace")
    return total_uploaded


def _staging_prefix(run_id: str = "") -> str:
    run_id = _resolve_staging_run_id(run_id)
    return f"staging/{run_id}" if run_id else "staging"


def _dedupe_preserving_order(paths: Sequence[str]) -> list[str]:
    seen = set()
    deduped = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def list_missing_staged_artifacts(
    rel_paths: Sequence[str],
    *,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    run_id: str = "",
) -> list[str]:
    """Return staged HF paths that are missing for this run."""
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    run_id = _resolve_staging_run_id(run_id)
    staging_prefix = _staging_prefix(run_id)
    repo_files = set(
        api.list_repo_files(
            repo_id=hf_repo_name,
            repo_type=hf_repo_type,
            token=token,
        )
    )
    return [
        f"{staging_prefix}/{rel_path}"
        for rel_path in _dedupe_preserving_order(rel_paths)
        if f"{staging_prefix}/{rel_path}" not in repo_files
    ]


def download_staged_artifacts_for_manifest(
    rel_paths: Sequence[str],
    *,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    run_id: str = "",
) -> list[tuple[Path, str]]:
    """Download staged HF artifacts for release-manifest checksums."""
    token = os.environ.get("HUGGING_FACE_TOKEN")
    run_id = _resolve_staging_run_id(run_id)
    staging_prefix = _staging_prefix(run_id)
    files_with_paths = []
    for rel_path in _dedupe_preserving_order(rel_paths):
        local_path = hf_hub_download(
            repo_id=hf_repo_name,
            filename=f"{staging_prefix}/{rel_path}",
            repo_type=hf_repo_type,
            token=token,
        )
        files_with_paths.append((Path(local_path), rel_path))
    return files_with_paths


def promote_staging_to_production_hf(
    files: List[str],
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    run_id: str = "",
    allow_noop: bool = False,
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
        run_id: Optional per-run scope for staged source files
        allow_noop: Treat an unchanged HF HEAD as success. This is useful
            when retrying a full-release promotion after the single HF copy
            commit already succeeded but later backends failed.

    Returns:
        Number of files promoted

    Raises:
        RuntimeError: If the commit was a no-op (HEAD unchanged)
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    run_id = _resolve_staging_run_id(run_id)
    staging_prefix = _staging_prefix(run_id)

    operations = []
    for rel_path in files:
        staging_path = f"{staging_prefix}/{rel_path}"
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
        commit_message=(
            f"Promote {len(files)} files from staging to production "
            f"for version {version}" + (f" ({run_id})" if run_id else "")
        ),
    )

    if result.oid == head_before:
        if allow_noop:
            logging.warning(
                "Promote commit was a no-op: HEAD stayed at %s. "
                "Treating as success for idempotent release retry.",
                head_before,
            )
            return len(files)
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
    run_id: str = "",
) -> int:
    """
    Clean up staging folder after successful promotion.

    Args:
        files: List of relative paths (e.g., "states/AL.h5")
        version: Version string for commit message
        hf_repo_name: HuggingFace repository
        hf_repo_type: Repository type
        run_id: Optional per-run scope for staged source files

    Returns:
        Number of files deleted

    Raises:
        RuntimeError: If the cleanup commit was a no-op (HEAD unchanged)
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    run_id = _resolve_staging_run_id(run_id)
    staging_prefix = _staging_prefix(run_id)

    existing_repo_files = None
    try:
        existing_repo_files = set(
            api.list_repo_files(
                repo_id=hf_repo_name,
                repo_type=hf_repo_type,
                token=token,
            )
        )
    except Exception as exc:
        logging.warning(
            "Could not list staged files before cleanup; attempting requested deletes: %s",
            exc,
        )

    operations = []
    for rel_path in files:
        staging_path = f"{staging_prefix}/{rel_path}"
        if existing_repo_files is not None and staging_path not in existing_repo_files:
            logging.info(
                "Skipping missing staged file during cleanup: %s", staging_path
            )
            continue
        operations.append(CommitOperationDelete(path_in_repo=staging_path))

    if not operations:
        logging.info("No staged files found to clean up.")
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
        commit_message=(
            f"Clean up staging after version {version} promotion"
            + (f" ({run_id})" if run_id else "")
        ),
    )

    if result.oid == head_before:
        raise RuntimeError(
            f"Cleanup commit was a no-op: HEAD stayed at {head_before}. "
            f"Staging files may not exist."
        )

    logging.info(f"Cleaned up {len(operations)} files from staging/")
    return len(operations)


def upload_from_hf_staging_to_gcs(
    rel_paths: List[str],
    version: str,
    gcs_bucket_name: str = "policyengine-us-data",
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    run_id: str = "",
) -> int:
    """Download files from HF staging/ and upload to GCS production paths.

    Args:
        rel_paths: Relative paths like "states/AL.h5", "districts/NC-01.h5"
        version: Version string for GCS metadata
        gcs_bucket_name: GCS bucket name
        hf_repo_name: HuggingFace repository name
        hf_repo_type: Repository type
        run_id: Optional per-run scope for staged source files

    Returns:
        Number of files uploaded
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    run_id = _resolve_staging_run_id(run_id)
    staging_prefix = _staging_prefix(run_id)

    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket = storage_client.bucket(gcs_bucket_name)

    uploaded = 0
    for rel_path in rel_paths:
        staging_filename = f"{staging_prefix}/{rel_path}"
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


def upload_final_version_manifest(
    *,
    version: str,
    released_paths: Sequence[str],
    run_id: str = "",
    hf_repo_name: str = "policyengine/policyengine-us-data",
) -> None:
    """Update version_manifest.json after a release is finalized."""
    from policyengine_us_data.utils.version_manifest import (
        HFVersionInfo,
        build_manifest,
        upload_manifest,
    )

    upload_manifest(
        build_manifest(
            version=version,
            blob_names=sorted(released_paths),
            hf_info=HFVersionInfo(repo=hf_repo_name, commit=version),
            run_id=run_id or None,
        )
    )


def _validation_report_candidates(run_id: str) -> list[str]:
    return [
        f"calibration/runs/{run_id}/diagnostics/{filename}"
        for filename in VALIDATION_REPORT_FILENAMES
    ]


def _resolve_validation_report_paths(
    *,
    repo_files: set[str],
    run_id: str,
    validation_report_paths: Optional[Sequence[str]],
) -> list[str]:
    if validation_report_paths is not None:
        return sorted(validation_report_paths)
    if not run_id:
        return []
    return sorted(
        path for path in _validation_report_candidates(run_id) if path in repo_files
    )


def _missing_release_completion_paths(
    *,
    repo_files: set[str],
    version: str,
    expected_paths: Sequence[str],
    validation_report_paths: Sequence[str],
) -> list[str]:
    required_paths = [
        *expected_paths,
        RELEASE_MANIFEST_PATH,
        f"releases/{version}/{RELEASE_MANIFEST_PATH}",
        TRACE_TRO_FILENAME,
        f"releases/{version}/{TRACE_TRO_FILENAME}",
        VERSION_MANIFEST_PATH,
        *validation_report_paths,
    ]
    return sorted(path for path in required_paths if path not in repo_files)


def upload_release_completion_marker_to_hf(
    *,
    version: str,
    run_id: str,
    released_paths: Sequence[str],
    expected_paths: Sequence[str],
    release_manifest: Mapping[str, Any],
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    promoted_hf: int = 0,
    uploaded_gcs: int = 0,
    create_tag: bool = False,
    validation_report_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Write the final release completion marker after all release writes."""
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    repo_files = set(
        api.list_repo_files(
            repo_id=hf_repo_name,
            repo_type=hf_repo_type,
            token=token,
        )
    )
    resolved_validation_report_paths = _resolve_validation_report_paths(
        repo_files=repo_files,
        run_id=run_id,
        validation_report_paths=validation_report_paths,
    )
    missing_paths = _missing_release_completion_paths(
        repo_files=repo_files,
        version=version,
        expected_paths=expected_paths,
        validation_report_paths=resolved_validation_report_paths,
    )
    if run_id and not resolved_validation_report_paths:
        missing_paths.append(
            f"calibration/runs/{run_id}/diagnostics/<validation report>"
        )
    if missing_paths:
        raise FileNotFoundError(
            "Cannot mark release complete; missing required release paths: "
            + ", ".join(sorted(missing_paths))
        )

    marker = build_release_completion_marker(
        version=version,
        run_id=run_id,
        hf_repo_name=hf_repo_name,
        hf_repo_type=hf_repo_type,
        release_manifest=release_manifest,
        released_paths=released_paths,
        validation_report_paths=resolved_validation_report_paths,
        promoted_hf=promoted_hf,
        uploaded_gcs=uploaded_gcs,
    )
    parent_commit = get_repo_head_revision(
        api=api,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
    )
    commit_info = hf_create_commit_with_retry(
        api=api,
        operations=[
            CommitOperationAdd(
                path_in_repo=release_completion_marker_path(version),
                path_or_fileobj=BytesIO(
                    serialize_release_completion_marker(marker),
                ),
            )
        ],
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Mark release {version} complete",
        parent_commit=parent_commit,
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
    return marker


def release_completion_marker_exists_on_hf(
    *,
    version: str,
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
) -> bool:
    """Return True only for a valid marker at the version tag."""
    token = os.environ.get("HUGGING_FACE_TOKEN")
    try:
        local_path = hf_hub_download(
            repo_id=hf_repo_name,
            filename=release_completion_marker_path(version),
            repo_type=hf_repo_type,
            revision=version,
            token=token,
        )
        with open(local_path, encoding="utf-8") as marker_file:
            marker = json.load(marker_file)
        validate_release_completion_marker(
            marker,
            version=version,
            hf_repo_name=hf_repo_name,
            hf_repo_type=hf_repo_type,
        )
    except (
        EntryNotFoundError,
        RevisionNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return True


def _full_release_promotion_dependencies() -> FullReleasePromotionDependencies:
    return FullReleasePromotionDependencies(
        dedupe_preserving_order=_dedupe_preserving_order,
        download_staged_artifacts_for_manifest=download_staged_artifacts_for_manifest,
        get_matching_finalized_release_manifest=get_matching_finalized_release_manifest,
        list_missing_staged_artifacts=list_missing_staged_artifacts,
        preflight_release_manifest_publish=preflight_release_manifest_publish,
        promote_staging_to_production_hf=promote_staging_to_production_hf,
        upload_from_hf_staging_to_gcs=upload_from_hf_staging_to_gcs,
        publish_release_manifest_to_hf=publish_release_manifest_to_hf,
        upload_final_version_manifest=upload_final_version_manifest,
        upload_release_completion_marker=upload_release_completion_marker_to_hf,
        release_completion_marker_exists=release_completion_marker_exists_on_hf,
        cleanup_staging_hf=cleanup_staging_hf,
    )


def promote_full_release_from_staging(
    *,
    rel_paths: Sequence[str],
    version: str,
    run_id: str = "",
    run_context: Optional[Dict] = None,
    files_with_paths: Optional[Sequence[Tuple[Path | str, str]]] = None,
    extra_cleanup_paths: Sequence[str] = (),
    gcs_bucket_name: str = "policyengine-us-data",
    hf_repo_name: str = "policyengine/policyengine-us-data",
    hf_repo_type: str = "model",
    cleanup_staging: bool = True,
) -> dict:
    """Promote one complete run-scoped staged release."""
    run_id = _resolve_staging_run_id(run_id)
    if not run_id:
        raise ValueError("run_id is required for full release promotion.")
    if not version:
        raise ValueError("version is required for full release promotion.")

    _apply_run_context_for_release(run_id, run_context)

    return promote_full_release(
        FullReleasePromotionConfig(
            rel_paths=rel_paths,
            version=version,
            run_id=run_id,
            files_with_paths=files_with_paths,
            extra_cleanup_paths=extra_cleanup_paths,
            gcs_bucket_name=gcs_bucket_name,
            hf_repo_name=hf_repo_name,
            hf_repo_type=hf_repo_type,
            cleanup_staging=cleanup_staging,
        ),
        deps=_full_release_promotion_dependencies(),
    )
