"""
Version registry for semver-based dataset versioning.

Provides typed structures and functions for versioned uploads,
downloads, and rollbacks across GCS and Hugging Face. All
versions are tracked in a single registry file
(version_manifest.json) on both backends.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import google.auth
from google.api_core.exceptions import NotFound
from google.cloud import storage
from huggingface_hub import (
    HfApi,
    CommitOperationAdd,
    hf_hub_download,
)

# -- Configuration -------------------------------------------------

REGISTRY_BLOB = "version_manifest.json"
GCS_BUCKET_NAME = "policyengine-us-data"
HF_REPO_NAME = "policyengine/policyengine-us-data"
HF_REPO_TYPE = "model"


# -- Types ---------------------------------------------------------


@dataclass
class HFVersionInfo:
    """Hugging Face backend location for a version."""

    repo: str
    commit: str

    def to_dict(self) -> dict[str, str]:
        return {"repo": self.repo, "commit": self.commit}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HFVersionInfo":
        return cls(repo=data["repo"], commit=data["commit"])


@dataclass
class GCSVersionInfo:
    """GCS backend location for a version."""

    bucket: str
    generations: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "generations": self.generations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GCSVersionInfo":
        return cls(
            bucket=data["bucket"],
            generations=data["generations"],
        )


@dataclass
class VersionManifest:
    """Single version entry tying semver to backend
    identifiers.

    Consumers interact only with the semver version string.
    HF commit SHAs and GCS generation numbers are internal
    implementation details resolved by this manifest.
    """

    version: str
    created_at: str
    hf: Optional[HFVersionInfo]
    gcs: GCSVersionInfo
    special_operation: Optional[str] = None
    roll_back_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": self.version,
            "created_at": self.created_at,
            "hf": self.hf.to_dict() if self.hf else None,
            "gcs": self.gcs.to_dict(),
        }
        if self.special_operation is not None:
            result["special_operation"] = self.special_operation
        if self.roll_back_version is not None:
            result["roll_back_version"] = self.roll_back_version
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionManifest":
        hf_data = data.get("hf")
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            hf=(HFVersionInfo.from_dict(hf_data) if hf_data else None),
            gcs=GCSVersionInfo.from_dict(data["gcs"]),
            special_operation=data.get("special_operation"),
            roll_back_version=data.get("roll_back_version"),
        )


@dataclass
class VersionRegistry:
    """Registry of all dataset versions.

    Contains a pointer to the current version and a list of
    all version manifests (most recent first).
    """

    current: str = ""
    versions: list[VersionManifest] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "versions": [v.to_dict() for v in self.versions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionRegistry":
        return cls(
            current=data["current"],
            versions=[VersionManifest.from_dict(v) for v in data["versions"]],
        )

    def get_version(self, version: str) -> VersionManifest:
        """Look up a specific version entry.

        Args:
            version: Semver version string.

        Returns:
            The matching VersionManifest.

        Raises:
            ValueError: If the version is not in the
                registry.
        """
        for v in self.versions:
            if v.version == version:
                return v
        available = [v.version for v in self.versions[:10]]
        raise ValueError(
            f"Version '{version}' not found in registry. "
            f"Available versions: {available}"
        )


# -- Internal helpers ----------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_gcs_bucket() -> storage.Bucket:
    """Return an authenticated GCS bucket handle."""
    credentials, project_id = google.auth.default()
    client = storage.Client(credentials=credentials, project=project_id)
    return client.bucket(GCS_BUCKET_NAME)


def _read_registry_from_gcs(
    bucket: storage.Bucket,
) -> VersionRegistry:
    """Read the version registry from GCS.

    Returns an empty registry if no registry exists yet.
    """
    blob = bucket.blob(REGISTRY_BLOB)
    try:
        content = blob.download_as_text()
    except NotFound:
        return VersionRegistry()
    return VersionRegistry.from_dict(json.loads(content))


def _upload_registry_to_gcs(
    bucket: storage.Bucket,
    registry: VersionRegistry,
) -> None:
    """Write the version registry to GCS."""
    data = json.dumps(registry.to_dict(), indent=2)
    blob = bucket.blob(REGISTRY_BLOB)
    blob.upload_from_string(data, content_type="application/json")
    logging.info("Uploaded registry to GCS " f"(current={registry.current}).")


def _upload_registry_to_hf(
    registry: VersionRegistry,
) -> None:
    """Write the version registry to Hugging Face."""
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    data = json.dumps(registry.to_dict(), indent=2)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        f.write(data)
        tmp_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=REGISTRY_BLOB,
            repo_id=HF_REPO_NAME,
            repo_type=HF_REPO_TYPE,
            token=token,
            commit_message=(
                "Update version registry " f"(current={registry.current})"
            ),
        )
        logging.info(
            f"Uploaded {REGISTRY_BLOB} to " f"HF repo {HF_REPO_NAME}."
        )
    finally:
        os.unlink(tmp_path)


def _restore_gcs_generations(
    bucket: storage.Bucket,
    old_generations: dict[str, int],
) -> dict[str, int]:
    """Copy old GCS generation blobs to live paths.

    Args:
        bucket: GCS bucket containing the blobs.
        old_generations: Map of blob path to old generation
            number.

    Returns:
        Map of blob path to new generation number.
    """
    new_generations: dict[str, int] = {}
    for file_path, generation in old_generations.items():
        source_blob = bucket.blob(file_path, generation=generation)
        bucket.copy_blob(source_blob, bucket, file_path)
        restored_blob = bucket.get_blob(file_path)
        new_generations[file_path] = restored_blob.generation
        logging.info(
            f"Restored {file_path}: generation "
            f"{generation} -> {restored_blob.generation}."
        )
    return new_generations


def _restore_hf_commit(
    old_manifest: VersionManifest,
    new_version: str,
) -> str:
    """Re-upload old HF data as a new commit and tag it.

    Args:
        old_manifest: The manifest of the version being
            restored.
        new_version: The new semver version string for
            tagging.

    Returns:
        The commit SHA of the new HF commit.
    """
    token = os.environ.get("HUGGING_FACE_TOKEN")
    api = HfApi()
    target_version = old_manifest.version

    operations = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for file_path in old_manifest.gcs.generations:
            hf_hub_download(
                repo_id=old_manifest.hf.repo,
                repo_type=HF_REPO_TYPE,
                filename=file_path,
                revision=old_manifest.hf.commit,
                local_dir=tmpdir,
                token=token,
            )
            downloaded = os.path.join(tmpdir, file_path)
            operations.append(
                CommitOperationAdd(
                    path_in_repo=file_path,
                    path_or_fileobj=downloaded,
                )
            )

        commit_info = api.create_commit(
            token=token,
            repo_id=HF_REPO_NAME,
            operations=operations,
            repo_type=HF_REPO_TYPE,
            commit_message=(
                f"Roll back to {target_version} " f"as {new_version}"
            ),
        )

    try:
        api.create_tag(
            token=token,
            repo_id=HF_REPO_NAME,
            tag=new_version,
            revision=commit_info.oid,
            repo_type=HF_REPO_TYPE,
        )
    except Exception as e:
        if "already exists" in str(e) or "409" in str(e):
            logging.warning(
                f"Tag {new_version} already exists. " "Skipping tag creation."
            )
        else:
            raise

    return commit_info.oid


# -- Public API ----------------------------------------------------


def build_manifest(
    version: str,
    blob_names: list[str],
    hf_info: Optional[HFVersionInfo] = None,
) -> VersionManifest:
    """Build a version manifest by reading generation
    numbers from uploaded blobs.

    Args:
        version: Semver version string.
        blob_names: List of blob paths to include.
        hf_info: Optional HF backend info to include.

    Returns:
        A VersionManifest with generation numbers for
        each blob.
    """
    bucket = _get_gcs_bucket()
    generations: dict[str, int] = {}
    for name in blob_names:
        blob = bucket.get_blob(name)
        if blob is None:
            raise ValueError(
                f"Blob '{name}' not found in bucket "
                f"'{bucket.name}' after upload."
            )
        generations[name] = blob.generation

    return VersionManifest(
        version=version,
        created_at=_utc_now_iso(),
        hf=hf_info,
        gcs=GCSVersionInfo(
            bucket=bucket.name,
            generations=generations,
        ),
    )


def upload_manifest(
    manifest: VersionManifest,
) -> None:
    """Append a version manifest to the registry and
    upload to both GCS and HF.

    Reads the existing registry from GCS (or starts fresh),
    prepends the new manifest, updates the current pointer,
    and writes the registry to both backends.

    Args:
        manifest: The version manifest to add.
    """
    bucket = _get_gcs_bucket()
    registry = _read_registry_from_gcs(bucket)
    registry.versions.insert(0, manifest)
    registry.current = manifest.version
    _upload_registry_to_gcs(bucket, registry)
    _upload_registry_to_hf(registry)


def get_current_version() -> Optional[str]:
    """Get the current version from the registry.

    Returns:
        The current semver version string, or None if no
        registry exists.
    """
    bucket = _get_gcs_bucket()
    registry = _read_registry_from_gcs(bucket)
    if not registry.current:
        return None
    return registry.current


def get_manifest(version: str) -> VersionManifest:
    """Get the manifest for a specific version.

    Args:
        version: Semver version string.

    Returns:
        The deserialized VersionManifest.

    Raises:
        ValueError: If the version is not in the registry.
    """
    bucket = _get_gcs_bucket()
    registry = _read_registry_from_gcs(bucket)
    return registry.get_version(version)


def list_versions() -> list[str]:
    """List all available versions.

    Returns:
        Sorted list of semver version strings.
    """
    bucket = _get_gcs_bucket()
    registry = _read_registry_from_gcs(bucket)
    return sorted(v.version for v in registry.versions)


def download_versioned_file(
    file_path: str,
    version: str,
    local_path: str,
) -> str:
    """Download a specific file at a specific version.

    Args:
        file_path: Path of the file within the bucket.
        version: Semver version string.
        local_path: Local path to save the file to.

    Returns:
        The local path where the file was saved.

    Raises:
        ValueError: If the version or file is not found.
    """
    bucket = _get_gcs_bucket()
    registry = _read_registry_from_gcs(bucket)
    manifest = registry.get_version(version)

    if file_path not in manifest.gcs.generations:
        raise ValueError(
            f"File '{file_path}' not found in manifest "
            f"for version '{version}'. Available files: "
            f"{list(manifest.gcs.generations.keys())[:10]}"
            "..."
        )

    generation = manifest.gcs.generations[file_path]
    blob = bucket.blob(file_path, generation=generation)

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)

    logging.info(
        f"Downloaded {file_path} at version {version} "
        f"(generation {generation}) to {local_path}."
    )
    return local_path


def rollback(
    target_version: str,
    new_version: str,
) -> VersionManifest:
    """Roll back by releasing a new version with old data.

    Treats rollback as a new release: data from
    target_version is copied to the live paths (creating
    new GCS generations), a new HF commit is created with
    the old data, and a new manifest is published under
    new_version with special_operation="roll-back".

    Args:
        target_version: Semver version to roll back to.
        new_version: New semver version to publish.

    Returns:
        The new VersionManifest for the rollback release.

    Raises:
        ValueError: If target_version is not in the
            registry.
    """
    bucket = _get_gcs_bucket()
    old_manifest = _read_registry_from_gcs(bucket).get_version(target_version)

    new_gens = _restore_gcs_generations(bucket, old_manifest.gcs.generations)
    hf_commit = (
        _restore_hf_commit(old_manifest, new_version)
        if old_manifest.hf
        else None
    )

    manifest = VersionManifest(
        version=new_version,
        created_at=_utc_now_iso(),
        hf=(
            HFVersionInfo(repo=HF_REPO_NAME, commit=hf_commit)
            if hf_commit
            else None
        ),
        gcs=GCSVersionInfo(
            bucket=GCS_BUCKET_NAME,
            generations=new_gens,
        ),
        special_operation="roll-back",
        roll_back_version=target_version,
    )
    upload_manifest(manifest)

    logging.info(
        f"Rolled back to {target_version} as new "
        f"version {new_version}. "
        f"Restored {len(new_gens)} files."
    )
    return manifest


# -- Consumer API --------------------------------------------------

_cached_registry: Optional[VersionRegistry] = None


def get_data_manifest() -> VersionRegistry:
    """Get the full version registry from HF.

    Fetches version_manifest.json from the Hugging Face
    repo and returns it as a VersionRegistry. The result
    is cached in memory after the first call.

    Returns:
        The full VersionRegistry.
    """
    global _cached_registry
    if _cached_registry is not None:
        return _cached_registry

    local_path = hf_hub_download(
        repo_id=HF_REPO_NAME,
        repo_type=HF_REPO_TYPE,
        filename=REGISTRY_BLOB,
    )
    with open(local_path) as f:
        data = json.load(f)

    _cached_registry = VersionRegistry.from_dict(data)
    return _cached_registry


def get_data_version() -> str:
    """Get the current deployed data version string.

    Convenience wrapper around get_data_manifest().

    Returns:
        The current semver version string.
    """
    return get_data_manifest().current
