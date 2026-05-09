from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from huggingface_hub import HfApi

from policyengine_us_data.utils.data_upload import (
    create_release_manifest_operations_from_manifest,
    get_repo_head_revision,
    hf_create_commit_with_retry,
)
from policyengine_us_data.utils.release_manifest import (
    build_release_manifest,
    serialize_release_manifest,
)


DEFAULT_HF_REPO_NAME = "policyengine/policyengine-us-data"
DEFAULT_HF_REPO_TYPE = "model"
DEFAULT_MODEL_PACKAGE_NAME = "policyengine-us"
DEFAULT_CORE_PACKAGE_NAME = "policyengine-core"
DEFAULT_NATIONAL_DATASET = "enhanced_cps_2024"


@dataclass(frozen=True)
class HfArtifactMetadata:
    path: str
    sha256: str
    size_bytes: int


def _artifact_key(path_in_repo: str) -> str:
    return str(PurePosixPath(path_in_repo).with_suffix(""))


def _artifact_kind(path_in_repo: str) -> str:
    suffix = PurePosixPath(path_in_repo).suffix.lower()
    if suffix == ".h5":
        return "microdata"
    if suffix == ".db":
        return "database"
    if suffix == ".npz":
        return "geography"
    if suffix == ".npy":
        return "weights"
    return "auxiliary"


def _require_lfs_artifact(entry: Any) -> HfArtifactMetadata:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        raise ValueError(f"{entry.path} is not an LFS artifact with checksum metadata.")
    sha256 = getattr(lfs, "sha256", None)
    size = getattr(lfs, "size", None) or getattr(entry, "size", None)
    if not sha256 or size is None:
        raise ValueError(f"{entry.path} is missing LFS sha256/size metadata.")
    return HfArtifactMetadata(
        path=entry.path,
        sha256=sha256,
        size_bytes=int(size),
    )


def collect_hf_artifact_metadata(
    *,
    version: str,
    artifact_paths: Sequence[str],
    hf_repo_name: str = DEFAULT_HF_REPO_NAME,
    hf_repo_type: str = DEFAULT_HF_REPO_TYPE,
    api: HfApi | None = None,
    token: str | None = None,
) -> list[HfArtifactMetadata]:
    """Collect artifact checksums from an existing Hugging Face revision."""
    if not artifact_paths:
        raise ValueError("At least one artifact path is required.")

    api = api or HfApi()
    requested_paths = set(artifact_paths)
    found: dict[str, HfArtifactMetadata] = {}
    for entry in api.list_repo_tree(
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        revision=version,
        recursive=True,
        expand=True,
        token=token,
    ):
        path = getattr(entry, "path", None)
        if path not in requested_paths:
            continue
        found[path] = _require_lfs_artifact(entry)
        if len(found) == len(requested_paths):
            break

    missing = sorted(requested_paths - set(found))
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s) at {hf_repo_name}@{version}: " + ", ".join(missing)
        )
    return [found[path] for path in artifact_paths]


def build_backfilled_release_manifest(
    *,
    version: str,
    artifacts: Sequence[HfArtifactMetadata],
    model_package_version: str,
    core_package_version: str,
    hf_repo_name: str = DEFAULT_HF_REPO_NAME,
    model_package_name: str = DEFAULT_MODEL_PACKAGE_NAME,
    model_package_git_sha: str | None = None,
    model_package_data_build_fingerprint: str | None = None,
    data_package_git_sha: str | None = None,
    default_national_dataset: str | None = DEFAULT_NATIONAL_DATASET,
) -> dict[str, Any]:
    """Build a certifiable release manifest from already-published HF artifacts."""
    manifest = build_release_manifest(
        files_with_repo_paths=[],
        version=version,
        repo_id=hf_repo_name,
        model_package_name=model_package_name,
        model_package_version=model_package_version,
        model_package_git_sha=model_package_git_sha,
        model_package_data_build_fingerprint=model_package_data_build_fingerprint,
        core_package_metadata={
            "name": DEFAULT_CORE_PACKAGE_NAME,
            "version": core_package_version,
        },
        data_package_git_sha=data_package_git_sha,
        build_id=f"policyengine-us-data-{version}",
    )
    manifest["artifacts"] = {
        _artifact_key(artifact.path): {
            "kind": _artifact_kind(artifact.path),
            "path": artifact.path,
            "repo_id": hf_repo_name,
            "revision": version,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in artifacts
    }
    if (
        default_national_dataset is not None
        and default_national_dataset in manifest["artifacts"]
    ):
        manifest.setdefault("default_datasets", {})["national"] = (
            default_national_dataset
        )
    return manifest


def upload_backfilled_release_manifest(
    manifest: Mapping[str, Any],
    *,
    version: str,
    hf_repo_name: str = DEFAULT_HF_REPO_NAME,
    hf_repo_type: str = DEFAULT_HF_REPO_TYPE,
    revision: str | None = None,
    create_pr: bool = False,
    token: str | None = None,
    api: HfApi | None = None,
) -> str:
    """Upload a backfilled manifest to a branch/revision without moving tags."""
    api = api or HfApi()
    token = token or os.environ.get("HUGGING_FACE_TOKEN")
    parent_commit = get_repo_head_revision(
        api=api,
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        revision=revision,
        token=token,
    )
    commit_info = hf_create_commit_with_retry(
        api=api,
        operations=create_release_manifest_operations_from_manifest(
            manifest,
            version=version,
            include_root_paths=False,
        ),
        repo_id=hf_repo_name,
        repo_type=hf_repo_type,
        token=token,
        commit_message=f"Backfill release manifest for version {version}",
        parent_commit=parent_commit,
        revision=revision,
        create_pr=create_pr,
    )
    return commit_info.oid


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_release_manifest(manifest))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a release_manifest.json from checksums on an existing "
            "Hugging Face policyengine-us-data revision."
        )
    )
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        required=True,
        help="Artifact path on Hugging Face. Repeat for multiple artifacts.",
    )
    parser.add_argument("--model-package-version", required=True)
    parser.add_argument("--core-package-version", required=True)
    parser.add_argument("--model-package-git-sha")
    parser.add_argument("--model-package-data-build-fingerprint")
    parser.add_argument("--data-package-git-sha")
    parser.add_argument("--hf-repo-name", default=DEFAULT_HF_REPO_NAME)
    parser.add_argument("--hf-repo-type", default=DEFAULT_HF_REPO_TYPE)
    parser.add_argument(
        "--default-national-dataset",
        default=DEFAULT_NATIONAL_DATASET,
        help="Artifact key to mark as default national dataset; use '' for none.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the generated manifest locally instead of printing JSON.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload release_manifest.json and TRACE TRO to Hugging Face.",
    )
    parser.add_argument(
        "--upload-revision",
        help="Branch/revision to receive the upload. Defaults to the repo default branch.",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Ask Hugging Face to create a PR for the upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    token = os.environ.get("HUGGING_FACE_TOKEN")
    artifacts = collect_hf_artifact_metadata(
        version=args.version,
        artifact_paths=args.artifacts,
        hf_repo_name=args.hf_repo_name,
        hf_repo_type=args.hf_repo_type,
        token=token,
    )
    manifest = build_backfilled_release_manifest(
        version=args.version,
        artifacts=artifacts,
        model_package_version=args.model_package_version,
        core_package_version=args.core_package_version,
        hf_repo_name=args.hf_repo_name,
        model_package_git_sha=args.model_package_git_sha,
        model_package_data_build_fingerprint=args.model_package_data_build_fingerprint,
        data_package_git_sha=args.data_package_git_sha,
        default_national_dataset=args.default_national_dataset or None,
    )
    if args.output is not None:
        _write_manifest(args.output, manifest)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.upload:
        commit_oid = upload_backfilled_release_manifest(
            manifest,
            version=args.version,
            hf_repo_name=args.hf_repo_name,
            hf_repo_type=args.hf_repo_type,
            revision=args.upload_revision,
            create_pr=args.create_pr,
            token=token,
        )
        print(f"Uploaded release manifest commit: {commit_oid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
