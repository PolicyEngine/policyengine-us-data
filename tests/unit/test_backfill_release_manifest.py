from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from policyengine_us_data.storage.backfill_release_manifest import (
    HfArtifactMetadata,
    build_backfilled_release_manifest,
    collect_hf_artifact_metadata,
    upload_backfilled_release_manifest,
)


def _repo_file(path: str, sha256: str, size: int):
    return SimpleNamespace(
        path=path,
        lfs=SimpleNamespace(sha256=sha256, size=size),
        size=size,
    )


def test_collect_hf_artifact_metadata_reads_lfs_checksums_in_requested_order():
    api = MagicMock()
    api.list_repo_tree.return_value = [
        _repo_file("states/AL.h5", "b" * 64, 2),
        _repo_file("enhanced_cps_2024.h5", "a" * 64, 1),
    ]

    artifacts = collect_hf_artifact_metadata(
        version="1.73.0",
        artifact_paths=["enhanced_cps_2024.h5", "states/AL.h5"],
        api=api,
    )

    assert artifacts == [
        HfArtifactMetadata(
            path="enhanced_cps_2024.h5",
            sha256="a" * 64,
            size_bytes=1,
        ),
        HfArtifactMetadata(path="states/AL.h5", sha256="b" * 64, size_bytes=2),
    ]
    api.list_repo_tree.assert_called_once_with(
        repo_id="policyengine/policyengine-us-data",
        repo_type="model",
        revision="1.73.0",
        recursive=True,
        expand=True,
        token=None,
    )


def test_collect_hf_artifact_metadata_rejects_missing_paths():
    api = MagicMock()
    api.list_repo_tree.return_value = [
        _repo_file("enhanced_cps_2024.h5", "a" * 64, 1),
    ]

    with pytest.raises(FileNotFoundError, match="states/AL.h5"):
        collect_hf_artifact_metadata(
            version="1.73.0",
            artifact_paths=["enhanced_cps_2024.h5", "states/AL.h5"],
            api=api,
        )


def test_build_backfilled_release_manifest_records_exact_core_metadata():
    manifest = build_backfilled_release_manifest(
        version="1.73.0",
        artifacts=[
            HfArtifactMetadata(
                path="enhanced_cps_2024.h5",
                sha256="a" * 64,
                size_bytes=1,
            ),
            HfArtifactMetadata(
                path="enhanced_cps_2024.clone_diagnostics.json",
                sha256="b" * 64,
                size_bytes=2,
            ),
        ],
        model_package_version="1.653.3",
        core_package_version="3.26.0",
        model_package_data_build_fingerprint="sha256:stable",
    )

    assert manifest["data_package"] == {
        "name": "policyengine-us-data",
        "version": "1.73.0",
    }
    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-us", "specifier": "==1.653.3"}
    ]
    assert manifest["compatible_core_packages"] == [
        {"name": "policyengine-core", "specifier": "==3.26.0"}
    ]
    assert manifest["default_datasets"] == {"national": "enhanced_cps_2024"}
    assert manifest["build"]["built_with_core_package"] == {
        "name": "policyengine-core",
        "version": "3.26.0",
    }
    assert manifest["build"]["built_with_model_package"]["core"] == {
        "name": "policyengine-core",
        "version": "3.26.0",
    }
    assert manifest["artifacts"]["enhanced_cps_2024"] == {
        "kind": "microdata",
        "path": "enhanced_cps_2024.h5",
        "repo_id": "policyengine/policyengine-us-data",
        "revision": "1.73.0",
        "sha256": "a" * 64,
        "size_bytes": 1,
    }
    assert manifest["artifacts"]["enhanced_cps_2024.clone_diagnostics"] == {
        "kind": "auxiliary",
        "path": "enhanced_cps_2024.clone_diagnostics.json",
        "repo_id": "policyengine/policyengine-us-data",
        "revision": "1.73.0",
        "sha256": "b" * 64,
        "size_bytes": 2,
    }


def test_build_backfilled_release_manifest_records_additional_core_compatibility():
    manifest = build_backfilled_release_manifest(
        version="1.73.0",
        artifacts=[
            HfArtifactMetadata(
                path="enhanced_cps_2024.h5",
                sha256="a" * 64,
                size_bytes=1,
            )
        ],
        model_package_version="1.653.3",
        core_package_version="3.26.0",
        compatible_core_package_versions=["3.26.1"],
    )

    assert manifest["build"]["built_with_core_package"] == {
        "name": "policyengine-core",
        "version": "3.26.0",
    }
    assert manifest["compatible_core_packages"] == [
        {"name": "policyengine-core", "specifier": "==3.26.0"},
        {"name": "policyengine-core", "specifier": "==3.26.1"},
    ]


def test_build_backfilled_release_manifest_records_additional_model_compatibility():
    manifest = build_backfilled_release_manifest(
        version="1.73.0",
        artifacts=[
            HfArtifactMetadata(
                path="enhanced_cps_2024.h5",
                sha256="a" * 64,
                size_bytes=1,
            )
        ],
        model_package_version="1.653.3",
        compatible_model_package_versions=["1.722.4"],
        core_package_version="3.26.0",
    )

    assert manifest["build"]["built_with_model_package"]["version"] == "1.653.3"
    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-us", "specifier": "==1.653.3"},
        {"name": "policyengine-us", "specifier": "==1.722.4"},
    ]


def test_upload_backfilled_release_manifest_uploads_manifest_and_trace(monkeypatch):
    api = MagicMock()
    api.create_commit.return_value = SimpleNamespace(oid="commit-sha")
    head_calls = []

    def fake_head(**kwargs):
        head_calls.append(kwargs)
        return "parent-sha"

    monkeypatch.setattr(
        "policyengine_us_data.storage.backfill_release_manifest.get_repo_head_revision",
        fake_head,
    )

    manifest = build_backfilled_release_manifest(
        version="1.73.0",
        artifacts=[
            HfArtifactMetadata(
                path="enhanced_cps_2024.h5",
                sha256="a" * 64,
                size_bytes=1,
            )
        ],
        model_package_version="1.653.3",
        core_package_version="3.26.0",
    )

    commit_sha = upload_backfilled_release_manifest(
        manifest,
        version="1.73.0",
        revision="backfill-branch",
        create_pr=True,
        token="token",
        api=api,
    )

    assert commit_sha == "commit-sha"
    assert head_calls == [
        {
            "api": api,
            "repo_id": "policyengine/policyengine-us-data",
            "repo_type": "model",
            "revision": "backfill-branch",
            "token": "token",
        }
    ]
    call = api.create_commit.call_args.kwargs
    assert call["revision"] == "backfill-branch"
    assert call["create_pr"] is True
    assert call["parent_commit"] == "parent-sha"
    operation_paths = [operation.path_in_repo for operation in call["operations"]]
    assert operation_paths == [
        "releases/1.73.0/release_manifest.json",
        "releases/1.73.0/trace.tro.jsonld",
    ]
