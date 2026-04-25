import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub import CommitOperationAdd

from policyengine_us_data.utils.data_upload import upload_files_to_hf
from policyengine_us_data.utils.data_upload import publish_release_manifest_to_hf
from policyengine_us_data.utils.release_manifest import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    build_release_manifest,
)
from policyengine_us_data.utils.trace_tro import TRACE_TRO_FILENAME


def _write_file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_build_release_manifest_tracks_uploaded_artifacts(tmp_path):
    national_bytes = b"national-dataset"
    state_bytes = b"state-dataset"
    national_path = _write_file(
        tmp_path / "enhanced_cps_2024.h5",
        national_bytes,
    )
    state_path = _write_file(tmp_path / "AL.h5", state_bytes)

    manifest = build_release_manifest(
        files_with_repo_paths=[
            (national_path, "enhanced_cps_2024.h5"),
            (state_path, "states/AL.h5"),
        ],
        version="1.73.0",
        repo_id="policyengine/policyengine-us-data",
        model_package_version="1.634.4",
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        created_at="2026-04-10T12:00:00Z",
    )

    assert manifest["data_package"] == {
        "name": "policyengine-us-data",
        "version": "1.73.0",
    }
    assert manifest["schema_version"] == RELEASE_MANIFEST_SCHEMA_VERSION
    assert manifest["compatible_model_packages"] == [
        {
            "name": "policyengine-us",
            "specifier": "==1.634.4",
        }
    ]
    assert manifest["build"] == {
        "build_id": "policyengine-us-data-1.73.0",
        "built_at": "2026-04-10T12:00:00Z",
        "built_with_model_package": {
            "name": "policyengine-us",
            "version": "1.634.4",
            "git_sha": "deadbeef",
            "data_build_fingerprint": "sha256:fingerprint",
        },
    }
    assert manifest["default_datasets"] == {"national": "enhanced_cps_2024"}

    assert manifest["artifacts"]["enhanced_cps_2024"] == {
        "kind": "microdata",
        "path": "enhanced_cps_2024.h5",
        "repo_id": "policyengine/policyengine-us-data",
        "revision": "1.73.0",
        "sha256": _sha256(national_bytes),
        "size_bytes": len(national_bytes),
    }
    assert manifest["artifacts"]["states/AL"] == {
        "kind": "microdata",
        "path": "states/AL.h5",
        "repo_id": "policyengine/policyengine-us-data",
        "revision": "1.73.0",
        "sha256": _sha256(state_bytes),
        "size_bytes": len(state_bytes),
    }


def test_build_release_manifest_adds_additional_compatible_specifiers(tmp_path):
    national_path = _write_file(
        tmp_path / "enhanced_cps_2024.h5",
        b"national-dataset",
    )

    manifest = build_release_manifest(
        files_with_repo_paths=[(national_path, "enhanced_cps_2024.h5")],
        version="1.83.3",
        repo_id="policyengine/policyengine-us-data",
        model_package_version="1.637.0",
        model_package_data_build_fingerprint="sha256:stable",
        additional_compatible_specifiers=(">=1.637.0,<2.0.0",),
        created_at="2026-04-18T12:00:00Z",
    )

    assert manifest["compatible_model_packages"] == [
        {"name": "policyengine-us", "specifier": "==1.637.0"},
        {"name": "policyengine-us", "specifier": ">=1.637.0,<2.0.0"},
    ]


def test_build_release_manifest_merges_existing_release_same_version(tmp_path):
    district_bytes = b"district-dataset"
    district_path = _write_file(tmp_path / "NC-01.h5", district_bytes)

    existing_manifest = {
        "data_package": {
            "name": "policyengine-us-data",
            "version": "1.73.0",
        },
        "compatible_model_packages": [
            {
                "name": "policyengine-us",
                "specifier": "==1.634.4",
            }
        ],
        "default_datasets": {"national": "enhanced_cps_2024"},
        "created_at": "2026-04-09T12:00:00Z",
        "artifacts": {
            "enhanced_cps_2024": {
                "kind": "microdata",
                "path": "enhanced_cps_2024.h5",
                "repo_id": "policyengine/policyengine-us-data",
                "revision": "1.73.0",
                "sha256": "abc",
                "size_bytes": 123,
            }
        },
    }

    manifest = build_release_manifest(
        files_with_repo_paths=[(district_path, "districts/NC-01.h5")],
        version="1.73.0",
        repo_id="policyengine/policyengine-us-data",
        model_package_version="1.634.4",
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        existing_manifest=existing_manifest,
        created_at="2026-04-10T12:00:00Z",
    )

    assert set(manifest["artifacts"]) == {"enhanced_cps_2024", "districts/NC-01"}
    assert manifest["default_datasets"] == {"national": "enhanced_cps_2024"}
    assert manifest["build"] == {
        "build_id": "policyengine-us-data-1.73.0",
        "built_at": "2026-04-10T12:00:00Z",
        "built_with_model_package": {
            "name": "policyengine-us",
            "version": "1.634.4",
            "git_sha": "deadbeef",
            "data_build_fingerprint": "sha256:fingerprint",
        },
    }
    assert manifest["artifacts"]["districts/NC-01"]["sha256"] == _sha256(district_bytes)


def test_upload_files_to_hf_adds_release_manifest_operations(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_cps_2024.h5",
        b"national-dataset",
    )

    mock_api = MagicMock()
    mock_api.create_commit.return_value = MagicMock(oid="commit-sha")

    with (
        patch("policyengine_us_data.utils.data_upload.HfApi", return_value=mock_api),
        patch(
            "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
            return_value=None,
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_model_package_build_metadata",
            return_value={
                "version": "1.634.4",
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        ),
        patch.dict(
            "policyengine_us_data.utils.data_upload.os.environ",
            {"HUGGING_FACE_TOKEN": "token"},
            clear=False,
        ),
    ):
        upload_files_to_hf(
            files=[dataset_path],
            version="1.73.0",
        )

    operations = mock_api.create_commit.call_args.kwargs["operations"]
    operation_paths = [operation.path_in_repo for operation in operations]

    assert "enhanced_cps_2024.h5" in operation_paths
    assert "release_manifest.json" in operation_paths
    assert "releases/1.73.0/release_manifest.json" in operation_paths
    assert TRACE_TRO_FILENAME in operation_paths
    assert f"releases/1.73.0/{TRACE_TRO_FILENAME}" in operation_paths

    release_ops = [
        operation
        for operation in operations
        if operation.path_in_repo.endswith("release_manifest.json")
    ]
    assert len(release_ops) == 2
    for operation in release_ops:
        assert isinstance(operation, CommitOperationAdd)
        assert isinstance(operation.path_or_fileobj, BytesIO)

    trace_ops = [
        operation
        for operation in operations
        if operation.path_in_repo.endswith(".jsonld")
    ]
    assert len(trace_ops) == 2
    for operation in trace_ops:
        assert isinstance(operation, CommitOperationAdd)
        assert isinstance(operation.path_or_fileobj, BytesIO)


def test_upload_files_to_hf_does_not_tag_until_finalize(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_cps_2024.h5",
        b"national-dataset",
    )

    mock_api = MagicMock()
    mock_api.create_commit.return_value = MagicMock(oid="commit-sha")

    with (
        patch("policyengine_us_data.utils.data_upload.HfApi", return_value=mock_api),
        patch(
            "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
            return_value=None,
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_model_package_build_metadata",
            return_value={
                "version": "1.634.4",
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        ),
        patch.dict(
            "policyengine_us_data.utils.data_upload.os.environ",
            {"HUGGING_FACE_TOKEN": "token"},
            clear=False,
        ),
    ):
        upload_files_to_hf(
            files=[dataset_path],
            version="1.73.0",
            create_tag=False,
        )

    mock_api.create_tag.assert_not_called()


def test_publish_release_manifest_to_hf_can_finalize_and_tag(tmp_path):
    state_path = _write_file(
        tmp_path / "AL.h5",
        b"state-dataset",
    )

    mock_api = MagicMock()
    mock_api.create_commit.return_value = MagicMock(oid="final-commit-sha")
    existing_manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": "policyengine-us-data",
            "version": "1.73.0",
        },
        "compatible_model_packages": [],
        "default_datasets": {"national": "enhanced_cps_2024"},
        "created_at": "2026-04-10T12:00:00Z",
        "build": {
            "build_id": "policyengine-us-data-1.73.0",
            "built_at": "2026-04-10T12:00:00Z",
        },
        "artifacts": {
            "enhanced_cps_2024": {
                "kind": "microdata",
                "path": "enhanced_cps_2024.h5",
                "repo_id": "policyengine/policyengine-us-data",
                "revision": "1.73.0",
                "sha256": "abc",
                "size_bytes": 123,
            }
        },
    }

    with (
        patch("policyengine_us_data.utils.data_upload.HfApi", return_value=mock_api),
        patch(
            "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
            side_effect=lambda *args, **kwargs: (
                None if kwargs.get("revision") == "1.73.0" else existing_manifest
            ),
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_model_package_build_metadata",
            return_value={
                "version": "1.634.4",
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        ),
        patch.dict(
            "policyengine_us_data.utils.data_upload.os.environ",
            {"HUGGING_FACE_TOKEN": "token"},
            clear=False,
        ),
    ):
        manifest = publish_release_manifest_to_hf(
            [(state_path, "states/AL.h5")],
            version="1.73.0",
            create_tag=True,
        )

    mock_api.create_tag.assert_called_once()
    assert manifest["build"] == {
        "build_id": "policyengine-us-data-1.73.0",
        "built_at": "2026-04-10T12:00:00Z",
        "built_with_model_package": {
            "name": "policyengine-us",
            "version": "1.634.4",
            "git_sha": "deadbeef",
            "data_build_fingerprint": "sha256:fingerprint",
        },
    }
