import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub import CommitOperationAdd

from policyengine_us_data.utils.data_upload import (
    load_release_manifest_from_hf,
    publish_release_manifest_to_hf,
    upload_files_to_hf,
)
from policyengine_us_data.utils.release_manifest import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    build_release_manifest,
)


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
        existing_manifest=existing_manifest,
        created_at="2026-04-10T12:00:00Z",
    )

    assert set(manifest["artifacts"]) == {"enhanced_cps_2024", "districts/NC-01"}
    assert manifest["default_datasets"] == {"national": "enhanced_cps_2024"}
    assert manifest["created_at"] == "2026-04-09T12:00:00Z"
    assert manifest["artifacts"]["districts/NC-01"]["sha256"] == _sha256(
        district_bytes
    )


def test_load_release_manifest_from_hf_uses_explicit_revision_when_requested(tmp_path):
    manifest_path = _write_file(
        tmp_path / "release_manifest.json",
        b'{"data_package": {"name": "policyengine-us-data", "version": "1.73.0"}}',
    )

    with patch(
        "policyengine_us_data.utils.data_upload.hf_hub_download",
        return_value=str(manifest_path),
    ) as mock_download:
        manifest = load_release_manifest_from_hf(
            version="1.73.0",
            revision="1.73.0",
        )

    assert manifest["data_package"]["version"] == "1.73.0"
    assert mock_download.call_args.kwargs["revision"] == "1.73.0"


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
            "policyengine_us_data.utils.data_upload.metadata.version",
            return_value="1.634.4",
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

    release_ops = [
        operation
        for operation in operations
        if operation.path_in_repo.endswith("release_manifest.json")
    ]
    assert len(release_ops) == 2
    for operation in release_ops:
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
            "policyengine_us_data.utils.data_upload.metadata.version",
            return_value="1.634.4",
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

    with (
        patch("policyengine_us_data.utils.data_upload.HfApi", return_value=mock_api),
        patch(
            "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
            return_value={
                "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
                "data_package": {
                    "name": "policyengine-us-data",
                    "version": "1.73.0",
                },
                "compatible_model_packages": [],
                "default_datasets": {"national": "enhanced_cps_2024"},
                "created_at": "2026-04-10T12:00:00Z",
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
            },
        ),
        patch(
            "policyengine_us_data.utils.data_upload.metadata.version",
            return_value="1.634.4",
        ),
        patch.dict(
            "policyengine_us_data.utils.data_upload.os.environ",
            {"HUGGING_FACE_TOKEN": "token"},
            clear=False,
        ),
    ):
        publish_release_manifest_to_hf(
            [(state_path, "states/AL.h5")],
            version="1.73.0",
            create_tag=True,
        )

    mock_api.create_tag.assert_called_once()


def test_upload_files_to_hf_fails_without_model_package_version(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_cps_2024.h5",
        b"national-dataset",
    )

    with (
        patch("policyengine_us_data.utils.data_upload.HfApi", return_value=MagicMock()),
        patch(
            "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
            return_value=None,
        ),
        patch(
            "policyengine_us_data.utils.data_upload.metadata.version",
            side_effect=RuntimeError("missing package"),
        ),
    ):
        with patch(
            "policyengine_us_data.utils.data_upload._get_model_package_version",
            side_effect=RuntimeError("missing package"),
        ):
            with patch.dict(
                "policyengine_us_data.utils.data_upload.os.environ",
                {"HUGGING_FACE_TOKEN": "token"},
                clear=False,
            ):
                try:
                    upload_files_to_hf(
                        files=[dataset_path],
                        version="1.73.0",
                    )
                except RuntimeError as exc:
                    assert "missing package" in str(exc)
                else:
                    raise AssertionError("Expected RuntimeError when model version is unavailable")


def test_publish_release_manifest_to_hf_rejects_finalized_release(tmp_path):
    state_path = _write_file(
        tmp_path / "AL.h5",
        b"state-dataset",
    )

    with patch(
        "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
        side_effect=[
            {
                "data_package": {
                    "name": "policyengine-us-data",
                    "version": "1.73.0",
                }
            }
        ],
    ):
        try:
            publish_release_manifest_to_hf(
                [(state_path, "states/AL.h5")],
                version="1.73.0",
                create_tag=True,
            )
        except RuntimeError as exc:
            assert "already finalized" in str(exc)
        else:
            raise AssertionError("Expected finalized release guard to raise")
