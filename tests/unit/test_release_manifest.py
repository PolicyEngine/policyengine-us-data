import hashlib
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub import CommitOperationAdd
import pytest

from policyengine_us_data.utils.data_upload import (
    load_release_manifest_from_hf,
    missing_release_prefixes,
    publish_release_manifest_to_hf,
    should_finalize_local_area_release,
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


EXPECTED_MODEL_PACKAGE_VERSION = "9.8.6"
EXPECTED_COMPATIBLE_MODEL_PACKAGES = [
    {"name": "policyengine-us", "specifier": f"=={EXPECTED_MODEL_PACKAGE_VERSION}"}
]
EXPECTED_CORE_PACKAGE_VERSION = "9.8.7"
EXPECTED_CORE_PACKAGE = {
    "name": "policyengine-core",
    "version": EXPECTED_CORE_PACKAGE_VERSION,
}
EXPECTED_COMPATIBLE_CORE_PACKAGES = [
    {"name": "policyengine-core", "specifier": f"=={EXPECTED_CORE_PACKAGE_VERSION}"}
]


def _build_local_area_manifest(
    *,
    states: int = 0,
    districts: int = 0,
    cities: int = 0,
) -> dict:
    artifacts = {}
    for index in range(states):
        artifacts[f"states/S{index:02d}"] = {"path": f"states/S{index:02d}.h5"}
    for index in range(districts):
        artifacts[f"districts/D{index:03d}"] = {"path": f"districts/D{index:03d}.h5"}
    for index in range(cities):
        artifacts[f"cities/C{index:03d}"] = {"path": f"cities/C{index:03d}.h5"}
    return {"artifacts": artifacts}


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
        model_package_version=EXPECTED_MODEL_PACKAGE_VERSION,
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        core_package_metadata=EXPECTED_CORE_PACKAGE,
        pipeline_run_id="run-123",
        data_package_git_sha="cafebabe",
        created_at="2026-04-10T12:00:00Z",
    )

    assert manifest["data_package"] == {
        "name": "policyengine-us-data",
        "version": "1.73.0",
    }
    assert manifest["schema_version"] == RELEASE_MANIFEST_SCHEMA_VERSION
    assert manifest["compatible_model_packages"] == EXPECTED_COMPATIBLE_MODEL_PACKAGES
    assert manifest["compatible_core_packages"] == EXPECTED_COMPATIBLE_CORE_PACKAGES
    assert manifest["build"] == {
        "build_id": "policyengine-us-data-1.73.0",
        "built_at": "2026-04-10T12:00:00Z",
        "metadata": {
            "pipeline_run_id": "run-123",
            "data_package_git_sha": "cafebabe",
        },
        "built_with_model_package": {
            "name": "policyengine-us",
            "version": EXPECTED_MODEL_PACKAGE_VERSION,
            "git_sha": "deadbeef",
            "data_build_fingerprint": "sha256:fingerprint",
            "core": EXPECTED_CORE_PACKAGE,
        },
        "built_with_core_package": EXPECTED_CORE_PACKAGE,
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
        "compatible_model_packages": EXPECTED_COMPATIBLE_MODEL_PACKAGES,
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
        model_package_version=EXPECTED_MODEL_PACKAGE_VERSION,
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        existing_manifest=existing_manifest,
        created_at="2026-04-10T12:00:00Z",
    )

    assert set(manifest["artifacts"]) == {"enhanced_cps_2024", "districts/NC-01"}
    assert manifest["default_datasets"] == {"national": "enhanced_cps_2024"}
    assert "created_at" not in manifest
    assert manifest["build"] == {
        "build_id": "policyengine-us-data-1.73.0",
        "built_at": "2026-04-10T12:00:00Z",
        "built_with_model_package": {
            "name": "policyengine-us",
            "version": EXPECTED_MODEL_PACKAGE_VERSION,
            "git_sha": "deadbeef",
            "data_build_fingerprint": "sha256:fingerprint",
        },
    }
    assert manifest["artifacts"]["districts/NC-01"]["sha256"] == _sha256(district_bytes)


def test_build_release_manifest_records_run_context(tmp_path):
    dataset_path = _write_file(
        tmp_path / "enhanced_cps_2024.h5",
        b"national-dataset",
    )

    manifest = build_release_manifest(
        files_with_repo_paths=[(dataset_path, "enhanced_cps_2024.h5")],
        version="1.73.0",
        repo_id="policyengine/policyengine-us-data",
        run_context={
            "run_id": "usdata-gha123-a1-abcdef12",
            "modal_app_name": "policyengine-us-data-pub-usdata-gha123-a1-abcdef12",
            "hf_staging_prefix": "staging/1.73.0/usdata-gha123-a1-abcdef12",
        },
        created_at="2026-04-10T12:00:00Z",
    )

    assert manifest["build"]["metadata"]["run_context"] == {
        "run_id": "usdata-gha123-a1-abcdef12",
        "modal_app_name": "policyengine-us-data-pub-usdata-gha123-a1-abcdef12",
        "hf_staging_prefix": "staging/1.73.0/usdata-gha123-a1-abcdef12",
    }


def test_build_release_manifest_validates_against_bundle_contract(tmp_path):
    policyengine_bundles = pytest.importorskip("policyengine_bundles")
    dataset_path = _write_file(
        tmp_path / "enhanced_cps_2024.h5",
        b"national-dataset",
    )
    zenodo_mirror = {
        "kind": "zenodo",
        "url": "https://zenodo.org/records/10000000/files/enhanced_cps_2024.h5",
        "doi": "10.5281/zenodo.10000000",
        "sha256": _sha256(b"national-dataset"),
        "deposited_at": "2026-04-21T12:00:00Z",
    }

    manifest = build_release_manifest(
        files_with_repo_paths=[(dataset_path, "enhanced_cps_2024.h5")],
        version="1.73.0",
        repo_id="policyengine/policyengine-us-data",
        run_context={
            "run_id": "usdata-gha123-a1-abcdef12",
            "modal_app_name": "policyengine-us-data-pub-usdata-gha123-a1-abcdef12",
            "hf_staging_prefix": "staging/1.73.0/usdata-gha123-a1-abcdef12",
        },
        model_package_version=EXPECTED_MODEL_PACKAGE_VERSION,
        model_package_git_sha="deadbeef",
        model_package_data_build_fingerprint="sha256:fingerprint",
        core_package_metadata=EXPECTED_CORE_PACKAGE,
        pipeline_run_id="run-123",
        data_package_git_sha="cafebabe",
        created_at="2026-04-10T12:00:00Z",
        preservation_mirrors_by_artifact={"enhanced_cps_2024": [zenodo_mirror]},
        preservation_dois=["10.5281/zenodo.10000000"],
    )

    policyengine_bundles.DataReleaseManifest.model_validate(manifest)


def test_build_release_manifest_omits_preservation_fields_when_not_provided(
    tmp_path,
):
    national_path = _write_file(tmp_path / "enhanced_cps_2024.h5", b"x")

    manifest = build_release_manifest(
        files_with_repo_paths=[(national_path, "enhanced_cps_2024.h5")],
        version="1.85.2",
        repo_id="policyengine/policyengine-us-data",
        created_at="2026-04-21T12:00:00Z",
    )

    assert "preservation_dois" not in manifest
    assert "preservation_mirrors" not in manifest["artifacts"]["enhanced_cps_2024"]


def test_build_release_manifest_records_preservation_mirrors_per_artifact(
    tmp_path,
):
    national_path = _write_file(tmp_path / "enhanced_cps_2024.h5", b"x")
    state_path = _write_file(tmp_path / "AL.h5", b"y")
    zenodo_mirror = {
        "kind": "zenodo",
        "url": "https://zenodo.org/records/10000000/files/enhanced_cps_2024.h5",
        "doi": "10.5281/zenodo.10000000",
        "sha256": _sha256(b"x"),
        "deposited_at": "2026-04-21T12:00:00Z",
    }

    manifest = build_release_manifest(
        files_with_repo_paths=[
            (national_path, "enhanced_cps_2024.h5"),
            (state_path, "states/AL.h5"),
        ],
        version="1.85.2",
        repo_id="policyengine/policyengine-us-data",
        created_at="2026-04-21T12:00:00Z",
        preservation_mirrors_by_artifact={"enhanced_cps_2024": [zenodo_mirror]},
        preservation_dois=["10.5281/zenodo.10000000"],
    )

    assert manifest["preservation_dois"] == ["10.5281/zenodo.10000000"]
    assert manifest["artifacts"]["enhanced_cps_2024"]["preservation_mirrors"] == [
        zenodo_mirror
    ]
    assert "preservation_mirrors" not in manifest["artifacts"]["states/AL"]


def test_build_release_manifest_skips_empty_mirror_lists(tmp_path):
    national_path = _write_file(tmp_path / "enhanced_cps_2024.h5", b"x")

    manifest = build_release_manifest(
        files_with_repo_paths=[(national_path, "enhanced_cps_2024.h5")],
        version="1.85.2",
        repo_id="policyengine/policyengine-us-data",
        created_at="2026-04-21T12:00:00Z",
        preservation_mirrors_by_artifact={"enhanced_cps_2024": []},
    )

    assert "preservation_mirrors" not in manifest["artifacts"]["enhanced_cps_2024"]


def test_build_release_manifest_drops_stale_preservation_dois(tmp_path):
    national_path = _write_file(tmp_path / "enhanced_cps_2024.h5", b"x")
    existing_manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": "policyengine-us-data",
            "version": "1.85.2",
        },
        "artifacts": {
            "enhanced_cps_2024": {
                "kind": "microdata",
                "path": "enhanced_cps_2024.h5",
                "repo_id": "policyengine/policyengine-us-data",
                "revision": "1.85.2",
                "sha256": "old",
                "size_bytes": 1,
            }
        },
        "preservation_dois": ["10.5281/zenodo.old"],
    }

    manifest = build_release_manifest(
        files_with_repo_paths=[(national_path, "enhanced_cps_2024.h5")],
        version="1.85.2",
        repo_id="policyengine/policyengine-us-data",
        existing_manifest=existing_manifest,
        created_at="2026-04-21T12:00:00Z",
    )

    assert "preservation_dois" not in manifest


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


def test_load_release_manifest_from_hf_raises_non_missing_download_errors():
    with patch(
        "policyengine_us_data.utils.data_upload.hf_hub_download",
        side_effect=RuntimeError("temporary Hugging Face failure"),
    ):
        with pytest.raises(RuntimeError, match="temporary Hugging Face failure"):
            load_release_manifest_from_hf(version="1.73.0", revision="1.73.0")


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
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_data_package_git_sha",
            return_value=None,
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
            "policyengine_us_data.utils.data_upload._get_model_package_build_metadata",
            return_value={
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_data_package_git_sha",
            return_value=None,
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
        "compatible_model_packages": EXPECTED_COMPATIBLE_MODEL_PACKAGES,
        "default_datasets": {"national": "enhanced_cps_2024"},
        "created_at": "2026-04-10T12:00:00Z",
        "build": {
            "build_id": "policyengine-us-data-1.73.0",
            "built_at": "2026-04-10T12:00:00Z",
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
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
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_data_package_git_sha",
            return_value=None,
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
            "version": EXPECTED_MODEL_PACKAGE_VERSION,
            "git_sha": "deadbeef",
            "data_build_fingerprint": "sha256:fingerprint",
        },
    }


def test_publish_release_manifest_records_bundle_build_metadata(tmp_path):
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
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
                "core": EXPECTED_CORE_PACKAGE,
            },
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_data_package_git_sha",
            return_value="cafebabe",
        ),
        patch.dict(
            "policyengine_us_data.utils.data_upload.os.environ",
            {"HUGGING_FACE_TOKEN": "token"},
            clear=False,
        ),
    ):
        manifest = publish_release_manifest_to_hf(
            [(dataset_path, "enhanced_cps_2024.h5")],
            version="1.73.0",
            pipeline_run_id="run-123",
        )

    assert manifest["compatible_core_packages"] == EXPECTED_COMPATIBLE_CORE_PACKAGES
    assert manifest["build"]["metadata"] == {
        "pipeline_run_id": "run-123",
        "data_package_git_sha": "cafebabe",
    }
    assert manifest["build"]["built_with_core_package"] == EXPECTED_CORE_PACKAGE
    assert (
        manifest["build"]["built_with_model_package"]["core"] == EXPECTED_CORE_PACKAGE
    )


def test_missing_release_prefixes_requires_full_local_area_bundle():
    existing_manifest = _build_local_area_manifest(states=1, districts=1)

    missing = missing_release_prefixes(
        existing_manifest=existing_manifest,
        new_repo_paths=["national/US.h5"],
    )

    assert missing == ["states/", "districts/", "cities/"]


def test_should_finalize_local_area_release_uses_combined_manifest_state():
    existing_manifest = _build_local_area_manifest(
        states=51,
        districts=435,
        cities=1,
    )

    with patch(
        "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
        return_value=existing_manifest,
    ):
        should_finalize, missing = should_finalize_local_area_release(
            version="1.73.0",
            new_repo_paths=["national/US.h5"],
        )

    assert should_finalize is True
    assert missing == []


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
                    raise AssertionError(
                        "Expected RuntimeError when model version is unavailable"
                    )


def test_publish_release_manifest_to_hf_rejects_finalized_release(tmp_path):
    state_path = _write_file(
        tmp_path / "AL.h5",
        b"state-dataset",
    )
    finalized_manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": "policyengine-us-data",
            "version": "1.73.0",
        },
        "compatible_model_packages": EXPECTED_COMPATIBLE_MODEL_PACKAGES,
        "default_datasets": {"national": "enhanced_cps_2024"},
        "created_at": "2026-04-10T12:00:00Z",
        "build": {
            "build_id": "policyengine-us-data-1.73.0",
            "built_at": "2026-04-10T12:00:00Z",
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        },
        "artifacts": {
            "states/AL": {
                "kind": "microdata",
                "path": "states/AL.h5",
                "repo_id": "policyengine/policyengine-us-data",
                "revision": "1.73.0",
                "sha256": _sha256(b"state-dataset"),
                "size_bytes": len(b"state-dataset"),
            }
        },
    }

    with (
        patch(
            "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
            side_effect=lambda *args, **kwargs: (
                finalized_manifest if kwargs.get("revision") == "1.73.0" else None
            ),
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_model_package_version",
            return_value=EXPECTED_MODEL_PACKAGE_VERSION,
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_model_package_build_metadata",
            return_value={
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        ),
    ):
        manifest = publish_release_manifest_to_hf(
            [(state_path, "states/AL.h5")],
            version="1.73.0",
            create_tag=True,
        )

    assert manifest == finalized_manifest


def test_publish_release_manifest_to_hf_rejects_mutating_finalized_release(tmp_path):
    state_path = _write_file(
        tmp_path / "AL.h5",
        b"state-dataset-v2",
    )
    finalized_manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "data_package": {
            "name": "policyengine-us-data",
            "version": "1.73.0",
        },
        "compatible_model_packages": EXPECTED_COMPATIBLE_MODEL_PACKAGES,
        "default_datasets": {"national": "enhanced_cps_2024"},
        "created_at": "2026-04-10T12:00:00Z",
        "build": {
            "build_id": "policyengine-us-data-1.73.0",
            "built_at": "2026-04-10T12:00:00Z",
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        },
        "artifacts": {
            "states/AL": {
                "kind": "microdata",
                "path": "states/AL.h5",
                "repo_id": "policyengine/policyengine-us-data",
                "revision": "1.73.0",
                "sha256": _sha256(b"state-dataset"),
                "size_bytes": len(b"state-dataset"),
            }
        },
    }

    with (
        patch(
            "policyengine_us_data.utils.data_upload.load_release_manifest_from_hf",
            side_effect=lambda *args, **kwargs: (
                finalized_manifest if kwargs.get("revision") == "1.73.0" else None
            ),
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_model_package_version",
            return_value=EXPECTED_MODEL_PACKAGE_VERSION,
        ),
        patch(
            "policyengine_us_data.utils.data_upload._get_model_package_build_metadata",
            return_value={
                "version": EXPECTED_MODEL_PACKAGE_VERSION,
                "git_sha": "deadbeef",
                "data_build_fingerprint": "sha256:fingerprint",
            },
        ),
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
