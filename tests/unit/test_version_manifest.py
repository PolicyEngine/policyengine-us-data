"""Tests for version manifest registry system."""

import json
from unittest.mock import MagicMock, patch, call

import pytest
from google.api_core.exceptions import NotFound

from policyengine_us_data.utils.version_manifest import (
    GCSVersionInfo,
    VersionManifest,
    VersionRegistry,
    build_manifest,
    upload_manifest,
    get_current_version,
    get_manifest,
    list_versions,
    download_versioned_file,
    rollback,
    get_data_manifest,
    get_data_version,
)
from tests.conftest import (
    make_mock_blob,
    setup_bucket_with_registry,
)

_MOD = "policyengine_us_data.utils.version_manifest"


# -- VersionManifest serialization tests ---------------------------


class TestVersionManifestSerialization:
    def test_to_dict(self, sample_manifest):
        result = sample_manifest.to_dict()

        assert result["version"] == "1.72.3"
        assert result["created_at"] == "2026-03-10T14:30:00Z"
        assert result["hf"]["repo"] == ("policyengine/policyengine-us-data")
        assert result["hf"]["commit"] == "abc123def456"
        assert result["gcs"]["bucket"] == ("policyengine-us-data")
        assert result["gcs"]["generations"]["enhanced_cps_2024.h5"] == 1710203948123456

    def test_from_dict(self, sample_manifest):
        data = {
            "version": "1.72.3",
            "created_at": "2026-03-10T14:30:00Z",
            "hf": {
                "repo": ("policyengine/policyengine-us-data"),
                "commit": "abc123def456",
            },
            "gcs": {
                "bucket": "policyengine-us-data",
                "generations": {
                    "enhanced_cps_2024.h5": (1710203948123456),
                    "cps_2024.h5": 1710203948234567,
                    "states/AL.h5": 1710203948345678,
                },
            },
        }
        result = VersionManifest.from_dict(data)

        assert result.version == "1.72.3"
        assert result.hf.commit == "abc123def456"
        assert result.hf.repo == ("policyengine/policyengine-us-data")
        assert result.gcs.generations["enhanced_cps_2024.h5"] == 1710203948123456
        assert result.gcs.bucket == "policyengine-us-data"

    def test_roundtrip(self, sample_manifest):
        roundtripped = VersionManifest.from_dict(sample_manifest.to_dict())

        assert roundtripped.version == (sample_manifest.version)
        assert roundtripped.created_at == (sample_manifest.created_at)
        assert roundtripped.hf.repo == (sample_manifest.hf.repo)
        assert roundtripped.hf.commit == (sample_manifest.hf.commit)
        assert roundtripped.gcs.bucket == (sample_manifest.gcs.bucket)
        assert roundtripped.gcs.generations == (sample_manifest.gcs.generations)

    def test_without_hf(self, sample_generations):
        manifest = VersionManifest(
            version="1.72.3",
            created_at="2026-03-10T14:30:00Z",
            hf=None,
            gcs=GCSVersionInfo(
                bucket="policyengine-us-data",
                generations=sample_generations,
            ),
        )
        data = manifest.to_dict()
        assert data["hf"] is None

        roundtripped = VersionManifest.from_dict(data)
        assert roundtripped.hf is None
        assert roundtripped.gcs.generations == (sample_generations)

    def test_special_operation_omitted_by_default(self, sample_manifest):
        data = sample_manifest.to_dict()
        assert "special_operation" not in data
        assert "roll_back_version" not in data

    def test_special_operation_included_when_set(
        self, sample_generations, sample_hf_info
    ):
        manifest = VersionManifest(
            version="1.73.0",
            created_at="2026-03-10T15:00:00Z",
            hf=sample_hf_info,
            gcs=GCSVersionInfo(
                bucket="policyengine-us-data",
                generations=sample_generations,
            ),
            special_operation="roll-back",
            roll_back_version="1.70.1",
        )
        data = manifest.to_dict()
        assert data["special_operation"] == "roll-back"
        assert data["roll_back_version"] == "1.70.1"

    def test_special_operation_roundtrip(self, sample_generations, sample_hf_info):
        manifest = VersionManifest(
            version="1.73.0",
            created_at="2026-03-10T15:00:00Z",
            hf=sample_hf_info,
            gcs=GCSVersionInfo(
                bucket="policyengine-us-data",
                generations=sample_generations,
            ),
            special_operation="roll-back",
            roll_back_version="1.70.1",
        )
        roundtripped = VersionManifest.from_dict(manifest.to_dict())
        assert roundtripped.special_operation == ("roll-back")
        assert roundtripped.roll_back_version == "1.70.1"

    def test_regular_manifest_has_no_special_operation(
        self,
    ):
        data = {
            "version": "1.72.3",
            "created_at": "2026-03-10T14:30:00Z",
            "hf": None,
            "gcs": {
                "bucket": "b",
                "generations": {"f.h5": 123},
            },
        }
        result = VersionManifest.from_dict(data)
        assert result.special_operation is None
        assert result.roll_back_version is None

    def test_pipeline_run_id_omitted_by_default(self, sample_manifest):
        data = sample_manifest.to_dict()
        assert "pipeline_run_id" not in data
        assert "diagnostics_path" not in data

    def test_pipeline_run_id_included_when_set(
        self, sample_generations, sample_hf_info
    ):
        manifest = VersionManifest(
            version="1.73.0",
            created_at="2026-03-10T15:00:00Z",
            hf=sample_hf_info,
            gcs=GCSVersionInfo(
                bucket="policyengine-us-data",
                generations=sample_generations,
            ),
            pipeline_run_id="1.73.0_abc12345_20260310",
            diagnostics_path=("calibration/runs/1.73.0_abc12345_20260310/diagnostics/"),
        )
        data = manifest.to_dict()
        assert data["pipeline_run_id"] == ("1.73.0_abc12345_20260310")
        assert "diagnostics/" in data["diagnostics_path"]

    def test_pipeline_run_id_roundtrip(self, sample_generations, sample_hf_info):
        manifest = VersionManifest(
            version="1.73.0",
            created_at="2026-03-10T15:00:00Z",
            hf=sample_hf_info,
            gcs=GCSVersionInfo(
                bucket="policyengine-us-data",
                generations=sample_generations,
            ),
            pipeline_run_id="1.73.0_abc12345_20260310",
            diagnostics_path="calibration/runs/x/diag/",
        )
        roundtripped = VersionManifest.from_dict(manifest.to_dict())
        assert roundtripped.pipeline_run_id == ("1.73.0_abc12345_20260310")
        assert roundtripped.diagnostics_path == ("calibration/runs/x/diag/")


# -- VersionRegistry serialization tests ---------------------------


class TestVersionRegistrySerialization:
    def test_to_dict(self, sample_registry):
        result = sample_registry.to_dict()

        assert result["current"] == "1.72.3"
        assert len(result["versions"]) == 1
        assert result["versions"][0]["version"] == "1.72.3"

    def test_from_dict(self, sample_manifest):
        data = {
            "current": "1.72.3",
            "versions": [sample_manifest.to_dict()],
        }
        result = VersionRegistry.from_dict(data)

        assert result.current == "1.72.3"
        assert len(result.versions) == 1
        assert result.versions[0].version == "1.72.3"
        assert result.versions[0].hf.commit == ("abc123def456")

    def test_roundtrip(self, sample_registry):
        roundtripped = VersionRegistry.from_dict(sample_registry.to_dict())
        assert roundtripped.current == (sample_registry.current)
        assert len(roundtripped.versions) == len(sample_registry.versions)
        assert roundtripped.versions[0].version == "1.72.3"

    def test_get_version(self, sample_registry):
        result = sample_registry.get_version("1.72.3")
        assert result.version == "1.72.3"
        assert result.hf.commit == "abc123def456"

    def test_get_version_not_found(self, sample_registry):
        with pytest.raises(ValueError, match="not found"):
            sample_registry.get_version("9.9.9")

    def test_empty_registry(self):
        registry = VersionRegistry()
        assert registry.current == ""
        assert registry.versions == []

        data = registry.to_dict()
        assert data == {"current": "", "versions": []}


# -- build_manifest tests ------------------------------------------


class TestBuildManifest:
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_structure(self, mock_get_bucket, mock_bucket):
        mock_get_bucket.return_value = mock_bucket
        blob_names = [
            "file_a.h5",
            "file_b.h5",
            "file_c.h5",
        ]
        mock_bucket.get_blob.side_effect = [
            make_mock_blob(100),
            make_mock_blob(200),
            make_mock_blob(300),
        ]

        result = build_manifest("1.72.3", blob_names)

        assert isinstance(result, VersionManifest)
        assert result.version == "1.72.3"
        assert result.created_at.endswith("Z")
        assert result.gcs.generations == {
            "file_a.h5": 100,
            "file_b.h5": 200,
            "file_c.h5": 300,
        }
        assert result.gcs.bucket == "policyengine-us-data"
        assert result.hf is None

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_with_subdirectories(self, mock_get_bucket, mock_bucket):
        mock_get_bucket.return_value = mock_bucket
        blob_names = [
            "states/AL.h5",
            "districts/CA-01.h5",
        ]
        mock_bucket.get_blob.side_effect = [
            make_mock_blob(111),
            make_mock_blob(222),
        ]

        result = build_manifest("1.72.3", blob_names)

        assert "states/AL.h5" in result.gcs.generations
        assert "districts/CA-01.h5" in result.gcs.generations
        assert result.gcs.generations["states/AL.h5"] == 111
        assert result.gcs.generations["districts/CA-01.h5"] == 222

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_with_hf_info(
        self,
        mock_get_bucket,
        mock_bucket,
        sample_hf_info,
    ):
        mock_get_bucket.return_value = mock_bucket
        mock_bucket.get_blob.return_value = make_mock_blob(999)

        result = build_manifest(
            "1.72.3",
            ["file.h5"],
            hf_info=sample_hf_info,
        )

        assert result.hf is not None
        assert result.hf.commit == "abc123def456"
        assert result.hf.repo == ("policyengine/policyengine-us-data")

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_missing_blob_raises(self, mock_get_bucket, mock_bucket):
        mock_get_bucket.return_value = mock_bucket
        mock_bucket.get_blob.return_value = None

        with pytest.raises(ValueError, match="not found"):
            build_manifest("1.72.3", ["missing.h5"])


# -- upload_manifest tests -----------------------------------------


class TestUploadManifest:
    def _setup_empty_registry(self, bucket):
        """Mock bucket with no existing registry."""
        written = {}

        def mock_blob(name):
            if name == "version_manifest.json":
                b = MagicMock()
                b.name = name
                b.download_as_text.side_effect = NotFound("Not found")
                written[name] = b
                return b
            b = MagicMock()
            b.name = name
            written[name] = b
            return b

        bucket.blob.side_effect = mock_blob
        return written

    @patch(f"{_MOD}._upload_registry_to_hf")
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_writes_registry_to_gcs(
        self,
        mock_get_bucket,
        mock_hf,
        mock_bucket,
        sample_manifest,
    ):
        mock_get_bucket.return_value = mock_bucket
        written = self._setup_empty_registry(mock_bucket)

        upload_manifest(sample_manifest)

        assert "version_manifest.json" in written
        blob = written["version_manifest.json"]
        written_json = blob.upload_from_string.call_args[0][0]
        registry_data = json.loads(written_json)

        assert registry_data["current"] == "1.72.3"
        assert len(registry_data["versions"]) == 1
        assert registry_data["versions"][0]["version"] == "1.72.3"

    @patch(f"{_MOD}._upload_registry_to_hf")
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_includes_hf_commit(
        self,
        mock_get_bucket,
        mock_hf,
        mock_bucket,
        sample_manifest,
    ):
        mock_get_bucket.return_value = mock_bucket
        written = self._setup_empty_registry(mock_bucket)

        upload_manifest(sample_manifest)

        blob = written["version_manifest.json"]
        written_json = blob.upload_from_string.call_args[0][0]
        registry_data = json.loads(written_json)

        assert registry_data["versions"][0]["hf"]["commit"] == "abc123def456"

    @patch(f"{_MOD}._upload_registry_to_hf")
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_appends_to_existing_registry(
        self,
        mock_get_bucket,
        mock_hf,
        mock_bucket,
        sample_manifest,
    ):
        mock_get_bucket.return_value = mock_bucket
        older = VersionManifest(
            version="1.72.2",
            created_at="2026-03-09T10:00:00Z",
            hf=None,
            gcs=GCSVersionInfo(
                bucket="policyengine-us-data",
                generations={"old.h5": 111},
            ),
        )
        existing_registry = VersionRegistry(current="1.72.2", versions=[older])
        existing_json = json.dumps(existing_registry.to_dict())
        written = {}

        def mock_blob(name):
            b = MagicMock()
            b.name = name
            b.download_as_text.return_value = existing_json
            written[name] = b
            return b

        mock_bucket.blob.side_effect = mock_blob

        upload_manifest(sample_manifest)

        blob = written["version_manifest.json"]
        written_json = blob.upload_from_string.call_args[0][0]
        registry_data = json.loads(written_json)

        assert registry_data["current"] == "1.72.3"
        assert len(registry_data["versions"]) == 2
        assert registry_data["versions"][0]["version"] == "1.72.3"
        assert registry_data["versions"][1]["version"] == "1.72.2"

    @patch(f"{_MOD}.os")
    @patch(f"{_MOD}.HfApi")
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_always_uploads_to_hf(
        self,
        mock_get_bucket,
        mock_hf_api_cls,
        mock_os,
        mock_bucket,
        sample_manifest,
    ):
        mock_get_bucket.return_value = mock_bucket
        mock_os.environ.get.return_value = "fake_token"
        mock_os.unlink = MagicMock()
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api

        blob = MagicMock()
        blob.download_as_text.side_effect = NotFound("Not found")
        mock_bucket.blob.return_value = blob

        upload_manifest(sample_manifest)

        mock_api.upload_file.assert_called_once()
        call_kwargs = mock_api.upload_file.call_args.kwargs
        assert call_kwargs["path_in_repo"] == ("version_manifest.json")
        assert call_kwargs["repo_id"] == ("policyengine/policyengine-us-data")


# -- get_current_version tests -------------------------------------


class TestGetCurrentVersion:
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_returns_version(
        self,
        mock_get_bucket,
        mock_bucket,
        sample_registry,
    ):
        mock_get_bucket.return_value = mock_bucket
        setup_bucket_with_registry(mock_bucket, sample_registry)

        result = get_current_version()

        assert result == "1.72.3"
        mock_bucket.blob.assert_called_with("version_manifest.json")

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_no_registry_returns_none(self, mock_get_bucket, mock_bucket):
        mock_get_bucket.return_value = mock_bucket
        blob = MagicMock()
        blob.download_as_text.side_effect = NotFound("Not found")
        mock_bucket.blob.return_value = blob

        result = get_current_version()

        assert result is None


# -- get_manifest tests ---------------------------------------------


class TestGetManifest:
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_specific_version(
        self,
        mock_get_bucket,
        mock_bucket,
        sample_registry,
    ):
        mock_get_bucket.return_value = mock_bucket
        setup_bucket_with_registry(mock_bucket, sample_registry)

        result = get_manifest("1.72.3")

        assert isinstance(result, VersionManifest)
        assert result.version == "1.72.3"
        assert result.hf.commit == "abc123def456"
        assert result.gcs.generations["enhanced_cps_2024.h5"] == 1710203948123456

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_nonexistent_version(
        self,
        mock_get_bucket,
        mock_bucket,
        sample_registry,
    ):
        mock_get_bucket.return_value = mock_bucket
        setup_bucket_with_registry(mock_bucket, sample_registry)

        with pytest.raises(ValueError, match="not found"):
            get_manifest("9.9.9")

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_no_registry_raises(self, mock_get_bucket, mock_bucket):
        mock_get_bucket.return_value = mock_bucket
        blob = MagicMock()
        blob.download_as_text.side_effect = NotFound("Not found")
        mock_bucket.blob.return_value = blob

        with pytest.raises(ValueError, match="not found"):
            get_manifest("1.72.3")


# -- list_versions tests -------------------------------------------


class TestListVersions:
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_returns_sorted(self, mock_get_bucket, mock_bucket):
        mock_get_bucket.return_value = mock_bucket
        v1 = VersionManifest(
            version="1.72.1",
            created_at="t1",
            hf=None,
            gcs=GCSVersionInfo(bucket="b", generations={"f.h5": 1}),
        )
        v2 = VersionManifest(
            version="1.72.3",
            created_at="t2",
            hf=None,
            gcs=GCSVersionInfo(bucket="b", generations={"f.h5": 2}),
        )
        v3 = VersionManifest(
            version="1.72.2",
            created_at="t3",
            hf=None,
            gcs=GCSVersionInfo(bucket="b", generations={"f.h5": 3}),
        )
        registry = VersionRegistry(current="1.72.3", versions=[v2, v3, v1])
        setup_bucket_with_registry(mock_bucket, registry)

        result = list_versions()

        assert result == [
            "1.72.1",
            "1.72.2",
            "1.72.3",
        ]

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_empty(self, mock_get_bucket, mock_bucket):
        mock_get_bucket.return_value = mock_bucket
        registry = VersionRegistry()
        setup_bucket_with_registry(mock_bucket, registry)

        result = list_versions()

        assert result == []


# -- download_versioned_file tests ---------------------------------


class TestDownloadVersionedFile:
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_downloads_correct_generation(
        self,
        mock_get_bucket,
        mock_bucket,
        sample_manifest,
        tmp_path,
    ):
        mock_get_bucket.return_value = mock_bucket
        registry = VersionRegistry(
            current="1.72.3",
            versions=[sample_manifest],
        )
        registry_json = json.dumps(registry.to_dict())

        def mock_blob(name, generation=None):
            if name == "version_manifest.json":
                blob = MagicMock()
                blob.download_as_text.return_value = registry_json
                return blob
            blob = MagicMock()
            blob.name = name
            blob.generation = generation
            return blob

        mock_bucket.blob.side_effect = mock_blob

        local_path = str(tmp_path / "AL.h5")
        download_versioned_file(
            "states/AL.h5",
            "1.72.3",
            local_path,
        )

        calls = mock_bucket.blob.call_args_list
        gen_call = [
            c
            for c in calls
            if c
            == call(
                "states/AL.h5",
                generation=1710203948345678,
            )
        ]
        assert len(gen_call) == 1

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_file_not_in_manifest(
        self,
        mock_get_bucket,
        mock_bucket,
        sample_manifest,
        tmp_path,
    ):
        mock_get_bucket.return_value = mock_bucket
        registry = VersionRegistry(
            current="1.72.3",
            versions=[sample_manifest],
        )
        setup_bucket_with_registry(mock_bucket, registry)

        with pytest.raises(ValueError, match="not found"):
            download_versioned_file(
                "nonexistent.h5",
                "1.72.3",
                str(tmp_path / "out.h5"),
            )


# -- rollback tests -------------------------------------------------


class TestRollback:
    @patch(f"{_MOD}.CommitOperationAdd")
    @patch(f"{_MOD}.hf_hub_download")
    @patch(f"{_MOD}.HfApi")
    @patch(f"{_MOD}.os")
    @patch(f"{_MOD}._get_gcs_bucket")
    def test_creates_new_version_with_old_data(
        self,
        mock_get_bucket,
        mock_os,
        mock_hf_api_cls,
        mock_hf_download,
        mock_commit_op,
        mock_bucket,
        sample_manifest,
    ):
        mock_get_bucket.return_value = mock_bucket
        mock_os.environ.get.return_value = "fake_token"
        mock_os.path.join = lambda *args: "/".join(args)
        mock_os.unlink = MagicMock()

        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        commit_info = MagicMock()
        commit_info.oid = "new_commit_sha"
        mock_api.create_commit.return_value = commit_info

        registry = VersionRegistry(
            current="1.72.3",
            versions=[sample_manifest],
        )
        registry_json = json.dumps(registry.to_dict())
        written = {}

        def mock_blob(name, generation=None):
            if name == "version_manifest.json":
                b = MagicMock()
                b.name = name
                b.download_as_text.return_value = registry_json
                written[name] = b
                return b
            blob = MagicMock()
            blob.name = name
            blob.generation = generation
            return blob

        mock_bucket.blob.side_effect = mock_blob

        new_gen_counter = iter([50001, 50002, 50003])

        def mock_get_blob(name):
            blob = MagicMock()
            blob.generation = next(new_gen_counter)
            return blob

        mock_bucket.get_blob.side_effect = mock_get_blob

        result = rollback(
            target_version="1.72.3",
            new_version="1.73.0",
        )

        assert isinstance(result, VersionManifest)
        assert result.version == "1.73.0"
        assert result.special_operation == "roll-back"
        assert result.roll_back_version == "1.72.3"

        assert mock_bucket.copy_blob.call_count == 3

        blob = written["version_manifest.json"]
        written_json = blob.upload_from_string.call_args[0][0]
        registry_data = json.loads(written_json)

        assert registry_data["current"] == "1.73.0"
        assert len(registry_data["versions"]) == 2
        assert registry_data["versions"][0]["version"] == "1.73.0"
        assert registry_data["versions"][0]["special_operation"] == "roll-back"

        mock_api.create_commit.assert_called_once()
        commit_msg = mock_api.create_commit.call_args.kwargs["commit_message"]
        assert "1.72.3" in commit_msg
        assert "1.73.0" in commit_msg
        mock_api.create_tag.assert_called_once()

    @patch(f"{_MOD}._get_gcs_bucket")
    def test_nonexistent_version(self, mock_get_bucket, mock_bucket):
        mock_get_bucket.return_value = mock_bucket
        blob = MagicMock()
        blob.download_as_text.side_effect = NotFound("Not found")
        mock_bucket.blob.return_value = blob

        with pytest.raises(ValueError, match="not found"):
            rollback(
                target_version="9.9.9",
                new_version="9.10.0",
            )


# -- Consumer API tests --------------------------------------------


class TestGetDataManifest:
    def setup_method(self):
        import policyengine_us_data.utils.version_manifest as mod

        mod._cached_registry = None

    def teardown_method(self):
        import policyengine_us_data.utils.version_manifest as mod

        mod._cached_registry = None

    @patch(f"{_MOD}.hf_hub_download")
    def test_returns_registry(self, mock_download, tmp_path):
        registry_data = {
            "current": "1.72.3",
            "versions": [
                {
                    "version": "1.72.3",
                    "created_at": ("2026-03-10T14:30:00Z"),
                    "hf": {
                        "repo": ("policyengine/policyengine-us-data"),
                        "commit": "abc123",
                    },
                    "gcs": {
                        "bucket": ("policyengine-us-data"),
                        "generations": {"file.h5": 12345},
                    },
                },
            ],
        }
        registry_file = tmp_path / "version_manifest.json"
        registry_file.write_text(json.dumps(registry_data))
        mock_download.return_value = str(registry_file)

        result = get_data_manifest()

        assert isinstance(result, VersionRegistry)
        assert result.current == "1.72.3"
        assert len(result.versions) == 1
        assert result.versions[0].hf.commit == "abc123"
        mock_download.assert_called_once_with(
            repo_id=("policyengine/policyengine-us-data"),
            repo_type="model",
            filename="version_manifest.json",
        )

    @patch(f"{_MOD}.hf_hub_download")
    def test_caches_result(self, mock_download, tmp_path):
        registry_data = {
            "current": "1.72.3",
            "versions": [
                {
                    "version": "1.72.3",
                    "created_at": ("2026-03-10T14:30:00Z"),
                    "hf": None,
                    "gcs": {
                        "bucket": "b",
                        "generations": {"f.h5": 1},
                    },
                },
            ],
        }
        registry_file = tmp_path / "version_manifest.json"
        registry_file.write_text(json.dumps(registry_data))
        mock_download.return_value = str(registry_file)

        first = get_data_manifest()
        second = get_data_manifest()

        assert first is second
        assert mock_download.call_count == 1

    @patch(f"{_MOD}.hf_hub_download")
    def test_get_data_version(self, mock_download, tmp_path):
        registry_data = {
            "current": "1.72.3",
            "versions": [
                {
                    "version": "1.72.3",
                    "created_at": ("2026-03-10T14:30:00Z"),
                    "hf": None,
                    "gcs": {
                        "bucket": "b",
                        "generations": {"f.h5": 1},
                    },
                },
            ],
        }
        registry_file = tmp_path / "version_manifest.json"
        registry_file.write_text(json.dumps(registry_data))
        mock_download.return_value = str(registry_file)

        result = get_data_version()

        assert result == "1.72.3"
