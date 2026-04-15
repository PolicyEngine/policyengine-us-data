import importlib
import sys
from types import ModuleType, SimpleNamespace


class _FakeCommitOperationAdd:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class _FakeCommitOperationCopy:
    def __init__(self, src_path_in_repo, path_in_repo):
        self.src_path_in_repo = src_path_in_repo
        self.path_in_repo = path_in_repo


class _FakeCommitOperationDelete:
    def __init__(self, path_in_repo):
        self.path_in_repo = path_in_repo


class _FakeBlob:
    def __init__(self, name):
        self.name = name
        self.uploaded_from = None
        self.metadata = None
        self.patch_called = False

    def upload_from_filename(self, filename):
        self.uploaded_from = filename

    def patch(self):
        self.patch_called = True


class _FakeBucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, name):
        blob = _FakeBlob(name)
        self.blobs[name] = blob
        return blob


def _load_data_upload_module():
    fake_google = ModuleType("google")
    fake_google_auth = ModuleType("google.auth")
    fake_google_cloud = ModuleType("google.cloud")
    fake_google_storage = ModuleType("google.cloud.storage")

    fake_google_auth.default = lambda: (object(), "test-project")
    fake_google_storage.Client = lambda credentials=None, project=None: SimpleNamespace(
        bucket=lambda _: _FakeBucket()
    )

    fake_google.auth = fake_google_auth
    fake_google.cloud = fake_google_cloud
    fake_google_cloud.storage = fake_google_storage

    sys.modules["google"] = fake_google
    sys.modules["google.auth"] = fake_google_auth
    sys.modules["google.cloud"] = fake_google_cloud
    sys.modules["google.cloud.storage"] = fake_google_storage
    sys.modules.pop("policyengine_us_data.utils.data_upload", None)
    return importlib.import_module("policyengine_us_data.utils.data_upload")


def test_upload_to_staging_hf_uses_run_scoped_staging_paths(tmp_path, monkeypatch):
    data_upload = _load_data_upload_module()
    local_file = tmp_path / "AL.h5"
    local_file.write_bytes(b"state")
    commit_operations = []

    monkeypatch.setattr(data_upload, "HfApi", lambda: object())
    monkeypatch.setattr(data_upload, "CommitOperationAdd", _FakeCommitOperationAdd)
    monkeypatch.setattr(
        data_upload,
        "hf_create_commit_with_retry",
        lambda **kwargs: (
            commit_operations.extend(kwargs["operations"])
            or SimpleNamespace(oid="after")
        ),
    )

    uploaded = data_upload.upload_to_staging_hf(
        [(local_file, "states/AL.h5")],
        version="1.73.0",
        run_id="run-123",
    )

    assert uploaded == 1
    assert [op.path_in_repo for op in commit_operations] == [
        "staging/run-123/states/AL.h5"
    ]


def test_upload_to_staging_hf_keeps_legacy_staging_prefix_without_run_id(
    tmp_path, monkeypatch
):
    data_upload = _load_data_upload_module()
    local_file = tmp_path / "AL.h5"
    local_file.write_bytes(b"state")
    commit_operations = []

    monkeypatch.setattr(data_upload, "HfApi", lambda: object())
    monkeypatch.setattr(data_upload, "CommitOperationAdd", _FakeCommitOperationAdd)
    monkeypatch.setattr(
        data_upload,
        "hf_create_commit_with_retry",
        lambda **kwargs: (
            commit_operations.extend(kwargs["operations"])
            or SimpleNamespace(oid="after")
        ),
    )

    uploaded = data_upload.upload_to_staging_hf(
        [(local_file, "states/AL.h5")],
        version="1.73.0",
    )

    assert uploaded == 1
    assert [op.path_in_repo for op in commit_operations] == ["staging/states/AL.h5"]


def test_promote_staging_to_production_hf_uses_run_scoped_source_only(
    monkeypatch,
):
    data_upload = _load_data_upload_module()
    commit_operations = []
    fake_api = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sha="before"))

    monkeypatch.setattr(data_upload, "HfApi", lambda: fake_api)
    monkeypatch.setattr(data_upload, "CommitOperationCopy", _FakeCommitOperationCopy)
    monkeypatch.setattr(
        data_upload,
        "hf_create_commit_with_retry",
        lambda **kwargs: (
            commit_operations.extend(kwargs["operations"])
            or SimpleNamespace(oid="after")
        ),
    )

    promoted = data_upload.promote_staging_to_production_hf(
        ["states/AL.h5"],
        version="1.73.0",
        run_id="run-123",
    )

    assert promoted == 1
    assert commit_operations[0].src_path_in_repo == "staging/run-123/states/AL.h5"
    assert commit_operations[0].path_in_repo == "states/AL.h5"


def test_cleanup_staging_hf_deletes_run_scoped_staging_paths(monkeypatch):
    data_upload = _load_data_upload_module()
    commit_operations = []
    fake_api = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sha="before"))

    monkeypatch.setattr(data_upload, "HfApi", lambda: fake_api)
    monkeypatch.setattr(
        data_upload, "CommitOperationDelete", _FakeCommitOperationDelete
    )
    monkeypatch.setattr(
        data_upload,
        "hf_create_commit_with_retry",
        lambda **kwargs: (
            commit_operations.extend(kwargs["operations"])
            or SimpleNamespace(oid="after")
        ),
    )

    deleted = data_upload.cleanup_staging_hf(
        ["states/AL.h5"],
        version="1.73.0",
        run_id="run-123",
    )

    assert deleted == 1
    assert [op.path_in_repo for op in commit_operations] == [
        "staging/run-123/states/AL.h5"
    ]


def test_upload_from_hf_staging_to_gcs_uses_run_scoped_hf_source_only(
    monkeypatch,
):
    data_upload = _load_data_upload_module()
    download_calls = []
    fake_bucket = _FakeBucket()
    fake_storage_client = SimpleNamespace(bucket=lambda _: fake_bucket)

    monkeypatch.setattr(
        data_upload,
        "hf_hub_download",
        lambda **kwargs: download_calls.append(kwargs) or "/tmp/AL.h5",
    )
    monkeypatch.setattr(
        data_upload.google.auth,
        "default",
        lambda: (object(), "test-project"),
    )
    monkeypatch.setattr(
        data_upload.storage,
        "Client",
        lambda credentials, project: fake_storage_client,
    )

    uploaded = data_upload.upload_from_hf_staging_to_gcs(
        ["states/AL.h5"],
        version="1.73.0",
        run_id="run-123",
    )

    assert uploaded == 1
    assert download_calls == [
        {
            "repo_id": "policyengine/policyengine-us-data",
            "filename": "staging/run-123/states/AL.h5",
            "repo_type": "model",
            "token": None,
        }
    ]
    blob = fake_bucket.blobs["states/AL.h5"]
    assert blob.name == "states/AL.h5"
    assert blob.uploaded_from == "/tmp/AL.h5"
    assert blob.metadata == {"version": "1.73.0"}
    assert blob.patch_called is True
