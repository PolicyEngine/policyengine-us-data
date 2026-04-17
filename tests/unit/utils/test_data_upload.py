import importlib
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

_DATA_UPLOAD_MODULE = None


def _install_fake_google_modules():
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

    sys.modules.setdefault("google", fake_google)
    sys.modules.setdefault("google.auth", fake_google_auth)
    sys.modules.setdefault("google.cloud", fake_google_cloud)
    sys.modules.setdefault("google.cloud.storage", fake_google_storage)


def _load_data_upload_module():
    global _DATA_UPLOAD_MODULE
    if _DATA_UPLOAD_MODULE is not None:
        return _DATA_UPLOAD_MODULE

    try:
        _DATA_UPLOAD_MODULE = importlib.import_module(
            "policyengine_us_data.utils.data_upload"
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "google",
            "google.auth",
            "google.cloud",
            "google.cloud.storage",
        }:
            raise
        _install_fake_google_modules()
        _DATA_UPLOAD_MODULE = importlib.import_module(
            "policyengine_us_data.utils.data_upload"
        )

    return _DATA_UPLOAD_MODULE


def _install_fake_hf(monkeypatch, tmp_path):
    data_upload = _load_data_upload_module()
    fake = SimpleNamespace(commits=[])

    monkeypatch.setattr(data_upload, "HfApi", lambda: fake)

    captured_ops = []

    def fake_commit(api, operations, repo_id, repo_type, token, commit_message):
        captured_ops.extend(operations)

    monkeypatch.setattr(data_upload, "hf_create_commit_with_retry", fake_commit)
    return data_upload, captured_ops


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


def _make_files(tmp_path, rel_paths):
    files = []
    for rel in rel_paths:
        local = tmp_path / Path(rel).name
        local.write_text("stub")
        files.append((local, rel))
    return files


def test_upload_to_staging_hf_accepts_run_id_kwarg(monkeypatch, tmp_path):
    data_upload, captured_ops = _install_fake_hf(monkeypatch, tmp_path)
    files = _make_files(tmp_path, ["states/AL.h5"])

    n = data_upload.upload_to_staging_hf(
        files,
        version="1.73.0",
        run_id="abc123",
    )

    assert n == 1
    assert len(captured_ops) == 1


def test_upload_to_staging_hf_run_id_scopes_staging_prefix(monkeypatch, tmp_path):
    data_upload, captured_ops = _install_fake_hf(monkeypatch, tmp_path)
    files = _make_files(tmp_path, ["states/AL.h5", "states/CA.h5"])

    data_upload.upload_to_staging_hf(files, version="1.73.0", run_id="abc123")

    assert [op.path_in_repo for op in captured_ops] == [
        "staging/abc123/states/AL.h5",
        "staging/abc123/states/CA.h5",
    ]


def test_upload_to_staging_hf_without_run_id_uses_bare_staging_prefix(
    monkeypatch, tmp_path
):
    data_upload, captured_ops = _install_fake_hf(monkeypatch, tmp_path)
    files = _make_files(tmp_path, ["states/AL.h5"])

    data_upload.upload_to_staging_hf(files, version="1.73.0")

    assert [op.path_in_repo for op in captured_ops] == ["staging/states/AL.h5"]


def test_promote_staging_to_production_hf_uses_run_scoped_source_only(monkeypatch):
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
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)

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
