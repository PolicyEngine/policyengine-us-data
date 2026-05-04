import importlib
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

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
    assert len(captured_ops) == 2
    assert captured_ops[0].path_in_repo == ("staging/abc123/_run_context.json")


def test_upload_to_staging_hf_run_id_scopes_staging_prefix(monkeypatch, tmp_path):
    data_upload, captured_ops = _install_fake_hf(monkeypatch, tmp_path)
    files = _make_files(tmp_path, ["states/AL.h5", "states/CA.h5"])

    data_upload.upload_to_staging_hf(files, version="1.73.0", run_id="abc123")

    assert [op.path_in_repo for op in captured_ops] == [
        "staging/abc123/_run_context.json",
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


def test_upload_to_staging_hf_uses_run_id_env(monkeypatch, tmp_path):
    monkeypatch.setenv("US_DATA_RUN_ID", "run-123")
    data_upload, captured_ops = _install_fake_hf(monkeypatch, tmp_path)
    files = _make_files(tmp_path, ["states/AL.h5"])

    data_upload.upload_to_staging_hf(files, version="1.73.0")

    assert [op.path_in_repo for op in captured_ops] == [
        "staging/run-123/_run_context.json",
        "staging/run-123/states/AL.h5",
    ]


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


def test_cleanup_staging_hf_skips_missing_staged_paths(monkeypatch):
    data_upload = _load_data_upload_module()
    fake_api = SimpleNamespace(list_repo_files=lambda **kwargs: [])

    monkeypatch.setattr(data_upload, "HfApi", lambda: fake_api)
    monkeypatch.setattr(
        data_upload, "CommitOperationDelete", _FakeCommitOperationDelete
    )
    monkeypatch.setattr(
        data_upload,
        "hf_create_commit_with_retry",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cleanup should not create an empty commit")
        ),
    )

    deleted = data_upload.cleanup_staging_hf(
        ["states/AL.h5"],
        version="1.73.0",
        run_id="run-123",
    )

    assert deleted == 0


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


def test_promote_staging_to_production_hf_allows_noop_for_release_retry(
    monkeypatch,
):
    data_upload = _load_data_upload_module()
    fake_api = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sha="same"))

    monkeypatch.setattr(data_upload, "HfApi", lambda: fake_api)
    monkeypatch.setattr(data_upload, "CommitOperationCopy", _FakeCommitOperationCopy)
    monkeypatch.setattr(
        data_upload,
        "hf_create_commit_with_retry",
        lambda **kwargs: SimpleNamespace(oid="same"),
    )

    promoted = data_upload.promote_staging_to_production_hf(
        ["states/AL.h5"],
        version="1.73.0",
        run_id="run-123",
        allow_noop=True,
    )

    assert promoted == 1


def test_promote_full_release_fails_before_writes_when_staging_missing(
    monkeypatch,
    tmp_path,
):
    data_upload = _load_data_upload_module()
    files = _make_files(tmp_path, ["states/AL.h5"])

    monkeypatch.setattr(
        data_upload,
        "list_missing_staged_artifacts",
        lambda *args, **kwargs: ["staging/run-123/states/AL.h5"],
    )
    monkeypatch.setattr(
        data_upload,
        "get_matching_finalized_release_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        data_upload,
        "promote_staging_to_production_hf",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("promotion should not run when staging is incomplete")
        ),
    )

    with pytest.raises(FileNotFoundError, match="Missing staged release artifacts"):
        data_upload.promote_full_release_from_staging(
            rel_paths=["states/AL.h5"],
            version="1.73.0",
            run_id="run-123",
            files_with_paths=files,
        )


def test_promote_full_release_orders_full_release_operations(
    monkeypatch,
    tmp_path,
):
    data_upload = _load_data_upload_module()
    rel_paths = ["cps_2024.h5", "states/AL.h5", "national/US.h5"]
    files = _make_files(tmp_path, rel_paths)
    calls = []

    monkeypatch.setattr(
        data_upload,
        "list_missing_staged_artifacts",
        lambda *args, **kwargs: calls.append("validate_staging") or [],
    )
    monkeypatch.setattr(
        data_upload,
        "get_matching_finalized_release_manifest",
        lambda *args, **kwargs: calls.append("check_finalized") or None,
    )
    monkeypatch.setattr(
        data_upload,
        "preflight_release_manifest_publish",
        lambda *args, **kwargs: calls.append("preflight_manifest") or (True, []),
    )
    monkeypatch.setattr(
        data_upload,
        "promote_staging_to_production_hf",
        lambda paths, **kwargs: calls.append("promote_hf") or len(paths),
    )
    monkeypatch.setattr(
        data_upload,
        "upload_from_hf_staging_to_gcs",
        lambda paths, **kwargs: calls.append("upload_gcs") or len(paths),
    )
    monkeypatch.setattr(
        data_upload,
        "publish_release_manifest_to_hf",
        lambda files_with_paths, **kwargs: calls.append("release_manifest")
        or {
            "artifacts": {
                Path(repo_path).with_suffix("").as_posix(): {"path": repo_path}
                for _, repo_path in files_with_paths
            }
        },
    )
    monkeypatch.setattr(
        data_upload,
        "upload_final_version_manifest",
        lambda **kwargs: calls.append("version_manifest"),
    )
    monkeypatch.setattr(
        data_upload,
        "cleanup_staging_hf",
        lambda paths, **kwargs: calls.append("cleanup_staging") or len(paths),
    )

    result = data_upload.promote_full_release_from_staging(
        rel_paths=rel_paths,
        version="1.73.0",
        run_id="run-123",
        files_with_paths=files,
        extra_cleanup_paths=["_run_context.json"],
    )

    assert calls == [
        "check_finalized",
        "validate_staging",
        "preflight_manifest",
        "promote_hf",
        "upload_gcs",
        "release_manifest",
        "version_manifest",
        "cleanup_staging",
    ]
    assert result["artifact_count"] == 3
    assert result["hf_promoted"] == 3
    assert result["gcs_uploaded"] == 3
    assert result["release_manifest_artifacts"] == 3


def test_promote_full_release_can_finish_registry_after_finalized_release(
    monkeypatch,
    tmp_path,
):
    data_upload = _load_data_upload_module()
    files = _make_files(tmp_path, ["states/AL.h5"])
    calls = []

    monkeypatch.setattr(
        data_upload,
        "get_matching_finalized_release_manifest",
        lambda *args, **kwargs: {"artifacts": {"states/AL": {"path": "states/AL.h5"}}},
    )
    monkeypatch.setattr(
        data_upload,
        "list_missing_staged_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("already-finalized retries should not require staging")
        ),
    )
    monkeypatch.setattr(
        data_upload,
        "upload_final_version_manifest",
        lambda **kwargs: calls.append(("version_manifest", kwargs["released_paths"])),
    )
    monkeypatch.setattr(
        data_upload,
        "cleanup_staging_hf",
        lambda paths, **kwargs: calls.append(("cleanup", list(paths))) or 0,
    )

    result = data_upload.promote_full_release_from_staging(
        rel_paths=["states/AL.h5"],
        version="1.73.0",
        run_id="run-123",
        files_with_paths=files,
    )

    assert result["already_finalized"] is True
    assert result["hf_promoted"] == 0
    assert result["gcs_uploaded"] == 0
    assert calls == [
        ("version_manifest", ["states/AL.h5"]),
        ("cleanup", ["states/AL.h5"]),
    ]
