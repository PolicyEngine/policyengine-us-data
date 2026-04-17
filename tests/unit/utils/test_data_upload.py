from pathlib import Path

from policyengine_us_data.utils import data_upload


class _FakeHfApi:
    def __init__(self):
        self.commits = []


def _install_fake_hf(monkeypatch, tmp_path):
    fake = _FakeHfApi()
    monkeypatch.setattr(data_upload, "HfApi", lambda: fake)

    captured_ops = []

    def fake_commit(api, operations, repo_id, repo_type, token, commit_message):
        captured_ops.extend(operations)

    monkeypatch.setattr(data_upload, "hf_create_commit_with_retry", fake_commit)
    return captured_ops


def _make_files(tmp_path, rel_paths):
    files = []
    for rel in rel_paths:
        local = tmp_path / Path(rel).name
        local.write_text("stub")
        files.append((local, rel))
    return files


def test_upload_to_staging_hf_accepts_run_id_kwarg(monkeypatch, tmp_path):
    captured_ops = _install_fake_hf(monkeypatch, tmp_path)
    files = _make_files(tmp_path, ["states/AL.h5"])

    n = data_upload.upload_to_staging_hf(
        files,
        version="1.73.0",
        run_id="abc123",
    )

    assert n == 1
    assert len(captured_ops) == 1


def test_upload_to_staging_hf_run_id_scopes_staging_prefix(monkeypatch, tmp_path):
    captured_ops = _install_fake_hf(monkeypatch, tmp_path)
    files = _make_files(tmp_path, ["states/AL.h5", "states/CA.h5"])

    data_upload.upload_to_staging_hf(files, version="1.73.0", run_id="abc123")

    assert [op.path_in_repo for op in captured_ops] == [
        "staging/abc123/states/AL.h5",
        "staging/abc123/states/CA.h5",
    ]


def test_upload_to_staging_hf_without_run_id_uses_bare_staging_prefix(
    monkeypatch, tmp_path
):
    captured_ops = _install_fake_hf(monkeypatch, tmp_path)
    files = _make_files(tmp_path, ["states/AL.h5"])

    data_upload.upload_to_staging_hf(files, version="1.73.0")

    assert [op.path_in_repo for op in captured_ops] == ["staging/states/AL.h5"]
