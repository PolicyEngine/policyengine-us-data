"""Tests for N10: deferred token check in ``utils.huggingface``.

Previously::

    TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
    if not TOKEN:
        raise ValueError(...)

at module import time. Any workflow that happened to import
``policyengine_us_data.utils.huggingface`` — docs builds, lightweight
CI checks, fully local calibration (issue #591), transitive imports
via ``raw_cache`` / ``datasets.sipp.sipp`` — died at import. The fix
defers the check into the upload paths (where the token is
genuinely required) and leaves the read side working against public
repos with no token.
"""

import importlib
import os

import pytest


def _reload_hf(monkeypatch, token):
    if token is None:
        monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HUGGING_FACE_TOKEN", token)
    import policyengine_us_data.utils.huggingface as hf

    return importlib.reload(hf)


def test_module_imports_without_token(monkeypatch):
    """Regression: importing utils.huggingface must not raise when
    HUGGING_FACE_TOKEN is unset."""
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
    import policyengine_us_data.utils.huggingface as hf

    hf_reloaded = importlib.reload(hf)
    assert hf_reloaded.get_token() is None


def test_get_token_reads_env(monkeypatch):
    hf = _reload_hf(monkeypatch, "hf_xyz123")
    assert hf.get_token() == "hf_xyz123"


def test_require_token_raises_with_action_message(monkeypatch):
    hf = _reload_hf(monkeypatch, None)
    with pytest.raises(ValueError, match="upload files"):
        hf._require_token("upload files to Hugging Face Hub")


def test_upload_raises_when_token_missing(monkeypatch):
    hf = _reload_hf(monkeypatch, None)
    with pytest.raises(ValueError, match="HUGGING_FACE_TOKEN"):
        hf.upload("dummy/local.h5", "org/repo", "dummy/remote.h5")


def test_upload_calibration_artifacts_no_ops_without_files_or_token(
    monkeypatch, tmp_path
):
    """If there are no files to upload, the function returns ``[]``
    and never checks the token (short-circuit)."""
    hf = _reload_hf(monkeypatch, None)
    result = hf.upload_calibration_artifacts(
        weights_path=str(tmp_path / "nonexistent.npy"),
        repo="org/repo",
        prefix="national_",
    )
    assert result == []


def test_download_uses_token_when_available(monkeypatch):
    """Sanity: ``download`` picks up the token from the env at call
    time so credentials rotate without a process restart."""
    hf = _reload_hf(monkeypatch, "hf_token_A")
    assert hf.get_token() == "hf_token_A"
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "hf_token_B")
    assert hf.get_token() == "hf_token_B"
