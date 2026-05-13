from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_pyproject(root: Path, version: str, name: str = "policyengine-us-data"):
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                f'name = "{name}"',
                f'version = "{version}"',
                "",
            ]
        )
    )


def test_bump_version_uses_next_rc_for_final_release(monkeypatch):
    module = _load_script(".github/bump_version.py", "bump_version_script_test")
    payload = {
        "releases": {
            "1.74.0rc1": [],
            "1.74.0rc2": [],
            "1.73.0rc9": [],
            "1.74.0": [],
        }
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        module, "json", types.SimpleNamespace(load=lambda response: payload)
    )
    monkeypatch.setattr(module, "urlopen", lambda url, timeout: FakeResponse())

    assert module.bump_version("1.73.0rc4", "patch") == "1.73.1"
    assert module.next_rc_version("policyengine_us_data", "1.74.0") == "1.74.0rc3"


def test_bump_version_starts_rc_sequence_when_pypi_package_is_missing(monkeypatch):
    module = _load_script(".github/bump_version.py", "bump_version_404_script_test")

    def raise_404(url, timeout):
        raise HTTPError(url, 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr(module, "urlopen", raise_404)

    assert module.next_rc_version("policyengine-us-data", "1.74.0") == "1.74.0rc1"


def test_bump_version_exits_when_pypi_history_cannot_be_read(monkeypatch, capsys):
    module = _load_script(".github/bump_version.py", "bump_version_error_script_test")

    def raise_url_error(url, timeout):
        raise URLError("offline")

    monkeypatch.setattr(module, "urlopen", raise_url_error)

    with pytest.raises(SystemExit):
        module.next_rc_version("policyengine-us-data", "1.74.0")

    assert "Could not fetch PyPI release history" in capsys.readouterr().err


def test_fetch_release_version_prints_stable_version(tmp_path, monkeypatch, capsys):
    module = _load_script(
        ".github/scripts/fetch_release_version.py",
        "fetch_release_version_script_test",
    )
    _write_pyproject(tmp_path, "1.74.0rc3")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    module.main()

    assert capsys.readouterr().out.strip() == "1.74.0"


def test_fetch_release_version_exits_on_unsupported_version(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_script(
        ".github/scripts/fetch_release_version.py",
        "fetch_release_version_error_script_test",
    )
    _write_pyproject(tmp_path, "1.74")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    with pytest.raises(SystemExit):
        module.main()

    assert "Unsupported version format: 1.74" in capsys.readouterr().err


def test_finalize_package_version_rewrites_rc_to_stable(tmp_path, monkeypatch, capsys):
    module = _load_script(
        ".github/scripts/finalize_package_version.py",
        "finalize_package_version_script_test",
    )
    _write_pyproject(tmp_path, "1.74.0rc3")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("US_DATA_RELEASE_VERSION", raising=False)

    module.main()

    assert 'version = "1.74.0"' in (tmp_path / "pyproject.toml").read_text()
    assert "Finalized package version: 1.74.0rc3 -> 1.74.0" in capsys.readouterr().out


def test_finalize_package_version_rejects_mismatched_release_env(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/finalize_package_version.py",
        "finalize_package_version_mismatch_script_test",
    )
    _write_pyproject(tmp_path, "1.74.0rc3")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("US_DATA_RELEASE_VERSION", "1.73.0")

    with pytest.raises(ValueError, match="must match the current package candidate"):
        module.main()

    assert 'version = "1.74.0rc3"' in (tmp_path / "pyproject.toml").read_text()


def test_resolve_run_context_ignores_removed_version_override(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/resolve_run_context.py",
        "resolve_run_context_script_test",
    )
    _write_pyproject(tmp_path, "1.75.0rc1")
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)

    assert module._candidate_version({"VERSION_OVERRIDE": "9.9.9"}) == "1.75.0rc1"
    assert (
        module._release_version(
            {"VERSION_OVERRIDE": "9.9.9"},
            candidate_version="1.75.0rc1",
        )
        == "1.75.0rc1"
    )


def test_promote_publication_script_does_not_pass_removed_version_override(
    monkeypatch,
):
    captured = {}

    class FakeRemoteFunction:
        def remote(self, **kwargs):
            captured["kwargs"] = kwargs
            return "promoted"

    class FakeFunction:
        @staticmethod
        def from_name(*args, **kwargs):
            captured["from_name"] = (args, kwargs)
            return FakeRemoteFunction()

    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(Function=FakeFunction),
    )
    module = _load_script(
        ".github/scripts/promote_publication_pipeline.py",
        "promote_publication_pipeline_script_test",
    )
    monkeypatch.setenv("US_DATA_RUN_ID", "run-123")
    monkeypatch.setenv("US_DATA_CANDIDATE_VERSION", "1.74.0rc3")
    monkeypatch.setenv("US_DATA_RELEASE_VERSION", "1.74.0")
    monkeypatch.setenv("CANDIDATE_VERSION", "1.74.0rc3")
    monkeypatch.setenv("RELEASE_VERSION", "1.74.0")
    monkeypatch.setenv("VERSION_OVERRIDE", "9.9.9")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")

    module.main()

    assert captured["kwargs"] == {
        "run_id": "run-123",
        "candidate_version": "1.74.0rc3",
        "release_version": "1.74.0",
    }
    assert "version" not in captured["kwargs"]
