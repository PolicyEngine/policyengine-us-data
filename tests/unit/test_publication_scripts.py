from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

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


def test_bump_version_computes_candidate_scope_without_mutating_pyproject(
    tmp_path,
):
    module = _load_script(".github/bump_version.py", "bump_version_script_test")
    _write_pyproject(tmp_path, "1.73.0")
    changelog_dir = tmp_path / "changelog.d"
    changelog_dir.mkdir()
    (changelog_dir / "123.added").write_text("Added a thing.\n")
    monkeypatch_root = tmp_path

    assert module.bump_version("1.73.0", "minor") == "1.74.0"
    module.write_publication_scope(
        monkeypatch_root / ".github_publication_scope.json",
        {
            "base_release_version": "1.73.0",
            "release_bump": "minor",
            "candidate_scope": "1.73.0-minor",
            "would_release_as_at_build_time": "1.74.0",
        },
    )

    assert 'version = "1.73.0"' in (tmp_path / "pyproject.toml").read_text()
    assert module.infer_bump(changelog_dir) == "minor"


def test_fetch_publication_scope_prints_requested_field(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_script(
        ".github/scripts/fetch_publication_scope.py",
        "fetch_publication_scope_script_test",
    )
    path = tmp_path / "publication_scope.json"
    path.write_text(
        json.dumps(
            {
                "base_release_version": "1.73.0",
                "release_bump": "minor",
                "candidate_scope": "1.73.0-minor",
                "would_release_as_at_build_time": "1.74.0",
            }
        )
    )
    monkeypatch.setattr(module, "PUBLICATION_SCOPE_PATH", path)
    monkeypatch.setattr(sys, "argv", ["fetch_publication_scope.py", "candidate_scope"])

    module.main()

    assert capsys.readouterr().out.strip() == "1.73.0-minor"


def test_fetch_publication_scope_exits_on_missing_field(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_script(
        ".github/scripts/fetch_publication_scope.py",
        "fetch_publication_scope_error_script_test",
    )
    path = tmp_path / "publication_scope.json"
    path.write_text(json.dumps({"candidate_scope": "1.73.0-minor"}))
    monkeypatch.setattr(module, "PUBLICATION_SCOPE_PATH", path)
    monkeypatch.setattr(sys, "argv", ["fetch_publication_scope.py", "release_bump"])

    with pytest.raises(SystemExit):
        module.main()

    assert "Publication scope file is missing required field" in capsys.readouterr().err


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


def test_finalize_package_version_rewrites_current_rc_to_stable(
    tmp_path,
    monkeypatch,
    capsys,
):
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


def test_finalize_package_version_accepts_promotion_time_release_version(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/finalize_package_version.py",
        "finalize_package_version_env_script_test",
    )
    _write_pyproject(tmp_path, "1.73.0")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("US_DATA_RELEASE_VERSION", "1.74.0")

    module.main()

    assert 'version = "1.74.0"' in (tmp_path / "pyproject.toml").read_text()


def test_resolve_run_context_uses_publication_scope(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/resolve_run_context.py",
        "resolve_run_context_script_test",
    )
    _write_pyproject(tmp_path, "1.75.0")
    scope_dir = tmp_path / ".github"
    scope_dir.mkdir()
    (scope_dir / "publication_scope.json").write_text(
        json.dumps(
            {
                "base_release_version": "1.75.0",
                "release_bump": "minor",
                "candidate_scope": "1.75.0-minor",
                "would_release_as_at_build_time": "1.76.0",
            }
        )
    )
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)

    assert module._base_release_version({}) == "1.75.0"
    assert module._release_bump({}) == "minor"
    assert (
        module._candidate_version(
            {},
            base_release_version="1.75.0",
            release_bump="minor",
        )
        == "1.75.0-minor"
    )
    assert module._release_version({}) == ""


def test_resolve_run_context_builds_candidate_scope_from_env(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/resolve_run_context.py",
        "resolve_run_context_env_script_test",
    )
    _write_pyproject(tmp_path, "1.75.0")
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)

    env = {
        "BASE_RELEASE_VERSION": "1.75.0",
        "RELEASE_BUMP": "patch",
    }

    assert module._base_release_version(env) == "1.75.0"
    assert module._release_bump(env) == "patch"
    assert (
        module._candidate_version(
            env,
            base_release_version="1.75.0",
            release_bump="patch",
        )
        == "1.75.0-patch"
    )


def test_promote_publication_script_derives_release_from_status(
    tmp_path,
    monkeypatch,
):
    captured = {"calls": []}

    class FakeRemoteFunction:
        def __init__(self, name):
            self.name = name

        def remote(self, *args, **kwargs):
            captured["calls"].append((self.name, args, kwargs))
            if self.name == "get_pipeline_status":
                return {
                    "run_manifest": {
                        "run_id": "run-123",
                        "candidate_version": "1.73.0-minor",
                        "base_release_version": "1.73.0",
                        "release_bump": "minor",
                        "run_context": {
                            "run_id": "run-123",
                            "candidate_version": "1.73.0-minor",
                            "base_release_version": "1.73.0",
                            "release_bump": "minor",
                        },
                    }
                }
            return "promoted"

    class FakeFunction:
        @staticmethod
        def from_name(app_name, function_name, **kwargs):
            captured["from_name"] = (app_name, function_name, kwargs)
            return FakeRemoteFunction(function_name)

    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(Function=FakeFunction),
    )
    module = _load_script(
        ".github/scripts/promote_publication_pipeline.py",
        "promote_publication_pipeline_script_test",
    )
    _write_pyproject(tmp_path, "1.73.0")
    github_env = tmp_path / "github_env"
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_ENV", str(github_env))
    monkeypatch.setenv("US_DATA_RUN_ID", "run-123")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    monkeypatch.setenv("VERSION_OVERRIDE", "9.9.9")

    module.main()

    assert captured["calls"] == [
        ("get_pipeline_status", ("run-123",), {}),
        (
            "promote_run",
            (),
            {
                "run_id": "run-123",
                "candidate_version": "1.73.0-minor",
                "release_version": "1.74.0",
            },
        ),
    ]
    assert "US_DATA_RELEASE_VERSION=1.74.0" in github_env.read_text()
    assert "VERSION_OVERRIDE" not in json.dumps(captured["calls"])


def test_promote_publication_script_requires_release_bump(
    tmp_path,
    monkeypatch,
):
    class FakeRemoteFunction:
        def __init__(self, name):
            self.name = name

        def remote(self, *args, **kwargs):
            return {
                "run_manifest": {
                    "run_id": "run-123",
                    "candidate_version": "1.73.0-minor",
                }
            }

    class FakeFunction:
        @staticmethod
        def from_name(app_name, function_name, **kwargs):
            return FakeRemoteFunction(function_name)

    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(Function=FakeFunction),
    )
    module = _load_script(
        ".github/scripts/promote_publication_pipeline.py",
        "promote_publication_pipeline_missing_bump_script_test",
    )
    _write_pyproject(tmp_path, "1.73.0")
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("US_DATA_RUN_ID", "run-123")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")

    with pytest.raises(RuntimeError, match="missing release_bump"):
        module.main()
