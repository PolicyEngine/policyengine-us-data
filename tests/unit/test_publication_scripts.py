from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
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


def _write_pyproject_with_policyengine_us(root: Path, dependency: str):
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "policyengine-us-data"',
                'version = "1.115.2"',
                "dependencies = [",
                f'    "{dependency}",',
                "]",
                "",
            ]
        )
    )


def _write_uv_lock_for_policyengine_us(
    root: Path,
    version: str,
    source: str = '{ registry = "https://pypi.org/simple" }',
):
    (root / "uv.lock").write_text(
        "\n".join(
            [
                "[[package]]",
                'name = "policyengine-us"',
                f'version = "{version}"',
                f"source = {source}",
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


def test_bump_version_script_runs_from_github_directory_without_installed_package(
    tmp_path,
):
    repo = tmp_path / "repo"
    script_dir = repo / ".github"
    package_utils_dir = repo / "policyengine_us_data" / "utils"
    changelog_dir = repo / "changelog.d"
    script_dir.mkdir(parents=True)
    package_utils_dir.mkdir(parents=True)
    changelog_dir.mkdir()
    shutil.copyfile(
        REPO_ROOT / ".github" / "bump_version.py", script_dir / "bump_version.py"
    )
    shutil.copyfile(
        REPO_ROOT / "policyengine_us_data" / "utils" / "run_context.py",
        package_utils_dir / "run_context.py",
    )
    shutil.copyfile(
        REPO_ROOT / "policyengine_us_data" / "utils" / "canonical_json.py",
        package_utils_dir / "canonical_json.py",
    )
    (repo / "policyengine_us_data" / "__init__.py").write_text("")
    (package_utils_dir / "__init__.py").write_text("")
    _write_pyproject(repo, "1.73.0")
    (changelog_dir / "123.added").write_text("Added a thing.\n")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["US_DATA_RUN_ID"] = "run-123"

    result = subprocess.run(
        [sys.executable, str(script_dir / "bump_version.py")],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((script_dir / "publication_scope.json").read_text()) == {
        "base_release_version": "1.73.0",
        "candidate_scope": "1.73.0-minor",
        "run_id": "run-123",
        "release_bump": "minor",
        "would_release_as_at_build_time": "1.74.0",
    }
    candidate_dir = script_dir / "publication_candidates" / "run-123"
    assert json.loads((candidate_dir / "publication_scope.json").read_text()) == {
        "base_release_version": "1.73.0",
        "candidate_scope": "1.73.0-minor",
        "run_id": "run-123",
        "release_bump": "minor",
        "would_release_as_at_build_time": "1.74.0",
    }
    assert (candidate_dir / "changelog.d" / "123.added").read_text() == (
        "Added a thing.\n"
    )
    assert not (changelog_dir / "123.added").exists()


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


def test_policyengine_us_dependency_check_passes_when_locked_to_latest(tmp_path):
    module = _load_script(
        ".github/scripts/check_policyengine_us_dependency.py",
        "check_policyengine_us_dependency_current_test",
    )
    _write_pyproject_with_policyengine_us(tmp_path, "policyengine-us==1.691.11")
    _write_uv_lock_for_policyengine_us(tmp_path, "1.691.11")

    assert module.check_dependency(tmp_path, latest_version="1.691.11") == []


def test_policyengine_us_dependency_check_flags_stale_lock(tmp_path):
    module = _load_script(
        ".github/scripts/check_policyengine_us_dependency.py",
        "check_policyengine_us_dependency_stale_test",
    )
    _write_pyproject_with_policyengine_us(tmp_path, "policyengine-us==1.691.10")
    _write_uv_lock_for_policyengine_us(tmp_path, "1.691.10")

    violations = module.check_dependency(tmp_path, latest_version="1.691.11")

    assert any("1.691.10" in violation for violation in violations)
    assert any("1.691.11" in violation for violation in violations)


def test_policyengine_us_dependency_check_flags_git_refs(tmp_path):
    module = _load_script(
        ".github/scripts/check_policyengine_us_dependency.py",
        "check_policyengine_us_dependency_git_test",
    )
    _write_pyproject_with_policyengine_us(
        tmp_path,
        "policyengine-us @ git+https://github.com/PolicyEngine/policyengine-us@abc",
    )
    _write_uv_lock_for_policyengine_us(
        tmp_path,
        "1.691.11",
        source='{ git = "https://github.com/PolicyEngine/policyengine-us?rev=abc#abc" }',
    )

    violations = module.check_dependency(tmp_path, latest_version="1.691.11")

    assert any("Git ref" in violation for violation in violations)


def test_policyengine_us_dependency_check_flags_non_exact_pyproject_pin(tmp_path):
    module = _load_script(
        ".github/scripts/check_policyengine_us_dependency.py",
        "check_policyengine_us_dependency_pyproject_test",
    )
    _write_pyproject_with_policyengine_us(tmp_path, "policyengine-us>=1.691.11")
    _write_uv_lock_for_policyengine_us(tmp_path, "1.691.11")

    violations = module.check_dependency(tmp_path, latest_version="1.691.11")

    assert any(
        "must pin policyengine-us==1.691.11" in violation for violation in violations
    )


def test_policyengine_us_dependency_check_allow_stale_exits_successfully(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/check_policyengine_us_dependency.py",
        "check_policyengine_us_dependency_allow_stale_test",
    )
    _write_pyproject_with_policyengine_us(tmp_path, "policyengine-us==1.691.10")
    _write_uv_lock_for_policyengine_us(tmp_path, "1.691.10")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_latest_pypi_version", lambda: "1.691.11")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_policyengine_us_dependency.py", "--mode", "fail"],
    )
    monkeypatch.setenv("POLICYENGINE_US_ALLOW_STALE", "true")

    assert module.main() == 0


def test_policyengine_us_dependency_check_allow_stale_keeps_local_errors_fatal(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/check_policyengine_us_dependency.py",
        "check_policyengine_us_dependency_allow_stale_local_error_test",
    )
    _write_pyproject_with_policyengine_us(
        tmp_path,
        "policyengine-us @ git+https://github.com/PolicyEngine/policyengine-us@abc",
    )
    _write_uv_lock_for_policyengine_us(
        tmp_path,
        "1.691.10",
        source='{ git = "https://github.com/PolicyEngine/policyengine-us?rev=abc#abc" }',
    )
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_latest_pypi_version", lambda: "1.691.11")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_policyengine_us_dependency.py", "--mode", "fail"],
    )
    monkeypatch.setenv("POLICYENGINE_US_ALLOW_STALE", "true")

    assert module.main() == 1


def test_data_release_version_check_passes_at_latest_release(tmp_path):
    module = _load_script(
        ".github/scripts/check_data_release_version.py",
        "check_data_release_version_current_test",
    )
    _write_pyproject(tmp_path, "1.115.3")

    assert (
        module.check_repository(
            tmp_path,
            finalized_release_version="1.115.3",
        )
        == []
    )


def test_data_release_version_check_flags_stale_package(tmp_path):
    module = _load_script(
        ".github/scripts/check_data_release_version.py",
        "check_data_release_version_stale_test",
    )
    _write_pyproject(tmp_path, "1.115.2")

    violations = module.check_repository(
        tmp_path,
        finalized_release_version="1.115.3",
    )

    assert any("1.115.2" in violation for violation in violations)
    assert any("1.115.3" in violation for violation in violations)


@pytest.mark.parametrize(
    ("package_version", "finalized_release_version", "expected_relation"),
    [
        ("1.115.3", "1.115.3", "current"),
        ("1.115.2", "1.115.3", "behind"),
        ("1.115.4", "1.115.3", "ahead"),
        ("1.115.3rc1", "1.115.3", "current"),
    ],
)
def test_data_release_version_state_relations(
    tmp_path,
    package_version,
    finalized_release_version,
    expected_relation,
):
    module = _load_script(
        ".github/scripts/check_data_release_version.py",
        f"check_data_release_version_{expected_relation}_state_test",
    )
    _write_pyproject(tmp_path, package_version)

    state = module.check_repository_state(
        tmp_path,
        finalized_release_version=finalized_release_version,
    )

    assert state.package_version == package_version
    assert state.finalized_release_version == finalized_release_version
    assert state.release_version_relation == expected_relation


def test_data_release_version_check_emits_github_outputs(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/check_data_release_version.py",
        "check_data_release_version_outputs_test",
    )
    _write_pyproject(tmp_path, "1.115.2")
    github_output = tmp_path / "github_output"
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr(module, "latest_hf_release_version", lambda url: "1.115.3")

    assert module.main(["--mode", "warn"]) == 0

    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs == {
        "package_version": "1.115.2",
        "finalized_release_version": "1.115.3",
        "release_version_relation": "behind",
    }


def test_data_release_version_check_emits_unknown_on_manifest_error(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_script(
        ".github/scripts/check_data_release_version.py",
        "check_data_release_version_unknown_outputs_test",
    )
    _write_pyproject(tmp_path, "1.115.2")
    github_output = tmp_path / "github_output"
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr(
        module,
        "latest_hf_release_version",
        lambda url: (_ for _ in ()).throw(OSError("manifest unavailable")),
    )

    assert module.main(["--mode", "warn"]) == 0

    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs == {
        "package_version": "1.115.2",
        "finalized_release_version": "",
        "release_version_relation": "unknown",
    }
    assert "manifest unavailable" in capsys.readouterr().err


def test_data_release_version_check_fails_on_invalid_local_version(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_script(
        ".github/scripts/check_data_release_version.py",
        "check_data_release_version_invalid_local_test",
    )
    _write_pyproject(tmp_path, "1.115")
    github_output = tmp_path / "github_output"
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr(module, "latest_hf_release_version", lambda url: "1.115.3")

    assert module.main(["--mode", "warn"]) == 1

    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs["release_version_relation"] == "unknown"
    assert "Unsupported version format: 1.115" in capsys.readouterr().err


def test_sync_finalized_data_release_version_updates_stale_pyproject(tmp_path):
    module = _load_script(
        ".github/scripts/sync_finalized_data_release_version.py",
        "sync_finalized_data_release_version_update_test",
    )
    _write_pyproject(tmp_path, "1.115.2")

    changed = module.sync_finalized_data_release_version(
        tmp_path,
        finalized_release_version="1.115.3",
    )

    assert changed is True
    assert 'version = "1.115.3"' in (tmp_path / "pyproject.toml").read_text()


def test_sync_finalized_data_release_version_leaves_current_pyproject(tmp_path):
    module = _load_script(
        ".github/scripts/sync_finalized_data_release_version.py",
        "sync_finalized_data_release_version_current_test",
    )
    _write_pyproject(tmp_path, "1.115.3")
    before = (tmp_path / "pyproject.toml").read_text()

    changed = module.sync_finalized_data_release_version(
        tmp_path,
        finalized_release_version="1.115.3",
    )

    assert changed is False
    assert (tmp_path / "pyproject.toml").read_text() == before


def test_sync_finalized_data_release_version_treats_matching_rc_as_current(
    tmp_path,
):
    module = _load_script(
        ".github/scripts/sync_finalized_data_release_version.py",
        "sync_finalized_data_release_version_rc_test",
    )
    _write_pyproject(tmp_path, "1.115.3rc1")
    before = (tmp_path / "pyproject.toml").read_text()

    changed = module.sync_finalized_data_release_version(
        tmp_path,
        finalized_release_version="1.115.3",
    )

    assert changed is False
    assert (tmp_path / "pyproject.toml").read_text() == before


def test_sync_then_bump_version_uses_synced_base_release(
    tmp_path,
    monkeypatch,
):
    sync_module = _load_script(
        ".github/scripts/sync_finalized_data_release_version.py",
        "sync_finalized_data_release_version_workflow_test",
    )
    bump_module = _load_script(
        ".github/bump_version.py",
        "bump_version_after_sync_script_test",
    )
    _write_pyproject(tmp_path, "1.115.4")
    changelog_dir = tmp_path / "changelog.d"
    changelog_dir.mkdir()
    (changelog_dir / "123.fixed").write_text("Fixed a thing.\n")
    monkeypatch.setattr(bump_module, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("US_DATA_RUN_ID", "run-123")

    assert (
        sync_module.sync_finalized_data_release_version(
            tmp_path,
            finalized_release_version="1.115.5",
        )
        is True
    )
    bump_module.main()

    assert (
        json.loads((tmp_path / ".github" / "publication_scope.json").read_text())[
            "base_release_version"
        ]
        == "1.115.5"
    )


def test_restore_publication_changelog_restores_candidate_snapshot(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/restore_publication_changelog.py",
        "restore_publication_changelog_script_test",
    )
    root_changelog = tmp_path / "changelog.d"
    snapshot = (
        tmp_path / ".github" / "publication_candidates" / "run-123" / "changelog.d"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "123.changed.md").write_text("Changed a thing.\n")
    monkeypatch.setattr(module, "ROOT_CHANGELOG_DIR", root_changelog)
    monkeypatch.setattr(
        module,
        "PUBLICATION_CANDIDATES_DIR",
        tmp_path / ".github" / "publication_candidates",
    )

    module.restore_candidate_changelog("run-123")

    assert (root_changelog / "123.changed.md").read_text() == "Changed a thing.\n"


def test_restore_publication_changelog_falls_back_to_scope_snapshot(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/restore_publication_changelog.py",
        "restore_publication_changelog_scope_fallback_test",
    )
    root_changelog = tmp_path / "changelog.d"
    publication_dir = tmp_path / ".github" / "publication_candidates"
    snapshot = publication_dir / "versioning-run" / "changelog.d"
    snapshot.mkdir(parents=True)
    (snapshot / "123.changed.md").write_text("Changed a thing.\n")
    scope_path = tmp_path / ".github" / "publication_scope.json"
    scope_path.write_text(
        json.dumps(
            {
                "base_release_version": "1.115.3",
                "candidate_scope": "1.115.3-patch",
                "release_bump": "patch",
                "run_id": "versioning-run",
            }
        )
    )
    monkeypatch.setenv("US_DATA_CANDIDATE_VERSION", "1.115.3-patch")
    monkeypatch.setenv("US_DATA_BASE_RELEASE_VERSION", "1.115.3")
    monkeypatch.setenv("US_DATA_RELEASE_BUMP", "patch")
    monkeypatch.setattr(module, "ROOT_CHANGELOG_DIR", root_changelog)
    monkeypatch.setattr(module, "PUBLICATION_SCOPE_PATH", scope_path)
    monkeypatch.setattr(module, "PUBLICATION_CANDIDATES_DIR", publication_dir)

    module.restore_candidate_changelog("pipeline-run")

    assert (root_changelog / "123.changed.md").read_text() == "Changed a thing.\n"


def test_restore_publication_changelog_rejects_unrelated_root_fragments(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/restore_publication_changelog.py",
        "restore_publication_changelog_conflict_script_test",
    )
    root_changelog = tmp_path / "changelog.d"
    root_changelog.mkdir()
    (root_changelog / "999.fixed.md").write_text("Unrelated fix.\n")
    snapshot = (
        tmp_path / ".github" / "publication_candidates" / "run-123" / "changelog.d"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "123.changed.md").write_text("Changed a thing.\n")
    monkeypatch.setattr(module, "ROOT_CHANGELOG_DIR", root_changelog)
    monkeypatch.setattr(
        module,
        "PUBLICATION_CANDIDATES_DIR",
        tmp_path / ".github" / "publication_candidates",
    )

    with pytest.raises(RuntimeError, match="do not match"):
        module.restore_candidate_changelog("run-123")


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


def test_resolve_run_context_prefers_run_scoped_publication_scope(
    tmp_path,
    monkeypatch,
):
    module = _load_script(
        ".github/scripts/resolve_run_context.py",
        "resolve_run_context_scoped_script_test",
    )
    _write_pyproject(tmp_path, "1.75.0")
    scope_dir = tmp_path / ".github"
    scoped_dir = scope_dir / "publication_candidates" / "run-123"
    scoped_dir.mkdir(parents=True)
    scope_dir.mkdir(exist_ok=True)
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
    (scoped_dir / "publication_scope.json").write_text(
        json.dumps(
            {
                "base_release_version": "1.75.0",
                "release_bump": "patch",
                "candidate_scope": "1.75.0-patch",
                "would_release_as_at_build_time": "1.75.1",
            }
        )
    )
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    env = {"US_DATA_RUN_ID": "run-123"}

    assert module._release_bump(env) == "patch"
    assert (
        module._candidate_version(
            env,
            base_release_version="1.75.0",
            release_bump="patch",
        )
        == "1.75.0-patch"
    )


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
    _write_pyproject(tmp_path, "9.9.9")
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


def test_promote_publication_script_fallback_release_uses_manifest_base(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(Function=types.SimpleNamespace()),
    )
    module = _load_script(
        ".github/scripts/promote_publication_pipeline.py",
        "promote_publication_pipeline_manifest_base_test",
    )
    _write_pyproject(tmp_path, "9.9.9")
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    context = module.RunContext.from_mapping(
        {"run_id": "run-123"},
        modal_app_name="app",
        modal_environment="main",
    )

    promoted_context = module._promotion_context_from_status(
        context,
        {
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
        },
    )

    assert promoted_context.release_version == "1.74.0"


def test_promote_publication_script_prefers_manifest_release_version(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(Function=types.SimpleNamespace()),
    )
    module = _load_script(
        ".github/scripts/promote_publication_pipeline.py",
        "promote_publication_pipeline_release_version_test",
    )
    _write_pyproject(tmp_path, "1.74.0")
    monkeypatch.setattr(module, "_REPO_ROOT", tmp_path)
    context = module.RunContext.from_mapping(
        {"run_id": "run-123"},
        modal_app_name="app",
        modal_environment="main",
    )

    promoted_context = module._promotion_context_from_status(
        context,
        {
            "run_manifest": {
                "run_id": "run-123",
                "candidate_version": "1.73.0-minor",
                "base_release_version": "1.73.0",
                "release_bump": "minor",
                "release_version": "1.74.0",
                "run_context": {
                    "run_id": "run-123",
                    "candidate_version": "1.73.0-minor",
                    "base_release_version": "1.73.0",
                    "release_bump": "minor",
                },
            }
        },
    )

    assert promoted_context.release_version == "1.74.0"


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
