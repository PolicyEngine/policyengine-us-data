"""Resolve run context for GitHub Actions workflows."""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policyengine_us_data.utils.run_context import (  # noqa: E402
    BASE_RELEASE_VERSION_ENV,
    CANDIDATE_SCOPE_ENV,
    CANDIDATE_VERSION_ENV,
    DEFAULT_MODAL_APP_PREFIX,
    DATA_PACKAGE_VERSION_ENV,
    RELEASE_BUMP_ENV,
    RELEASE_VERSION_ENV,
    RUN_ID_ENV,
    RunContext,
    build_candidate_scope,
    build_modal_resource_name,
    build_run_id,
    resolve_base_release_version,
    resolve_release_bump,
)


def _append_key_values(path_env: str, values: dict[str, str]) -> None:
    output_path = os.environ.get(path_env)
    if not output_path:
        return
    with Path(output_path).open("a") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def _github_actions_run_id(env: Mapping[str, str]) -> str:
    if not env.get("GITHUB_RUN_ID"):
        return ""
    return build_run_id(
        github_run_id=env.get("GITHUB_RUN_ID", ""),
        github_run_attempt=env.get("GITHUB_RUN_ATTEMPT", "1"),
        github_sha=env.get("GITHUB_SHA", ""),
    )


def _pyproject_version() -> str:
    pyproject_path = _REPO_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return ""
    with pyproject_path.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def _publication_scope() -> dict[str, str]:
    path = _REPO_ROOT / ".github" / "publication_scope.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _base_release_version(env: Mapping[str, str]) -> str:
    scope = _publication_scope()
    value = (
        env.get(BASE_RELEASE_VERSION_ENV)
        or env.get("BASE_RELEASE_VERSION", "")
        or scope.get("base_release_version", "")
    )
    if value:
        return resolve_base_release_version(value, env={})
    return ""


def _release_bump(env: Mapping[str, str]) -> str:
    scope = _publication_scope()
    value = (
        env.get(RELEASE_BUMP_ENV)
        or env.get("RELEASE_BUMP", "")
        or scope.get("release_bump", "")
    )
    if value:
        return resolve_release_bump(value, env={})
    return ""


def _candidate_version(
    env: Mapping[str, str],
    *,
    base_release_version: str = "",
    release_bump: str = "",
) -> str:
    scope = _publication_scope()
    version = (
        env.get(CANDIDATE_SCOPE_ENV)
        or env.get(CANDIDATE_VERSION_ENV)
        or env.get(DATA_PACKAGE_VERSION_ENV)
        or env.get("CANDIDATE_SCOPE", "")
        or env.get("CANDIDATE_VERSION", "")
        or scope.get("candidate_scope", "")
    )
    if version:
        return version
    if base_release_version and release_bump:
        return build_candidate_scope(base_release_version, release_bump)
    return _pyproject_version()


def _release_version(env: Mapping[str, str]) -> str:
    return env.get(RELEASE_VERSION_ENV) or env.get("RELEASE_VERSION", "")


def main() -> None:
    env = os.environ
    app_prefix = env.get("US_DATA_MODAL_APP_PREFIX", DEFAULT_MODAL_APP_PREFIX)
    run_id = env.get(RUN_ID_ENV, "")
    base_release_version = _base_release_version(env)
    release_bump = _release_bump(env)
    candidate_version = _candidate_version(
        env,
        base_release_version=base_release_version,
        release_bump=release_bump,
    )
    context = RunContext.from_env(
        run_id=run_id or _github_actions_run_id(env),
        candidate_version=candidate_version,
        release_version=_release_version(env),
        base_release_version=base_release_version,
        release_bump=release_bump,
        modal_app_prefix=app_prefix,
    )
    if not context.run_id:
        raise RuntimeError(
            "Could not resolve run ID. Set US_DATA_RUN_ID or run "
            "inside GitHub Actions with GITHUB_RUN_ID."
        )

    pipeline_volume_name = os.environ.get(
        "US_DATA_PIPELINE_VOLUME_NAME",
        build_modal_resource_name(
            context.run_id,
            prefix="pipeline-artifacts",
        ),
    )
    staging_volume_name = os.environ.get(
        "US_DATA_STAGING_VOLUME_NAME",
        build_modal_resource_name(
            context.run_id,
            prefix="local-area-staging",
        ),
    )
    checkpoint_volume_name = os.environ.get(
        "US_DATA_CHECKPOINT_VOLUME_NAME",
        build_modal_resource_name(
            context.run_id,
            prefix="data-build-checkpoints",
        ),
    )
    context = RunContext.from_mapping(
        {
            **context.to_dict(),
            "pipeline_volume_name": pipeline_volume_name,
            "staging_volume_name": staging_volume_name,
            "checkpoint_volume_name": checkpoint_volume_name,
        },
        modal_app_name=context.modal_app_name,
        modal_environment=context.modal_environment,
    )

    outputs = {
        "run_id": context.run_id,
        "modal_app_name": context.modal_app_name,
        "modal_environment": context.modal_environment,
        "hf_staging_prefix": context.hf_staging_prefix,
        "candidate_version": context.candidate_version,
        "release_version": context.release_version,
        "base_release_version": context.base_release_version,
        "release_bump": context.release_bump,
        "github_run_url": context.github_run_url,
        "pipeline_volume_name": context.pipeline_volume_name,
        "staging_volume_name": context.staging_volume_name,
        "checkpoint_volume_name": context.checkpoint_volume_name,
    }
    _append_key_values("GITHUB_OUTPUT", outputs)
    _append_key_values("GITHUB_ENV", context.export_env())
    print(context.to_json())


if __name__ == "__main__":
    main()
