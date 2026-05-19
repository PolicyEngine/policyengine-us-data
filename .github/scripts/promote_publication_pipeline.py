"""Promote a completed run-scoped US data pipeline from GitHub Actions."""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policyengine_us_data.utils.run_context import (  # noqa: E402
    RunContext,
    release_version_from_bump,
    stable_release_version,
)


def _current_package_version() -> str:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        return stable_release_version(tomllib.load(file)["project"]["version"])


def _modal_function(app_name: str, function_name: str, environment_name: str):
    if environment_name:
        return modal.Function.from_name(
            app_name,
            function_name,
            environment_name=environment_name,
        )
    return modal.Function.from_name(app_name, function_name)


def _manifest_field(manifest: dict, key: str) -> str:
    value = manifest.get(key)
    if value:
        return str(value)
    run_context = manifest.get("run_context") or {}
    value = run_context.get(key)
    return str(value) if value else ""


def _promotion_context_from_status(context: RunContext, status: dict) -> RunContext:
    manifest = status.get("run_manifest") or {}
    if not manifest:
        raise RuntimeError(
            "Could not read run_manifest from pipeline status. "
            "The run must have a completed run manifest before promotion."
        )
    candidate_version = _manifest_field(manifest, "candidate_version")
    release_bump = _manifest_field(manifest, "release_bump")
    base_release_version = _manifest_field(manifest, "base_release_version")
    if not candidate_version:
        raise RuntimeError("Run manifest is missing candidate_version.")
    if not release_bump:
        raise RuntimeError("Run manifest is missing release_bump.")
    release_version = _manifest_field(manifest, "release_version")
    if not release_version:
        release_version = release_version_from_bump(
            _current_package_version(),
            release_bump,
        )
    return RunContext.from_mapping(
        manifest.get("run_context"),
        run_id=context.run_id,
        modal_app_name=context.modal_app_name,
        modal_environment=context.modal_environment,
        candidate_version=candidate_version,
        release_version=release_version,
        base_release_version=base_release_version,
        release_bump=release_bump,
    )


def _append_env(context: RunContext) -> None:
    env_path = os.environ.get("GITHUB_ENV")
    if not env_path:
        return
    values = {
        **context.export_env(),
        "CANDIDATE_VERSION": context.candidate_version,
        "RELEASE_VERSION": context.release_version,
        "BASE_RELEASE_VERSION": context.base_release_version,
        "RELEASE_BUMP": context.release_bump,
    }
    with Path(env_path).open("a") as handle:
        for key, value in values.items():
            if value:
                handle.write(f"{key}={value}\n")


def _append_summary(result: str, context: RunContext) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with Path(summary_path).open("a") as handle:
        handle.write("## Publication Promoted\n\n")
        handle.write("| Field | Value |\n")
        handle.write("|-------|-------|\n")
        handle.write(f"| Run ID | `{context.run_id}` |\n")
        handle.write(f"| Candidate scope | `{context.candidate_version}` |\n")
        handle.write(f"| Release version | `{context.release_version}` |\n")
        handle.write(f"| Modal app | `{context.modal_app_name}` |\n")
        handle.write(f"| Modal environment | `{context.modal_environment}` |\n")
        handle.write(f"| HF staging | `{context.hf_staging_prefix}` |\n")
        handle.write("\n")
        handle.write("```text\n")
        handle.write(result)
        handle.write("\n```\n")


def main() -> None:
    context = RunContext.from_env()
    if not context.run_id:
        raise RuntimeError("US_DATA_RUN_ID is required to promote a publication run.")

    app_name = context.modal_app_name or "policyengine-us-data-pipeline"
    environment_name = context.modal_environment or os.environ.get("MODAL_ENVIRONMENT")
    get_pipeline_status = _modal_function(
        app_name,
        "get_pipeline_status",
        environment_name,
    )
    status = get_pipeline_status.remote(context.run_id)
    context = _promotion_context_from_status(context, status)
    _append_env(context)
    promote_run = _modal_function(app_name, "promote_run", environment_name)

    kwargs = {
        "run_id": context.run_id,
        "candidate_version": context.candidate_version,
        "release_version": context.release_version,
    }

    print("Promoting publication run.")
    print(f"Run ID: {context.run_id}")
    print(f"Candidate scope: {context.candidate_version}")
    print(f"Base release version: {context.base_release_version}")
    print(f"Release bump: {context.release_bump}")
    print(f"Release version: {context.release_version}")
    print(f"Modal app: {app_name}")
    print(f"Modal environment: {environment_name}")
    print(f"HF staging prefix: {context.hf_staging_prefix}")
    print(f"Request: {json.dumps(kwargs, sort_keys=True)}")
    result = promote_run.remote(**kwargs)
    print(result)
    _append_summary(result, context)


if __name__ == "__main__":
    main()
