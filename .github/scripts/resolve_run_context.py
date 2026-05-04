"""Resolve run context for GitHub Actions workflows."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policyengine_us_data.utils.run_context import (  # noqa: E402
    DEFAULT_MODAL_APP_PREFIX,
    RUN_ID_ENV,
    RunContext,
    build_modal_resource_name,
    build_run_id,
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


def main() -> None:
    env = os.environ
    app_prefix = env.get("US_DATA_MODAL_APP_PREFIX", DEFAULT_MODAL_APP_PREFIX)
    run_id = env.get(RUN_ID_ENV, "") or env.get("RUN_ID", "")
    context = RunContext.from_env(
        run_id=run_id or _github_actions_run_id(env),
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
