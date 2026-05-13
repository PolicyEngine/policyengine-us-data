"""Promote a completed run-scoped US data pipeline from GitHub Actions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policyengine_us_data.utils.run_context import RunContext  # noqa: E402


def _append_summary(result: str, context: RunContext) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with Path(summary_path).open("a") as handle:
        handle.write("## Publication Promoted\n\n")
        handle.write("| Field | Value |\n")
        handle.write("|-------|-------|\n")
        handle.write(f"| Run ID | `{context.run_id}` |\n")
        handle.write(f"| Candidate version | `{context.candidate_version}` |\n")
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
    if environment_name:
        promote_run = modal.Function.from_name(
            app_name,
            "promote_run",
            environment_name=environment_name,
        )
    else:
        promote_run = modal.Function.from_name(app_name, "promote_run")

    kwargs = {"run_id": context.run_id}
    if os.environ.get("CANDIDATE_VERSION"):
        kwargs["candidate_version"] = context.candidate_version
    if os.environ.get("RELEASE_VERSION"):
        kwargs["release_version"] = context.release_version

    print("Promoting publication run.")
    print(f"Run ID: {context.run_id}")
    print(f"Candidate version: {context.candidate_version}")
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
