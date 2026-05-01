import os
import sys
from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policyengine_us_data.utils.publication_context import PublicationContext  # noqa: E402


def _as_bool(value: str) -> bool:
    return value.lower() == "true"


def _append_summary(function_call_id: str, context: PublicationContext) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with Path(summary_path).open("a") as handle:
        handle.write("## Pipeline Launched\n\n")
        handle.write("| Field | Value |\n")
        handle.write("|-------|-------|\n")
        handle.write(f"| GPU | `{os.environ['GPU']}` |\n")
        handle.write(
            "| Epochs | "
            f"`{os.environ['EPOCHS']}` / "
            f"`{os.environ['NATIONAL_EPOCHS']}` |\n"
        )
        handle.write(f"| Publication ID | `{context.publication_id}` |\n")
        handle.write(f"| Modal app | `{context.modal_app_name}` |\n")
        handle.write(f"| Modal environment | `{context.modal_environment}` |\n")
        handle.write(f"| HF staging | `{context.hf_staging_prefix}` |\n")
        handle.write(f"| Function call ID | `{function_call_id}` |\n\n")
        handle.write("**[Monitor on Modal Dashboard](https://modal.com/apps)**\n")


def main() -> None:
    context = PublicationContext.from_env()
    app_name = context.modal_app_name or "policyengine-us-data-pipeline"
    environment_name = context.modal_environment or os.environ.get("MODAL_ENVIRONMENT")
    kwargs = {
        "branch": os.environ.get("PIPELINE_BRANCH", "main"),
        "gpu": os.environ["GPU"],
        "epochs": int(os.environ["EPOCHS"]),
        "national_epochs": int(os.environ["NATIONAL_EPOCHS"]),
        "num_workers": int(os.environ["NUM_WORKERS"]),
        "skip_national": _as_bool(os.environ["SKIP_NATIONAL"]),
        "resume_run_id": os.environ.get("RESUME_RUN_ID") or None,
        "version_override": os.environ.get("VERSION_OVERRIDE", ""),
        "publication_id": context.publication_id,
        "publication_context": context.to_dict(),
        "modal_app_name": context.modal_app_name,
        "modal_environment": context.modal_environment,
    }
    if environment_name:
        run_pipeline = modal.Function.from_name(
            app_name,
            "run_pipeline",
            environment_name=environment_name,
        )
    else:
        run_pipeline = modal.Function.from_name(app_name, "run_pipeline")
    function_call = run_pipeline.spawn(**kwargs)
    print("Pipeline spawned.")
    print(f"Publication ID: {context.publication_id}")
    print(f"Modal app: {app_name}")
    print(f"Modal environment: {environment_name}")
    print(f"HF staging prefix: {context.hf_staging_prefix}")
    print(f"Function call ID: {function_call.object_id}")
    _append_summary(function_call.object_id, context)


if __name__ == "__main__":
    main()
