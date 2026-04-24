import os
from pathlib import Path

import modal


def _as_bool(value: str) -> bool:
    return value.lower() == "true"


def _append_summary(function_call_id: str) -> None:
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
        handle.write(f"| Function call ID | `{function_call_id}` |\n\n")
        handle.write("**[Monitor on Modal Dashboard](https://modal.com/apps)**\n")


def main() -> None:
    app_name = os.environ.get("MODAL_APP_NAME", "policyengine-us-data-pipeline")
    environment_name = os.environ.get("MODAL_ENVIRONMENT")
    kwargs = {
        "branch": os.environ.get("PIPELINE_BRANCH", "main"),
        "gpu": os.environ["GPU"],
        "epochs": int(os.environ["EPOCHS"]),
        "national_epochs": int(os.environ["NATIONAL_EPOCHS"]),
        "num_workers": int(os.environ["NUM_WORKERS"]),
        "skip_national": _as_bool(os.environ["SKIP_NATIONAL"]),
        "resume_run_id": os.environ.get("RESUME_RUN_ID") or None,
        "version_override": os.environ.get("VERSION_OVERRIDE", ""),
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
    print(f"Function call ID: {function_call.object_id}")
    _append_summary(function_call.object_id)


if __name__ == "__main__":
    main()
