"""Modal entrypoint for production long-run projection builds.

The GitHub Actions wrapper is useful as an orchestrator, but hosted runners
do not have enough memory for the production projection path. This module runs
the same package script on a large Modal worker, streams logs, stores outputs on
a Modal volume, and optionally uploads the resulting artifacts to HF staging.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as image  # noqa: E402
from policyengine_us_data.utils.run_context import (  # noqa: E402
    sanitize_run_id,
    staging_prefix,
)

app = modal.App(
    os.environ.get("US_DATA_LONG_TERM_MODAL_APP_NAME")
    or os.environ.get("US_DATA_MODAL_APP_NAME")
    or "policyengine-us-data-long-term"
)

hf_secret = modal.Secret.from_name("huggingface-token")
_OUTPUT_VOLUME_NAME = os.environ.get(
    "US_DATA_LONG_TERM_VOLUME",
    "policyengine-us-data-long-term",
)
output_volume = modal.Volume.from_name(
    _OUTPUT_VOLUME_NAME,
    create_if_missing=True,
)

_LONG_TERM_DIR = (
    "/root/policyengine-us-data/policyengine_us_data/datasets/cps/long_term"
)
_REPO_ROOT = "/root/policyengine-us-data"
_LONG_TERM_PRODUCTION_MODULE = (
    "policyengine_us_data.datasets.cps.long_term.run_long_term_production"
)
_OUTPUT_MOUNT = Path("/outputs")


def _local_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_local,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _local_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_local,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _validate_local_source(source_sha: str, *, allow_dirty_source: bool) -> None:
    if allow_dirty_source:
        return
    if _local_git_dirty():
        raise ValueError(
            "The local policyengine-us-data checkout has uncommitted changes. "
            "Commit and pass its SHA with --source-sha before running Modal, or "
            "rerun with --allow-dirty-source for an explicitly non-publishable "
            "experiment."
        )
    local_sha = _local_git_sha()
    if not local_sha:
        raise ValueError(
            "Could not resolve the local policyengine-us-data git SHA; pass "
            "--allow-dirty-source only for an explicitly non-publishable experiment."
        )
    if local_sha != source_sha:
        raise ValueError(
            "The requested source_sha does not match the local checkout that Modal "
            f"will package: {source_sha} != {local_sha}. Check out the exact "
            "source SHA before running production."
        )


def _append_optional_value(
    command: list[str],
    flag: str,
    value: str | int | float | None,
) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def _build_command(
    *,
    years: str,
    jobs: int,
    output_dir: Path,
    profile: str,
    target_source: str,
    tax_assumption: str,
    run_id: str,
    source_sha: str,
    upload_to_hf_staging: bool,
    base_dataset: str,
    allow_validation_failures: bool,
    keep_temp: bool,
    support_augmentation_profile: str,
    support_augmentation_target_year: int | None,
    support_augmentation_align_to_run_year: bool,
    support_augmentation_start_year: int | None,
    support_augmentation_top_n_targets: int | None,
    support_augmentation_donors_per_target: int | None,
    support_augmentation_max_distance: float | None,
    support_augmentation_clone_weight_scale: float | None,
    support_augmentation_blueprint_base_weight_scale: float | None,
    support_augmentation_sanitize_worker_non_target_income: bool,
    support_augmentation_sanitize_clone_non_target_income: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        _LONG_TERM_PRODUCTION_MODULE,
        "--years",
        years,
        "--jobs",
        str(jobs),
        "--output-dir",
        str(output_dir),
        "--profile",
        profile,
        "--target-source",
        target_source,
        "--tax-assumption",
        tax_assumption,
        "--run-id",
        run_id,
        "--source-sha",
        source_sha,
    ]
    _append_optional_value(command, "--base-dataset", base_dataset)
    _append_optional_value(
        command,
        "--support-augmentation-profile",
        support_augmentation_profile,
    )
    _append_optional_value(
        command,
        "--support-augmentation-target-year",
        support_augmentation_target_year,
    )
    if support_augmentation_align_to_run_year:
        command.append("--support-augmentation-align-to-run-year")
    _append_optional_value(
        command,
        "--support-augmentation-start-year",
        support_augmentation_start_year,
    )
    _append_optional_value(
        command,
        "--support-augmentation-top-n-targets",
        support_augmentation_top_n_targets,
    )
    _append_optional_value(
        command,
        "--support-augmentation-donors-per-target",
        support_augmentation_donors_per_target,
    )
    _append_optional_value(
        command,
        "--support-augmentation-max-distance",
        support_augmentation_max_distance,
    )
    _append_optional_value(
        command,
        "--support-augmentation-clone-weight-scale",
        support_augmentation_clone_weight_scale,
    )
    _append_optional_value(
        command,
        "--support-augmentation-blueprint-base-weight-scale",
        support_augmentation_blueprint_base_weight_scale,
    )
    if support_augmentation_sanitize_worker_non_target_income:
        command.append("--support-augmentation-sanitize-worker-non-target-income")
    if support_augmentation_sanitize_clone_non_target_income:
        command.append("--support-augmentation-sanitize-clone-non-target-income")
    if allow_validation_failures:
        command.append("--allow-validation-failures")
    if keep_temp:
        command.append("--keep-temp")
    if upload_to_hf_staging:
        command.append("--upload-to-hf-staging")
    return command


def _stream_command(command: list[str], env: dict[str, str]) -> None:
    printable = " ".join(command)
    print(f"Running long-run projection command:\n{printable}", flush=True)
    process = subprocess.Popen(
        command,
        cwd="/root/policyengine-us-data",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _output_files(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    )


def _commit_output_volume(*, suppress_errors: bool = False) -> None:
    try:
        output_volume.commit()
    except Exception as exc:
        if suppress_errors:
            print(
                "WARNING: failed to commit long-run Modal output volume after "
                f"subprocess failure: {exc}",
                flush=True,
            )
            return
        raise


@app.function(
    image=image,
    timeout=12 * 60 * 60,
    cpu=16,
    memory=131072,
    volumes={str(_OUTPUT_MOUNT): output_volume},
    secrets=[hf_secret],
)
def build_long_term_projection(
    *,
    years: str,
    run_id: str,
    source_sha: str,
    jobs: int = 1,
    profile: str = "ss-payroll-tob",
    target_source: str = "trustees_2025_current_law",
    tax_assumption: str = "trustees-2025-core-thresholds-v1",
    base_dataset: str = "",
    upload_to_hf_staging: bool = False,
    allow_validation_failures: bool = False,
    keep_temp: bool = False,
    clear_output: bool = True,
    support_augmentation_profile: str = "",
    support_augmentation_target_year: int | None = None,
    support_augmentation_align_to_run_year: bool = False,
    support_augmentation_start_year: int | None = None,
    support_augmentation_top_n_targets: int | None = None,
    support_augmentation_donors_per_target: int | None = None,
    support_augmentation_max_distance: float | None = None,
    support_augmentation_clone_weight_scale: float | None = None,
    support_augmentation_blueprint_base_weight_scale: float | None = None,
    support_augmentation_sanitize_worker_non_target_income: bool = False,
    support_augmentation_sanitize_clone_non_target_income: bool = False,
) -> dict[str, object]:
    if not run_id:
        raise ValueError("run_id is required.")
    if not source_sha:
        raise ValueError("source_sha is required.")

    run_id = sanitize_run_id(run_id)
    output_dir = _OUTPUT_MOUNT / run_id
    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = _build_command(
        years=years,
        jobs=jobs,
        output_dir=output_dir,
        profile=profile,
        target_source=target_source,
        tax_assumption=tax_assumption,
        run_id=run_id,
        source_sha=source_sha,
        upload_to_hf_staging=upload_to_hf_staging,
        base_dataset=base_dataset,
        allow_validation_failures=allow_validation_failures,
        keep_temp=keep_temp,
        support_augmentation_profile=support_augmentation_profile,
        support_augmentation_target_year=support_augmentation_target_year,
        support_augmentation_align_to_run_year=(support_augmentation_align_to_run_year),
        support_augmentation_start_year=support_augmentation_start_year,
        support_augmentation_top_n_targets=support_augmentation_top_n_targets,
        support_augmentation_donors_per_target=(support_augmentation_donors_per_target),
        support_augmentation_max_distance=support_augmentation_max_distance,
        support_augmentation_clone_weight_scale=(
            support_augmentation_clone_weight_scale
        ),
        support_augmentation_blueprint_base_weight_scale=(
            support_augmentation_blueprint_base_weight_scale
        ),
        support_augmentation_sanitize_worker_non_target_income=(
            support_augmentation_sanitize_worker_non_target_income
        ),
        support_augmentation_sanitize_clone_non_target_income=(
            support_augmentation_sanitize_clone_non_target_income
        ),
    )
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": (
            f"{_REPO_ROOT}:{_LONG_TERM_DIR}:{os.environ.get('PYTHONPATH', '')}"
        ),
        "US_DATA_RUN_ID": run_id,
        "GITHUB_REPOSITORY": "PolicyEngine/policyengine-us-data",
        "GITHUB_SHA": source_sha,
        "US_DATA_GITHUB_RUN_URL": os.environ.get("US_DATA_GITHUB_RUN_URL", ""),
    }
    try:
        _stream_command(command, env)
    finally:
        _commit_output_volume(suppress_errors=sys.exc_info()[0] is not None)
    return {
        "run_id": run_id,
        "source_sha": source_sha,
        "output_dir": str(output_dir),
        "files": _output_files(output_dir),
        "hf_staging_prefix": (
            f"{staging_prefix(run_id, version=source_sha)}/long_term"
            if upload_to_hf_staging
            else None
        ),
    }


@app.local_entrypoint()
def main(
    years: str = "2026",
    run_id: str = "long-term-projection",
    source_sha: str = "",
    jobs: int = 1,
    profile: str = "ss-payroll-tob",
    target_source: str = "trustees_2025_current_law",
    tax_assumption: str = "trustees-2025-core-thresholds-v1",
    base_dataset: str = "",
    upload_to_hf_staging: bool = False,
    allow_validation_failures: bool = False,
    keep_temp: bool = False,
    clear_output: bool = True,
    support_augmentation_profile: str = "",
    support_augmentation_target_year: int | None = None,
    support_augmentation_align_to_run_year: bool = False,
    support_augmentation_start_year: int | None = None,
    support_augmentation_top_n_targets: int | None = None,
    support_augmentation_donors_per_target: int | None = None,
    support_augmentation_max_distance: float | None = None,
    support_augmentation_clone_weight_scale: float | None = None,
    support_augmentation_blueprint_base_weight_scale: float | None = None,
    support_augmentation_sanitize_worker_non_target_income: bool = False,
    support_augmentation_sanitize_clone_non_target_income: bool = False,
    spawn: bool = False,
    allow_dirty_source: bool = False,
) -> None:
    if not source_sha:
        source_sha = os.environ.get("GITHUB_SHA", "") or _local_git_sha()
    if not source_sha:
        raise ValueError("source_sha is required; pass --source-sha.")
    _validate_local_source(
        source_sha,
        allow_dirty_source=allow_dirty_source,
    )
    run_id = sanitize_run_id(run_id)
    kwargs = {
        "years": years,
        "run_id": run_id,
        "source_sha": source_sha,
        "jobs": jobs,
        "profile": profile,
        "target_source": target_source,
        "tax_assumption": tax_assumption,
        "base_dataset": base_dataset,
        "upload_to_hf_staging": upload_to_hf_staging,
        "allow_validation_failures": allow_validation_failures,
        "keep_temp": keep_temp,
        "clear_output": clear_output,
        "support_augmentation_profile": support_augmentation_profile,
        "support_augmentation_target_year": support_augmentation_target_year,
        "support_augmentation_align_to_run_year": (
            support_augmentation_align_to_run_year
        ),
        "support_augmentation_start_year": support_augmentation_start_year,
        "support_augmentation_top_n_targets": support_augmentation_top_n_targets,
        "support_augmentation_donors_per_target": (
            support_augmentation_donors_per_target
        ),
        "support_augmentation_max_distance": support_augmentation_max_distance,
        "support_augmentation_clone_weight_scale": (
            support_augmentation_clone_weight_scale
        ),
        "support_augmentation_blueprint_base_weight_scale": (
            support_augmentation_blueprint_base_weight_scale
        ),
        "support_augmentation_sanitize_worker_non_target_income": (
            support_augmentation_sanitize_worker_non_target_income
        ),
        "support_augmentation_sanitize_clone_non_target_income": (
            support_augmentation_sanitize_clone_non_target_income
        ),
    }
    if spawn:
        call = build_long_term_projection.spawn(**kwargs)
        payload = {
            "function_call_id": call.object_id,
            "run_id": run_id,
            "source_sha": source_sha,
            "modal_volume": _OUTPUT_VOLUME_NAME,
            "volume_output_prefix": run_id,
        }
    else:
        payload = build_long_term_projection.remote(**kwargs)
    print(json.dumps(payload, indent=2, sort_keys=True))
