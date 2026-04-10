from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as base_image  # noqa: E402

app = modal.App("policyengine-us-data-long-term-probe")

hf_secret = modal.Secret.from_name("huggingface-token")

image = base_image

_LONG_TERM_DIR = "/root/policyengine-us-data/policyengine_us_data/datasets/cps/long_term"
_VENV_PYTHON = "/root/policyengine-us-data/.venv/bin/python"


def _run_long_term_json_command(script_name: str, *args: str) -> str:
    command = [_VENV_PYTHON, f"{_LONG_TERM_DIR}/{script_name}", *args]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": _LONG_TERM_DIR,
        },
    )
    return result.stdout


@app.function(
    image=image,
    timeout=60 * 60,
    cpu=8,
    memory=65536,
    secrets=[hf_secret],
)
def assess_publishable_probe_json(
    *,
    years_csv: str = "2075",
    profile: str = "ss-payroll-tob",
    target_source: str = "trustees_2025_current_law",
    base_dataset_path: str = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
) -> str:
    return _run_long_term_json_command(
        "assess_publishable_horizon.py",
        "--profile",
        profile,
        "--target-source",
        target_source,
        "--years",
        years_csv,
        "--base-dataset",
        base_dataset_path,
    )


@app.function(
    image=image,
    timeout=60 * 60,
    cpu=8,
    memory=65536,
    secrets=[hf_secret],
)
def assess_augmented_publishable_probe_json(
    *,
    years_csv: str = "2075",
    profile: str = "ss-payroll-tob",
    target_source: str = "trustees_2025_current_law",
    support_augmentation_profile: str = "late-clone-v1",
    base_dataset_path: str = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
) -> str:
    return _run_long_term_json_command(
        "assess_augmented_publishability.py",
        "--profile",
        profile,
        "--target-source",
        target_source,
        "--years",
        years_csv,
        "--base-dataset",
        base_dataset_path,
        "--support-augmentation",
        support_augmentation_profile,
    )


@app.local_entrypoint()
def main(
    years: str = "2075",
    profile: str = "ss-payroll-tob",
    target_source: str = "trustees_2025_current_law",
    support_augmentation_profile: str = "",
) -> None:
    if support_augmentation_profile:
        payload = assess_augmented_publishable_probe_json.remote(
            years_csv=years,
            profile=profile,
            target_source=target_source,
            support_augmentation_profile=support_augmentation_profile,
        )
    else:
        payload = assess_publishable_probe_json.remote(
            years_csv=years,
            profile=profile,
            target_source=target_source,
        )
    print(payload)
