from __future__ import annotations

import json
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


@app.function(
    image=image,
    timeout=60 * 60,
    cpu=8,
    memory=32768,
    secrets=[hf_secret],
)
def assess_publishable_probe_json(
    *,
    years_csv: str = "2075",
    profile: str = "ss-payroll-tob",
    target_source: str = "trustees_2025_current_law",
    base_dataset_path: str = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
) -> str:
    from policyengine_us_data.datasets.cps.long_term.assess_publishable_horizon import (
        assess_years,
    )

    years = [int(value.strip()) for value in years_csv.split(",") if value.strip()]
    rows = assess_years(
        years=years,
        profile_name=profile,
        target_source=target_source,
        base_dataset_path=base_dataset_path,
    )
    return json.dumps(rows, indent=2, sort_keys=True)


@app.local_entrypoint()
def main(
    years: str = "2075",
    profile: str = "ss-payroll-tob",
    target_source: str = "trustees_2025_current_law",
) -> None:
    payload = assess_publishable_probe_json.remote(
        years_csv=years,
        profile=profile,
        target_source=target_source,
    )
    print(payload)
