"""Modal status functions for deployed pipeline runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_baked = "/root/policyengine-us-data"
_local = str(Path(__file__).resolve().parent.parent)
for _p in (_baked, _local):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_app.images import cpu_image as image  # noqa: E402
from modal_app.step_manifests.state import PIPELINE_MOUNT, RUNS_DIR  # noqa: E402
from modal_app.step_manifests.status import build_pipeline_status_payload  # noqa: E402

app = modal.App(
    os.environ.get("US_DATA_PIPELINE_STATUS_APP_NAME")
    or os.environ.get("US_DATA_MODAL_APP_NAME")
    or "policyengine-us-data-pipeline-status"
)

pipeline_volume = modal.Volume.from_name(
    os.environ.get("US_DATA_PIPELINE_VOLUME_NAME", "pipeline-artifacts"),
    create_if_missing=True,
    version=2,
)
status_image = image.pip_install("fastapi")


@app.function(
    image=image,
    timeout=60,
    volumes={PIPELINE_MOUNT: pipeline_volume},
)
def get_pipeline_status(
    run_id: str,
) -> dict:
    """Get structured JSON status for a pipeline run."""

    pipeline_volume.reload()
    return build_pipeline_status_payload(run_id)


@app.function(
    image=status_image,
    timeout=60,
    volumes={PIPELINE_MOUNT: pipeline_volume},
)
@modal.fastapi_endpoint(
    method="GET",
    docs=False,
    requires_proxy_auth=True,
)
def pipeline_status_endpoint(
    run_id: str,
) -> dict:
    """Protected HTTP endpoint for structured pipeline status."""

    pipeline_volume.reload()
    return build_pipeline_status_payload(run_id)


@app.function(
    image=image,
    timeout=60,
    volumes={PIPELINE_MOUNT: pipeline_volume},
)
def pipeline_status_snippet(
    run_id: str = None,
) -> str:
    """Get human-readable pipeline status."""

    pipeline_volume.reload()
    runs_dir = Path(RUNS_DIR)

    if not runs_dir.exists():
        return "No pipeline runs found."

    if run_id:
        payload = build_pipeline_status_payload(run_id)
        if payload["status"] == "not_found":
            return payload["message"]
        run_manifest = payload["run_manifest"]
        lines = [
            f"Run: {payload['run_id']}",
            f"  Branch:  {run_manifest['branch']}",
            f"  SHA:     {run_manifest['sha'][:12]}",
            f"  Version: {run_manifest['version']}",
            f"  Status:  {payload['status']}",
            f"  Started: {run_manifest['started_at']}",
        ]
        if payload["error"]:
            error = payload["error"]
            lines.append(
                f"  Error:   {error['error_type']}: {error.get('message', '')[:200]}"
            )
            if error.get("record_path"):
                lines.append(f"  Error record: {error['record_path']}")
        if payload["stage_manifests"]:
            lines.append("  Step manifests:")
            for item in payload["stage_manifests"]:
                manifest = item["manifest"]
                duration = (
                    manifest["duration_s"]
                    if manifest.get("duration_s") is not None
                    else "?"
                )
                reuse = manifest.get("reuse_decision", "not_applicable")
                lines.append(
                    f"    {manifest['step_id']}: {duration}s "
                    f"({manifest['status']}, {reuse})"
                )
        return "\n".join(lines)

    runs = []
    for entry in sorted(runs_dir.iterdir()):
        manifest_path = entry / "run_manifest.json"
        if manifest_path.exists():
            data = build_pipeline_status_payload(entry.name)
            run_manifest = data["run_manifest"]
            runs.append(
                f"  {data['run_id']}: "
                f"{data['status']} "
                f"(branch={run_manifest['branch']}, "
                f"v={run_manifest['version']})"
            )

    if not runs:
        return "No pipeline runs found."

    return "Pipeline runs:\n" + "\n".join(runs)
