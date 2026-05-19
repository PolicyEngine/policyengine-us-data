"""Stable Modal app for discovering deployed US data pipeline runs."""

from __future__ import annotations

import asyncio
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
from modal_app.pipeline_discovery_core import (  # noqa: E402
    DeployedPipelineRunsPayloadDict,
    ModalAppRecord,
    RawRecord,
    build_deployed_pipeline_runs_payload,
)
from policyengine_us_data.utils.run_context import (  # noqa: E402
    DEFAULT_MODAL_ENVIRONMENT,
)


DISCOVERY_APP_NAME = "policyengine-us-data-pipeline-status"
modal_token_secret = modal.Secret.from_name("modal-token")

app = modal.App(
    os.environ.get("US_DATA_PIPELINE_DISCOVERY_APP_NAME") or DISCOVERY_APP_NAME
)
status_image = image.pip_install("fastapi")


def _modal_environment(explicit: str = "") -> str:
    return (
        explicit
        or os.environ.get("US_DATA_MODAL_ENVIRONMENT")
        or os.environ.get("MODAL_ENVIRONMENT")
        or DEFAULT_MODAL_ENVIRONMENT
    )


def _modal_state_name(api_pb2, state: int) -> str:
    try:
        return api_pb2.AppState.Name(state).removeprefix("APP_STATE_").lower()
    except ValueError:
        return str(state)


async def _list_modal_app_records_async(environment_name: str) -> list[ModalAppRecord]:
    from modal.client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()
    resp = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=environment_name)
    )
    return [
        {
            "app_id": item.app_id,
            "name": item.name or item.description,
            "description": item.description,
            "state": _modal_state_name(api_pb2, item.state),
            "tasks": item.n_running_tasks,
            "created_at": item.created_at,
            "stopped_at": item.stopped_at,
        }
        for item in resp.apps
    ]


def _list_modal_app_records(environment_name: str) -> list[ModalAppRecord]:
    return asyncio.run(_list_modal_app_records_async(environment_name))


def _pipeline_status_lookup(
    *,
    app_name: str,
    run_id: str,
    environment_name: str,
) -> RawRecord:
    status_fn = modal.Function.from_name(
        app_name,
        "get_pipeline_status",
        environment_name=environment_name,
    )
    payload = status_fn.remote(run_id)
    return payload if isinstance(payload, dict) else {}


def _build_deployed_pipeline_runs(
    *,
    limit: int | str | None,
    status: str,
    branch: str,
    include_unreachable: bool,
    modal_environment: str,
) -> DeployedPipelineRunsPayloadDict:
    environment_name = _modal_environment(modal_environment)
    app_records = _list_modal_app_records(environment_name)
    payload = build_deployed_pipeline_runs_payload(
        app_records,
        lambda candidate: _pipeline_status_lookup(
            app_name=candidate.app_name,
            run_id=candidate.run_id,
            environment_name=environment_name,
        ),
        limit=limit,
        status=status,
        branch=branch,
        include_unreachable=include_unreachable,
        modal_environment=environment_name,
    )
    return payload.to_dict()


@app.function(image=image, timeout=180, secrets=[modal_token_secret])
def list_deployed_pipeline_runs(
    limit: int = 25,
    status: str = "",
    branch: str = "",
    include_unreachable: bool = True,
    modal_environment: str = "",
) -> DeployedPipelineRunsPayloadDict:
    """Return deployed publication pipeline runs discovered from Modal app names."""

    return _build_deployed_pipeline_runs(
        limit=limit,
        status=status,
        branch=branch,
        include_unreachable=include_unreachable,
        modal_environment=modal_environment,
    )


@app.function(image=status_image, timeout=180, secrets=[modal_token_secret])
@modal.fastapi_endpoint(
    method="GET",
    docs=False,
    requires_proxy_auth=True,
)
def deployed_pipeline_runs_endpoint(
    limit: int = 25,
    status: str = "",
    branch: str = "",
    include_unreachable: bool = True,
    modal_environment: str = "",
) -> DeployedPipelineRunsPayloadDict:
    """Protected HTTP endpoint for deployed publication pipeline discovery."""

    return _build_deployed_pipeline_runs(
        limit=limit,
        status=status,
        branch=branch,
        include_unreachable=include_unreachable,
        modal_environment=modal_environment,
    )
