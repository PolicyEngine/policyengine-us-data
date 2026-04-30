"""Deployed Modal E2E test for the tiny Stage 1-5 to H5 handoff."""

from __future__ import annotations

import os
import uuid

import pytest

modal = pytest.importorskip("modal")

PIPELINE_APP_NAME = os.environ.get(
    "MODAL_APP_NAME",
    "policyengine-us-data-pipeline",
)
LOCAL_AREA_APP_NAME = os.environ.get(
    "MODAL_LOCAL_AREA_APP_NAME",
    "policyengine-us-data-local-area",
)
HARNESS_APP_NAME = os.environ.get(
    "MODAL_H5_TEST_HARNESS_APP_NAME",
    "policyengine-us-data-h5-test-harness",
)
MODAL_ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT", "main")

pytestmark = pytest.mark.integration


def _require_modal_tokens() -> None:
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        pytest.skip("Modal credentials are required for deployed pipeline E2E tests")


def _function(app_name: str, function_name: str):
    return modal.Function.from_name(
        app_name,
        function_name,
        environment_name=MODAL_ENVIRONMENT,
    )


def _run_id(label: str) -> str:
    return f"0.0.0_{label}_{uuid.uuid4().hex[:10]}"


def test_deployed_modal_pipeline_accepts_tiny_stage_1_to_5_artifact_shape():
    _require_modal_tokens()

    run_id = _run_id("stage15-handoff")
    seed = _function(HARNESS_APP_NAME, "seed_tiny_pipeline_case")
    preflight = _function(HARNESS_APP_NAME, "preflight_h5_case")
    inspect = _function(HARNESS_APP_NAME, "inspect_h5_outputs")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")
    publish = _function(PIPELINE_APP_NAME, "run_seeded_h5_publish_seam")
    validate = _function(LOCAL_AREA_APP_NAME, "validate_staging")

    try:
        seeded = seed.remote(run_id, "saved_geography_success")
        preflight_result = preflight.remote(run_id, n_clones=seeded["n_clones"])

        assert seeded["stage_5_source"].endswith(
            "source_imputed_stratified_extended_cps.h5"
        )
        assert preflight_result["geography_source"] == "saved_geography"
        assert len(preflight_result["fingerprint"]) == 16

        publish_result = publish.remote(
            branch="main",
            n_clones=seeded["n_clones"],
            run_id=run_id,
            regional_work_items=[
                {
                    "type": "district",
                    "id": seeded["expected_district_name"],
                    "weight": 1,
                },
                {"type": "state", "id": "NC", "weight": 1},
            ],
        )
        regional_result = publish_result["regional"]
        national_result = publish_result["national"]

        assert regional_result["message"].endswith("Upload skipped.")
        assert len(regional_result["fingerprint"]) == 16
        assert national_result["message"].endswith("Upload skipped.")
        assert len(national_result["fingerprint"]) == 16
        assert national_result["fingerprint"] != regional_result["fingerprint"]

        manifest = validate.remote(branch="main", run_id=run_id, version="0.0.0")
        assert manifest["totals"]["districts"] == 1
        assert manifest["totals"]["states"] == 1
        assert manifest["totals"]["national"] == 1
        assert "districts/NC-01.h5" in manifest["files"]
        assert "states/NC.h5" in manifest["files"]
        assert "national/US.h5" in manifest["files"]

        inspection = inspect.remote(
            run_id,
            ["districts/NC-01.h5", "states/NC.h5", "national/US.h5"],
        )
        for relative_path in (
            "districts/NC-01.h5",
            "states/NC.h5",
            "national/US.h5",
        ):
            assert inspection["outputs"][relative_path]["exists"] is True
            assert inspection["outputs"][relative_path]["size_bytes"] > 0
            assert (
                inspection["outputs"][relative_path]["variables"]["household_id"][
                    "rows"
                ]
                > 0
            )
            assert (
                inspection["outputs"][relative_path]["variables"]["person_id"]["rows"]
                > 0
            )
    finally:
        cleanup.remote(run_id)
