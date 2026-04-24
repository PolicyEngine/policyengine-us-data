"""Tiny-fixture Modal end-to-end tests for the H5 publication path."""

from __future__ import annotations

import os
import uuid

import pytest

modal = pytest.importorskip("modal")

PIPELINE_APP_NAME = os.environ.get(
    "MODAL_LOCAL_AREA_APP_NAME",
    "policyengine-us-data-local-area",
)
HARNESS_APP_NAME = os.environ.get(
    "MODAL_H5_TEST_HARNESS_APP_NAME",
    "policyengine-us-data-h5-test-harness",
)
MODAL_ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT", "main")

DISTRICT_NAME = "NC-01"

pytestmark = pytest.mark.integration


def _require_modal_tokens() -> None:
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        pytest.skip("Modal credentials are required for optimized H5 tests")


def _function(app_name: str, function_name: str):
    return modal.Function.from_name(
        app_name,
        function_name,
        environment_name=MODAL_ENVIRONMENT,
    )


def _run_id(label: str) -> str:
    return f"0.0.0_{label}_{uuid.uuid4().hex[:10]}"


def _district_work_item() -> list[dict[str, object]]:
    return [{"type": "district", "id": DISTRICT_NAME, "weight": 1}]


def test_saved_geography_case_builds_and_stages_one_h5():
    _require_modal_tokens()

    run_id = _run_id("saved-geo")
    seed = _function(HARNESS_APP_NAME, "seed_h5_case")
    preflight = _function(HARNESS_APP_NAME, "preflight_h5_case")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")
    build = _function(PIPELINE_APP_NAME, "build_areas_worker")
    validate = _function(PIPELINE_APP_NAME, "validate_staging")

    try:
        seeded = seed.remote(run_id, "saved_geography_success")
        preflight_result = preflight.remote(run_id, n_clones=1)

        assert preflight_result["geography_source"] == "saved_geography"

        build_result = build.remote(
            branch="main",
            run_id=run_id,
            work_items=_district_work_item(),
            calibration_inputs=preflight_result["calibration_inputs"],
            validate=False,
        )

        assert build_result["failed"] == []
        assert build_result["errors"] == []
        assert build_result["completed"] == [
            f"district:{seeded['expected_district_name']}"
        ]

        manifest = validate.remote(branch="main", run_id=run_id, version="0.0.0")
        assert manifest["totals"]["districts"] == 1
        assert "districts/NC-01.h5" in manifest["files"]
    finally:
        cleanup.remote(run_id)


def test_package_fallback_case_builds_and_stages_one_h5():
    _require_modal_tokens()

    run_id = _run_id("package")
    seed = _function(HARNESS_APP_NAME, "seed_h5_case")
    preflight = _function(HARNESS_APP_NAME, "preflight_h5_case")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")
    build = _function(PIPELINE_APP_NAME, "build_areas_worker")
    validate = _function(PIPELINE_APP_NAME, "validate_staging")

    try:
        seeded = seed.remote(run_id, "package_fallback_success")
        preflight_result = preflight.remote(run_id, n_clones=1)

        assert preflight_result["geography_source"] == "calibration_package"

        build_result = build.remote(
            branch="main",
            run_id=run_id,
            work_items=_district_work_item(),
            calibration_inputs=preflight_result["calibration_inputs"],
            validate=False,
        )

        assert build_result["failed"] == []
        assert build_result["errors"] == []
        assert build_result["completed"] == [
            f"district:{seeded['expected_district_name']}"
        ]

        manifest = validate.remote(branch="main", run_id=run_id, version="0.0.0")
        assert manifest["totals"]["districts"] == 1
        assert "districts/NC-01.h5" in manifest["files"]
    finally:
        cleanup.remote(run_id)


def test_checkpoint_manifest_name_mismatch_fails_in_preflight():
    _require_modal_tokens()

    run_id = _run_id("checkpoint-mismatch")
    seed = _function(HARNESS_APP_NAME, "seed_h5_case")
    preflight = _function(HARNESS_APP_NAME, "preflight_h5_case")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")

    try:
        seed.remote(run_id, "checkpoint_name_mismatch")
        with pytest.raises(Exception, match="calibration_checkpoint\\.pt not found"):
            preflight.remote(run_id, n_clones=1)
    finally:
        cleanup.remote(run_id)


def test_misnamed_package_breaks_worker_geography_resolution():
    _require_modal_tokens()

    run_id = _run_id("package-mismatch")
    seed = _function(HARNESS_APP_NAME, "seed_h5_case")
    preflight = _function(HARNESS_APP_NAME, "preflight_h5_case")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")
    build = _function(PIPELINE_APP_NAME, "build_areas_worker")

    try:
        seed.remote(run_id, "misnamed_package")
        preflight_result = preflight.remote(run_id, n_clones=1)

        assert preflight_result["geography_source"] is None

        build_result = build.remote(
            branch="main",
            run_id=run_id,
            work_items=_district_work_item(),
            calibration_inputs=preflight_result["calibration_inputs"],
            validate=False,
        )

        assert build_result["completed"] == []
        assert build_result["failed"] == [f"district:{DISTRICT_NAME}"]
        assert build_result["errors"]
        assert (
            "No saved calibration geography found" in build_result["errors"][0]["error"]
        )
    finally:
        cleanup.remote(run_id)
