"""Deployed Modal integration tests for the tiny H5-builder pipeline."""

from __future__ import annotations

import os
import uuid

import pytest

modal = pytest.importorskip("modal")

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
        pytest.skip("Modal credentials are required for deployed H5 integration tests")


def _function(app_name: str, function_name: str):
    return modal.Function.from_name(
        app_name,
        function_name,
        environment_name=MODAL_ENVIRONMENT,
    )


def _run_id(label: str) -> str:
    return f"0.0.0_{label}_{uuid.uuid4().hex[:10]}"


def _work_items(*area_types: str) -> list[dict[str, object]]:
    items = {
        "district": {"type": "district", "id": "NC-01", "weight": 1},
        "state": {"type": "state", "id": "NC", "weight": 1},
        "national": {"type": "national", "id": "US", "weight": 1},
    }
    return [items[area_type] for area_type in area_types]


def _assert_output_contract(inspection: dict, relative_paths: tuple[str, ...]) -> None:
    assert inspection["run_dir_exists"] is True
    for relative_path in relative_paths:
        output = inspection["outputs"][relative_path]
        assert output["exists"] is True
        assert output["size_bytes"] > 0
        for variable in (
            "household_id",
            "person_id",
            "household_weight",
            "state_fips",
            "congressional_district_geoid",
        ):
            assert output["variables"][variable]["exists"] is True
            assert output["variables"][variable]["rows"] > 0


def test_saved_geography_h5_pipeline_builds_regional_and_national_outputs():
    _require_modal_tokens()

    run_id = _run_id("h5-saved-geo")
    seed = _function(HARNESS_APP_NAME, "seed_h5_case")
    preflight = _function(HARNESS_APP_NAME, "preflight_h5_case")
    inspect = _function(HARNESS_APP_NAME, "inspect_h5_outputs")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")
    build = _function(LOCAL_AREA_APP_NAME, "build_areas_worker")
    validate = _function(LOCAL_AREA_APP_NAME, "validate_staging")

    try:
        seed.remote(run_id, "saved_geography_success")
        preflight_result = preflight.remote(run_id, n_clones=1)

        assert preflight_result["geography_source"] == "saved_geography"
        assert len(preflight_result["fingerprint"]) == 16

        regional_result = build.remote(
            branch="main",
            run_id=run_id,
            scope="regional",
            work_items=_work_items("district", "state"),
            calibration_inputs=preflight_result["calibration_inputs"],
            validate=False,
        )
        national_result = build.remote(
            branch="main",
            run_id=run_id,
            scope="national",
            work_items=_work_items("national"),
            calibration_inputs=preflight_result["calibration_inputs"],
            validate=False,
        )

        assert regional_result["failed"] == []
        assert regional_result["errors"] == []
        assert regional_result["completed"] == [
            "district:NC-01",
            "state:NC",
        ]
        assert national_result["failed"] == []
        assert national_result["errors"] == []
        assert national_result["completed"] == ["national:US"]

        manifest = validate.remote(branch="main", run_id=run_id, version="0.0.0")
        assert manifest["totals"]["districts"] == 1
        assert manifest["totals"]["states"] == 1
        assert "districts/NC-01.h5" in manifest["files"]
        assert "states/NC.h5" in manifest["files"]

        inspection = inspect.remote(
            run_id,
            ["districts/NC-01.h5", "states/NC.h5", "national/US.h5"],
        )
        _assert_output_contract(
            inspection,
            ("districts/NC-01.h5", "states/NC.h5", "national/US.h5"),
        )
        assert inspection["manifest_exists"] is True
    finally:
        cleanup.remote(run_id)


def test_deployed_regional_coordinator_builds_override_requests_from_seeded_artifacts():
    _require_modal_tokens()

    run_id = _run_id("h5-coordinator")
    seed = _function(HARNESS_APP_NAME, "seed_h5_case")
    inspect = _function(HARNESS_APP_NAME, "inspect_h5_outputs")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")
    coordinate = _function(LOCAL_AREA_APP_NAME, "coordinate_publish")

    try:
        seeded = seed.remote(run_id, "saved_geography_success")

        result = coordinate.remote(
            branch="main",
            num_workers=1,
            skip_upload=True,
            n_clones=seeded["n_clones"],
            validate=False,
            run_id=run_id,
            work_items_override=_work_items("district", "state"),
        )

        assert result["message"].endswith("Upload skipped.")
        assert result["reuse_measurement"]["expected_outputs"] == 2
        assert result["reuse_measurement"]["invalid_outputs"] == 0

        inspection = inspect.remote(
            run_id,
            ["districts/NC-01.h5", "states/NC.h5"],
        )
        _assert_output_contract(
            inspection,
            ("districts/NC-01.h5", "states/NC.h5"),
        )
    finally:
        cleanup.remote(run_id)


def test_package_fallback_h5_pipeline_builds_district_output():
    _require_modal_tokens()

    run_id = _run_id("h5-package")
    seed = _function(HARNESS_APP_NAME, "seed_h5_case")
    preflight = _function(HARNESS_APP_NAME, "preflight_h5_case")
    inspect = _function(HARNESS_APP_NAME, "inspect_h5_outputs")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")
    build = _function(LOCAL_AREA_APP_NAME, "build_areas_worker")
    validate = _function(LOCAL_AREA_APP_NAME, "validate_staging")

    try:
        seed.remote(run_id, "package_fallback_success")
        preflight_result = preflight.remote(run_id, n_clones=1)

        assert preflight_result["geography_source"] == "calibration_package"
        assert len(preflight_result["fingerprint"]) == 16

        build_result = build.remote(
            branch="main",
            run_id=run_id,
            scope="regional",
            work_items=_work_items("district"),
            calibration_inputs=preflight_result["calibration_inputs"],
            validate=False,
        )

        assert build_result["failed"] == []
        assert build_result["errors"] == []
        assert build_result["completed"] == ["district:NC-01"]

        manifest = validate.remote(branch="main", run_id=run_id, version="0.0.0")
        assert manifest["totals"]["districts"] == 1
        assert "districts/NC-01.h5" in manifest["files"]

        inspection = inspect.remote(run_id, ["districts/NC-01.h5"])
        _assert_output_contract(inspection, ("districts/NC-01.h5",))
    finally:
        cleanup.remote(run_id)


def test_missing_geography_h5_pipeline_fails_clearly():
    _require_modal_tokens()

    run_id = _run_id("h5-package-mismatch")
    seed = _function(HARNESS_APP_NAME, "seed_h5_case")
    preflight = _function(HARNESS_APP_NAME, "preflight_h5_case")
    cleanup = _function(HARNESS_APP_NAME, "cleanup_h5_case")
    build = _function(LOCAL_AREA_APP_NAME, "build_areas_worker")

    try:
        seed.remote(run_id, "misnamed_package")
        preflight_result = preflight.remote(run_id, n_clones=1)

        assert preflight_result["geography_source"] is None

        build_result = build.remote(
            branch="main",
            run_id=run_id,
            scope="regional",
            work_items=_work_items("district"),
            calibration_inputs=preflight_result["calibration_inputs"],
            validate=False,
        )

        assert build_result["completed"] == []
        assert build_result["failed"] == ["district:NC-01"]
        assert build_result["errors"]
        assert (
            "No saved calibration geography found" in build_result["errors"][0]["error"]
        )
    finally:
        cleanup.remote(run_id)
