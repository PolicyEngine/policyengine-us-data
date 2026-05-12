"""Integration tests for the deployed Modal pipeline app.

These tests focus on image/runtime seams rather than the full data build.
They verify that the deployed pipeline image can boot, import critical
packages, and launch key Python entrypoints with the interpreter active
inside the container.
"""

import os

import pytest
import requests

modal = pytest.importorskip("modal")

APP_NAME = os.environ.get("MODAL_APP_NAME", "policyengine-us-data-pipeline")
MODAL_ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT", "main")

pytestmark = pytest.mark.integration


def _require_modal_tokens() -> None:
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        pytest.skip("Modal credentials are required for deployed-image seam tests")


def _modal_proxy_auth_headers() -> dict[str, str]:
    key = os.environ.get("MODAL_PROXY_TOKEN_ID") or os.environ.get("MODAL_TOKEN_ID")
    secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET") or os.environ.get(
        "MODAL_TOKEN_SECRET"
    )
    if not (key and secret):
        pytest.skip("Modal proxy auth credentials are required for HTTP seam tests")
    return {
        "Modal-Key": key,
        "Modal-Secret": secret,
    }


def test_pipeline_image_runtime_seams():
    _require_modal_tokens()

    fn = modal.Function.from_name(
        APP_NAME,
        "verify_runtime_seams",
        environment_name=MODAL_ENVIRONMENT,
    )
    result = fn.remote()

    assert result["paths"]["repo_root_exists"] is True
    assert result["paths"]["target_config_exists"] is True
    assert result["paths"]["working_directory_is_repo_root"] is True
    assert result["paths"]["all_expected_files_exist"] is True
    assert result["paths"]["expected_files"] == {
        "pyproject.toml": True,
        "uv.lock": True,
        "modal_app/worker_script.py": True,
        "modal_app/local_area.py": True,
        "modal_app/h5_test_harness.py": True,
        "modal_app/step_manifests/specs.py": True,
        "modal_app/step_manifests/state.py": True,
        "modal_app/step_manifests/store.py": True,
        "modal_app/step_manifests/errors.py": True,
        "modal_app/step_manifests/status.py": True,
        "modal_app/fixtures/h5_cases.py": True,
        "tests/integration/test_fixture_50hh.h5": True,
        "policyengine_us_data/calibration/target_config.yaml": True,
        "policyengine_us_data/calibration/target_config_full.yaml": True,
        "policyengine_us_data/utils/run_context.py": True,
        "policyengine_us_data/utils/step_manifest.py": True,
    }

    for module_name in (
        "google.cloud.storage",
        "pandas",
        "h5py",
        "huggingface_hub",
        "modal_app.fixtures.h5_cases",
        "modal_app.h5_test_harness",
        "modal_app.local_area",
        "modal_app.remote_calibration_runner",
        "modal_app.step_manifests.specs",
        "modal_app.step_manifests.state",
        "modal_app.step_manifests.store",
        "modal_app.step_manifests.errors",
        "modal_app.step_manifests.status",
        "numpy",
        "policyengine_us",
        "policyengine_us_data",
        "policyengine_us_data.utils.run_context",
        "policyengine_us_data.utils.step_manifest",
        "modal_app.worker_script",
        "spm_calculator",
        "sqlalchemy",
    ):
        assert result["imports"][module_name]["ok"] is True

    assert result["interpreter"]["child_matches_parent"] is True
    assert result["interpreter"]["child_cwd_is_repo_root"] is True
    assert result["subprocess"]["worker_import"]["returncode"] == 0
    assert result["subprocess"]["worker_help"]["returncode"] == 0
    assert result["subprocess"]["local_area_import"]["returncode"] == 0
    assert result["subprocess"]["calibration_help"]["returncode"] == 0
    checkpoint_policy = result["calibration_optimizer_checkpoint_policy"]
    assert checkpoint_policy["runner_exposes_checkpoint_name"] is False
    assert checkpoint_policy["runner_passes_checkpoint_output"] is False
    assert checkpoint_policy["runner_collects_checkpoint_path"] is False


def test_pipeline_status_callable_reports_missing_run():
    _require_modal_tokens()

    fn = modal.Function.from_name(
        APP_NAME,
        "get_pipeline_status",
        environment_name=MODAL_ENVIRONMENT,
    )
    result = fn.remote("missing-run-for-status-seam")

    assert result["status"] == "not_found"
    assert result["run_id"] == "missing-run-for-status-seam"
    assert result["stage_manifests"] == []


def test_pipeline_status_http_endpoint_reports_missing_run():
    _require_modal_tokens()
    headers = _modal_proxy_auth_headers()

    fn = modal.Function.from_name(
        APP_NAME,
        "pipeline_status_endpoint",
        environment_name=MODAL_ENVIRONMENT,
    )
    endpoint = fn.get_web_url()
    assert endpoint

    response = requests.get(
        endpoint,
        params={"run_id": "missing-run-for-status-http-seam"},
        headers=headers,
        timeout=30,
    )

    assert response.status_code == 200, response.text[:500]
    result = response.json()
    assert result["status"] == "not_found"
    assert result["run_id"] == "missing-run-for-status-http-seam"
    assert result["stage_manifests"] == []
    assert result["error"] is None
