"""Optimized integration tests for the deployed Modal pipeline app.

These tests focus on image/runtime seams rather than the full data build.
They verify that the deployed pipeline image can boot, import critical
packages, and launch key Python entrypoints with the interpreter active
inside the container.
"""

import os

import pytest

modal = pytest.importorskip("modal")

APP_NAME = os.environ.get("MODAL_APP_NAME", "policyengine-us-data-pipeline")
MODAL_ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT", "main")

pytestmark = pytest.mark.integration


def _require_modal_tokens() -> None:
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        pytest.skip("Modal credentials are required for deployed-image seam tests")


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

    for module_name in (
        "pandas",
        "h5py",
        "policyengine_us_data",
        "modal_app.worker_script",
    ):
        assert result["imports"][module_name]["ok"] is True

    assert result["interpreter"]["child_matches_parent"] is True
    assert result["subprocess"]["worker_help"]["returncode"] == 0
    assert result["subprocess"]["calibration_help"]["returncode"] == 0
