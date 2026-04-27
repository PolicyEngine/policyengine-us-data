import json

from tests.unit.fixtures.test_modal_local_area import load_local_area_module


def test_build_promote_national_publish_script_imports_version_manifest_helpers():
    local_area = load_local_area_module()

    script = local_area._build_promote_national_publish_script(
        version="1.73.0",
        run_id="1.73.0_deadbeef_20260411",
        rel_paths=["national/US.h5"],
    )

    assert "from policyengine_us_data.utils.version_manifest import (" in script
    assert "HFVersionInfo" in script
    assert "build_manifest" in script
    assert "upload_manifest" in script


def test_build_promote_publish_script_finalizes_complete_release():
    local_area = load_local_area_module()

    script = local_area._build_promote_publish_script(
        version="1.73.0",
        run_id="1.73.0_deadbeef_20260411",
        rel_paths=["states/AL.h5", "districts/AL-01.h5", "cities/NYC.h5"],
    )

    assert "should_finalize_local_area_release" in script
    assert "create_tag=should_finalize" in script
    assert "upload_manifest(" in script


def test_validate_artifacts_ignores_deprecated_checkpoint_entries(tmp_path):
    local_area = load_local_area_module()
    weights = tmp_path / "calibration_weights.npy"
    weights.write_bytes(b"weights")
    checkpoint = tmp_path / "calibration_weights.checkpoint.pt"
    checkpoint.write_bytes(b"stale-checkpoint")
    config = tmp_path / "unified_run_config.json"
    config.write_text(
        json.dumps(
            {
                "artifacts": {
                    "calibration_weights.npy": (
                        "sha256:"
                        "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848"
                        "cf1a13eafdf33b2c"
                    ),
                    "calibration_checkpoint.pt": "sha256:not-a-real-checksum",
                    "calibration_weights.checkpoint.pt": (
                        "sha256:also-not-a-real-checksum"
                    ),
                }
            }
        )
    )

    local_area.validate_artifacts(config, tmp_path)
