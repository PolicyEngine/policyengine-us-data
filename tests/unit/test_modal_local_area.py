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
