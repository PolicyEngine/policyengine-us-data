import importlib
import sys
from types import ModuleType, SimpleNamespace


def _load_local_area_module():
    fake_modal = ModuleType("modal")

    class _FakeApp:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    fake_modal.App = _FakeApp
    fake_modal.Secret = SimpleNamespace(from_name=lambda *args, **kwargs: object())
    fake_modal.Volume = SimpleNamespace(from_name=lambda *args, **kwargs: object())

    fake_images = ModuleType("modal_app.images")
    fake_images.cpu_image = object()

    fake_resilience = ModuleType("modal_app.resilience")
    fake_resilience.reconcile_run_dir_fingerprint = lambda *args, **kwargs: None

    sys.modules["modal"] = fake_modal
    sys.modules["modal_app.images"] = fake_images
    sys.modules["modal_app.resilience"] = fake_resilience
    sys.modules.pop("modal_app.local_area", None)
    return importlib.import_module("modal_app.local_area")


def test_build_promote_national_publish_script_imports_version_manifest_helpers():
    local_area = _load_local_area_module()

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
    local_area = _load_local_area_module()

    script = local_area._build_promote_publish_script(
        version="1.73.0",
        run_id="1.73.0_deadbeef_20260411",
        rel_paths=["states/AL.h5", "districts/AL-01.h5", "cities/NYC.h5"],
    )

    assert "should_finalize_local_area_release" in script
    assert "create_tag=should_finalize" in script
    assert "upload_manifest(" in script
