"""Fixture helpers for `test_modal_local_area.py`."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

__test__ = False


def load_local_area_module():
    """Import `modal_app.local_area` with minimal fake Modal dependencies."""

    fake_modal = ModuleType("modal")
    fake_policyengine = ModuleType("policyengine_us_data")
    fake_calibration = ModuleType("policyengine_us_data.calibration")
    fake_local_h5 = ModuleType("policyengine_us_data.calibration.local_h5")
    fake_partitioning = ModuleType(
        "policyengine_us_data.calibration.local_h5.partitioning"
    )
    fake_policyengine.__path__ = []
    fake_calibration.__path__ = []
    fake_local_h5.__path__ = []

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
    fake_partitioning.partition_weighted_work_items = (
        lambda *args, **kwargs: []
    )

    sys.modules["modal"] = fake_modal
    sys.modules["modal_app.images"] = fake_images
    sys.modules["modal_app.resilience"] = fake_resilience
    sys.modules["policyengine_us_data"] = fake_policyengine
    sys.modules["policyengine_us_data.calibration"] = fake_calibration
    sys.modules["policyengine_us_data.calibration.local_h5"] = fake_local_h5
    sys.modules[
        "policyengine_us_data.calibration.local_h5.partitioning"
    ] = fake_partitioning
    sys.modules.pop("modal_app.local_area", None)
    return importlib.import_module("modal_app.local_area")
