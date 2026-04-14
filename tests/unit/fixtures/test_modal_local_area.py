"""Fixture helpers for `test_modal_local_area.py`."""

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

__test__ = False


@contextmanager
def _patched_module_registry(overrides: dict[str, ModuleType]):
    """Temporarily replace selected `sys.modules` entries for one import."""

    sentinel = object()
    previous = {
        name: sys.modules.get(name, sentinel)
        for name in [*overrides.keys(), "modal_app.local_area"]
    }

    try:
        for name, module in overrides.items():
            sys.modules[name] = module
        sys.modules.pop("modal_app.local_area", None)
        yield
    finally:
        for name, module in previous.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_local_area_module():
    """Import `modal_app.local_area` with scoped fake Modal dependencies."""

    fake_modal = ModuleType("modal")
    fake_policyengine = ModuleType("policyengine_us_data")
    fake_calibration = ModuleType("policyengine_us_data.calibration")
    fake_local_h5 = ModuleType("policyengine_us_data.calibration.local_h5")
    fake_partitioning = ModuleType(
        "policyengine_us_data.calibration.local_h5.partitioning"
    )
    fake_fingerprinting = ModuleType(
        "policyengine_us_data.calibration.local_h5.fingerprinting"
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
    fake_partitioning.partition_weighted_work_items = lambda *args, **kwargs: []
    fake_fingerprinting.PublishingInputBundle = object

    class _FakeFingerprintingService:
        def build_traceability(self, *args, **kwargs):
            return object()

        def compute_scope_fingerprint(self, *args, **kwargs):
            return "fake-fingerprint"

    fake_fingerprinting.FingerprintingService = _FakeFingerprintingService

    with _patched_module_registry(
        {
            "modal": fake_modal,
            "modal_app.images": fake_images,
            "modal_app.resilience": fake_resilience,
            "policyengine_us_data": fake_policyengine,
            "policyengine_us_data.calibration": fake_calibration,
            "policyengine_us_data.calibration.local_h5": fake_local_h5,
            "policyengine_us_data.calibration.local_h5.fingerprinting": (
                fake_fingerprinting
            ),
            "policyengine_us_data.calibration.local_h5.partitioning": (
                fake_partitioning
            ),
        }
    ):
        return importlib.import_module("modal_app.local_area")
