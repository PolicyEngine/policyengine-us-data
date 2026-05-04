"""Helpers for importing `modal_app.local_area` in tests."""

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


def load_local_area_module(*, stub_policyengine: bool = True):
    """Import `modal_app.local_area` with scoped fake Modal dependencies."""

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

    overrides = {
        "modal": fake_modal,
        "modal_app.images": fake_images,
        "modal_app.resilience": fake_resilience,
    }

    if stub_policyengine:
        fake_policyengine = ModuleType("policyengine_us_data")
        fake_calibration = ModuleType("policyengine_us_data.calibration")
        fake_build_outputs = ModuleType("policyengine_us_data.build_outputs")
        fake_pipeline_metadata = ModuleType("policyengine_us_data.pipeline_metadata")
        fake_pipeline_schema = ModuleType("policyengine_us_data.pipeline_schema")
        fake_utils = ModuleType("policyengine_us_data.utils")
        fake_run_context = ModuleType("policyengine_us_data.utils.run_context")
        fake_partitioning = ModuleType(
            "policyengine_us_data.build_outputs.partitioning"
        )
        fake_fingerprinting = ModuleType(
            "policyengine_us_data.build_outputs.fingerprinting"
        )
        fake_policyengine.__path__ = []
        fake_calibration.__path__ = []
        fake_build_outputs.__path__ = []
        fake_utils.__path__ = []

        class _FakePipelineNode:
            def __init__(self, *args, **kwargs):
                pass

        def _fake_pipeline_node(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        fake_pipeline_metadata.pipeline_node = _fake_pipeline_node
        fake_pipeline_schema.PipelineNode = _FakePipelineNode
        fake_run_context.resolve_run_id = lambda explicit="", **kwargs: explicit
        fake_partitioning.partition_weighted_work_items = lambda *args, **kwargs: []
        fake_fingerprinting.PublishingInputBundle = object

        class _FakeFingerprintingService:
            def build_traceability(self, *args, **kwargs):
                return object()

            def compute_scope_fingerprint(self, *args, **kwargs):
                return "fake-fingerprint"

        fake_fingerprinting.FingerprintingService = _FakeFingerprintingService
        overrides.update(
            {
                "policyengine_us_data": fake_policyengine,
                "policyengine_us_data.calibration": fake_calibration,
                "policyengine_us_data.pipeline_metadata": fake_pipeline_metadata,
                "policyengine_us_data.pipeline_schema": fake_pipeline_schema,
                "policyengine_us_data.utils": fake_utils,
                "policyengine_us_data.utils.run_context": fake_run_context,
                "policyengine_us_data.build_outputs": fake_build_outputs,
                "policyengine_us_data.build_outputs.fingerprinting": (
                    fake_fingerprinting
                ),
                "policyengine_us_data.build_outputs.partitioning": (fake_partitioning),
            }
        )

    with _patched_module_registry(overrides):
        return importlib.import_module("modal_app.local_area")
