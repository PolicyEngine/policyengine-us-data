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
        fake_bootstrap = ModuleType("policyengine_us_data.build_outputs.bootstrap")
        fake_fingerprinting = ModuleType(
            "policyengine_us_data.build_outputs.fingerprinting"
        )
        fake_worker_inputs = ModuleType(
            "policyengine_us_data.build_outputs.worker_inputs"
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

        class _FakeWorkerBootstrapBuilder:
            def build(self, *args, **kwargs):
                return SimpleNamespace(
                    manifest_path=kwargs.get("artifacts_dir", "") / "bootstrap.json"
                    if kwargs.get("artifacts_dir")
                    else "bootstrap.json"
                )

        fake_bootstrap.WorkerBootstrapBuilder = _FakeWorkerBootstrapBuilder
        fake_fingerprinting.PublishingInputBundle = object

        class _FakeWorkerCalibrationInputs:
            def __init__(
                self,
                *,
                weights_path,
                dataset_path,
                database_path,
                geography_path=None,
                calibration_package_path=None,
                run_config_path=None,
                n_clones=430,
                seed=42,
            ):
                self.weights_path = weights_path
                self.dataset_path = dataset_path
                self.database_path = database_path
                self.geography_path = geography_path
                self.calibration_package_path = calibration_package_path
                self.run_config_path = run_config_path
                self.n_clones = n_clones
                self.seed = seed

            @classmethod
            def from_artifact_paths(cls, **kwargs):
                for key in (
                    "geography_path",
                    "calibration_package_path",
                    "run_config_path",
                ):
                    path = kwargs.get(key)
                    if path is not None and not path.exists():
                        kwargs[key] = None
                return cls(**kwargs)

            @classmethod
            def from_wire_dict(cls, payload):
                if isinstance(payload, cls):
                    return payload
                return cls(
                    weights_path=payload["weights"],
                    dataset_path=payload["dataset"],
                    database_path=payload["database"],
                    geography_path=payload.get("geography"),
                    calibration_package_path=payload.get("calibration_package"),
                    run_config_path=payload.get("run_config"),
                    n_clones=payload.get("n_clones", 430),
                    seed=payload.get("seed", 42),
                )

            def to_worker_cli_args(self):
                args = [
                    "--weights-path",
                    str(self.weights_path),
                    "--dataset-path",
                    str(self.dataset_path),
                    "--db-path",
                    str(self.database_path),
                    "--n-clones",
                    str(self.n_clones),
                    "--seed",
                    str(self.seed),
                ]
                if self.geography_path is not None:
                    args.extend(["--geography-path", str(self.geography_path)])
                if self.calibration_package_path is not None:
                    args.extend(
                        [
                            "--calibration-package-path",
                            str(self.calibration_package_path),
                        ]
                    )
                if self.run_config_path is not None:
                    args.extend(["--run-config-path", str(self.run_config_path)])
                return args

            def to_wire_dict(self):
                payload = {
                    "weights": str(self.weights_path),
                    "dataset": str(self.dataset_path),
                    "database": str(self.database_path),
                    "n_clones": self.n_clones,
                    "seed": self.seed,
                }
                if self.geography_path is not None:
                    payload["geography"] = str(self.geography_path)
                if self.calibration_package_path is not None:
                    payload["calibration_package"] = str(self.calibration_package_path)
                if self.run_config_path is not None:
                    payload["run_config"] = str(self.run_config_path)
                return payload

        fake_worker_inputs.WorkerCalibrationInputs = _FakeWorkerCalibrationInputs

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
                "policyengine_us_data.build_outputs.bootstrap": fake_bootstrap,
                "policyengine_us_data.build_outputs.fingerprinting": (
                    fake_fingerprinting
                ),
                "policyengine_us_data.build_outputs.partitioning": (fake_partitioning),
                "policyengine_us_data.build_outputs.worker_inputs": (
                    fake_worker_inputs
                ),
            }
        )

    with _patched_module_registry(overrides):
        return importlib.import_module("modal_app.local_area")
