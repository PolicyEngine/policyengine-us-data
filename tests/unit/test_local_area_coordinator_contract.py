import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_area_module(monkeypatch):
    class FakeModalFunction:
        def __init__(self, fn):
            self.fn = fn
            self.object_id = "fake-modal-fn"

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def remote(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def spawn(self, *args, **kwargs):
            result = self.fn(*args, **kwargs)
            return types.SimpleNamespace(object_id="fake-handle", get=lambda: result)

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **_kwargs):
            def decorator(fn):
                return FakeModalFunction(fn)

            return decorator

        def local_entrypoint(self, **_kwargs):
            def decorator(fn):
                return fn

            return decorator

    fake_modal = types.ModuleType("modal")
    fake_modal.App = FakeApp
    fake_modal.Secret = types.SimpleNamespace(
        from_name=lambda *args, **kwargs: object()
    )
    fake_modal.Volume = types.SimpleNamespace(
        from_name=lambda *args, **kwargs: types.SimpleNamespace(
            reload=lambda: None,
            commit=lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    images_module = types.ModuleType("modal_app.images")
    images_module.cpu_image = object()
    monkeypatch.setitem(sys.modules, "modal_app.images", images_module)

    resilience_module = types.ModuleType("modal_app.resilience")
    resilience_module.reconcile_run_dir_fingerprint = lambda *_a, **_k: "initialized"
    monkeypatch.setitem(sys.modules, "modal_app.resilience", resilience_module)

    package = types.ModuleType("policyengine_us_data")
    package.__path__ = []
    calibration_package = types.ModuleType("policyengine_us_data.calibration")
    calibration_package.__path__ = []
    local_h5_package = types.ModuleType("policyengine_us_data.calibration.local_h5")
    local_h5_package.__path__ = []
    monkeypatch.setitem(sys.modules, "policyengine_us_data", package)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration",
        calibration_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5",
        local_h5_package,
    )

    fingerprinting_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.fingerprinting"
    )
    fingerprinting_module.FingerprintService = type("FingerprintService", (), {})
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.fingerprinting",
        fingerprinting_module,
    )

    package_geo_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.package_geography"
    )
    package_geo_module.require_calibration_package_path = lambda path: Path(path)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.package_geography",
        package_geo_module,
    )

    partitioning_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.partitioning"
    )
    partitioning_module.partition_weighted_work_items = (
        lambda work_items, _num_workers, _completed: [work_items]
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.partitioning",
        partitioning_module,
    )

    module_path = Path(__file__).resolve().parents[2] / "modal_app" / "local_area.py"
    return _load_module("local_area_under_test", module_path)


def _translated_path_factory(tmp_path):
    real_path = Path
    pipeline_root = tmp_path / "pipeline"
    staging_root = tmp_path / "staging"

    def translate(value):
        raw = str(value)
        if raw == "/pipeline":
            return pipeline_root
        if raw.startswith("/pipeline/"):
            return pipeline_root / raw.removeprefix("/pipeline/")
        if raw == "/staging":
            return staging_root
        if raw.startswith("/staging/"):
            return staging_root / raw.removeprefix("/staging/")
        return real_path(value)

    return translate


def _prepare_artifacts(tmp_path, run_id: str):
    artifacts = tmp_path / "pipeline" / "artifacts" / run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in (
        "calibration_weights.npy",
        "national_calibration_weights.npy",
        "source_imputed_stratified_extended_cps.h5",
        "policy_data.db",
        "calibration_package.pkl",
    ):
        (artifacts / name).write_bytes(b"x")
    return artifacts


class _FakeFingerprintService:
    def __init__(self, digest="actualfp"):
        self.digest = digest

    def create_publish_fingerprint(self, **_kwargs):
        return types.SimpleNamespace(digest=self.digest)


def test_coordinate_publish_requires_calibration_package(monkeypatch):
    local_area = _load_local_area_module(monkeypatch)
    monkeypatch.setattr(local_area, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(local_area, "setup_repo", lambda _branch: None)
    monkeypatch.setattr(local_area, "get_version", lambda: "1.0.0")
    monkeypatch.setattr(
        local_area,
        "require_calibration_package_path",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing package")),
    )

    with pytest.raises(FileNotFoundError, match="missing package"):
        local_area.coordinate_publish(
            branch="main",
            num_workers=1,
            skip_upload=True,
            validate=False,
            run_id="run1",
        )


def test_coordinate_publish_rejects_pinned_fingerprint_mismatch(
    monkeypatch, tmp_path
):
    local_area = _load_local_area_module(monkeypatch)
    monkeypatch.setattr(local_area, "Path", _translated_path_factory(tmp_path))
    _prepare_artifacts(tmp_path, "run1")
    monkeypatch.setattr(local_area, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(local_area, "setup_repo", lambda _branch: None)
    monkeypatch.setattr(local_area, "get_version", lambda: "1.0.0")
    monkeypatch.setattr(local_area, "validate_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr(
        local_area,
        "require_calibration_package_path",
        lambda path: local_area.Path(path),
    )
    monkeypatch.setattr(local_area, "_derive_canonical_n_clones", lambda **_k: 3)
    monkeypatch.setattr(
        local_area,
        "FingerprintService",
        lambda: _FakeFingerprintService(digest="actualfp"),
    )

    with pytest.raises(RuntimeError, match="Pinned fingerprint does not match"):
        local_area.coordinate_publish(
            branch="main",
            num_workers=1,
            skip_upload=True,
            validate=False,
            run_id="run1",
            expected_fingerprint="expectedfp",
        )


def test_coordinate_publish_returns_validation_errors_in_skip_upload_mode(
    monkeypatch, tmp_path
):
    local_area = _load_local_area_module(monkeypatch)
    monkeypatch.setattr(local_area, "Path", _translated_path_factory(tmp_path))
    _prepare_artifacts(tmp_path, "run1")
    monkeypatch.setattr(local_area, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(local_area, "setup_repo", lambda _branch: None)
    monkeypatch.setattr(local_area, "get_version", lambda: "1.0.0")
    monkeypatch.setattr(local_area, "validate_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr(
        local_area,
        "require_calibration_package_path",
        lambda path: local_area.Path(path),
    )
    monkeypatch.setattr(local_area, "_derive_canonical_n_clones", lambda **_k: 3)
    monkeypatch.setattr(
        local_area,
        "FingerprintService",
        lambda: _FakeFingerprintService(digest="actualfp"),
    )
    monkeypatch.setattr(
        local_area,
        "reconcile_run_dir_fingerprint",
        lambda *_a, **_k: "initialized",
    )
    monkeypatch.setattr(
        local_area,
        "run_phase",
        lambda *_a, **_k: (
            {"state:CA"},
            [],
            [{"area_type": "state", "area_id": "CA"}],
            [{"item": "state:CA", "error": "validator crashed"}],
        ),
    )

    monkeypatch.setattr(
        local_area.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "states": ["CA"],
                    "districts": [],
                    "cities": [],
                    "cds": [],
                }
            ),
            stderr="",
        ),
    )

    result = local_area.coordinate_publish(
        branch="main",
        num_workers=1,
        skip_upload=True,
        validate=False,
        run_id="run1",
    )

    assert result["fingerprint"] == "actualfp"
    assert result["validation_rows"] == [{"area_type": "state", "area_id": "CA"}]
    assert result["validation_errors"] == [
        {"item": "state:CA", "error": "validator crashed"}
    ]


def test_coordinate_national_publish_returns_worker_validation_errors(
    monkeypatch, tmp_path
):
    local_area = _load_local_area_module(monkeypatch)
    monkeypatch.setattr(local_area, "Path", _translated_path_factory(tmp_path))
    _prepare_artifacts(tmp_path, "run1")
    monkeypatch.setattr(local_area, "setup_gcp_credentials", lambda: None)
    monkeypatch.setattr(local_area, "setup_repo", lambda _branch: None)
    monkeypatch.setattr(local_area, "get_version", lambda: "1.0.0")
    monkeypatch.setattr(local_area, "validate_artifacts", lambda *_a, **_k: None)
    monkeypatch.setattr(
        local_area,
        "require_calibration_package_path",
        lambda path: local_area.Path(path),
    )
    monkeypatch.setattr(local_area, "_derive_canonical_n_clones", lambda **_k: 3)

    def fake_remote(**_kwargs):
        output_path = local_area.Path("/staging/run1/national/US.h5")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-h5")
        return {
            "completed": ["national:US"],
            "failed": [],
            "errors": [],
            "validation_errors": [
                {"item": "national:US", "error": "validator crashed"}
            ],
        }

    monkeypatch.setattr(
        local_area,
        "build_areas_worker",
        types.SimpleNamespace(remote=fake_remote),
    )
    monkeypatch.setattr(
        local_area.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(returncode=0, stdout="Done", stderr=""),
    )

    result = local_area.coordinate_national_publish(
        branch="main",
        validate=False,
        run_id="run1",
    )

    assert result["validation_errors"] == [
        {"item": "national:US", "error": "validator crashed"}
    ]
