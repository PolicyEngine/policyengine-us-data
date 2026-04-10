import importlib.util
import json
from dataclasses import dataclass
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

    contracts_spec = importlib.util.spec_from_file_location(
        "policyengine_us_data.calibration.local_h5.contracts",
        Path(__file__).resolve().parents[2]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
        / "contracts.py",
    )
    contracts_module = importlib.util.module_from_spec(contracts_spec)
    assert contracts_spec is not None
    assert contracts_spec.loader is not None
    sys.modules[contracts_spec.name] = contracts_module
    contracts_spec.loader.exec_module(contracts_module)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.contracts",
        contracts_module,
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
    package_geo_module.CalibrationPackageGeographyLoader = type(
        "CalibrationPackageGeographyLoader", (), {}
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
    partitioning_module.work_item_key = (
        lambda item: f"{item['type']}:{item['id']}"
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.partitioning",
        partitioning_module,
    )

    area_catalog_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.area_catalog"
    )
    area_catalog_module.USAreaCatalog = type("USAreaCatalog", (), {})
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.area_catalog",
        area_catalog_module,
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


@dataclass(frozen=True)
class _FakeEntry:
    request: object
    weight: int

    @property
    def key(self):
        return f"{self.request.area_type}:{self.request.area_id}"

    def to_partition_item(self):
        return {
            "type": self.request.area_type,
            "id": self.request.area_id,
            "weight": self.weight,
        }


class _FakeCatalog:
    def __init__(self, entries=None, national_entry=None):
        self._entries = entries or ()
        self._national_entry = national_entry

    def resolved_regional_entries(self, *_args, **_kwargs):
        return self._entries

    def resolved_national_entry(self):
        return self._national_entry


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
    request = local_area.AreaBuildRequest(
        area_type="state",
        area_id="CA",
        display_name="CA",
        output_relative_path="states/CA.h5",
    )
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
    reconcile_calls = []
    monkeypatch.setattr(
        local_area,
        "reconcile_run_dir_fingerprint",
        lambda *_a, **kwargs: (
            reconcile_calls.append(kwargs),
            "initialized",
        )[1],
    )
    monkeypatch.setattr(
        local_area,
        "_load_catalog_geography",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        local_area,
        "USAreaCatalog",
        lambda: _FakeCatalog(entries=(_FakeEntry(request=request, weight=1),)),
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

    result = local_area.coordinate_publish(
        branch="main",
        num_workers=1,
        skip_upload=True,
        validate=False,
        run_id="run1",
    )

    assert result["fingerprint"] == "actualfp"
    assert reconcile_calls == [{"scope": "regional"}]
    assert result["validation_rows"] == [{"area_type": "state", "area_id": "CA"}]
    assert result["validation_errors"] == [
        {"item": "state:CA", "error": "validator crashed"}
    ]


def test_coordinate_publish_raises_on_build_failures_even_if_files_exist(
    monkeypatch, tmp_path
):
    local_area = _load_local_area_module(monkeypatch)
    request = local_area.AreaBuildRequest(
        area_type="state",
        area_id="CA",
        display_name="CA",
        output_relative_path="states/CA.h5",
    )
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
        "_load_catalog_geography",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        local_area,
        "USAreaCatalog",
        lambda: _FakeCatalog(entries=(_FakeEntry(request=request, weight=1),)),
    )
    monkeypatch.setattr(
        local_area,
        "run_phase",
        lambda *_a, **_k: (
            {"state:CA"},
            [{"type": "build_failure", "item": "state:CA", "error": "bad output"}],
            [],
            [],
        ),
    )

    with pytest.raises(RuntimeError, match="build failure"):
        local_area.coordinate_publish(
            branch="main",
            num_workers=1,
            skip_upload=True,
            validate=False,
            run_id="run1",
        )


def test_coordinate_publish_raises_on_worker_issues_even_if_files_exist(
    monkeypatch, tmp_path
):
    local_area = _load_local_area_module(monkeypatch)
    request = local_area.AreaBuildRequest(
        area_type="state",
        area_id="CA",
        display_name="CA",
        output_relative_path="states/CA.h5",
    )
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
        "_load_catalog_geography",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        local_area,
        "USAreaCatalog",
        lambda: _FakeCatalog(entries=(_FakeEntry(request=request, weight=1),)),
    )
    monkeypatch.setattr(
        local_area,
        "run_phase",
        lambda *_a, **_k: (
            {"state:CA"},
            [{"type": "worker_issue", "item": "worker", "error": "subprocess failed"}],
            [],
            [],
        ),
    )

    with pytest.raises(RuntimeError, match="worker issue"):
        local_area.coordinate_publish(
            branch="main",
            num_workers=1,
            skip_upload=True,
            validate=False,
            run_id="run1",
        )


def test_coordinate_national_publish_returns_worker_validation_errors(
    monkeypatch, tmp_path
):
    local_area = _load_local_area_module(monkeypatch)
    national_request = local_area.AreaBuildRequest.national()
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
        "USAreaCatalog",
        lambda: _FakeCatalog(
            national_entry=_FakeEntry(request=national_request, weight=1)
        ),
    )
    monkeypatch.setattr(
        local_area,
        "FingerprintService",
        lambda: _FakeFingerprintService(digest="natfp"),
    )
    reconcile_calls = []
    monkeypatch.setattr(
        local_area,
        "reconcile_run_dir_fingerprint",
        lambda *_a, **kwargs: (
            reconcile_calls.append(kwargs),
            "initialized",
        )[1],
    )

    def fake_remote(**_kwargs):
        output_path = local_area.Path("/staging/run1/national/US.h5")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-h5")
        return {
            "completed": [
                {
                    "request": national_request.to_dict(),
                    "build_status": "completed",
                    "output_path": str(output_path),
                    "build_error": None,
                    "validation": {
                        "status": "error",
                        "rows": [],
                        "issues": [
                            {
                                "code": "validation_exception",
                                "message": "validator crashed",
                                "severity": "error",
                                "details": {},
                            }
                        ],
                        "summary": {},
                    },
                }
            ],
            "failed": [],
            "worker_issues": [],
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
        {
            "item": "national:US",
            "error": "validator crashed",
            "code": "validation_exception",
            "details": {},
        }
    ]
    assert result["fingerprint"] == "natfp"
    assert reconcile_calls == [{"scope": "national"}]


def test_coordinate_national_publish_rejects_pinned_fingerprint_mismatch(
    monkeypatch, tmp_path
):
    local_area = _load_local_area_module(monkeypatch)
    national_request = local_area.AreaBuildRequest.national()
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
        "USAreaCatalog",
        lambda: _FakeCatalog(
            national_entry=_FakeEntry(request=national_request, weight=1)
        ),
    )
    monkeypatch.setattr(
        local_area,
        "FingerprintService",
        lambda: _FakeFingerprintService(digest="actualfp"),
    )

    with pytest.raises(RuntimeError, match="Pinned fingerprint does not match"):
        local_area.coordinate_national_publish(
            branch="main",
            validate=False,
            run_id="run1",
            expected_fingerprint="expectedfp",
        )


def test_coordinate_national_publish_resumes_without_rebuilding(
    monkeypatch, tmp_path
):
    local_area = _load_local_area_module(monkeypatch)
    national_request = local_area.AreaBuildRequest.national()
    monkeypatch.setattr(local_area, "Path", _translated_path_factory(tmp_path))
    _prepare_artifacts(tmp_path, "run1")
    output_path = tmp_path / "staging" / "run1" / "national" / "US.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"existing-h5")
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
        "USAreaCatalog",
        lambda: _FakeCatalog(
            national_entry=_FakeEntry(request=national_request, weight=1)
        ),
    )
    monkeypatch.setattr(
        local_area,
        "FingerprintService",
        lambda: _FakeFingerprintService(digest="natfp"),
    )
    monkeypatch.setattr(
        local_area,
        "reconcile_run_dir_fingerprint",
        lambda *_a, **_k: "resume",
    )
    monkeypatch.setattr(
        local_area,
        "build_areas_worker",
        types.SimpleNamespace(
            remote=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("worker should not be called on resume")
            )
        ),
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
        expected_fingerprint="natfp",
    )

    assert result["fingerprint"] == "natfp"


def test_run_phase_aggregates_structured_worker_results(monkeypatch):
    local_area = _load_local_area_module(monkeypatch)
    request = local_area.AreaBuildRequest(
        area_type="state",
        area_id="CA",
        display_name="CA",
        output_relative_path="states/CA.h5",
    )
    entry = _FakeEntry(request=request, weight=2)
    run_dir = Path("/staging/run1")
    monkeypatch.setattr(local_area, "get_completed_from_volume", lambda _run_dir: {"state:CA"})
    monkeypatch.setattr(local_area.staging_volume, "reload", lambda: None)

    def fake_spawn(**kwargs):
        assert kwargs["requests"] == [request.to_dict()]
        payload = {
            "completed": [
                {
                    "request": request.to_dict(),
                    "build_status": "completed",
                    "output_path": "/staging/run1/states/CA.h5",
                    "build_error": None,
                    "validation": {
                        "status": "failed",
                        "rows": [
                            {
                                "target_name": "population",
                                "sanity_check": "FAIL",
                                "rel_abs_error": 0.2,
                            }
                        ],
                        "issues": [],
                        "summary": {
                            "n_targets": 1,
                            "n_sanity_fail": 1,
                            "mean_rel_abs_error": 0.2,
                        },
                    },
                }
            ],
            "failed": [],
            "worker_issues": [],
        }
        return types.SimpleNamespace(object_id="fake-handle", get=lambda: payload)

    monkeypatch.setattr(
        local_area,
        "build_areas_worker",
        types.SimpleNamespace(spawn=fake_spawn),
    )

    completed, errors, validation_rows, validation_errors = local_area.run_phase(
        "All areas",
        entries=[entry],
        num_workers=1,
        completed=set(),
        branch="main",
        run_id="run1",
        calibration_inputs={"weights": "w", "dataset": "d", "database": "db"},
        run_dir=run_dir,
        validate=True,
    )

    assert completed == {"state:CA"}
    assert errors == []
    assert validation_rows == [
        {
            "target_name": "population",
            "sanity_check": "FAIL",
            "rel_abs_error": 0.2,
        }
    ]
    assert validation_errors == []
