import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock


def _module_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[2].joinpath(*parts)


def _install_fake_modal(monkeypatch):
    modal = types.ModuleType("modal")

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def include(self, _other):
            return None

        def function(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def local_entrypoint(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    class FakeSecret:
        @staticmethod
        def from_name(_name):
            return object()

    class FakeVolume:
        @staticmethod
        def from_name(_name, create_if_missing=False):
            return MagicMock()

    modal.App = FakeApp
    modal.Secret = FakeSecret
    modal.Volume = FakeVolume
    monkeypatch.setitem(sys.modules, "modal", modal)


def _load_pipeline_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]

    modal_app_package = types.ModuleType("modal_app")
    modal_app_package.__path__ = [str(repo_root / "modal_app")]
    monkeypatch.setitem(sys.modules, "modal_app", modal_app_package)

    images_module = types.ModuleType("modal_app.images")
    images_module.cpu_image = object()
    monkeypatch.setitem(sys.modules, "modal_app.images", images_module)

    data_build_module = types.ModuleType("modal_app.data_build")
    data_build_module.app = object()
    data_build_module.build_datasets = object()
    monkeypatch.setitem(sys.modules, "modal_app.data_build", data_build_module)

    calibration_module = types.ModuleType("modal_app.remote_calibration_runner")
    calibration_module.app = object()
    calibration_module.build_package_remote = object()
    calibration_module.PACKAGE_GPU_FUNCTIONS = {}
    monkeypatch.setitem(
        sys.modules,
        "modal_app.remote_calibration_runner",
        calibration_module,
    )

    local_area_module = types.ModuleType("modal_app.local_area")
    local_area_module.app = object()
    local_area_module.coordinate_publish = object()
    local_area_module.coordinate_national_publish = object()
    local_area_module.promote_publish = object()
    local_area_module.promote_national_publish = object()
    monkeypatch.setitem(sys.modules, "modal_app.local_area", local_area_module)

    policyengine_root = types.ModuleType("policyengine_us_data")
    policyengine_root.__path__ = [str(repo_root / "policyengine_us_data")]
    monkeypatch.setitem(sys.modules, "policyengine_us_data", policyengine_root)

    calibration_package = types.ModuleType("policyengine_us_data.calibration")
    calibration_package.__path__ = [
        str(repo_root / "policyengine_us_data" / "calibration")
    ]
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration",
        calibration_package,
    )

    local_h5_package = types.ModuleType("policyengine_us_data.calibration.local_h5")
    local_h5_package.__path__ = [
        str(repo_root / "policyengine_us_data" / "calibration" / "local_h5")
    ]
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5",
        local_h5_package,
    )

    validation_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.validation"
    )

    def tag_validation_errors(errors, source):
        return [{**error, "source": source} for error in errors]

    validation_module.tag_validation_errors = tag_validation_errors
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.validation",
        validation_module,
    )

    _install_fake_modal(monkeypatch)

    module_path = _module_path("modal_app", "pipeline.py")
    spec = importlib.util.spec_from_file_location(
        "modal_app.pipeline_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_meta(pipeline_module):
    return pipeline_module.RunMetadata(
        run_id="test_run",
        branch="main",
        sha="abc123",
        version="1.0.0",
        start_time="2026-03-19T12:00:00Z",
        status="running",
    )


def test_write_validation_diagnostics_writes_outputs_and_meta(monkeypatch, tmp_path):
    pipeline = _load_pipeline_module(monkeypatch)
    runs_dir = tmp_path / "runs"
    meta = _make_meta(pipeline)
    mock_vol = MagicMock()

    regional_result = {
        "validation_rows": [
            {
                "area_type": "state",
                "area_id": "CA",
                "district": "",
                "variable": "household_count",
                "target_name": "household_count",
                "period": 2024,
                "target_value": 100.0,
                "sim_value": 110.0,
                "error": 10.0,
                "rel_error": 0.1,
                "abs_error": 10.0,
                "rel_abs_error": 0.1,
                "sanity_check": "FAIL",
                "sanity_reason": "too_high",
                "in_training": True,
            }
        ],
        "validation_errors": [
            {
                "item": "state:CA",
                "error": "regional validator crashed",
                "code": "validation_exception",
                "details": {"traceback": "tb-regional"},
            }
        ],
    }
    national_result = {
        "national_validation": "national validation output",
        "validation_errors": [
            {
                "item": "national:US",
                "error": "national validator crashed",
                "code": "validation_exception",
                "details": {"traceback": "tb-national"},
            }
        ],
    }

    monkeypatch.setattr(pipeline, "RUNS_DIR", str(runs_dir))

    pipeline._write_validation_diagnostics(
        run_id="test_run",
        regional_result=regional_result,
        national_result=national_result,
        meta=meta,
        vol=mock_vol,
    )

    diag_dir = runs_dir / "test_run" / "diagnostics"
    csv_path = diag_dir / "validation_results.csv"
    errors_path = diag_dir / "validation_errors.json"
    national_path = diag_dir / "national_validation.txt"
    meta_path = runs_dir / "test_run" / "meta.json"

    assert csv_path.exists()
    assert errors_path.exists()
    assert national_path.exists()
    assert meta_path.exists()

    csv_lines = csv_path.read_text().strip().splitlines()
    assert len(csv_lines) == 2
    assert "area_type,area_id" in csv_lines[0]
    assert "state,CA" in csv_lines[1]

    assert json.loads(errors_path.read_text()) == [
        {
            "item": "state:CA",
            "error": "regional validator crashed",
            "code": "validation_exception",
            "details": {"traceback": "tb-regional"},
            "source": "regional",
        },
        {
            "item": "national:US",
            "error": "national validator crashed",
            "code": "validation_exception",
            "details": {"traceback": "tb-national"},
            "source": "national",
        },
    ]
    assert national_path.read_text() == "national validation output"

    assert meta.step_timings["validation"] == {
        "total_targets": 1,
        "sanity_failures": 1,
        "mean_rel_abs_error": 0.1,
        "validation_errors": 2,
        "worst_areas": [
            {
                "area": "state:CA",
                "mean_rae": 0.1,
                "sanity_fails": 1,
            }
        ],
    }

    saved_meta = json.loads(meta_path.read_text())
    assert saved_meta["step_timings"]["validation"]["validation_errors"] == 2
    assert mock_vol.commit.call_count == 2


def test_write_validation_diagnostics_skips_when_no_data(monkeypatch, tmp_path):
    pipeline = _load_pipeline_module(monkeypatch)
    runs_dir = tmp_path / "runs"
    meta = _make_meta(pipeline)
    mock_vol = MagicMock()

    monkeypatch.setattr(pipeline, "RUNS_DIR", str(runs_dir))

    pipeline._write_validation_diagnostics(
        run_id="test_run",
        regional_result={},
        national_result={},
        meta=meta,
        vol=mock_vol,
    )

    assert not (runs_dir / "test_run").exists()
    assert meta.step_timings == {}
    mock_vol.commit.assert_not_called()
