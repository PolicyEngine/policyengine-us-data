import importlib.util
from pathlib import Path
import sys
import types


def _load_validation_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
        / "validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "worker_script_validation_helpers",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_worker_script_module(monkeypatch):
    validation_module = _load_validation_module()

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
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.validation",
        validation_module,
    )

    module_path = (
        Path(__file__).resolve().parents[3] / "modal_app" / "worker_script.py"
    )
    spec = importlib.util.spec_from_file_location("worker_script_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_in_subprocess_passes_dataset_path(monkeypatch):
    worker_script = _load_worker_script_module(monkeypatch)
    calls = {}

    fake_policyengine_us = types.ModuleType("policyengine_us")

    class FakeMicrosimulation:
        def __init__(self, dataset):
            calls["microsimulation_dataset"] = dataset

    fake_policyengine_us.Microsimulation = FakeMicrosimulation
    monkeypatch.setitem(sys.modules, "policyengine_us", fake_policyengine_us)

    fake_sqlalchemy = types.ModuleType("sqlalchemy")

    def fake_create_engine(url):
        calls["engine_url"] = url
        return "fake-engine"

    fake_sqlalchemy.create_engine = fake_create_engine
    monkeypatch.setitem(sys.modules, "sqlalchemy", fake_sqlalchemy)

    fake_validate_staging = types.ModuleType(
        "policyengine_us_data.calibration.validate_staging"
    )

    def fake_build_variable_entity_map(sim):
        calls["variable_entity_map_sim"] = sim
        return {"household_weight": "household"}

    def fake_validate_area(**kwargs):
        calls["validate_area_kwargs"] = kwargs
        return [{"sanity_check": "PASS", "rel_abs_error": 0.1}]

    fake_validate_staging._build_variable_entity_map = fake_build_variable_entity_map
    fake_validate_staging.validate_area = fake_validate_area
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.validate_staging",
        fake_validate_staging,
    )

    rows = worker_script._validate_in_subprocess(
        h5_path="/tmp/CA.h5",
        area_type="states",
        area_id="CA",
        display_id="CA",
        area_targets=[{"target": "population"}],
        area_training=[True],
        constraints_map={1: []},
        db_path="/tmp/policy_data.db",
        period=2024,
    )

    assert rows == [{"sanity_check": "PASS", "rel_abs_error": 0.1}]
    assert calls["microsimulation_dataset"] == "/tmp/CA.h5"
    assert calls["engine_url"] == "sqlite:////tmp/policy_data.db"
    assert calls["validate_area_kwargs"]["dataset_path"] == "/tmp/CA.h5"
    assert calls["validate_area_kwargs"]["sim"] is calls["variable_entity_map_sim"]


def test_record_validation_success_writes_rows_and_summary(monkeypatch):
    worker_script = _load_worker_script_module(monkeypatch)
    results = {"validation_rows": [], "validation_summary": {}}

    worker_script._record_validation_success(
        results,
        "state:CA",
        [
            {"sanity_check": "PASS", "rel_abs_error": 0.1},
            {"sanity_check": "FAIL", "rel_abs_error": 0.3},
        ],
    )

    assert len(results["validation_rows"]) == 2
    assert results["validation_summary"]["state:CA"] == {
        "n_targets": 2,
        "n_sanity_fail": 1,
        "mean_rel_abs_error": 0.2,
    }


def test_record_validation_error_writes_structured_entry(monkeypatch):
    worker_script = _load_worker_script_module(monkeypatch)
    results = {"validation_errors": []}

    try:
        raise ValueError("broken validator")
    except ValueError as error:
        entry = worker_script._record_validation_error(
            results,
            "district:CA-12",
            error,
        )

    assert entry["item"] == "district:CA-12"
    assert entry["error"] == "broken validator"
    assert "ValueError" in entry["traceback"]
    assert results["validation_errors"] == [entry]
