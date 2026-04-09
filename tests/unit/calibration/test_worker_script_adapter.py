import importlib.util
import json
from pathlib import Path
import sys
import types


def _module_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[3].joinpath(*parts)


def _load_worker_script_module():
    module_path = _module_path("modal_app", "worker_script.py")
    spec = importlib.util.spec_from_file_location(
        "worker_script_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_requires_requests_or_work_items(monkeypatch):
    worker_script = _load_worker_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker_script.py",
            "--weights-path",
            "/tmp/weights.npy",
            "--dataset-path",
            "/tmp/source.h5",
            "--db-path",
            "/tmp/policy_data.db",
            "--output-dir",
            "/tmp/output",
        ],
    )

    try:
        worker_script.main()
    except ValueError as error:
        assert str(error) == "Either --requests-json or --work-items is required"
    else:
        raise AssertionError("Expected ValueError when no request input is provided")


def test_main_delegates_to_worker_service_and_emits_structured_json(
    monkeypatch,
    capsys,
):
    worker_script = _load_worker_script_module()
    calls = {}

    takeup = types.ModuleType("policyengine_us_data.utils.takeup")
    takeup.SIMPLE_TAKEUP_VARS = [{"variable": "snap"}]
    monkeypatch.setitem(sys.modules, "policyengine_us_data.utils.takeup", takeup)

    contracts = types.ModuleType("policyengine_us_data.calibration.local_h5.contracts")

    class FakeAreaBuildRequest:
        @staticmethod
        def from_dict(payload):
            return {"parsed_request": dict(payload)}

    class FakeValidationPolicy:
        def __init__(self, enabled=True):
            self.enabled = enabled

    contracts.AreaBuildRequest = FakeAreaBuildRequest
    contracts.ValidationPolicy = FakeValidationPolicy
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.contracts",
        contracts,
    )

    package_geography = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.package_geography"
    )
    package_geography.require_calibration_package_path = lambda path: Path(path)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.package_geography",
        package_geography,
    )

    worker_service = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.worker_service"
    )

    class FakeWorkerResult:
        def to_dict(self):
            return {
                "completed": [],
                "failed": [],
                "worker_issues": [],
            }

    class FakeWorkerSession:
        @classmethod
        def load(cls, **kwargs):
            calls["session_kwargs"] = kwargs
            return types.SimpleNamespace(
                requested_n_clones=430,
                n_clones=430,
                geography_source="package",
                geography_warnings=(),
                geography=types.SimpleNamespace(n_clones=430, n_records=2),
                source_snapshot=types.SimpleNamespace(n_households=2),
            )

    class FakeLocalH5WorkerService:
        def run(self, session, requests, initial_failures=()):
            calls["service_run"] = {
                "session": session,
                "requests": requests,
                "initial_failures": initial_failures,
            }
            return FakeWorkerResult()

    worker_service.WorkerSession = FakeWorkerSession
    worker_service.LocalH5WorkerService = FakeLocalH5WorkerService
    worker_service.load_validation_context = lambda **kwargs: None
    worker_service.build_requests_from_work_items = (
        lambda *args, **kwargs: ((), ())
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.worker_service",
        worker_service,
    )

    publish_local_area = types.ModuleType(
        "policyengine_us_data.calibration.publish_local_area"
    )
    publish_local_area.AT_LARGE_DISTRICTS = {0, 98}
    publish_local_area.NYC_COUNTY_FIPS = {"36061"}
    publish_local_area.SUB_ENTITIES = ["tax_unit", "spm_unit"]
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.publish_local_area",
        publish_local_area,
    )

    calibration_utils = types.ModuleType(
        "policyengine_us_data.calibration.calibration_utils"
    )
    calibration_utils.STATE_CODES = {1: "AL"}
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.calibration_utils",
        calibration_utils,
    )

    source_dataset = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.source_dataset"
    )

    class FakeReader:
        def __init__(self, sub_entities):
            calls["reader_sub_entities"] = tuple(sub_entities)

    source_dataset.PolicyEngineDatasetReader = FakeReader
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.source_dataset",
        source_dataset,
    )

    requests_json = json.dumps(
        [
            {
                "area_type": "state",
                "area_id": "CA",
                "display_name": "CA",
                "output_relative_path": "states/CA.h5",
            }
        ]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker_script.py",
            "--requests-json",
            requests_json,
            "--weights-path",
            "/tmp/weights.npy",
            "--dataset-path",
            "/tmp/source.h5",
            "--db-path",
            "/tmp/policy_data.db",
            "--output-dir",
            "/tmp/output",
        ],
    )

    worker_script.main()
    captured = capsys.readouterr()

    assert calls["reader_sub_entities"] == ("tax_unit", "spm_unit")
    assert calls["session_kwargs"]["weights_path"] == Path("/tmp/weights.npy")
    assert calls["session_kwargs"]["dataset_path"] == Path("/tmp/source.h5")
    assert calls["service_run"]["requests"] == (
        {
            "parsed_request": {
                "area_type": "state",
                "area_id": "CA",
                "display_name": "CA",
                "output_relative_path": "states/CA.h5",
            }
        },
    )
    assert json.loads(captured.out) == {
        "completed": [],
        "failed": [],
        "worker_issues": [],
    }
