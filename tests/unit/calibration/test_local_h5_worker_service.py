import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np


def _module_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[3].joinpath(*parts)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_package_hierarchy(monkeypatch):
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

    contracts = _load_module(
        "policyengine_us_data.calibration.local_h5.contracts",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "contracts.py",
        ),
    )
    _load_module(
        "policyengine_us_data.calibration.local_h5.validation",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "validation.py",
        ),
    )
    _load_module(
        "policyengine_us_data.calibration.local_h5.weights",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "weights.py",
        ),
    )

    builder_module = types.ModuleType("policyengine_us_data.calibration.local_h5.builder")
    builder_module.LocalAreaDatasetBuilder = type("LocalAreaDatasetBuilder", (), {})
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.builder",
        builder_module,
    )

    package_geo_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.package_geography"
    )
    package_geo_module.CalibrationPackageGeographyLoader = type(
        "CalibrationPackageGeographyLoader", (), {}
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.package_geography",
        package_geo_module,
    )

    source_dataset_module = types.ModuleType(
        "policyengine_us_data.calibration.local_h5.source_dataset"
    )

    @dataclass(frozen=True)
    class FakeSourceDatasetSnapshot:
        dataset_path: Path
        time_period: int
        household_ids: np.ndarray

        @property
        def n_households(self):
            return int(len(self.household_ids))

    source_dataset_module.SourceDatasetSnapshot = FakeSourceDatasetSnapshot
    source_dataset_module.PolicyEngineDatasetReader = type(
        "PolicyEngineDatasetReader", (), {}
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.source_dataset",
        source_dataset_module,
    )

    writer_module = types.ModuleType("policyengine_us_data.calibration.local_h5.writer")
    writer_module.H5Writer = type("H5Writer", (), {})
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us_data.calibration.local_h5.writer",
        writer_module,
    )

    worker_service = _load_module(
        "policyengine_us_data.calibration.local_h5.worker_service",
        _module_path(
            "policyengine_us_data",
            "calibration",
            "local_h5",
            "worker_service.py",
        ),
    )
    return contracts, source_dataset_module, worker_service


def test_worker_session_loads_source_geography_and_weights_once(monkeypatch, tmp_path):
    contracts, source_dataset_module, worker_service = _install_fake_package_hierarchy(
        monkeypatch
    )
    ValidationPolicy = contracts.ValidationPolicy
    WorkerSession = worker_service.WorkerSession
    ValidationContext = worker_service.ValidationContext

    weights_path = tmp_path / "weights.npy"
    np.save(weights_path, np.asarray([1.0, 0.0, 2.0, 0.0], dtype=float))
    dataset_path = tmp_path / "source.h5"
    output_dir = tmp_path / "output"

    snapshot = source_dataset_module.SourceDatasetSnapshot(
        dataset_path=dataset_path,
        time_period=2024,
        household_ids=np.asarray([10, 20]),
    )
    source_calls = []
    geo_calls = []

    class FakeSourceReader:
        def load(self, path):
            source_calls.append(Path(path))
            return snapshot

    class FakeGeographyLoader:
        def resolve_for_weights(
            self,
            *,
            package_path,
            weights_length,
            n_records,
            n_clones,
            seed,
            allow_seed_fallback,
        ):
            geo_calls.append(
                {
                    "package_path": package_path,
                    "weights_length": weights_length,
                    "n_records": n_records,
                    "n_clones": n_clones,
                    "seed": seed,
                    "allow_seed_fallback": allow_seed_fallback,
                }
            )
            return types.SimpleNamespace(
                geography=types.SimpleNamespace(n_records=2, n_clones=2),
                source="package",
                warnings=("exact geography",),
            )

    validation_context = ValidationContext(
        validation_targets=(),
        training_mask_full=np.asarray([], dtype=bool),
        constraints_map={},
        db_path=tmp_path / "policy_data.db",
        period=2024,
    )

    session = WorkerSession.load(
        weights_path=weights_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        calibration_package_path=tmp_path / "calibration_package.pkl",
        requested_n_clones=430,
        seed=99,
        takeup_filter=("snap", "wic"),
        validation_policy=ValidationPolicy(enabled=False),
        validation_context=validation_context,
        source_reader=FakeSourceReader(),
        geography_loader=FakeGeographyLoader(),
    )

    assert source_calls == [dataset_path]
    assert geo_calls == [
        {
            "package_path": tmp_path / "calibration_package.pkl",
            "weights_length": 4,
            "n_records": 2,
            "n_clones": 2,
            "seed": 99,
            "allow_seed_fallback": True,
        }
    ]
    np.testing.assert_array_equal(session.weights, np.asarray([1.0, 0.0, 2.0, 0.0]))
    assert session.source_snapshot is snapshot
    assert session.geography_source == "package"
    assert session.geography_warnings == ("exact geography",)
    assert session.n_clones == 2
    assert session.takeup_filter == ("snap", "wic")


def test_local_h5_worker_service_handles_mixed_chunk_results(monkeypatch):
    contracts, source_dataset_module, worker_service = _install_fake_package_hierarchy(
        monkeypatch
    )
    AreaBuildRequest = contracts.AreaBuildRequest
    AreaFilter = contracts.AreaFilter
    ValidationResult = contracts.ValidationResult
    LocalH5WorkerService = worker_service.LocalH5WorkerService
    ValidationContext = worker_service.ValidationContext
    WorkerSession = worker_service.WorkerSession

    snapshot = source_dataset_module.SourceDatasetSnapshot(
        dataset_path=Path("/tmp/source.h5"),
        time_period=2024,
        household_ids=np.asarray([10, 20]),
    )
    session = WorkerSession(
        source_snapshot=snapshot,
        weights=np.asarray([1.0, 0.0, 0.0, 2.0], dtype=float),
        geography=types.SimpleNamespace(),
        output_dir=Path("/tmp/output"),
        takeup_filter=("snap",),
        validation_policy=contracts.ValidationPolicy(enabled=True),
        validation_context=ValidationContext(
            validation_targets=(),
            training_mask_full=np.asarray([], dtype=bool),
            constraints_map={},
            db_path=Path("/tmp/policy_data.db"),
            period=2024,
        ),
    )
    request_ok = AreaBuildRequest(
        area_type="state",
        area_id="CA",
        display_name="CA",
        output_relative_path="states/CA.h5",
    )
    request_fail = AreaBuildRequest(
        area_type="district",
        area_id="CA-12",
        display_name="CA-12",
        output_relative_path="districts/CA-12.h5",
        filters=(
            AreaFilter(
                geography_field="cd_geoid",
                op="in",
                value=("0612",),
            ),
        ),
    )

    builder_calls = []

    class FakeBuilder:
        def build(self, **kwargs):
            builder_calls.append(kwargs)
            if kwargs["filters"] == request_fail.filters:
                raise ValueError("builder exploded")
            return types.SimpleNamespace(
                payload="payload",
                selection=types.SimpleNamespace(),
                reindexed=types.SimpleNamespace(),
                time_period=2024,
            )

    writer_calls = []

    class FakeWriter:
        def write_payload(self, payload, output_path):
            writer_calls.append(("write", payload, Path(output_path)))
            return Path(output_path)

        def verify_output(self, output_path, *, time_period):
            writer_calls.append(("verify", Path(output_path), time_period))
            return {"household_count": 1}

    validator_calls = []

    def fake_validator(output_path, request, session_obj):
        validator_calls.append((Path(output_path), request.area_id, session_obj))
        return ValidationResult(
            status="passed",
            rows=(),
            summary={"n_targets": 0, "n_sanity_fail": 0, "mean_rel_abs_error": 0.0},
        )

    service = LocalH5WorkerService(
        builder=FakeBuilder(),
        writer=FakeWriter(),
        validator=fake_validator,
    )

    result = service.run(session, (request_ok, request_fail))

    assert [item.request.area_id for item in result.completed] == ["CA"]
    assert [item.request.area_id for item in result.failed] == ["CA-12"]
    assert result.failed[0].build_error == "builder exploded"
    assert builder_calls[0]["source"] is snapshot
    assert builder_calls[1]["source"] is snapshot
    assert writer_calls == [
        ("write", "payload", Path("/tmp/output/states/CA.h5")),
        ("verify", Path("/tmp/output/states/CA.h5"), 2024),
    ]
    assert validator_calls[0][1] == "CA"


def test_local_h5_worker_service_records_validation_exception(monkeypatch):
    contracts, source_dataset_module, worker_service = _install_fake_package_hierarchy(
        monkeypatch
    )
    AreaBuildRequest = contracts.AreaBuildRequest
    LocalH5WorkerService = worker_service.LocalH5WorkerService
    ValidationContext = worker_service.ValidationContext
    WorkerSession = worker_service.WorkerSession

    snapshot = source_dataset_module.SourceDatasetSnapshot(
        dataset_path=Path("/tmp/source.h5"),
        time_period=2024,
        household_ids=np.asarray([10]),
    )
    session = WorkerSession(
        source_snapshot=snapshot,
        weights=np.asarray([1.0], dtype=float),
        geography=types.SimpleNamespace(),
        output_dir=Path("/tmp/output"),
        validation_policy=contracts.ValidationPolicy(enabled=True),
        validation_context=ValidationContext(
            validation_targets=(),
            training_mask_full=np.asarray([], dtype=bool),
            constraints_map={},
            db_path=Path("/tmp/policy_data.db"),
            period=2024,
        ),
    )
    request = AreaBuildRequest.national()

    class FakeBuilder:
        def build(self, **_kwargs):
            return types.SimpleNamespace(
                payload="payload",
                selection=types.SimpleNamespace(),
                reindexed=types.SimpleNamespace(),
                time_period=2024,
            )

    class FakeWriter:
        def write_payload(self, payload, output_path):
            return Path(output_path)

        def verify_output(self, output_path, *, time_period):
            return {}

    def exploding_validator(*_args, **_kwargs):
        raise RuntimeError("validator crashed")

    service = LocalH5WorkerService(
        builder=FakeBuilder(),
        writer=FakeWriter(),
        validator=exploding_validator,
    )
    result = service.run(session, (request,))

    assert len(result.completed) == 1
    validation = result.completed[0].validation
    assert validation.status == "error"
    assert validation.issues[0].code == "validation_exception"
    assert validation.issues[0].message == "validator crashed"


def test_worker_result_to_legacy_dict_flattens_structured_results(monkeypatch):
    contracts, _, worker_service = _install_fake_package_hierarchy(monkeypatch)
    AreaBuildRequest = contracts.AreaBuildRequest
    AreaBuildResult = contracts.AreaBuildResult
    ValidationIssue = contracts.ValidationIssue
    ValidationResult = contracts.ValidationResult
    WorkerResult = contracts.WorkerResult

    completed = AreaBuildResult(
        request=AreaBuildRequest(
            area_type="state",
            area_id="CA",
            display_name="CA",
            output_relative_path="states/CA.h5",
        ),
        build_status="completed",
        output_path=Path("/tmp/output/states/CA.h5"),
        validation=ValidationResult(
            status="failed",
            rows=(
                {"sanity_check": "FAIL", "rel_abs_error": 0.3},
                {"sanity_check": "PASS", "rel_abs_error": 0.1},
            ),
            summary={"n_targets": 2, "n_sanity_fail": 1, "mean_rel_abs_error": 0.2},
        ),
    )
    failed = AreaBuildResult(
        request=AreaBuildRequest(
            area_type="district",
            area_id="CA-12",
            display_name="CA-12",
            output_relative_path="districts/CA-12.h5",
        ),
        build_status="failed",
        build_error="build crashed",
    )
    result = WorkerResult(
        completed=(completed,),
        failed=(failed,),
        worker_issues=(
            ValidationIssue(
                code="session_warning",
                message="stale cache",
                severity="warning",
                details={"path": "/tmp/cache"},
            ),
        ),
    )

    payload = worker_service.worker_result_to_legacy_dict(result)

    assert payload["completed"] == ["state:CA"]
    assert payload["failed"] == ["district:CA-12"]
    assert payload["validation_rows"] == [
        {"sanity_check": "FAIL", "rel_abs_error": 0.3},
        {"sanity_check": "PASS", "rel_abs_error": 0.1},
    ]
    assert payload["validation_summary"]["state:CA"] == {
        "n_targets": 2,
        "n_sanity_fail": 1,
        "mean_rel_abs_error": 0.2,
    }
    assert payload["errors"][0] == {
        "item": "district:CA-12",
        "error": "build crashed",
    }
    assert payload["errors"][1] == {
        "item": "worker",
        "error": "stale cache",
        "code": "session_warning",
        "details": {"path": "/tmp/cache"},
    }


def test_build_requests_from_work_items_handles_city_and_invalid_items(monkeypatch):
    contracts, _, worker_service = _install_fake_package_hierarchy(monkeypatch)

    geography = types.SimpleNamespace(
        cd_geoid=np.asarray(["3607", "3610", "0101"], dtype=str),
        county_fips=np.asarray(["36061", "36047", "01001"], dtype=str),
    )

    requests, failures = worker_service.build_requests_from_work_items(
        (
            {"type": "city", "id": "NYC"},
            {"type": "unknown", "id": "mystery"},
        ),
        geography=geography,
        state_codes={36: "NY", 1: "AL"},
        at_large_districts={0, 98},
        nyc_county_fips={"36061", "36047", "36081"},
    )

    assert len(requests) == 1
    city_request = requests[0]
    assert city_request.area_type == "city"
    assert city_request.output_relative_path == "cities/NYC.h5"
    assert city_request.validation_geo_level == "district"
    assert city_request.validation_geographic_ids == ("3607", "3610")
    assert city_request.filters[0].geography_field == "county_fips"
    assert len(failures) == 1
    assert failures[0].request.area_type == "custom"
    assert failures[0].build_error == "Unknown item type: unknown"
