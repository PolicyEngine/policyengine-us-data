import json
import importlib.util
from pathlib import Path
import sys

import pytest


def _load_contracts_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
        / "contracts.py"
    )
    spec = importlib.util.spec_from_file_location("local_h5_contracts", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load_contracts_module()
AreaBuildRequest = contracts.AreaBuildRequest
AreaBuildResult = contracts.AreaBuildResult
AreaFilter = contracts.AreaFilter
PublishingInputBundle = contracts.PublishingInputBundle
ValidationIssue = contracts.ValidationIssue
ValidationPolicy = contracts.ValidationPolicy
ValidationResult = contracts.ValidationResult
WorkerResult = contracts.WorkerResult


def test_area_filter_validates_eq_vs_in_shape():
    AreaFilter(geography_field="state_fips", op="eq", value=6)
    AreaFilter(geography_field="county_fips", op="in", value=("06037", "06059"))

    with pytest.raises(ValueError, match="must be a tuple"):
        AreaFilter(geography_field="county_fips", op="in", value="06037")

    with pytest.raises(ValueError, match="must not be a tuple"):
        AreaFilter(geography_field="state_fips", op="eq", value=(6, 12))


def test_area_build_request_national_defaults():
    request = AreaBuildRequest.national()

    assert request.area_type == "national"
    assert request.area_id == "US"
    assert request.output_relative_path == "national/US.h5"
    assert request.validation_geo_level == "national"
    assert request.validation_geographic_ids == ("US",)
    assert request.filters == ()


def test_area_build_request_requires_validation_level_if_ids_provided():
    with pytest.raises(ValueError, match="validation_geo_level"):
        AreaBuildRequest(
            area_type="district",
            area_id="CA-12",
            display_name="CA-12",
            output_relative_path="districts/CA-12.h5",
            validation_geographic_ids=("612",),
        )


def test_publishing_input_bundle_required_paths_and_json_dict():
    bundle = PublishingInputBundle(
        weights_path=Path("/tmp/weights.npy"),
        source_dataset_path=Path("/tmp/source.h5"),
        target_db_path=Path("/tmp/policy_data.db"),
        calibration_package_path=Path("/tmp/calibration_package.pkl"),
        run_config_path=Path("/tmp/config.json"),
        run_id="1.0.0_abc",
        version="1.0.0",
        n_clones=430,
        seed=42,
    )

    assert bundle.required_paths() == (
        Path("/tmp/weights.npy"),
        Path("/tmp/source.h5"),
        Path("/tmp/policy_data.db"),
        Path("/tmp/calibration_package.pkl"),
    )
    assert json.loads(json.dumps(bundle.to_dict()))["version"] == "1.0.0"


def test_validation_policy_defaults_are_conservative():
    policy = ValidationPolicy()

    assert policy.enabled is True
    assert policy.fail_on_exception is False
    assert policy.fail_on_validation_failure is False
    assert policy.run_sanity_checks is True
    assert policy.run_target_validation is True
    assert policy.run_national_validation is True


def test_completed_area_build_result_requires_output_path_and_no_build_error():
    request = AreaBuildRequest.national()

    with pytest.raises(ValueError, match="requires output_path"):
        AreaBuildResult(
            request=request,
            build_status="completed",
        )

    with pytest.raises(ValueError, match="must not include build_error"):
        AreaBuildResult(
            request=request,
            build_status="completed",
            output_path=Path("/tmp/US.h5"),
            build_error="should not be here",
        )


def test_failed_area_build_result_requires_build_error():
    request = AreaBuildRequest.national()

    with pytest.raises(ValueError, match="requires build_error"):
        AreaBuildResult(
            request=request,
            build_status="failed",
        )


def test_worker_result_enforces_completed_and_failed_buckets():
    request = AreaBuildRequest.national()
    completed = AreaBuildResult(
        request=request,
        build_status="completed",
        output_path=Path("/tmp/US.h5"),
    )
    failed = AreaBuildResult(
        request=request,
        build_status="failed",
        build_error="boom",
    )

    result = WorkerResult(completed=(completed,), failed=(failed,))
    assert result.all_results() == (completed, failed)

    with pytest.raises(ValueError, match="completed"):
        WorkerResult(completed=(failed,), failed=())

    with pytest.raises(ValueError, match="failed"):
        WorkerResult(completed=(), failed=(completed,))


def test_worker_result_and_validation_result_are_json_serializable():
    request = AreaBuildRequest(
        area_type="state",
        area_id="CA",
        display_name="California",
        output_relative_path="states/CA.h5",
        filters=(AreaFilter(geography_field="state_fips", op="eq", value=6),),
        validation_geo_level="state",
        validation_geographic_ids=("6",),
        metadata={"takeup_filter": "snap,ssi"},
    )
    validation = ValidationResult(
        status="failed",
        rows=({"target_name": "population", "rel_abs_error": 0.12},),
        issues=(
            ValidationIssue(
                code="sanity_fail",
                message="population exceeded ceiling",
                severity="error",
                details={"target_name": "population"},
            ),
        ),
        summary={"n_targets": 1, "n_fail": 1},
    )
    completed = AreaBuildResult(
        request=request,
        build_status="completed",
        output_path=Path("/tmp/states/CA.h5"),
        validation=validation,
    )
    worker_result = WorkerResult(
        completed=(completed,),
        failed=(),
        worker_issues=(
            ValidationIssue(
                code="partial_validation",
                message="one validator retried",
                severity="warning",
            ),
        ),
    )

    payload = worker_result.to_dict()
    roundtrip = json.loads(json.dumps(payload))

    assert roundtrip["completed"][0]["request"]["area_id"] == "CA"
    assert roundtrip["completed"][0]["validation"]["status"] == "failed"
    assert roundtrip["completed"][0]["output_path"] == "/tmp/states/CA.h5"
    assert roundtrip["worker_issues"][0]["severity"] == "warning"


def test_contracts_round_trip_from_dict():
    request = AreaBuildRequest(
        area_type="district",
        area_id="CA-12",
        display_name="CA-12",
        output_relative_path="districts/CA-12.h5",
        filters=(
            AreaFilter(geography_field="cd_geoid", op="in", value=("612",)),
        ),
        validation_geo_level="district",
        validation_geographic_ids=("612",),
        metadata={"source": "catalog"},
    )
    validation = ValidationResult(
        status="error",
        issues=(
            ValidationIssue(
                code="validation_exception",
                message="validator crashed",
                severity="error",
                details={"traceback": "boom"},
            ),
        ),
        summary={"n_targets": 0},
    )
    result = WorkerResult(
        completed=(
            AreaBuildResult(
                request=request,
                build_status="completed",
                output_path=Path("/tmp/districts/CA-12.h5"),
                validation=validation,
            ),
        ),
        failed=(),
    )

    restored_request = AreaBuildRequest.from_dict(request.to_dict())
    restored_result = WorkerResult.from_dict(result.to_dict())

    assert restored_request == request
    assert restored_result.completed[0].request == request
    assert restored_result.completed[0].validation.issues[0].code == "validation_exception"
