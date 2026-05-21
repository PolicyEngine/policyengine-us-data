from tests.unit.fixtures.calibration_package_stage_contract import (
    TARGET_CONFIG_PATH,
    calibration_package_contract,
    calibration_package_parameters,
    calibration_package_payload,
    calibration_package_payload_with_block_geoids,
    calibration_package_payload_with_cd_geoids,
    calibration_package_payload_without_geography,
    contract_input_paths,
    empty_matrix_calibration_package_payload,
    write_calibration_package_payload,
    write_non_mapping_calibration_package_payload,
)
from tests.unit.fixtures.geography import checksum_block_geoids, checksum_cd_geoids
from policyengine_us_data.stage_contracts import (
    StageContract,
    contract_from_json,
    contract_to_json,
    GeographyAssignmentSummary,
)
from policyengine_us_data.stage_contracts.calibration_package import (
    CALIBRATION_PACKAGE_CONTRACT_FILENAME,
    CalibrationPackageParameters,
    CalibrationPackageSummary,
    build_calibration_package_contract,
    load_calibration_package_payload,
    summarize_calibration_package,
    summarize_geography_assignment,
    validate_calibration_package_contract,
    validate_persisted_calibration_package_contract,
    write_calibration_package_contract,
)
from policyengine_us_data.utils.geography_checksum import (
    canonical_geography_checksum,
    hash_string_array,
)


def test_calibration_package_contract_records_stage_2_handoff(tmp_path):
    contract = calibration_package_contract(tmp_path)

    assert contract.stage_id == "2_build_calibration_package"
    assert contract.contract_type == "calibration_package"
    assert contract.run_id == "run-a"
    assert contract.code_sha == "abc123"
    assert contract.package_version == "1.98.2"
    assert {artifact.logical_name for artifact in contract.inputs} == {
        "source_imputed_stratified_extended_cps",
        "policy_data_db",
    }
    assert {artifact.logical_name for artifact in contract.outputs} == {
        "calibration_package"
    }
    assert all(artifact.sha256.startswith("sha256:") for artifact in contract.inputs)
    assert contract.outputs[0].media_type == "application/python-pickle"


def test_calibration_package_parameters_parse_runtime_args():
    params = CalibrationPackageParameters.from_runtime_args(
        workers=8,
        n_clones=430,
        target_config_path=TARGET_CONFIG_PATH,
        target_config_sha256="sha256:target-config",
        target_config_mode="explicit",
        skip_county=True,
        skip_source_impute=True,
        skip_takeup_rerandomize=False,
        chunked_matrix=True,
        chunk_size=25_000,
        parallel=True,
        num_matrix_workers=50,
    )

    assert params.to_dict() == {
        "chunk_size": 25_000,
        "chunked_matrix": True,
        "n_clones": 430,
        "num_matrix_workers": 50,
        "parallel_matrix": True,
        "skip_county": True,
        "skip_source_impute": True,
        "skip_takeup_rerandomize": False,
        "target_config": TARGET_CONFIG_PATH,
        "target_config_mode": "explicit",
        "target_config_sha256": "sha256:target-config",
        "workers": None,
    }


def test_calibration_package_parameters_require_identity_for_config_modes():
    try:
        CalibrationPackageParameters.from_runtime_args(
            workers=8,
            n_clones=430,
            target_config_path=TARGET_CONFIG_PATH,
            target_config_sha256=None,
            target_config_mode="explicit",
            skip_county=True,
            skip_source_impute=True,
            skip_takeup_rerandomize=False,
            chunked_matrix=False,
            chunk_size=25_000,
            parallel=False,
            num_matrix_workers=50,
        )
    except ValueError as exc:
        assert "target_config and target_config_sha256" in str(exc)
    else:
        raise AssertionError("Explicit target config mode should require checksum")


def test_calibration_package_parameters_accept_legacy_identity_fields_missing():
    params = CalibrationPackageParameters.from_dict(
        {
            "chunk_size": None,
            "chunked_matrix": False,
            "n_clones": 430,
            "num_matrix_workers": None,
            "parallel_matrix": False,
            "skip_county": True,
            "skip_source_impute": True,
            "skip_takeup_rerandomize": False,
            "target_config": TARGET_CONFIG_PATH,
            "workers": 8,
        }
    )

    assert params.target_config == TARGET_CONFIG_PATH
    assert params.target_config_mode is None
    assert params.target_config_sha256 is None


def test_calibration_package_parameters_reject_inconsistent_chunk_shape():
    try:
        CalibrationPackageParameters(
            workers=8,
            n_clones=430,
            target_config=None,
            target_config_sha256=None,
            target_config_mode="all_active_targets",
            skip_county=True,
            skip_source_impute=True,
            skip_takeup_rerandomize=False,
            chunked_matrix=True,
            chunk_size=25_000,
            parallel_matrix=False,
            num_matrix_workers=None,
        )
    except ValueError as exc:
        assert "workers must be None" in str(exc)
    else:
        raise AssertionError("Inconsistent chunked parameter shape should fail")


def test_calibration_package_summary_round_trips_through_schema():
    summary = summarize_calibration_package(calibration_package_payload())

    assert isinstance(summary, CalibrationPackageSummary)
    assert CalibrationPackageSummary.from_dict(summary.to_dict()) == summary


def test_geography_assignment_summary_round_trips_through_schema():
    summary = summarize_geography_assignment(calibration_package_payload())

    assert isinstance(summary, GeographyAssignmentSummary)
    assert GeographyAssignmentSummary.from_dict(summary.to_dict()) == summary
    assert summary.source_kind == "calibration_package"
    assert summary.n_records == 1
    assert summary.n_clones == 3
    assert summary.n_rows == 3
    assert summary.block_geoid_sha256.startswith("sha256:")
    assert summary.cd_geoid_sha256.startswith("sha256:")
    assert summary.canonical_geography_sha256.startswith("sha256:")


def test_geography_assignment_summary_hashes_are_dtype_width_independent():
    narrow = checksum_block_geoids(dtype="<U9")
    wide = checksum_block_geoids(dtype="<U15")
    narrow_cd = checksum_cd_geoids(dtype="<U4")
    wide_cd = checksum_cd_geoids(dtype="<U10")

    assert hash_string_array(narrow) == hash_string_array(wide)
    assert canonical_geography_checksum(
        block_geoid=narrow,
        cd_geoid=narrow_cd,
        n_records=1,
        n_clones=2,
    ) == canonical_geography_checksum(
        block_geoid=wide,
        cd_geoid=wide_cd,
        n_records=1,
        n_clones=2,
    )


def test_geography_assignment_summary_allows_unavailable_package_geography():
    package = calibration_package_payload_without_geography()

    summary = summarize_geography_assignment(package)

    assert summary.source_kind == "unavailable"
    assert summary.n_records == 1
    assert summary.n_clones == 3
    assert summary.n_rows is None
    assert summary.canonical_geography_sha256 is None


def test_calibration_package_contract_rejects_invalid_parameter_mapping(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)

    try:
        build_calibration_package_contract(
            package_path=package_path,
            dataset_path=dataset_path,
            db_path=db_path,
            package=package,
            parameters={"workers": 1},
            run_id="run-a",
            completed_at="2026-05-08T12:02:00Z",
        )
    except ValueError as exc:
        assert "missing required key" in str(exc)
    else:
        raise AssertionError("Invalid parameter mapping should fail")


def test_calibration_package_contract_normalizes_empty_run_id_to_none(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)

    contract = build_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="",
        completed_at="2026-05-08T12:02:00Z",
    )

    assert contract.run_id is None


def test_calibration_package_contract_records_matrix_summary(tmp_path):
    contract = calibration_package_contract(tmp_path)

    summary = contract.metadata["package_summary"]

    assert summary["matrix_shape"] == (2, 3)
    assert summary["matrix_nnz"] == 3
    assert summary["matrix_density"] == 0.5
    assert summary["n_targets"] == 2
    assert summary["target_name_count"] == 2
    assert summary["target_config_sha256"] == "sha256:target-config"
    assert summary["n_clones"] == 3
    assert summary["seed"] == 42
    assert summary["matrix_builder"] == "chunked"
    assert summary["has_initial_weights"] is True
    assert summary["has_cd_geoid"] is True
    assert summary["has_block_geoid"] is True
    assert summary["cd_geoid_length"] == 3
    assert summary["block_geoid_length"] == 3


def test_calibration_package_contract_records_geography_assignment(tmp_path):
    contract = calibration_package_contract(tmp_path)

    geography = contract.metadata["geography_assignment"]

    assert geography["source_kind"] == "calibration_package"
    assert geography["n_records"] == 1
    assert geography["n_clones"] == 3
    assert geography["n_rows"] == 3
    assert geography["has_block_geoid"] is True
    assert geography["has_cd_geoid"] is True
    assert geography["block_geoid_length"] == 3
    assert geography["cd_geoid_length"] == 3
    assert geography["block_geoid_sha256"].startswith("sha256:")
    assert geography["cd_geoid_sha256"].startswith("sha256:")
    assert geography["canonical_geography_sha256"].startswith("sha256:")


def test_calibration_package_contract_records_single_substage(tmp_path):
    contract = calibration_package_contract(tmp_path)

    assert len(contract.substages) == 1
    substage = contract.substages[0]
    assert substage.substage_id == "2a_matrix_build_calibration_target_construction"
    assert substage.status == "completed"
    assert substage.reuse_mode == "handoff"
    assert substage.outputs[0].logical_name == "calibration_package"
    assert substage.metadata == {}


def test_calibration_package_contract_json_round_trip_is_deterministic(tmp_path):
    contract = calibration_package_contract(tmp_path)

    payload = contract_to_json(contract)
    restored = contract_from_json(payload)

    assert restored == contract
    assert contract_to_json(restored) == payload


def test_calibration_package_contract_fingerprint_changes_with_geography(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    first = build_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )

    changed_path = tmp_path / "changed_calibration_package.pkl"
    changed_package = calibration_package_payload_with_block_geoids()
    write_calibration_package_payload(changed_path, changed_package)
    second = build_calibration_package_contract(
        package_path=changed_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=changed_package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )

    assert (
        first.metadata["geography_assignment"]
        != second.metadata["geography_assignment"]
    )
    assert (
        first.fingerprint.material["geography_assignment"]
        == first.metadata["geography_assignment"]
    )
    assert (
        second.fingerprint.material["geography_assignment"]
        == second.metadata["geography_assignment"]
    )
    assert (
        first.fingerprint.material["geography_assignment"]
        != second.fingerprint.material["geography_assignment"]
    )
    assert first.fingerprint != second.fingerprint


def test_calibration_package_summary_omits_bulky_payloads():
    summary = summarize_calibration_package(calibration_package_payload()).to_dict()

    assert "X_sparse" not in summary
    assert "targets_df" not in summary
    assert "target_names" not in summary
    assert "initial_weights" not in summary
    assert "cd_geoid" not in summary
    assert "block_geoid" not in summary


def test_calibration_package_geography_summary_rejects_mismatched_arrays():
    package = calibration_package_payload_with_cd_geoids(("0101", "0102"))

    try:
        summarize_geography_assignment(package)
    except ValueError as exc:
        assert "mismatched" in str(exc)
    else:
        raise AssertionError("Mismatched geography arrays should fail")


def test_calibration_package_summary_handles_empty_matrix():
    package = empty_matrix_calibration_package_payload()

    summary = summarize_calibration_package(package).to_dict()

    assert summary["matrix_shape"] == (0, 0)
    assert summary["matrix_nnz"] == 0
    assert summary["matrix_density"] == 0.0


def test_calibration_package_contract_rejects_missing_artifact(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = calibration_package_payload()

    try:
        build_calibration_package_contract(
            package_path=package_path,
            dataset_path=dataset_path,
            db_path=db_path,
            package=package,
            parameters=calibration_package_parameters(),
            run_id="run-a",
            completed_at="2026-05-08T12:02:00Z",
        )
    except FileNotFoundError as exc:
        assert "calibration package" in str(exc)
    else:
        raise AssertionError("Missing calibration package should fail")


def test_write_and_validate_calibration_package_contract(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)

    contract = write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )

    contract_path = tmp_path / CALIBRATION_PACKAGE_CONTRACT_FILENAME
    assert contract_path.exists()
    validated = validate_calibration_package_contract(
        package_path=package_path,
        contract_path=contract_path,
        package=package,
        dataset_path=dataset_path,
        db_path=db_path,
    )
    assert validated == contract


def test_validate_persisted_calibration_package_contract_loads_pickle(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    contract = write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )

    validated = validate_persisted_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
    )

    assert validated == contract


def test_write_and_validate_calibration_package_contract_without_geography(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(
        package_path,
        calibration_package_payload_without_geography(),
    )

    contract = write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )

    validated = validate_calibration_package_contract(
        package_path=package_path,
        package=package,
        dataset_path=dataset_path,
        db_path=db_path,
    )
    assert validated == contract
    assert contract.metadata["geography_assignment"]["source_kind"] == "unavailable"


def test_validate_calibration_package_contract_fails_on_stale_summary(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )
    changed_package = empty_matrix_calibration_package_payload()

    try:
        validate_calibration_package_contract(
            package_path=package_path,
            package=changed_package,
        )
    except ValueError as exc:
        assert "summary does not match" in str(exc)
    else:
        raise AssertionError("Stale calibration package summary should fail")


def test_validate_calibration_package_contract_fails_on_stale_geography(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )
    changed_package = calibration_package_payload_with_block_geoids()

    try:
        validate_calibration_package_contract(
            package_path=package_path,
            package=changed_package,
        )
    except ValueError as exc:
        assert "geography assignment" in str(exc)
    else:
        raise AssertionError("Stale calibration package geography should fail")


def test_validate_calibration_package_contract_fails_on_contract_geography(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    contract = write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )
    payload = contract.to_dict()
    payload["metadata"]["geography_assignment"]["canonical_geography_sha256"] = (
        "sha256:" + "0" * 64
    )
    contract_path = tmp_path / CALIBRATION_PACKAGE_CONTRACT_FILENAME
    contract_path.write_text(
        contract_to_json(StageContract.from_dict(payload)),
        encoding="utf-8",
    )

    try:
        validate_calibration_package_contract(
            package_path=package_path,
            contract_path=contract_path,
            package=package,
        )
    except ValueError as exc:
        assert "geography assignment" in str(exc)
    else:
        raise AssertionError("Stale contract geography should fail")


def test_validate_calibration_package_contract_fails_on_package_checksum(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )
    package_path.write_bytes(b"not the original pickle")

    try:
        validate_calibration_package_contract(package_path=package_path)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("Stale calibration package checksum should fail")


def test_validate_calibration_package_contract_requires_package_for_summary(tmp_path):
    dataset_path, db_path, package_path = contract_input_paths(tmp_path)
    package = write_calibration_package_payload(package_path)
    write_calibration_package_contract(
        package_path=package_path,
        dataset_path=dataset_path,
        db_path=db_path,
        package=package,
        parameters=calibration_package_parameters(),
        run_id="run-a",
        completed_at="2026-05-08T12:02:00Z",
    )

    try:
        validate_calibration_package_contract(package_path=package_path)
    except ValueError as exc:
        assert "package is required" in str(exc)
    else:
        raise AssertionError("Package payload should be required for full validation")


def test_load_calibration_package_payload_rejects_non_mapping(tmp_path):
    package_path = tmp_path / "calibration_package.pkl"
    write_non_mapping_calibration_package_payload(package_path)

    try:
        load_calibration_package_payload(package_path)
    except ValueError as exc:
        assert "must contain a mapping" in str(exc)
    else:
        raise AssertionError("Non-mapping package payload should fail")
