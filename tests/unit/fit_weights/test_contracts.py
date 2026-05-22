from pathlib import Path

from policyengine_us_data.fit_weights import (
    FITTED_WEIGHTS_CONTRACT_SCHEMA_VERSION,
    FitScope,
    FittedWeightsContractBuilder,
    FittedWeightsInputBundle,
    fitted_weights_contract_filename,
)
from policyengine_us_data.stage_contracts import contract_from_json, contract_to_json
from policyengine_us_data.stage_contracts.stages import STAGE_3_FIT_WEIGHTS
from policyengine_us_data.utils.step_manifest import sha256_file


def test_regional_contract_shape(
    stage2_contract_fixture,
    scoped_fit_files,
    fitted_weights_parameters: dict,
) -> None:
    files = scoped_fit_files(FitScope.REGIONAL)
    contract = _build_contract(
        stage2_contract_fixture,
        files,
        fitted_weights_parameters,
    )

    assert contract.stage_id == STAGE_3_FIT_WEIGHTS
    assert contract.contract_type == "fitted_weights"
    assert contract.run_id == "run-a"
    assert contract.parameters["scope"] == "regional"
    assert contract.metadata["scope"] == "regional"
    assert contract.metadata["schema_version"] == FITTED_WEIGHTS_CONTRACT_SCHEMA_VERSION
    assert contract.metadata["weight_summary"]["shape"] == (3,)
    assert contract.metadata["diagnostics_summary"]["diagnostics"]["row_count"] == 2
    assert contract.substages[0].substage_id == "3a_weight_fitting_regional"
    assert {artifact.logical_name for artifact in contract.inputs} == {
        "calibration_package",
        "calibration_package_contract",
    }
    assert {artifact.logical_name for artifact in contract.outputs} == {
        "fitted_weights_regional_weights",
        "fitted_weights_regional_geography",
        "fitted_weights_regional_run_config",
        "fitted_weights_regional_diagnostics",
        "fitted_weights_regional_epoch_log",
    }


def test_national_contract_shape(
    stage2_contract_fixture,
    scoped_fit_files,
    fitted_weights_parameters: dict,
) -> None:
    files = scoped_fit_files(FitScope.NATIONAL)
    params = {
        **fitted_weights_parameters,
        "scope": "national",
        "lambda_l0": 1e-4,
    }

    contract = _build_contract(stage2_contract_fixture, files, params)

    assert contract.parameters["scope"] == "national"
    assert contract.metadata["scope"] == "national"
    assert contract.substages[0].substage_id == "3b_weight_fitting_national"
    assert {artifact.logical_name for artifact in contract.outputs} == {
        "fitted_weights_national_weights",
        "fitted_weights_national_geography",
        "fitted_weights_national_run_config",
        "fitted_weights_national_diagnostics",
        "fitted_weights_national_epoch_log",
    }
    assert fitted_weights_contract_filename(FitScope.NATIONAL) == (
        "fitted_weights_national_contract.json"
    )


def test_contract_fingerprint_tracks_solver_parameters(
    stage2_contract_fixture,
    scoped_fit_files,
    fitted_weights_parameters: dict,
) -> None:
    files = scoped_fit_files(FitScope.REGIONAL)
    first = _build_contract(
        stage2_contract_fixture,
        files,
        fitted_weights_parameters,
    )
    second = _build_contract(
        stage2_contract_fixture,
        files,
        {**fitted_weights_parameters, "epochs": 3},
    )

    assert first.fingerprint.value != second.fingerprint.value


def test_contract_references_stage_2_package_contract_checksum(
    stage2_contract_fixture,
    scoped_fit_files,
    fitted_weights_parameters: dict,
) -> None:
    files = scoped_fit_files(FitScope.REGIONAL)
    contract = _build_contract(
        stage2_contract_fixture,
        files,
        fitted_weights_parameters,
    )

    contract_input = next(
        artifact
        for artifact in contract.inputs
        if artifact.logical_name == "calibration_package_contract"
    )
    assert contract_input.sha256 == (
        f"sha256:{sha256_file(stage2_contract_fixture.contract_path)}"
    )
    assert contract.metadata["package_contract_checksum"] == contract_input.sha256


def test_contract_round_trips_through_generic_stage_contract(
    tmp_path: Path,
    stage2_contract_fixture,
    scoped_fit_files,
    fitted_weights_parameters: dict,
) -> None:
    files = scoped_fit_files(FitScope.REGIONAL)
    builder = _builder(stage2_contract_fixture, files, fitted_weights_parameters)
    contract_path = builder.write(tmp_path / "fitted_weights_regional_contract.json")

    contract = contract_from_json(contract_path.read_text())

    assert contract == contract_from_json(contract_to_json(contract))
    assert contract.fingerprint.value.startswith("sha256:")


def _builder(
    stage2_contract_fixture,
    files,
    parameters: dict,
) -> FittedWeightsContractBuilder:
    return FittedWeightsContractBuilder(
        scope=files.scope,
        input_bundle=FittedWeightsInputBundle(
            scope=files.scope,
            calibration_package_path=stage2_contract_fixture.package_path,
            calibration_package_contract_path=stage2_contract_fixture.contract_path,
        ),
        parameters=parameters,
        artifacts_root=files.artifacts_root,
        diagnostics_root=files.diagnostics_root,
        run_id="run-a",
        started_at="2026-05-08T12:00:00+00:00",
        completed_at="2026-05-08T12:01:00+00:00",
        modal_call_id="fc-123",
    )


def _build_contract(
    stage2_contract_fixture,
    files,
    parameters: dict,
):
    return _builder(stage2_contract_fixture, files, parameters).build()
