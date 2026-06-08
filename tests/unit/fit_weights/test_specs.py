import pytest

from policyengine_us_data.fit_weights import (
    FIT_TARGET_CONFIG_PATH,
    NATIONAL_FIT_LAMBDA_L0,
    REGIONAL_FIT_LAMBDA_L0,
    FitHyperparameters,
    FitScope,
    fitted_weights_spec_for_scope,
)


def test_manifest_parameters_preserve_current_runtime_shape() -> None:
    spec = fitted_weights_spec_for_scope(FitScope.REGIONAL)

    params = spec.manifest_parameters(gpu="T4", epochs=200)

    assert params["scope"] == "regional"
    assert params["gpu"] == "T4"
    assert params["epochs"] == 200
    assert params["target_config"] == FIT_TARGET_CONFIG_PATH
    assert params["beta"] == pytest.approx(0.65)
    assert params["lambda_l0"] == pytest.approx(REGIONAL_FIT_LAMBDA_L0)
    assert params["lambda_l2"] == pytest.approx(1e-8)
    assert params["log_freq"] == 100
    assert params["fit_parameter_identity"].startswith("sha256:")


def test_national_spec_tracks_current_lambda_l0() -> None:
    spec = fitted_weights_spec_for_scope("national")

    assert spec.hyperparameters.lambda_l0 == pytest.approx(NATIONAL_FIT_LAMBDA_L0)
    assert spec.runtime_kwargs()["lambda_l0"] == pytest.approx(0.0)


def test_fit_parameter_identity_is_deterministic() -> None:
    spec = fitted_weights_spec_for_scope(FitScope.REGIONAL)

    first = spec.parameter_identity(gpu="A10", epochs=5)
    second = spec.parameter_identity(gpu="A10", epochs=5)

    assert first == second


def test_fit_parameter_identity_tracks_scope() -> None:
    regional = fitted_weights_spec_for_scope(FitScope.REGIONAL)
    national = fitted_weights_spec_for_scope(FitScope.NATIONAL)

    assert regional.parameter_identity(gpu="T4", epochs=1) != (
        national.parameter_identity(gpu="T4", epochs=1)
    )


def test_fit_scope_rejects_unknown_scopes() -> None:
    with pytest.raises(ValueError, match="Unknown fit scope"):
        FitScope.parse("county")


def test_fit_hyperparameters_reject_invalid_identity_values() -> None:
    with pytest.raises(ValueError, match="lambda_l0"):
        FitHyperparameters(
            target_config=FIT_TARGET_CONFIG_PATH,
            beta=0.65,
            lambda_l0=-1.0,
            lambda_l2=1e-8,
            log_freq=100,
        )
