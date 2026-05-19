"""Stage 3 fitted-weight specifications and artifact identities."""

from policyengine_us_data.fit_weights.artifacts import (
    FitArtifactLocation,
    FitArtifactRole,
    FitArtifactSpec,
    ScopedFitArtifacts,
    fit_artifacts_for_scope,
)
from policyengine_us_data.fit_weights.bundles import (
    FitResultBytes,
    FitWeightsBuildContext,
    FittedWeightsInputBundle,
    FittedWeightsOutputBundle,
    MissingFitWeightsOutputError,
)
from policyengine_us_data.fit_weights.specs import (
    FIT_BETA,
    FIT_LOG_FREQ,
    FIT_TARGET_CONFIG_PATH,
    FIT_WEIGHTS_SPEC_SCHEMA_VERSION,
    NATIONAL_FIT_LAMBDA_L0,
    NATIONAL_FIT_LAMBDA_L2,
    REGIONAL_FIT_LAMBDA_L0,
    REGIONAL_FIT_LAMBDA_L2,
    FitHyperparameters,
    FitScope,
    FittedWeightsSpec,
    fitted_weights_spec_for_scope,
)

__all__ = [
    "FIT_BETA",
    "FIT_LOG_FREQ",
    "FIT_TARGET_CONFIG_PATH",
    "FIT_WEIGHTS_SPEC_SCHEMA_VERSION",
    "NATIONAL_FIT_LAMBDA_L0",
    "NATIONAL_FIT_LAMBDA_L2",
    "REGIONAL_FIT_LAMBDA_L0",
    "REGIONAL_FIT_LAMBDA_L2",
    "FitArtifactLocation",
    "FitArtifactRole",
    "FitArtifactSpec",
    "FitHyperparameters",
    "FitResultBytes",
    "FitScope",
    "FitWeightsBuildContext",
    "FittedWeightsInputBundle",
    "FittedWeightsOutputBundle",
    "FittedWeightsSpec",
    "MissingFitWeightsOutputError",
    "ScopedFitArtifacts",
    "fit_artifacts_for_scope",
    "fitted_weights_spec_for_scope",
]
