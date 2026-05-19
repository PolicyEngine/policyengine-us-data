"""Scoped Stage 3 fitted-weight parameter specifications."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.stage_contracts.fingerprints import fingerprint_material

FIT_WEIGHTS_SPEC_SCHEMA_VERSION = "1"
FIT_TARGET_CONFIG_PATH = "policyengine_us_data/calibration/target_config.yaml"
FIT_BETA = 0.65
FIT_LOG_FREQ = 100
REGIONAL_FIT_LAMBDA_L0 = 1e-7
REGIONAL_FIT_LAMBDA_L2 = 1e-8
NATIONAL_FIT_LAMBDA_L0 = 1e-4
NATIONAL_FIT_LAMBDA_L2 = 1e-12


class FitScope(str, Enum):
    """Supported fitted-weight output scopes."""

    REGIONAL = "regional"
    NATIONAL = "national"

    @classmethod
    def parse(cls, value: "FitScope | str") -> "FitScope":
        """Return a `FitScope` or raise a clear error for unknown scopes."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            allowed = ", ".join(scope.value for scope in cls)
            raise ValueError(
                f"Unknown fit scope {value!r}; expected one of {allowed}"
            ) from exc


@dataclass(frozen=True)
class FitHyperparameters:
    """Hyperparameters that define one fitted-weight optimization run."""

    target_config: str
    beta: float
    lambda_l0: float
    lambda_l2: float
    log_freq: int
    learning_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.target_config:
            raise ValueError("target_config must be non-empty")
        for name in ("beta", "lambda_l0", "lambda_l2"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.log_freq <= 0:
            raise ValueError("log_freq must be positive")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive when supplied")

    def to_manifest_parameters(self) -> dict[str, Any]:
        """Return deterministic manifest parameters for reuse checks."""

        params = {
            "target_config": self.target_config,
            "beta": self.beta,
            "lambda_l0": self.lambda_l0,
            "lambda_l2": self.lambda_l2,
            "learning_rate": self.learning_rate,
            "log_freq": self.log_freq,
        }
        return {key: value for key, value in params.items() if value is not None}

    def to_runtime_kwargs(self) -> dict[str, Any]:
        """Return Modal remote-call keyword arguments for this fit."""

        return self.to_manifest_parameters()


@dataclass(frozen=True)
class FittedWeightsSpec:
    """A scoped Stage 3 fitted-weight run specification."""

    scope: FitScope | str
    hyperparameters: FitHyperparameters
    schema_version: str = FIT_WEIGHTS_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", FitScope.parse(self.scope))
        if self.schema_version != FIT_WEIGHTS_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported fitted-weight spec schema version: {self.schema_version}"
            )

    def parameter_identity(self, *, gpu: str, epochs: int) -> str:
        """Return a deterministic identity for fit parameters and scope."""

        material = {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "gpu": gpu,
            "epochs": int(epochs),
            "hyperparameters": self.hyperparameters.to_manifest_parameters(),
        }
        return fingerprint_material(material).value

    def manifest_parameters(
        self,
        *,
        gpu: str,
        epochs: int,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return step-manifest parameters for this scoped fit."""

        params = {
            "scope": self.scope.value,
            "gpu": gpu,
            "epochs": epochs,
            **self.hyperparameters.to_manifest_parameters(),
            "fit_parameter_identity": self.parameter_identity(
                gpu=gpu,
                epochs=epochs,
            ),
        }
        if extra:
            params.update(dict(extra))
        return params

    def runtime_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments passed to the remote fit function."""

        return self.hyperparameters.to_runtime_kwargs()


REGIONAL_FITTED_WEIGHTS_SPEC = FittedWeightsSpec(
    scope=FitScope.REGIONAL,
    hyperparameters=FitHyperparameters(
        target_config=FIT_TARGET_CONFIG_PATH,
        beta=FIT_BETA,
        lambda_l0=REGIONAL_FIT_LAMBDA_L0,
        lambda_l2=REGIONAL_FIT_LAMBDA_L2,
        log_freq=FIT_LOG_FREQ,
    ),
)
NATIONAL_FITTED_WEIGHTS_SPEC = FittedWeightsSpec(
    scope=FitScope.NATIONAL,
    hyperparameters=FitHyperparameters(
        target_config=FIT_TARGET_CONFIG_PATH,
        beta=FIT_BETA,
        lambda_l0=NATIONAL_FIT_LAMBDA_L0,
        lambda_l2=NATIONAL_FIT_LAMBDA_L2,
        log_freq=FIT_LOG_FREQ,
    ),
)


@pipeline_node(
    PipelineNode(
        id="fitted_weights_spec",
        label="Fitted Weights Spec",
        node_type="library",
        description=("Scoped Stage 3 fit parameters and deterministic reuse identity."),
        source_file="policyengine_us_data/fit_weights/specs.py",
        status="current",
        stability="moving",
        pathways=["fit_weights", "reuse_identity"],
        artifacts_in=["calibration_package.pkl"],
        artifacts_out=["fit_parameter_identity"],
        validation_commands=["uv run pytest tests/unit/fit_weights/test_specs.py"],
    )
)
def fitted_weights_spec_for_scope(scope: FitScope | str) -> FittedWeightsSpec:
    """Return the current fitted-weight spec for a regional or national scope."""

    scope = FitScope.parse(scope)
    if scope == FitScope.REGIONAL:
        return REGIONAL_FITTED_WEIGHTS_SPEC
    if scope == FitScope.NATIONAL:
        return NATIONAL_FITTED_WEIGHTS_SPEC
    raise ValueError(f"Unknown fit scope: {scope!r}")


__all__ = [
    "FIT_BETA",
    "FIT_LOG_FREQ",
    "FIT_TARGET_CONFIG_PATH",
    "FIT_WEIGHTS_SPEC_SCHEMA_VERSION",
    "NATIONAL_FIT_LAMBDA_L0",
    "NATIONAL_FIT_LAMBDA_L2",
    "REGIONAL_FIT_LAMBDA_L0",
    "REGIONAL_FIT_LAMBDA_L2",
    "FitHyperparameters",
    "FitScope",
    "FittedWeightsSpec",
    "fitted_weights_spec_for_scope",
]
