"""Scoped Stage 3 fitted-weight artifact identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from policyengine_us_data.fit_weights.specs import FitScope
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode


class FitArtifactLocation(str, Enum):
    """Pipeline-volume location for a fitted-weight artifact."""

    ARTIFACTS = "artifacts"
    DIAGNOSTICS = "diagnostics"


class FitArtifactRole(str, Enum):
    """Logical role for one fitted-weight artifact."""

    WEIGHTS = "weights"
    GEOGRAPHY = "geography"
    RUN_CONFIG = "run_config"
    DIAGNOSTICS = "diagnostics"
    EPOCH_LOG = "epoch_log"


@dataclass(frozen=True)
class FitArtifactSpec:
    """Filename and remote-result mapping for one Stage 3 artifact."""

    role: FitArtifactRole
    filename: str
    location: FitArtifactLocation
    result_key: str | None = None
    required: bool = True

    def path_under(self, root: str | Path) -> Path:
        """Return this artifact path under an artifacts or diagnostics root."""

        return Path(root) / self.filename


@dataclass(frozen=True)
class ScopedFitArtifacts:
    """The artifact contract for one fitted-weight scope."""

    scope: FitScope | str
    weights: FitArtifactSpec
    geography: FitArtifactSpec
    run_config: FitArtifactSpec
    diagnostics: FitArtifactSpec
    epoch_log: FitArtifactSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", FitScope.parse(self.scope))

    def artifact_specs(self) -> tuple[FitArtifactSpec, ...]:
        """Return primary Stage 3 output artifacts."""

        return (self.weights, self.geography, self.run_config)

    def diagnostic_specs(self) -> tuple[FitArtifactSpec, ...]:
        """Return diagnostic Stage 3 artifacts archived under run diagnostics."""

        return (self.diagnostics, self.epoch_log, self.run_config)

    def artifact_paths(self, artifacts_root: str | Path) -> list[Path]:
        """Return primary artifact paths under the pipeline artifacts root."""

        return [
            artifact.path_under(artifacts_root) for artifact in self.artifact_specs()
        ]

    def diagnostic_result_filenames(self) -> dict[str, str]:
        """Map remote result byte keys to diagnostic archive filenames."""

        return {
            artifact.result_key: artifact.filename
            for artifact in self.diagnostic_specs()
            if artifact.result_key is not None
        }


REGIONAL_FIT_ARTIFACTS = ScopedFitArtifacts(
    scope=FitScope.REGIONAL,
    weights=FitArtifactSpec(
        role=FitArtifactRole.WEIGHTS,
        filename="calibration_weights.npy",
        location=FitArtifactLocation.ARTIFACTS,
        result_key="weights",
    ),
    geography=FitArtifactSpec(
        role=FitArtifactRole.GEOGRAPHY,
        filename="geography_assignment.npz",
        location=FitArtifactLocation.ARTIFACTS,
        result_key="geography",
    ),
    run_config=FitArtifactSpec(
        role=FitArtifactRole.RUN_CONFIG,
        filename="unified_run_config.json",
        location=FitArtifactLocation.ARTIFACTS,
        result_key="config",
    ),
    diagnostics=FitArtifactSpec(
        role=FitArtifactRole.DIAGNOSTICS,
        filename="unified_diagnostics.csv",
        location=FitArtifactLocation.DIAGNOSTICS,
        result_key="log",
        required=False,
    ),
    epoch_log=FitArtifactSpec(
        role=FitArtifactRole.EPOCH_LOG,
        filename="calibration_log.csv",
        location=FitArtifactLocation.DIAGNOSTICS,
        result_key="cal_log",
        required=False,
    ),
)
NATIONAL_FIT_ARTIFACTS = ScopedFitArtifacts(
    scope=FitScope.NATIONAL,
    weights=FitArtifactSpec(
        role=FitArtifactRole.WEIGHTS,
        filename="national_calibration_weights.npy",
        location=FitArtifactLocation.ARTIFACTS,
        result_key="weights",
    ),
    geography=FitArtifactSpec(
        role=FitArtifactRole.GEOGRAPHY,
        filename="national_geography_assignment.npz",
        location=FitArtifactLocation.ARTIFACTS,
        result_key="geography",
    ),
    run_config=FitArtifactSpec(
        role=FitArtifactRole.RUN_CONFIG,
        filename="national_unified_run_config.json",
        location=FitArtifactLocation.ARTIFACTS,
        result_key="config",
    ),
    diagnostics=FitArtifactSpec(
        role=FitArtifactRole.DIAGNOSTICS,
        filename="national_unified_diagnostics.csv",
        location=FitArtifactLocation.DIAGNOSTICS,
        result_key="log",
        required=False,
    ),
    epoch_log=FitArtifactSpec(
        role=FitArtifactRole.EPOCH_LOG,
        filename="national_calibration_log.csv",
        location=FitArtifactLocation.DIAGNOSTICS,
        result_key="cal_log",
        required=False,
    ),
)


@pipeline_node(
    PipelineNode(
        id="fitted_weights_artifacts",
        label="Fitted Weights Artifacts",
        node_type="library",
        description=(
            "Canonical regional and national fitted-weight artifact filenames."
        ),
        source_file="policyengine_us_data/fit_weights/artifacts.py",
        status="current",
        stability="moving",
        pathways=["fit_weights", "artifact_identity"],
        artifacts_in=["remote fit result bytes"],
        artifacts_out=[
            "calibration_weights.npy",
            "geography_assignment.npz",
            "unified_run_config.json",
            "national_calibration_weights.npy",
            "national_geography_assignment.npz",
            "national_unified_run_config.json",
        ],
        validation_commands=["uv run pytest tests/unit/fit_weights/test_artifacts.py"],
    )
)
def fit_artifacts_for_scope(scope: FitScope | str) -> ScopedFitArtifacts:
    """Return canonical fitted-weight artifacts for a regional or national scope."""

    scope = FitScope.parse(scope)
    if scope == FitScope.REGIONAL:
        return REGIONAL_FIT_ARTIFACTS
    if scope == FitScope.NATIONAL:
        return NATIONAL_FIT_ARTIFACTS
    raise ValueError(f"Unknown fit scope: {scope!r}")


__all__ = [
    "FitArtifactLocation",
    "FitArtifactRole",
    "FitArtifactSpec",
    "ScopedFitArtifacts",
    "fit_artifacts_for_scope",
]
