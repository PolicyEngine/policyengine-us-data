"""Scoped Stage 3 fitted-weight input and output bundles."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Mapping

from policyengine_us_data.fit_weights.artifacts import (
    ScopedFitArtifacts,
    fit_artifacts_for_scope,
)
from policyengine_us_data.fit_weights.specs import FitScope
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode


class MissingFitWeightsOutputError(ValueError):
    """Raised when remote fit bytes omit required weights."""


@dataclass(frozen=True)
class FitWeightsBuildContext:
    """Run-scoped filesystem context for Stage 3 fitted-weight artifacts."""

    run_id: str
    artifacts_root: Path
    diagnostics_root: Path


@dataclass(frozen=True)
class FittedWeightsInputBundle:
    """Scoped Stage 3 input paths consumed before fitting starts."""

    scope: FitScope | str
    calibration_package_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", FitScope.parse(self.scope))
        object.__setattr__(
            self,
            "calibration_package_path",
            Path(self.calibration_package_path),
        )

    def artifact_identity_paths(self) -> dict[str, Path]:
        """Return paths used for Stage 3 input identity calculation."""

        return {"calibration_package": self.calibration_package_path}


@dataclass(frozen=True)
class FitResultBytes:
    """Compatibility transport model for current remote fit result bytes."""

    weights: bytes
    geography: bytes | None = None
    diagnostics: bytes | None = None
    epoch_log: bytes | None = None
    run_config: bytes | None = None

    @classmethod
    def from_mapping(cls, result_bytes: Mapping[str, bytes | None]) -> "FitResultBytes":
        """Build transport bytes from the current remote result dictionary."""

        weights = result_bytes.get("weights")
        if weights is None:
            raise MissingFitWeightsOutputError(
                "Fitted-weight result is missing required weights bytes."
            )
        return cls(
            weights=weights,
            geography=result_bytes.get("geography"),
            diagnostics=result_bytes.get("log"),
            epoch_log=result_bytes.get("cal_log"),
            run_config=result_bytes.get("config"),
        )

    def to_result_dict(self) -> dict[str, bytes | None]:
        """Return the legacy result dictionary shape used by Modal adapters."""

        return {
            "weights": self.weights,
            "geography": self.geography,
            "log": self.diagnostics,
            "cal_log": self.epoch_log,
            "config": self.run_config,
        }

    def bytes_for_result_key(self, result_key: str | None) -> bytes | None:
        """Return bytes for an artifact spec result key."""

        return self.to_result_dict().get(result_key or "")


@pipeline_node(
    PipelineNode(
        id="fitted_weights_output_bundle",
        label="Fitted Weights Output Bundle",
        node_type="library",
        description="Scoped Stage 3 result bytes before artifact file writes.",
        source_file="policyengine_us_data/fit_weights/bundles.py",
        status="current",
        stability="moving",
        pathways=["fit_weights", "artifact_identity"],
        artifacts_in=["remote fit result bytes"],
        artifacts_out=["scoped fitted-weight artifact writes"],
        validation_commands=["uv run pytest tests/unit/fit_weights/test_bundles.py"],
    )
)
@dataclass(frozen=True)
class FittedWeightsOutputBundle:
    """Scoped output bundle created before Stage 3 bytes become files."""

    scope: FitScope | str
    result: FitResultBytes
    artifacts: ScopedFitArtifacts
    run_id: str = ""

    def __post_init__(self) -> None:
        scope = FitScope.parse(self.scope)
        object.__setattr__(self, "scope", scope)
        if self.artifacts.scope != scope:
            raise ValueError(
                "Output bundle scope does not match artifact catalog: "
                f"{scope.value} != {self.artifacts.scope.value}"
            )

    @classmethod
    def from_result_bytes(
        cls,
        *,
        scope: FitScope | str,
        result_bytes: Mapping[str, bytes | None],
        run_id: str = "",
    ) -> "FittedWeightsOutputBundle":
        """Build a scoped bundle from the current remote result dictionary."""

        scope = FitScope.parse(scope)
        return cls(
            scope=scope,
            result=FitResultBytes.from_mapping(result_bytes),
            artifacts=fit_artifacts_for_scope(scope),
            run_id=run_id,
        )

    def write_artifacts(self, batch, artifacts_rel: str) -> list[str]:
        """Write present primary artifacts to a Modal batch upload object."""

        written: list[str] = []
        for artifact in self.artifacts.artifact_specs():
            data = self.result.bytes_for_result_key(artifact.result_key)
            if data is None:
                continue
            destination = f"{artifacts_rel}/{artifact.filename}"
            batch.put_file(BytesIO(data), destination)
            written.append(destination)
        return written

    def artifact_paths(self, artifacts_root: str | Path) -> list[Path]:
        """Return expected primary artifact paths under a pipeline artifact root."""

        return self.artifacts.artifact_paths(artifacts_root)

    def diagnostic_result_bytes(self) -> dict[str, bytes | None]:
        """Return only diagnostics belonging to this output scope."""

        return {
            artifact.result_key: self.result.bytes_for_result_key(artifact.result_key)
            for artifact in self.artifacts.diagnostic_specs()
            if artifact.result_key is not None
        }


__all__ = [
    "FitResultBytes",
    "FitWeightsBuildContext",
    "FittedWeightsInputBundle",
    "FittedWeightsOutputBundle",
    "MissingFitWeightsOutputError",
]
