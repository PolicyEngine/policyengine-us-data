"""Scoped Stage 3 fitted-weight input and output bundles."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
import warnings

from policyengine_us_data.calibration_package.specs import (
    CALIBRATION_PACKAGE_CONTRACT_FILENAME,
)
from policyengine_us_data.fit_weights.artifacts import (
    ScopedFitArtifacts,
    fit_artifacts_for_scope,
)
from policyengine_us_data.fit_weights.specs import FitScope
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.stage_contracts import StageContract
from policyengine_us_data.stage_contracts.io import read_contract
from policyengine_us_data.stage_contracts.stages import (
    STAGE_2_BUILD_CALIBRATION_PACKAGE,
    contract_type_for_stage,
)
from policyengine_us_data.utils.step_manifest import sha256_file

STAGE_2_CALIBRATION_PACKAGE_CONTRACT_TYPE = contract_type_for_stage(
    STAGE_2_BUILD_CALIBRATION_PACKAGE
)


class MissingFitWeightsOutputError(ValueError):
    """Raised when remote fit bytes omit required fitted-weight artifacts."""


class FittedWeightsInputContractError(ValueError):
    """Raised when Stage 3 cannot establish Stage 2 package identity."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FitWeightsBuildContext:
    """Run-scoped filesystem context for Stage 3 fitted-weight artifacts."""

    run_id: str
    artifacts_root: Path
    diagnostics_root: Path


@dataclass(frozen=True)
class FittedWeightsInputIdentity:
    """Checksum-backed Stage 2 package identity consumed by Stage 3."""

    calibration_package_sha256: str
    calibration_package_size_bytes: int
    stage2_contract_mode: str
    calibration_package_contract_sha256: str | None = None
    calibration_package_contract_size_bytes: int | None = None
    calibration_package_contract_fingerprint: str | None = None
    calibration_package_contract_run_id: str | None = None

    def to_manifest_parameters(self) -> dict[str, Any]:
        """Return fit manifest parameters that identify the Stage 2 package."""

        params: dict[str, Any] = {
            "calibration_package_sha256": self.calibration_package_sha256,
            "calibration_package_size_bytes": self.calibration_package_size_bytes,
            "stage2_contract_mode": self.stage2_contract_mode,
            "calibration_package_contract_sha256": (
                self.calibration_package_contract_sha256
            ),
            "calibration_package_contract_size_bytes": (
                self.calibration_package_contract_size_bytes
            ),
            "calibration_package_contract_fingerprint": (
                self.calibration_package_contract_fingerprint
            ),
            "calibration_package_contract_run_id": (
                self.calibration_package_contract_run_id
            ),
        }
        return {key: value for key, value in params.items() if value is not None}


@dataclass(frozen=True)
class FittedWeightsInputBundle:
    """Scoped Stage 3 input paths and Stage 2 package identity."""

    scope: FitScope | str
    calibration_package_path: Path
    calibration_package_contract_path: Path | None = None
    allow_legacy_no_contract: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", FitScope.parse(self.scope))
        package_path = Path(self.calibration_package_path)
        contract_path = (
            Path(self.calibration_package_contract_path)
            if self.calibration_package_contract_path is not None
            else package_path.with_name(CALIBRATION_PACKAGE_CONTRACT_FILENAME)
        )
        object.__setattr__(self, "calibration_package_path", package_path)
        object.__setattr__(self, "calibration_package_contract_path", contract_path)

    def artifact_identity_paths(self) -> dict[str, Path]:
        """Return paths used for Stage 3 input identity calculation."""

        paths = {"calibration_package": self.calibration_package_path}
        contract_path = self.calibration_package_contract_path
        if contract_path is not None and (
            not self.allow_legacy_no_contract or contract_path.exists()
        ):
            paths["calibration_package_contract"] = contract_path
        return paths

    def stage2_identity(self) -> FittedWeightsInputIdentity:
        """Validate and return the Stage 2 package identity for fitting."""

        package_path = self.calibration_package_path
        if not package_path.exists():
            raise FittedWeightsInputContractError(
                f"Missing calibration package artifact: {package_path}",
                code="missing_calibration_package",
            )
        if not package_path.is_file():
            raise FittedWeightsInputContractError(
                f"Calibration package artifact is not a file: {package_path}",
                code="invalid_calibration_package_path",
            )

        package_sha256 = f"sha256:{sha256_file(package_path)}"
        package_size_bytes = package_path.stat().st_size
        contract_path = self.calibration_package_contract_path
        if contract_path is None or not contract_path.exists():
            if self.allow_legacy_no_contract:
                warnings.warn(
                    "Proceeding with Stage 3 fitting without "
                    f"{CALIBRATION_PACKAGE_CONTRACT_FILENAME}; this legacy "
                    "manual fallback records only the package checksum.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return FittedWeightsInputIdentity(
                    calibration_package_sha256=package_sha256,
                    calibration_package_size_bytes=package_size_bytes,
                    stage2_contract_mode="legacy_no_contract",
                )
            raise FittedWeightsInputContractError(
                "Missing Stage 2 calibration package contract: "
                f"{contract_path or CALIBRATION_PACKAGE_CONTRACT_FILENAME}",
                code="missing_stage2_contract",
            )
        if not contract_path.is_file():
            raise FittedWeightsInputContractError(
                f"Stage 2 calibration package contract is not a file: {contract_path}",
                code="invalid_stage2_contract_path",
            )

        contract = _read_stage2_contract(contract_path)
        _assert_stage2_contract_matches_package(
            contract=contract,
            package_path=package_path,
            package_sha256=package_sha256,
            package_size_bytes=package_size_bytes,
        )
        return FittedWeightsInputIdentity(
            calibration_package_sha256=package_sha256,
            calibration_package_size_bytes=package_size_bytes,
            stage2_contract_mode="stage2_contract",
            calibration_package_contract_sha256=f"sha256:{sha256_file(contract_path)}",
            calibration_package_contract_size_bytes=contract_path.stat().st_size,
            calibration_package_contract_fingerprint=contract.fingerprint.value,
            calibration_package_contract_run_id=contract.run_id,
        )

    def stage2_identity_parameters(self) -> dict[str, Any]:
        """Return manifest parameters for the validated Stage 2 identity."""

        return self.stage2_identity().to_manifest_parameters()


def _read_stage2_contract(contract_path: Path) -> StageContract:
    try:
        contract = read_contract(contract_path)
    except Exception as exc:
        raise FittedWeightsInputContractError(
            f"Could not read Stage 2 calibration package contract: {contract_path}",
            code="invalid_stage2_contract",
        ) from exc
    if contract.stage_id != STAGE_2_BUILD_CALIBRATION_PACKAGE:
        raise FittedWeightsInputContractError(
            f"Invalid Stage 2 contract stage_id: {contract.stage_id!r}",
            code="invalid_stage2_contract",
        )
    if contract.contract_type != STAGE_2_CALIBRATION_PACKAGE_CONTRACT_TYPE:
        raise FittedWeightsInputContractError(
            f"Invalid Stage 2 contract type: {contract.contract_type!r}",
            code="invalid_stage2_contract",
        )
    return contract


def _assert_stage2_contract_matches_package(
    *,
    contract: StageContract,
    package_path: Path,
    package_sha256: str,
    package_size_bytes: int,
) -> None:
    package_artifacts = [
        artifact
        for artifact in contract.outputs
        if artifact.logical_name == "calibration_package"
    ]
    if len(package_artifacts) != 1:
        raise FittedWeightsInputContractError(
            "Stage 2 contract must declare exactly one calibration_package output; "
            f"found {len(package_artifacts)}.",
            code="invalid_stage2_contract",
        )
    package_artifact = package_artifacts[0]
    if package_artifact.sha256 != package_sha256:
        raise FittedWeightsInputContractError(
            "Stage 2 calibration package contract checksum mismatch for "
            f"{package_path}: {package_artifact.sha256!r} != {package_sha256!r}",
            code="stage2_contract_package_mismatch",
        )
    if package_artifact.size_bytes != package_size_bytes:
        raise FittedWeightsInputContractError(
            "Stage 2 calibration package contract size mismatch for "
            f"{package_path}: {package_artifact.size_bytes!r} != "
            f"{package_size_bytes!r}",
            code="stage2_contract_package_mismatch",
        )


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
                if artifact.required:
                    raise MissingFitWeightsOutputError(
                        "Fitted-weight result is missing required "
                        f"{self.scope.value} {artifact.role.value} bytes "
                        f"for {artifact.filename}."
                    )
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
    "FittedWeightsInputContractError",
    "FittedWeightsInputBundle",
    "FittedWeightsInputIdentity",
    "FittedWeightsOutputBundle",
    "MissingFitWeightsOutputError",
]
