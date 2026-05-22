"""Scoped Stage 3 fitted-weight contract builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from policyengine_us_data.fit_weights.artifacts import (
    FitArtifactRole,
    fit_artifacts_for_scope,
)
from policyengine_us_data.fit_weights.bundles import FittedWeightsInputBundle
from policyengine_us_data.fit_weights.specs import FitScope
from policyengine_us_data.stage_contracts import ArtifactRef, StageContract
from policyengine_us_data.stage_contracts.execution import (
    ExecutionRecord,
    ReuseSummary,
)
from policyengine_us_data.stage_contracts.fingerprints import fingerprint_material
from policyengine_us_data.stage_contracts.io import write_contract
from policyengine_us_data.stage_contracts.stages import (
    STAGE_3_FIT_WEIGHTS,
    contract_type_for_stage,
)
from policyengine_us_data.stage_contracts.substages import SubstageRecord
from policyengine_us_data.utils.step_manifest import sha256_file, utc_now

FITTED_WEIGHTS_CONTRACT_SCHEMA_VERSION = "1"
FITTED_WEIGHTS_CONTRACT_TYPE = contract_type_for_stage(STAGE_3_FIT_WEIGHTS)
FITTED_WEIGHTS_CONTRACT_FILENAMES = {
    FitScope.REGIONAL: "fitted_weights_regional_contract.json",
    FitScope.NATIONAL: "fitted_weights_national_contract.json",
}
FITTED_WEIGHTS_SUBSTAGE_IDS = {
    FitScope.REGIONAL: "3a_weight_fitting_regional",
    FitScope.NATIONAL: "3b_weight_fitting_national",
}


def fitted_weights_contract_filename(scope: FitScope | str) -> str:
    """Return the scoped Stage 3 contract filename."""

    return FITTED_WEIGHTS_CONTRACT_FILENAMES[FitScope.parse(scope)]


def fitted_weights_contract_path(
    *,
    scope: FitScope | str,
    artifacts_root: str | Path,
) -> Path:
    """Return the scoped Stage 3 contract path under an artifacts root."""

    return Path(artifacts_root) / fitted_weights_contract_filename(scope)


@dataclass(frozen=True)
class FittedWeightsContractBuilder:
    """Build a semantic contract for one scoped Stage 3 fit."""

    scope: FitScope | str
    input_bundle: FittedWeightsInputBundle
    parameters: Mapping[str, Any]
    artifacts_root: Path
    diagnostics_root: Path
    run_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None
    modal_call_id: str | None = None
    code_sha: str | None = None
    package_version: str | None = None
    target_metadata_paths: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", FitScope.parse(self.scope))
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root))
        object.__setattr__(self, "diagnostics_root", Path(self.diagnostics_root))

    @property
    def contract_path(self) -> Path:
        """Return the default contract file path for this scoped fit."""

        return fitted_weights_contract_path(
            scope=self.scope,
            artifacts_root=self.artifacts_root,
        )

    def build(self) -> StageContract:
        """Build the Stage 3 contract from existing fit artifacts."""

        inputs = tuple(self._input_artifacts())
        outputs = tuple(self._output_artifacts())
        metadata = self._metadata()
        fingerprint = fingerprint_material(
            {
                "stage_id": STAGE_3_FIT_WEIGHTS,
                "contract_type": FITTED_WEIGHTS_CONTRACT_TYPE,
                "schema_version": FITTED_WEIGHTS_CONTRACT_SCHEMA_VERSION,
                "scope": self.scope.value,
                "inputs": inputs,
                "outputs": outputs,
                "parameters": dict(self.parameters),
                "metadata": metadata,
            }
        )
        execution = ExecutionRecord(
            status="completed",
            started_at=self.started_at,
            completed_at=self.completed_at or utc_now(),
            duration_s=self.duration_s,
            modal_call_id=self.modal_call_id,
            reuse_decision="computed",
            reuse_summary=ReuseSummary(
                expected_outputs=len(outputs),
                recomputed_outputs=len(outputs),
            ),
        )
        substage = SubstageRecord(
            substage_id=FITTED_WEIGHTS_SUBSTAGE_IDS[self.scope],
            status="completed",
            inputs=inputs,
            outputs=outputs,
            parameters=dict(self.parameters),
            fingerprint=fingerprint,
            reuse_mode="handoff",
            metadata={"scope": self.scope.value},
        )
        return StageContract(
            contract_type=FITTED_WEIGHTS_CONTRACT_TYPE,
            stage_id=STAGE_3_FIT_WEIGHTS,
            run_id=self.run_id,
            created_at=execution.completed_at or utc_now(),
            code_sha=self.code_sha,
            package_version=self.package_version,
            inputs=inputs,
            outputs=outputs,
            parameters=dict(self.parameters),
            fingerprint=fingerprint,
            substages=(substage,),
            execution=execution,
            metadata=metadata,
        )

    def write(self, path: str | Path | None = None) -> Path:
        """Write the scoped Stage 3 contract and return its path."""

        contract_path = Path(path) if path is not None else self.contract_path
        write_contract(self.build(), contract_path)
        return contract_path

    def _input_artifacts(self) -> list[ArtifactRef]:
        artifacts = [
            _artifact_ref(
                logical_name="calibration_package",
                path=self.input_bundle.calibration_package_path,
                artifact_family="calibration_package",
                role="input",
                scope=self.scope.value,
            )
        ]
        contract_path = self.input_bundle.calibration_package_contract_path
        if contract_path is not None and Path(contract_path).exists():
            artifacts.append(
                _artifact_ref(
                    logical_name="calibration_package_contract",
                    path=contract_path,
                    artifact_family="stage_contract",
                    role="input",
                    scope=self.scope.value,
                )
            )
        for logical_name, path in sorted(self.target_metadata_paths.items()):
            if Path(path).exists():
                artifacts.append(
                    _artifact_ref(
                        logical_name=logical_name,
                        path=path,
                        artifact_family="calibration_target_metadata",
                        role="input",
                        scope=self.scope.value,
                    )
                )
        return artifacts

    def _output_artifacts(self) -> list[ArtifactRef]:
        scoped_artifacts = fit_artifacts_for_scope(self.scope)
        artifacts: list[ArtifactRef] = []
        for spec in scoped_artifacts.artifact_specs():
            artifacts.append(
                _artifact_ref(
                    logical_name=_logical_output_name(self.scope, spec.role),
                    path=spec.path_under(self.artifacts_root),
                    artifact_family="fitted_weights",
                    role=spec.role.value,
                    scope=self.scope.value,
                    location=spec.location.value,
                )
            )
        for spec in scoped_artifacts.diagnostic_specs():
            if spec.role == FitArtifactRole.RUN_CONFIG:
                continue
            path = spec.path_under(self.diagnostics_root)
            if not path.exists():
                continue
            artifacts.append(
                _artifact_ref(
                    logical_name=_logical_output_name(self.scope, spec.role),
                    path=path,
                    artifact_family="fitted_weights",
                    role=spec.role.value,
                    scope=self.scope.value,
                    location=spec.location.value,
                )
            )
        return artifacts

    def _metadata(self) -> dict[str, Any]:
        scoped_artifacts = fit_artifacts_for_scope(self.scope)
        identity = self.input_bundle.stage2_identity()
        weights_path = scoped_artifacts.weights.path_under(self.artifacts_root)
        geography_path = scoped_artifacts.geography.path_under(self.artifacts_root)
        diagnostics = {}
        for spec in scoped_artifacts.diagnostic_specs():
            if spec.role == FitArtifactRole.RUN_CONFIG:
                continue
            path = spec.path_under(self.diagnostics_root)
            if path.exists():
                diagnostics[spec.role.value] = _csv_summary(path)
        return {
            "schema_version": FITTED_WEIGHTS_CONTRACT_SCHEMA_VERSION,
            "scope": self.scope.value,
            "package_checksum": identity.calibration_package_sha256,
            "package_contract_checksum": identity.calibration_package_contract_sha256,
            "package_contract_fingerprint": (
                identity.calibration_package_contract_fingerprint
            ),
            "weight_summary": _npy_summary(weights_path),
            "geography_checksum": f"sha256:{sha256_file(geography_path)}",
            "geography_size_bytes": geography_path.stat().st_size,
            "diagnostics_summary": diagnostics,
            "target_metadata_available": any(
                Path(path).exists() for path in self.target_metadata_paths.values()
            ),
        }


def _logical_output_name(scope: FitScope, role: FitArtifactRole) -> str:
    return f"fitted_weights_{scope.value}_{role.value}"


def _artifact_ref(
    *,
    logical_name: str,
    path: str | Path,
    artifact_family: str,
    role: str,
    scope: str,
    location: str | None = None,
) -> ArtifactRef:
    artifact_path = Path(path)
    metadata = {
        "artifact_family": artifact_family,
        "scope": scope,
        "role": role,
    }
    if location is not None:
        metadata["location"] = location
    return ArtifactRef(
        logical_name=logical_name,
        uri=artifact_path.resolve().as_uri(),
        sha256=f"sha256:{sha256_file(artifact_path)}",
        size_bytes=artifact_path.stat().st_size,
        media_type=_media_type_for_path(artifact_path),
        metadata=metadata,
    )


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".npy":
        return "application/x-numpy"
    if suffix == ".npz":
        return "application/x-numpy-zip"
    if suffix == ".pkl":
        return "application/python-pickle"
    return "application/octet-stream"


def _npy_summary(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "count": int(array.size),
        "sha256": f"sha256:{sha256_file(path)}",
    }
    if array.size:
        summary.update(
            {
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "sum": float(np.sum(array)),
            }
        )
    return summary


def _csv_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    return {
        "sha256": f"sha256:{sha256_file(path)}",
        "size_bytes": path.stat().st_size,
        "row_count": max(line_count - 1, 0),
    }


__all__ = [
    "FITTED_WEIGHTS_CONTRACT_SCHEMA_VERSION",
    "FittedWeightsContractBuilder",
    "fitted_weights_contract_filename",
    "fitted_weights_contract_path",
]
