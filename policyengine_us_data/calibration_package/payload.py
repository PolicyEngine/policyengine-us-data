"""Typed reader and writer boundary for Stage 2 package payloads."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.geography_checksum import (
    canonical_geography_checksum,
    hash_string_array,
)
from policyengine_us_data.utils.step_manifest import sha256_file

from .specs import CALIBRATION_PACKAGE_FILENAME, CALIBRATION_PACKAGE_METADATA_FILENAME

if TYPE_CHECKING:
    from policyengine_us_data.stage_contracts.calibration_package_schema import (
        CalibrationPackageSummary,
        GeographyAssignmentSummary,
    )

REQUIRED_PACKAGE_KEYS: frozenset[str] = frozenset(
    {"X_sparse", "targets_df", "target_names", "metadata"}
)
LEGACY_MISSING_GEOGRAPHY_WARNING = (
    "legacy packages without block_geoid/cd_geoid cannot prove geography assignment"
)


@pipeline_node(
    PipelineNode(
        id="stage2_payload_boundary",
        label="Stage 2 Package Payload",
        node_type="library",
        description="Typed access to the calibration_package.pkl matrix, targets, metadata, geography arrays, and compatibility warnings.",
        source_file="policyengine_us_data/calibration_package/payload.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=[CALIBRATION_PACKAGE_FILENAME],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_payload.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class CalibrationPackagePayload:
    """Typed access to the dictionary persisted in `calibration_package.pkl`."""

    X_sparse: Any
    targets_df: Any
    target_names: Any
    metadata: Mapping[str, Any]
    initial_weights: Any | None = None
    cd_geoid: Any | None = None
    block_geoid: Any | None = None
    compatibility_warnings: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        package: Mapping[str, Any],
        *,
        require_required_keys: bool = True,
    ) -> "CalibrationPackagePayload":
        """Validate and wrap a legacy package mapping."""

        if not isinstance(package, Mapping):
            raise ValueError("Calibration package pickle must contain a mapping")
        missing = sorted(REQUIRED_PACKAGE_KEYS - set(package))
        if missing and require_required_keys:
            raise ValueError(f"Calibration package missing required key: {missing[0]}")
        metadata = package.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError("Calibration package metadata must be a mapping")
        cd_geoid = package.get("cd_geoid")
        block_geoid = package.get("block_geoid")
        warnings: list[str] = []
        if cd_geoid is None and block_geoid is None:
            warnings.append(LEGACY_MISSING_GEOGRAPHY_WARNING)
        return cls(
            X_sparse=package.get("X_sparse"),
            targets_df=package.get("targets_df"),
            target_names=package.get("target_names"),
            metadata=dict(metadata),
            initial_weights=package.get("initial_weights"),
            cd_geoid=cd_geoid,
            block_geoid=block_geoid,
            compatibility_warnings=tuple(warnings),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the pickle-compatible package dictionary."""

        return {
            "X_sparse": self.X_sparse,
            "targets_df": self.targets_df,
            "target_names": self.target_names,
            "metadata": dict(self.metadata),
            "initial_weights": self.initial_weights,
            "cd_geoid": self.cd_geoid,
            "block_geoid": self.block_geoid,
        }

    def summary(self) -> CalibrationPackageSummary:
        """Return the contract-safe package summary."""

        from policyengine_us_data.stage_contracts.calibration_package_schema import (
            CalibrationPackageSummary,
        )

        try:
            n_targets, n_columns = self.X_sparse.shape
        except (AttributeError, ValueError) as exc:
            raise ValueError("X_sparse must expose a two-dimensional shape") from exc
        if not hasattr(self.X_sparse, "nnz"):
            raise ValueError("X_sparse must expose nnz")

        n_targets = int(n_targets)
        n_columns = int(n_columns)
        nnz = int(self.X_sparse.nnz)
        density = nnz / (n_targets * n_columns) if n_targets * n_columns else 0.0

        return CalibrationPackageSummary(
            matrix_shape=(n_targets, n_columns),
            matrix_nnz=nnz,
            matrix_density=float(density),
            n_targets=int(len(self.targets_df)),
            n_columns=n_columns,
            target_name_count=int(len(self.target_names)),
            dataset_sha256=self.metadata_string("dataset_sha256"),
            db_sha256=self.metadata_string("db_sha256"),
            target_config_path=self.metadata_string("target_config_path"),
            target_config_sha256=self.metadata_string("target_config_sha256"),
            n_clones=self.metadata_int("n_clones"),
            seed=self.metadata_int("seed"),
            base_n_records=self.metadata_int("base_n_records"),
            package_scope=self.metadata_string("package_scope"),
            matrix_builder=self.metadata_string("matrix_builder"),
            chunk_size=self.metadata_int("chunk_size"),
            chunk_dir=self.metadata_string("chunk_dir"),
            has_initial_weights=self.initial_weights is not None,
            has_cd_geoid=self.cd_geoid is not None,
            has_block_geoid=self.block_geoid is not None,
            cd_geoid_length=_optional_len(self.cd_geoid),
            block_geoid_length=_optional_len(self.block_geoid),
        )

    def geography_summary(self) -> GeographyAssignmentSummary:
        """Return the contract-safe geography assignment summary."""

        from policyengine_us_data.stage_contracts.calibration_package_schema import (
            GeographyAssignmentSummary,
        )

        n_records = self.metadata_int("base_n_records")
        n_clones = self.metadata_int("n_clones")
        has_blocks = self.block_geoid is not None
        has_cds = self.cd_geoid is not None

        if not has_blocks and not has_cds:
            return GeographyAssignmentSummary(
                source_kind="unavailable",
                n_records=n_records,
                n_clones=n_clones,
                n_rows=None,
                has_block_geoid=False,
                has_cd_geoid=False,
                block_geoid_length=None,
                cd_geoid_length=None,
                block_geoid_sha256=None,
                cd_geoid_sha256=None,
                canonical_geography_sha256=None,
            )
        if not has_blocks or not has_cds:
            raise ValueError(
                "Calibration package geography requires both block_geoid and cd_geoid"
            )
        if n_records is None or n_clones is None:
            raise ValueError(
                "Calibration package geography requires metadata base_n_records and n_clones"
            )
        if n_records <= 0 or n_clones <= 0:
            raise ValueError(
                "Calibration package geography requires positive base_n_records and n_clones"
            )

        block_geoids = _one_dimensional_string_array(self.block_geoid, "block_geoid")
        cd_geoids = _one_dimensional_string_array(self.cd_geoid, "cd_geoid")
        n_rows = int(len(block_geoids))
        if n_rows == 0:
            raise ValueError("Calibration package geography arrays must be non-empty")
        if len(cd_geoids) != n_rows:
            raise ValueError(
                "Calibration package geography has mismatched block_geoid and cd_geoid "
                f"lengths: {n_rows} != {len(cd_geoids)}"
            )
        if n_records * n_clones != n_rows:
            raise ValueError(
                "Calibration package geography length does not match metadata: "
                f"{n_rows} rows for {n_records} records x {n_clones} clones"
            )

        return GeographyAssignmentSummary(
            source_kind="calibration_package",
            n_records=n_records,
            n_clones=n_clones,
            n_rows=n_rows,
            has_block_geoid=True,
            has_cd_geoid=True,
            block_geoid_length=n_rows,
            cd_geoid_length=int(len(cd_geoids)),
            block_geoid_sha256=hash_string_array(block_geoids),
            cd_geoid_sha256=hash_string_array(cd_geoids),
            canonical_geography_sha256=canonical_geography_checksum(
                block_geoid=block_geoids,
                cd_geoid=cd_geoids,
                n_records=n_records,
                n_clones=n_clones,
            ),
        )

    def metadata_string(self, key: str) -> str | None:
        """Return a metadata value coerced to a string, preserving nulls."""

        value = self.metadata.get(key)
        if value is None:
            return None
        return str(value)

    def metadata_int(self, key: str) -> int | None:
        """Return a metadata value coerced to an integer, preserving nulls."""

        value = self.metadata.get(key)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"Calibration package metadata {key!r} must be an integer")
        return int(value)


@pipeline_node(
    PipelineNode(
        id="stage2_payload_reader",
        label="Stage 2 Payload Reader",
        node_type="library",
        description="Load calibration_package.pkl through the typed Stage 2 payload boundary and expose checksum/summary material.",
        source_file="policyengine_us_data/calibration_package/payload.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_in=[CALIBRATION_PACKAGE_FILENAME],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_payload.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class CalibrationPackageReader:
    """Read typed Stage 2 package payloads from disk."""

    package_path: Path

    def read(self) -> CalibrationPackagePayload:
        """Load and validate the persisted package payload."""

        with Path(self.package_path).open("rb") as handle:
            package = pickle.load(handle)
        return CalibrationPackagePayload.from_mapping(package)

    def checksum(self) -> str:
        """Return the package file checksum used for reuse comparisons."""

        return f"sha256:{sha256_file(Path(self.package_path))}"

    def summary(self) -> CalibrationPackageSummary:
        """Read the package and return its summary."""

        return self.read().summary()


@pipeline_node(
    PipelineNode(
        id="stage2_payload_writer",
        label="Stage 2 Payload Writer",
        node_type="library",
        description="Persist calibration_package.pkl and derive calibration_package_meta.json from typed payload and contract material.",
        source_file="policyengine_us_data/calibration_package/payload.py",
        status="current",
        stability="moving",
        pathways=["calibration_package"],
        artifacts_out=[
            CALIBRATION_PACKAGE_FILENAME,
            CALIBRATION_PACKAGE_METADATA_FILENAME,
        ],
        validation_commands=[
            "uv run pytest tests/unit/calibration_package/test_payload.py"
        ],
    )
)
@dataclass(frozen=True, kw_only=True)
class CalibrationPackageWriter:
    """Write typed Stage 2 package payloads and metadata sidecars."""

    package_path: Path

    def write(self, payload: CalibrationPackagePayload) -> Path:
        """Persist a package payload using the legacy pickle format."""

        output_path = Path(self.package_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(payload.to_mapping(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_path

    def write_metadata_sidecar(
        self,
        payload: CalibrationPackagePayload,
        *,
        contract: Any | None = None,
        sidecar_path: str | Path | None = None,
    ) -> Path:
        """Write `calibration_package_meta.json` from typed payload material."""

        output_path = (
            Path(sidecar_path)
            if sidecar_path is not None
            else Path(self.package_path).with_name(
                CALIBRATION_PACKAGE_METADATA_FILENAME
            )
        )
        sidecar_payload = {
            **dict(payload.metadata),
            "package_sha256": f"sha256:{sha256_file(Path(self.package_path))}",
            "package_summary": payload.summary().to_dict(),
            "geography_assignment": payload.geography_summary().to_dict(),
            "compatibility_warnings": list(payload.compatibility_warnings),
        }
        if contract is not None:
            sidecar_payload["contract"] = {
                "stage_id": getattr(contract, "stage_id", None),
                "contract_type": getattr(contract, "contract_type", None),
                "fingerprint": (
                    contract.fingerprint.to_dict()
                    if getattr(contract, "fingerprint", None) is not None
                    else None
                ),
            }
        output_path.write_text(
            json.dumps(sidecar_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path


def _optional_len(value: Any) -> int | None:
    if value is None:
        return None
    return int(len(value))


def _one_dimensional_string_array(value: Any, key: str) -> Any:
    import numpy as np

    array = np.asarray(value, dtype=str)
    if array.ndim != 1:
        raise ValueError(f"Calibration package geography {key} must be one-dimensional")
    if np.any(array == ""):
        raise ValueError(f"Calibration package geography {key} contains empty values")
    return array


__all__ = [
    "LEGACY_MISSING_GEOGRAPHY_WARNING",
    "REQUIRED_PACKAGE_KEYS",
    "CalibrationPackagePayload",
    "CalibrationPackageReader",
    "CalibrationPackageWriter",
]
