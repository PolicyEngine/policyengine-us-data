"""Coordinator-owned provenance and resumability logic for local H5 publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from policyengine_us_data.pipeline_metadata import pipeline_node

from .geography_loader import CalibrationGeographyLoader
from .simulation_access import calculate_variable_values

FingerprintScope = Literal["regional", "national"]
ScopeFingerprint = str
ArtifactMetadata = Mapping[str, Any]

__all__ = [
    "ArtifactIdentity",
    "ArtifactMetadata",
    "FingerprintScope",
    "FingerprintingService",
    "PublishingInputBundle",
    "ScopeFingerprint",
    "TraceabilityBundle",
]


@pipeline_node(
    id="local_h5_publishing_input_bundle",
    label="PublishingInputBundle",
    node_type="library",
    description="Input artifact and run metadata bundle for one local H5 publish scope.",
    source_file="policyengine_us_data/build_outputs/fingerprinting.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
)
@dataclass(frozen=True)
class PublishingInputBundle:
    """Input artifact bundle for one local H5 publication scope.

    The bundle is the library-level contract used by coordinators before
    fingerprinting. Paths point to local files already materialized in the
    worker or orchestration environment.

    Attributes:
        weights_path: Path to `calibration_weights.npy`.
        source_dataset_path: Path to the source-imputed dataset H5.
        target_db_path: Optional `policy_data.db` path used for validation.
        exact_geography_path: Optional saved `geography_assignment.npz` path.
        calibration_package_path: Optional `calibration_package.pkl` path used
            as a geography fallback.
        run_config_path: Optional run configuration JSON with code and model
            build metadata.
        run_id: Pipeline run identifier.
        version: Package or release version associated with the publish.
        n_clones: Expected clone count, when known.
        seed: Geography assignment seed used by the build.
        legacy_blocks_path: Optional legacy `stacked_blocks.npy` fallback.
    """

    weights_path: Path
    source_dataset_path: Path
    target_db_path: Path | None
    exact_geography_path: Path | None
    calibration_package_path: Path | None
    run_config_path: Path | None
    run_id: str
    version: str
    n_clones: int | None
    seed: int
    legacy_blocks_path: Path | None = None


@pipeline_node(
    id="local_h5_artifact_identity",
    label="ArtifactIdentity",
    node_type="library",
    description="Stable content identity for one local H5 publication input artifact.",
    source_file="policyengine_us_data/build_outputs/fingerprinting.py",
    status="current",
    stability="stable",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_fingerprinting.py"
    ],
)
@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable identity for an input artifact used by traceability.

    Attributes:
        logical_name: Semantic artifact name, such as `"weights"` or
            `"source_dataset"`.
        path: Physical artifact path when the artifact exists in the local
            runtime.
        sha256: Content digest prefixed with `"sha256:"`.
        size_bytes: Artifact size in bytes, when available.
        metadata: Additional normalized metadata, for example canonical
            geography checksum or source kind.
    """

    logical_name: str
    path: Path | None
    sha256: str | None
    size_bytes: int | None = None
    metadata: ArtifactMetadata = field(default_factory=dict)


@pipeline_node(
    id="local_h5_traceability_bundle",
    label="TraceabilityBundle",
    node_type="library",
    description="Provenance and resumability material for one local H5 publish scope.",
    source_file="policyengine_us_data/build_outputs/fingerprinting.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_fingerprinting.py"
    ],
)
@dataclass(frozen=True)
class TraceabilityBundle:
    """Full provenance record for one local H5 publish scope.

    Attributes:
        scope: Publish scope being fingerprinted, currently `"regional"` or
            `"national"`.
        weights: Identity of the calibration weights artifact.
        source_dataset: Identity of the source-imputed H5 artifact.
        exact_geography: Identity of the geography source used for clone
            selection, if available.
        target_db: Optional target database identity.
        calibration_package: Optional calibration package identity.
        run_config: Optional run configuration identity.
        code_version: Normalized code version metadata extracted from run
            config.
        model_build: Normalized model build metadata extracted from run config.
        metadata: Scope-level metadata such as run ID, version, clone count, and
            seed.
    """

    scope: FingerprintScope
    weights: ArtifactIdentity
    source_dataset: ArtifactIdentity
    exact_geography: ArtifactIdentity | None = None
    target_db: ArtifactIdentity | None = None
    calibration_package: ArtifactIdentity | None = None
    run_config: ArtifactIdentity | None = None
    code_version: ArtifactMetadata = field(default_factory=dict)
    model_build: ArtifactMetadata = field(default_factory=dict)
    metadata: ArtifactMetadata = field(default_factory=dict)

    def resumability_material(self) -> ArtifactMetadata:
        """Return the normalized fields that control staged-output validity.

        Returns:
            Stable mapping suitable for deterministic JSON hashing. The payload
            intentionally excludes non-control fields such as display version and
            run ID.
        """

        geography_sha = None
        if self.exact_geography is not None:
            geography_sha = self.exact_geography.metadata.get("canonical_sha256")
            if geography_sha is None:
                geography_sha = self.exact_geography.sha256

        return {
            "scope": self.scope,
            "weights_sha256": self.weights.sha256,
            "source_dataset_sha256": self.source_dataset.sha256,
            "exact_geography_sha256": geography_sha,
            "target_db_sha256": (
                self.target_db.sha256 if self.target_db is not None else None
            ),
            "n_clones": self.metadata.get("n_clones"),
            "seed": self.metadata.get("seed"),
            "policyengine_us_locked_version": self.model_build.get("locked_version"),
            "policyengine_us_git_commit": self.model_build.get("git_commit"),
        }


@pipeline_node(
    id="local_h5_traceability",
    label="FingerprintingService",
    node_type="library",
    description="Build traceability bundles and deterministic scope fingerprints for local H5 publication.",
    source_file="policyengine_us_data/build_outputs/fingerprinting.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/build_outputs/test_fingerprinting.py"
    ],
)
class FingerprintingService:
    """Build traceability bundles and derive deterministic scope fingerprints.

    The service centralizes provenance rules for local H5 publishing. It avoids
    importing heavy simulation machinery until a fallback record-count inference
    is needed.
    """

    def __init__(
        self,
        *,
        geography_loader: CalibrationGeographyLoader | None = None,
    ) -> None:
        """Create a fingerprinting service.

        Args:
            geography_loader: Optional loader used to resolve exact geography
                artifacts. Supplying this is useful in tests or alternate
                storage adapters.
        """

        self._geography_loader = geography_loader or CalibrationGeographyLoader()

    def build_traceability(
        self,
        *,
        inputs: PublishingInputBundle,
        scope: FingerprintScope,
    ) -> TraceabilityBundle:
        """Build a traceability bundle from current publish inputs.

        Args:
            inputs: File paths and run metadata for the publish scope.
            scope: Scope being published.

        Returns:
            A complete traceability bundle with content identities for required
            and available optional artifacts.

        Raises:
            FileNotFoundError: If a required artifact path does not exist.
            ValueError: If geography fallback metadata is inconsistent.
        """

        run_config_payload = self._load_json(inputs.run_config_path)
        return TraceabilityBundle(
            scope=scope,
            weights=self._build_artifact_identity("weights", inputs.weights_path),
            source_dataset=self._build_artifact_identity(
                "source_dataset",
                inputs.source_dataset_path,
            ),
            exact_geography=self._build_geography_identity(inputs),
            target_db=self._build_optional_artifact_identity(
                "target_db",
                inputs.target_db_path,
            ),
            calibration_package=self._build_optional_artifact_identity(
                "calibration_package",
                inputs.calibration_package_path,
            ),
            run_config=self._build_optional_artifact_identity(
                "run_config",
                inputs.run_config_path,
            ),
            code_version=self._extract_code_version(run_config_payload),
            model_build=self._extract_model_build(run_config_payload),
            metadata={
                "run_id": inputs.run_id,
                "version": inputs.version,
                "n_clones": inputs.n_clones,
                "seed": inputs.seed,
            },
        )

    def compute_scope_fingerprint(
        self, traceability: TraceabilityBundle
    ) -> ScopeFingerprint:
        """Hash normalized resumability material into a short fingerprint.

        Args:
            traceability: Traceability bundle whose resumability material should
                be hashed.

        Returns:
            First 16 hexadecimal characters of the SHA-256 digest over
            normalized resumability material.
        """

        payload = json.dumps(
            traceability.resumability_material(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _build_artifact_identity(
        self,
        logical_name: str,
        path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactIdentity:
        actual_path = Path(path)
        if not actual_path.exists():
            raise FileNotFoundError(
                f"Expected {logical_name} artifact at {actual_path}"
            )
        return ArtifactIdentity(
            logical_name=logical_name,
            path=actual_path,
            sha256=self._sha256_file(actual_path),
            size_bytes=actual_path.stat().st_size,
            metadata=dict(metadata or {}),
        )

    def _build_optional_artifact_identity(
        self,
        logical_name: str,
        path: Path | None,
    ) -> ArtifactIdentity | None:
        if path is None:
            return None
        actual_path = Path(path)
        if not actual_path.exists():
            return None
        return self._build_artifact_identity(logical_name, actual_path)

    def _build_geography_identity(
        self,
        inputs: PublishingInputBundle,
    ) -> ArtifactIdentity | None:
        resolved = self._geography_loader.resolve_source(
            weights_path=inputs.weights_path,
            geography_path=inputs.exact_geography_path,
            blocks_path=inputs.legacy_blocks_path,
            calibration_package_path=inputs.calibration_package_path,
        )
        if resolved is None:
            return None

        metadata = {
            "source_kind": resolved.kind,
            "canonical_sha256": self._geography_loader.compute_canonical_checksum(
                weights_path=inputs.weights_path,
                n_records=self._infer_n_records(
                    weights_path=inputs.weights_path,
                    source_dataset_path=inputs.source_dataset_path,
                    n_clones=inputs.n_clones,
                ),
                n_clones=inputs.n_clones,
                geography_path=inputs.exact_geography_path,
                blocks_path=inputs.legacy_blocks_path,
                calibration_package_path=inputs.calibration_package_path,
            ),
        }
        return self._build_artifact_identity(
            "exact_geography",
            resolved.path,
            metadata=metadata,
        )

    def _extract_code_version(
        self, run_config_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "git_commit": run_config_payload.get("git_commit"),
            "git_branch": run_config_payload.get("git_branch"),
            "git_dirty": run_config_payload.get("git_dirty"),
        }

    def _extract_model_build(
        self, run_config_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "locked_version": run_config_payload.get("package_version"),
            "git_commit": run_config_payload.get("git_commit"),
        }

    def _load_json(self, path: Path | None) -> Mapping[str, Any]:
        if path is None:
            return {}
        actual_path = Path(path)
        if not actual_path.exists():
            return {}
        with open(actual_path) as handle:
            return json.load(handle)

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _infer_n_records(
        self,
        *,
        weights_path: Path,
        source_dataset_path: Path,
        n_clones: int | None,
    ) -> int:
        if n_clones is not None:
            import numpy as np

            weights = np.load(weights_path, mmap_mode="r")
            if len(weights) % n_clones == 0:
                return int(len(weights) // n_clones)

        from policyengine_us import Microsimulation

        simulation = Microsimulation(dataset=str(source_dataset_path))
        household_ids = calculate_variable_values(
            simulation,
            "household_id",
            map_to="household",
        )
        return int(len(household_ids))
