"""Coordinator-owned provenance and resumability logic for local H5 publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from .geography_loader import CalibrationGeographyLoader

FingerprintScope = Literal["regional", "national"]


@dataclass(frozen=True)
class PublishingInputBundle:
    """File-system and run metadata needed to publish one H5 scope."""

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


@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable identity for one input artifact used by traceability and resume."""

    logical_name: str
    path: Path | None
    sha256: str | None
    size_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceabilityBundle:
    """Full provenance record for one publish scope."""

    scope: FingerprintScope
    weights: ArtifactIdentity
    source_dataset: ArtifactIdentity
    exact_geography: ArtifactIdentity | None = None
    target_db: ArtifactIdentity | None = None
    calibration_package: ArtifactIdentity | None = None
    run_config: ArtifactIdentity | None = None
    code_version: Mapping[str, Any] = field(default_factory=dict)
    model_build: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resumability_material(self) -> Mapping[str, Any]:
        """Return the normalized subset that controls staged-output validity."""

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


class FingerprintingService:
    """Build traceability bundles and derive scope fingerprints from them."""

    def __init__(
        self,
        *,
        geography_loader: CalibrationGeographyLoader | None = None,
    ) -> None:
        self._geography_loader = geography_loader or CalibrationGeographyLoader()

    def build_traceability(
        self,
        *,
        inputs: PublishingInputBundle,
        scope: FingerprintScope,
    ) -> TraceabilityBundle:
        """Build a traceability bundle from current publish inputs."""

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

    def compute_scope_fingerprint(self, traceability: TraceabilityBundle) -> str:
        """Hash normalized resumability material into a short scope fingerprint."""

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
            raise FileNotFoundError(f"Expected {logical_name} artifact at {actual_path}")
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

    def _extract_code_version(self, run_config_payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "git_commit": run_config_payload.get("git_commit"),
            "git_branch": run_config_payload.get("git_branch"),
            "git_dirty": run_config_payload.get("git_dirty"),
        }

    def _extract_model_build(self, run_config_payload: Mapping[str, Any]) -> dict[str, Any]:
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
        return int(
            len(simulation.calculate("household_id", map_to="household").values)
        )
