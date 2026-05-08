"""Fixture helpers for worker-bootstrap tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from policyengine_us_data.build_outputs.fingerprinting import (
    ArtifactIdentity,
    PublishingInputBundle,
    TraceabilityBundle,
)
from policyengine_us_data.build_outputs.geography_loader import ResolvedGeographySource
from policyengine_us_data.build_outputs.source_dataset import (
    SourceDatasetSnapshot,
)
from tests.support.build_outputs.source_dataset import (
    FakeHolder,
    FakeSimulation,
)

__test__ = False


@dataclass(frozen=True)
class BootstrapTestArtifacts:
    """Tiny local files and objects for bootstrap tests."""

    inputs: PublishingInputBundle
    snapshot: SourceDatasetSnapshot
    geography_path: Path
    n_records: int
    n_clones: int


def make_bootstrap_test_artifacts(
    tmp_path: Path,
    *,
    n_records: int = 2,
    n_clones: int = 2,
) -> BootstrapTestArtifacts:
    """Create tiny files and source snapshot for bootstrap tests."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_path = tmp_path / "source_imputed_stratified_extended_cps.h5"
    weights_path = tmp_path / "calibration_weights.npy"
    db_path = tmp_path / "policy_data.db"
    geography_path = tmp_path / "geography_assignment.npz"
    run_config_path = tmp_path / "unified_run_config.json"

    dataset_path.write_bytes(b"source-dataset")
    np.save(weights_path, np.arange(1, n_records * n_clones + 1, dtype=np.float32))
    db_path.write_bytes(b"sqlite")
    geography_path.write_bytes(b"geography")
    run_config_path.write_text('{"package_version": "1.0.0", "git_commit": "abc"}')

    snapshot = SourceDatasetSnapshot.from_simulation(
        dataset_path,
        FakeSimulation(
            {
                "household_id": FakeHolder({2023: np.arange(10, 10 + n_records)}),
                "person_household_id": FakeHolder(
                    {2023: np.arange(10, 10 + n_records)}
                ),
                "tax_unit_id": FakeHolder({2023: np.arange(100, 100 + n_records)}),
                "spm_unit_id": FakeHolder({2023: np.arange(200, 200 + n_records)}),
                "family_id": FakeHolder({2023: np.arange(300, 300 + n_records)}),
                "marital_unit_id": FakeHolder({2023: np.arange(400, 400 + n_records)}),
                "person_tax_unit_id": FakeHolder(
                    {2023: np.arange(100, 100 + n_records)}
                ),
                "person_spm_unit_id": FakeHolder(
                    {2023: np.arange(200, 200 + n_records)}
                ),
                "person_family_id": FakeHolder({2023: np.arange(300, 300 + n_records)}),
                "person_marital_unit_id": FakeHolder(
                    {2023: np.arange(400, 400 + n_records)}
                ),
            }
        ),
    )
    inputs = PublishingInputBundle(
        weights_path=weights_path,
        source_dataset_path=dataset_path,
        target_db_path=db_path,
        exact_geography_path=geography_path,
        calibration_package_path=None,
        run_config_path=run_config_path,
        run_id="run-123",
        version="0.0.0",
        n_clones=n_clones,
        seed=42,
    )
    return BootstrapTestArtifacts(
        inputs=inputs,
        snapshot=snapshot,
        geography_path=geography_path,
        n_records=n_records,
        n_clones=n_clones,
    )


class FakeDatasetReader:
    """Dataset reader test double returning a prepared snapshot."""

    def __init__(self, snapshot: SourceDatasetSnapshot):
        self.snapshot = snapshot
        self.loaded_paths: list[Path] = []

    def load(self, dataset_path: Path) -> SourceDatasetSnapshot:
        self.loaded_paths.append(Path(dataset_path))
        return self.snapshot


class FakeGeographyLoader:
    """Geography loader test double with deterministic metadata."""

    def __init__(self, artifacts: BootstrapTestArtifacts):
        self.artifacts = artifacts

    def resolve_source(self, **kwargs):
        return ResolvedGeographySource(
            kind="saved_geography",
            path=self.artifacts.geography_path,
        )

    def load(self, **kwargs):
        return SimpleNamespace(
            n_records=self.artifacts.n_records,
            n_clones=self.artifacts.n_clones,
        )

    def compute_canonical_checksum(self, **kwargs):
        return "sha256:canonical-geography"


class FakeFingerprintingService:
    """Fingerprinting service test double with real traceability shape."""

    def build_traceability(self, *, inputs, scope):
        return TraceabilityBundle(
            scope=scope,
            weights=ArtifactIdentity(
                logical_name="weights",
                path=inputs.weights_path,
                sha256="sha256:weights",
                size_bytes=inputs.weights_path.stat().st_size,
            ),
            source_dataset=ArtifactIdentity(
                logical_name="source_dataset",
                path=inputs.source_dataset_path,
                sha256="sha256:source",
                size_bytes=inputs.source_dataset_path.stat().st_size,
            ),
            exact_geography=ArtifactIdentity(
                logical_name="exact_geography",
                path=inputs.exact_geography_path,
                sha256="sha256:geography",
                size_bytes=inputs.exact_geography_path.stat().st_size,
                metadata={
                    "source_kind": "saved_geography",
                    "canonical_sha256": "sha256:canonical-geography",
                },
            ),
            target_db=ArtifactIdentity(
                logical_name="target_db",
                path=inputs.target_db_path,
                sha256="sha256:db",
                size_bytes=inputs.target_db_path.stat().st_size,
            ),
            run_config=ArtifactIdentity(
                logical_name="run_config",
                path=inputs.run_config_path,
                sha256="sha256:config",
                size_bytes=inputs.run_config_path.stat().st_size,
            ),
            metadata={
                "run_id": inputs.run_id,
                "version": inputs.version,
                "n_clones": inputs.n_clones,
                "seed": inputs.seed,
            },
        )

    def compute_scope_fingerprint(self, traceability):
        return f"{traceability.scope}-fingerprint"
