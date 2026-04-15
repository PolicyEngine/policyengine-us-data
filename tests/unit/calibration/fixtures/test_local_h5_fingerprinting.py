"""Fixture helpers for ``test_local_h5_fingerprinting.py``."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import h5py
import numpy as np

from tests.unit.calibration.fixtures.test_local_h5_geography_loader import (
    write_saved_geography,
)

__test__ = False

_FINGERPRINTING_EXPORTS = None


def load_fingerprinting_exports():
    """Load the fingerprinting module without replacing shared package modules."""

    global _FINGERPRINTING_EXPORTS
    if _FINGERPRINTING_EXPORTS is not None:
        return _FINGERPRINTING_EXPORTS

    module = importlib.import_module(
        "policyengine_us_data.calibration.local_h5.fingerprinting"
    )
    _FINGERPRINTING_EXPORTS = {
        "module": module,
        "ArtifactIdentity": module.ArtifactIdentity,
        "FingerprintingService": module.FingerprintingService,
        "PublishingInputBundle": module.PublishingInputBundle,
        "TraceabilityBundle": module.TraceabilityBundle,
    }
    return _FINGERPRINTING_EXPORTS


def write_source_dataset(
    path: Path,
    *,
    n_records: int,
    person_records: int | None = None,
) -> None:
    """Write a minimal HDF5 dataset with a ``person`` entity."""

    person_count = person_records if person_records is not None else n_records
    with h5py.File(path, "w") as handle:
        person = handle.create_group("person")
        person.create_dataset("person_id", data=np.arange(person_count, dtype=np.int32))


def write_run_config(path: Path, *, package_version: str = "1.0.0") -> None:
    """Write a minimal run-config payload with provenance fields."""

    payload = {
        "git_commit": "deadbeefcafebabe",
        "git_branch": "main",
        "git_dirty": False,
        "package_version": package_version,
    }
    path.write_text(json.dumps(payload))


def write_artifact_file(path: Path, content: bytes) -> None:
    """Write one small binary artifact for traceability tests."""

    path.write_bytes(content)


def make_publishing_inputs(
    bundle_cls,
    *,
    tmp_path: Path,
    n_records: int = 2,
    person_records: int | None = None,
    n_clones: int = 2,
    seed: int = 42,
    package_version: str = "1.0.0",
):
    """Create a fully-populated publishing input bundle for tests."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    weights_path = tmp_path / "calibration_weights.npy"
    dataset_path = tmp_path / "source.h5"
    db_path = tmp_path / "policy_data.db"
    geography_path = tmp_path / "geography_assignment.npz"
    run_config_path = tmp_path / "unified_run_config.json"

    np.save(weights_path, np.arange(n_records * n_clones, dtype=float) + 1.0)
    write_source_dataset(
        dataset_path,
        n_records=n_records,
        person_records=person_records,
    )
    write_artifact_file(db_path, b"fake-db")
    write_saved_geography(
        geography_path,
        n_records=n_records,
        n_clones=n_clones,
    )
    write_run_config(run_config_path, package_version=package_version)

    return bundle_cls(
        weights_path=weights_path,
        source_dataset_path=dataset_path,
        target_db_path=db_path,
        exact_geography_path=geography_path,
        calibration_package_path=None,
        run_config_path=run_config_path,
        run_id="run-123",
        version="1.2.3",
        n_clones=n_clones,
        seed=seed,
    )
