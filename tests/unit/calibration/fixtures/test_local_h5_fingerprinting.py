"""Fixture helpers for ``test_local_h5_fingerprinting.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np

from tests.unit.calibration.fixtures.test_local_h5_geography_loader import (
    write_saved_geography,
)

__test__ = False


def _ensure_package(name: str, path: Path) -> None:
    """Register a synthetic package so relative imports resolve locally."""

    package = sys.modules.get(name)
    if package is None:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
        return
    package.__path__ = [str(path)]


def _load_module(name: str, path: Path):
    """Load one module from disk under a specific fully-qualified name."""

    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_fingerprinting_exports():
    """Load the local H5 fingerprinting module under a synthetic package name."""

    repo_root = Path(__file__).resolve().parents[4]
    local_h5_root = (
        repo_root
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
    )
    calibration_root = repo_root / "policyengine_us_data" / "calibration"
    storage_root = repo_root / "policyengine_us_data" / "storage"
    package_name = "local_h5_fingerprinting_fixture"
    policyengine_package = "policyengine_us_data"
    calibration_package = "policyengine_us_data.calibration"

    for name in list(sys.modules):
        if (
            name == package_name
            or name.startswith(f"{package_name}.")
            or name == policyengine_package
            or name.startswith(f"{policyengine_package}.")
        ):
            sys.modules.pop(name, None)

    _ensure_package(package_name, local_h5_root)
    _ensure_package(policyengine_package, repo_root / "policyengine_us_data")
    _ensure_package(calibration_package, calibration_root)
    _load_module(
        "policyengine_us_data.storage",
        storage_root / "__init__.py",
    )
    _load_module(
        "policyengine_us_data.calibration.clone_and_assign",
        calibration_root / "clone_and_assign.py",
    )
    _load_module(
        f"{package_name}.geography_loader",
        local_h5_root / "geography_loader.py",
    )
    module = _load_module(
        f"{package_name}.fingerprinting",
        local_h5_root / "fingerprinting.py",
    )
    return {
        "module": module,
        "ArtifactIdentity": module.ArtifactIdentity,
        "FingerprintingService": module.FingerprintingService,
        "PublishingInputBundle": module.PublishingInputBundle,
        "TraceabilityBundle": module.TraceabilityBundle,
    }


def write_source_dataset(path: Path, *, n_records: int) -> None:
    """Write a minimal HDF5 dataset with a ``person`` entity."""

    with h5py.File(path, "w") as handle:
        person = handle.create_group("person")
        person.create_dataset("person_id", data=np.arange(n_records, dtype=np.int32))


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

    np.save(weights_path, np.array([1.0, 2.0, 3.0]))
    write_source_dataset(dataset_path, n_records=n_records)
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
