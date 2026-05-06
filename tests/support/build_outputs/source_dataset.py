"""Fixture helpers for build-output source dataset tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _ensure_package(name: str, path: Path) -> None:
    """Register a synthetic package so local imports resolve from disk."""

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


def load_source_dataset_exports():
    """Load the source dataset module under a synthetic package name."""

    build_outputs_root = (
        Path(__file__).resolve().parents[3] / "policyengine_us_data" / "build_outputs"
    )
    package_name = "build_outputs_source_dataset_fixture"

    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)

    _ensure_package(package_name, build_outputs_root)
    source_dataset_module = _load_module(
        f"{package_name}.source_dataset",
        build_outputs_root / "source_dataset.py",
    )
    return {
        "module": source_dataset_module,
        "EntityGraph": source_dataset_module.EntityGraph,
    }


def make_entity_graph_arrays():
    """Return small valid entity graph arrays with two households."""

    return {
        "household_ids": np.array([10, 20], dtype=np.int64),
        "person_household_ids": np.array([10, 10, 20], dtype=np.int64),
        "subentity_ids": {
            "tax_unit": np.array([100, 200], dtype=np.int64),
            "spm_unit": np.array([300, 400], dtype=np.int64),
        },
        "person_subentity_ids": {
            "tax_unit": np.array([100, 100, 200], dtype=np.int64),
            "spm_unit": np.array([300, 300, 400], dtype=np.int64),
        },
    }
