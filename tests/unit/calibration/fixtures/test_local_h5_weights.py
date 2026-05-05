"""Fixture helpers for ``test_local_h5_weights.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

__test__ = False


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


def load_weights_exports():
    """Load the local H5 weights module under a synthetic package name."""

    local_h5_root = (
        Path(__file__).resolve().parents[4]
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
    )
    package_name = "local_h5_weights_fixture"

    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)

    _ensure_package(package_name, local_h5_root)
    weights_module = _load_module(
        f"{package_name}.weights",
        local_h5_root / "weights.py",
    )
    return {
        "module": weights_module,
        "CloneWeightMatrix": weights_module.CloneWeightMatrix,
        "make_weight_vector": make_weight_vector,
    }


def make_weight_vector(length: int) -> np.ndarray:
    """Create a small deterministic numeric vector for shape tests."""

    return np.arange(length, dtype=float) + 1.0
