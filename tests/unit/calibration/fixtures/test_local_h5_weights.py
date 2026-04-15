"""Fixture helpers for ``test_local_h5_weights.py``."""

from __future__ import annotations

import importlib

import numpy as np

__test__ = False

_WEIGHTS_EXPORTS = None


def load_weights_exports():
    """Load the weights module without replacing shared package modules."""

    global _WEIGHTS_EXPORTS
    if _WEIGHTS_EXPORTS is not None:
        return _WEIGHTS_EXPORTS

    module = importlib.import_module(
        "policyengine_us_data.calibration.local_h5.weights"
    )
    _WEIGHTS_EXPORTS = {
        "module": module,
        "CloneWeightMatrix": module.CloneWeightMatrix,
    }
    return _WEIGHTS_EXPORTS


def make_weight_vector(length: int) -> np.ndarray:
    """Create a small deterministic numeric vector for shape tests."""

    return np.arange(length, dtype=float) + 1.0
