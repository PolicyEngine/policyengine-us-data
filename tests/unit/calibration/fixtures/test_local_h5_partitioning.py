"""Fixture helpers for ``test_local_h5_partitioning.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__test__ = False


def _load_partitioning_module():
    """Load the pure partitioning module directly from disk."""

    repo_root = Path(__file__).resolve().parents[4]
    module_path = (
        repo_root
        / "policyengine_us_data"
        / "calibration"
        / "local_h5"
        / "partitioning.py"
    )
    spec = importlib.util.spec_from_file_location(
        "local_h5_partitioning",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def flatten_chunks(chunks):
    """Flatten worker chunks into a single item list for assertions."""

    return [item for chunk in chunks for item in chunk]


def load_partitioning_exports():
    """Load the partitioning module and return its public exports."""

    module = _load_partitioning_module()
    return {
        "module": module,
        "flatten_chunks": flatten_chunks,
        "partition_weighted_work_items": module.partition_weighted_work_items,
        "work_item_key": module.work_item_key,
    }
