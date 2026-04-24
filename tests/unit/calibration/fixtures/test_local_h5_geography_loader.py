"""Fixture helpers for ``test_local_h5_geography_loader.py``."""

from __future__ import annotations

import importlib
from pathlib import Path

__test__ = False

_GEOGRAPHY_LOADER_EXPORTS = None


def load_geography_loader_exports():
    """Load the geography loader without replacing shared package modules."""

    global _GEOGRAPHY_LOADER_EXPORTS
    if _GEOGRAPHY_LOADER_EXPORTS is not None:
        return _GEOGRAPHY_LOADER_EXPORTS

    module = importlib.import_module(
        "policyengine_us_data.calibration.local_h5.geography_loader"
    )
    _GEOGRAPHY_LOADER_EXPORTS = {
        "module": module,
        "CalibrationGeographyLoader": module.CalibrationGeographyLoader,
        "ResolvedGeographySource": module.ResolvedGeographySource,
    }
    return _GEOGRAPHY_LOADER_EXPORTS


def write_saved_geography(path: Path, *, n_records: int, n_clones: int) -> None:
    """Write a small saved geography artifact for loader tests."""

    clone_and_assign = importlib.import_module(
        "policyengine_us_data.calibration.clone_and_assign"
    )

    total_rows = n_records * n_clones
    block_geoids = ["010010000001", "010010000002"] * n_clones
    cd_geoids = ["101", "102"] * n_clones
    clone_and_assign.save_geography(
        clone_and_assign.GeographyAssignment(
            block_geoid=block_geoids[:total_rows],
            cd_geoid=cd_geoids[:total_rows],
            county_fips=["01001"] * total_rows,
            state_fips=[1] * total_rows,
            n_records=n_records,
            n_clones=n_clones,
        ),
        path,
    )
