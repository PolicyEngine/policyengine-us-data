"""Fixture helpers for unit geography loader tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from tests.support.build_outputs.geography_loader import (
    load_geography_loader_exports,
    write_saved_geography,
)

from .geography import (
    LOADER_BLOCK_GEOIDS,
    LOADER_CD_GEOIDS,
    stage_2_geography_package,
)

__test__ = False

_EXPORTS = load_geography_loader_exports()
geography_loader_module = _EXPORTS["module"]
CalibrationGeographyIndex = _EXPORTS["CalibrationGeographyIndex"]
CalibrationGeographyLoader = _EXPORTS["CalibrationGeographyLoader"]


class ReconstructGeographySpy:
    """Spy for geography reconstruction fallback tests."""

    def __init__(self) -> None:
        self.block_geoids: tuple[str, ...] | None = None
        self.n_records: int | None = None
        self.n_clones: int | None = None

    def __call__(self, *, block_geoids, n_records, n_clones):
        self.block_geoids = tuple(block_geoids)
        self.n_records = n_records
        self.n_clones = n_clones
        return "reconstructed"


def write_saved_geography_artifact(
    path: Path,
    *,
    n_records: int = 2,
    n_clones: int = 2,
) -> None:
    """Write a saved geography assignment artifact."""

    write_saved_geography(path, n_records=n_records, n_clones=n_clones)


def write_legacy_blocks_artifact(path: Path) -> None:
    """Write a legacy stacked blocks artifact."""

    np.save(path, np.array(LOADER_BLOCK_GEOIDS))


def write_loader_calibration_package(path: Path) -> dict:
    """Write and return a package-shaped geography payload."""

    package = {
        "block_geoid": np.array(LOADER_BLOCK_GEOIDS),
        "cd_geoid": np.array(LOADER_CD_GEOIDS),
    }
    with path.open("wb") as handle:
        pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return package


def write_stage_2_loader_calibration_package(path: Path) -> dict:
    """Write and return a Stage 2 metadata-bearing geography package."""

    package = stage_2_geography_package()
    with path.open("wb") as handle:
        pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return package


def patch_reconstruct_geography_from_blocks(monkeypatch) -> ReconstructGeographySpy:
    """Patch block reconstruction and return the spy."""

    spy = ReconstructGeographySpy()
    monkeypatch.setattr(
        geography_loader_module,
        "reconstruct_geography_from_blocks",
        spy,
    )
    return spy
