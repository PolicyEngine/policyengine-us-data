"""Fixture data and writers for unit geography tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

__test__ = False

CHECKSUM_BLOCK_GEOIDS = ("010010001", "010010002")
CHECKSUM_CD_GEOIDS = ("0101", "0102")
CHECKSUM_COUNTY_FIPS = ("01001",)
CHECKSUM_STATE_FIPS = (1,)

LOADER_BLOCK_GEOIDS = (
    "010010000001",
    "010010000002",
    "010010000001",
    "010010000002",
)
LOADER_CD_GEOIDS = ("101", "102", "101", "102")


def checksum_block_geoids(*, dtype: str = "<U9") -> np.ndarray:
    """Return small block GEOIDs with the requested NumPy string dtype."""

    return np.array(CHECKSUM_BLOCK_GEOIDS, dtype=dtype)


def checksum_cd_geoids(*, dtype: str = "<U4") -> np.ndarray:
    """Return small CD GEOIDs with the requested NumPy string dtype."""

    return np.array(CHECKSUM_CD_GEOIDS, dtype=dtype)


def write_weights(path: Path) -> None:
    """Write a small calibration weights artifact."""

    np.save(path, np.array([1.0, 2.0]))


def stage_2_geography_package() -> dict:
    """Return a package-shaped geography payload with Stage 2 metadata."""

    return {
        "metadata": {
            "base_n_records": 2,
            "n_clones": 2,
        },
        "block_geoid": np.array(LOADER_BLOCK_GEOIDS, dtype="<U15"),
        "cd_geoid": np.array(LOADER_CD_GEOIDS, dtype="<U10"),
    }


def write_stage_2_geography_package(path: Path) -> dict:
    """Write and return a package-shaped geography payload."""

    package = stage_2_geography_package()
    with path.open("wb") as handle:
        pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return package
