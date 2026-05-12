"""Canonical checksums for calibration geography assignments."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def hash_string_array(values: Any) -> str:
    """Hash one-dimensional string values independent of numpy dtype width."""

    array = np.asarray(values, dtype=str)
    if array.ndim != 1:
        raise ValueError("string array must be one-dimensional")
    digest = hashlib.sha256()
    digest.update(b"policyengine-us-data:string-array:v1")
    digest.update(len(array).to_bytes(8, byteorder="big", signed=False))
    for value in array:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def canonical_geography_checksum(
    *,
    block_geoid: Any,
    cd_geoid: Any,
    county_fips: Any | None = None,
    state_fips: Any | None = None,
    n_records: int,
    n_clones: int,
) -> str:
    """Hash normalized geography independent of source artifact format."""

    block_geoids = np.asarray(block_geoid, dtype=str)
    cd_geoids = np.asarray(cd_geoid, dtype=str)
    if block_geoids.ndim != 1:
        raise ValueError("block_geoid must be one-dimensional")
    if cd_geoids.ndim != 1:
        raise ValueError("cd_geoid must be one-dimensional")
    if len(block_geoids) != len(cd_geoids):
        raise ValueError("block_geoid and cd_geoid must have the same length")

    if county_fips is None:
        county_fips = np.fromiter(
            (str(block)[:5] for block in block_geoids),
            dtype="U5",
            count=len(block_geoids),
        )
    else:
        county_fips = np.asarray(county_fips, dtype=str)

    if state_fips is None:
        try:
            state_fips = np.fromiter(
                (int(str(block)[:2]) for block in block_geoids),
                dtype=np.int32,
                count=len(block_geoids),
            )
        except ValueError as exc:
            raise ValueError(
                "block_geoid values must start with numeric state FIPS"
            ) from exc
    else:
        state_fips = np.asarray(state_fips, dtype=np.int32)

    if np.asarray(county_fips).ndim != 1:
        raise ValueError("county_fips must be one-dimensional")
    if np.asarray(state_fips).ndim != 1:
        raise ValueError("state_fips must be one-dimensional")
    if len(county_fips) != len(block_geoids):
        raise ValueError("county_fips must have the same length as block_geoid")
    if len(state_fips) != len(block_geoids):
        raise ValueError("state_fips must have the same length as block_geoid")

    digest = hashlib.sha256()
    digest.update(b"policyengine-us-data:geography-assignment:v1")
    for values in (block_geoids, cd_geoids, county_fips):
        digest.update(hash_string_array(values).encode("utf-8"))
    digest.update(b"state_fips")
    digest.update(
        np.ascontiguousarray(np.asarray(state_fips, dtype=np.int32)).tobytes()
    )
    digest.update(str(int(n_records)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(int(n_clones)).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"
