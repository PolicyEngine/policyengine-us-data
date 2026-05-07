"""Deterministic fingerprint helpers for stage contracts."""

from __future__ import annotations

from dataclasses import is_dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from policyengine_us_data.utils.canonical_json import canonical_json_bytes

from .core import CONTRACT_FINGERPRINT_ALGORITHM, Fingerprint


def canonicalize_for_fingerprint(value: Any) -> Any:
    """Normalize supported values into deterministic JSON primitives."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return canonicalize_for_fingerprint(value.to_dict())
    if is_dataclass(value):
        raise TypeError("Dataclass values must expose to_dict() before fingerprinting")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize_for_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [canonicalize_for_fingerprint(item) for item in value]
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Fingerprint material floats must be finite")
        return value
    if isinstance(value, str | int | bool) or value is None:
        return value
    raise TypeError(f"Unsupported fingerprint material value: {type(value).__name__}")


def fingerprint_material(material: Mapping[str, Any]) -> Fingerprint:
    """Hash canonicalized semantic material into a `Fingerprint`."""

    canonical_material = canonicalize_for_fingerprint(material)
    payload = canonical_json_bytes(
        canonical_material,
        compact=True,
        trailing_newline=False,
    )
    return Fingerprint(
        algorithm=CONTRACT_FINGERPRINT_ALGORITHM,
        value=f"sha256:{sha256(payload).hexdigest()}",
        material=canonical_material,
    )
