"""Deterministic fingerprint helpers for stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from policyengine_us_data.utils.canonical_json import canonical_json_bytes

from ._coercion import (
    freeze_mapping,
    jsonable_value,
    mapping_value,
    require_non_empty,
    required_string,
    schema_version,
    validate_schema_version,
)
from .constants import (
    CONTRACT_FINGERPRINT_ALGORITHM,
    CONTRACT_SCHEMA_VERSION,
    ContractPayload,
)


@dataclass(frozen=True, kw_only=True)
class Fingerprint:
    """Canonical hash for stage contract semantic material."""

    value: str
    material: Mapping[str, Any]
    algorithm: str = CONTRACT_FINGERPRINT_ALGORITHM
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, self.__class__.__name__)
        object.__setattr__(
            self,
            "algorithm",
            require_non_empty(self.algorithm, "algorithm"),
        )
        object.__setattr__(self, "value", require_non_empty(self.value, "value"))
        object.__setattr__(
            self,
            "material",
            freeze_mapping(self.material, "material"),
        )

    def to_dict(self) -> ContractPayload:
        return {
            "algorithm": self.algorithm,
            "value": self.value,
            "material": jsonable_value(self.material),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fingerprint":
        return cls(
            algorithm=(
                required_string(data, "algorithm")
                if "algorithm" in data
                else CONTRACT_FINGERPRINT_ALGORITHM
            ),
            value=required_string(data, "value"),
            material=mapping_value(data, "material"),
            schema_version=schema_version(data),
        )


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
