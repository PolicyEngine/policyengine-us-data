"""Shared helpers used by typed local H5 contract modules.

This module stays private because it exists only to support contract
serialization and should not become part of the public migration
surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def jsonable_contract_value(value: Any) -> Any:
    """Convert contract values into JSON-serializable primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable_contract_value(item) for item in value]
    if isinstance(value, list):
        return [jsonable_contract_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): jsonable_contract_value(item)
            for key, item in value.items()
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value
