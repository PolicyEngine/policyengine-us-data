"""Shared coercion helpers for typed release-promotion results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from policyengine_us_data.stage_contracts._coercion import require_non_empty


def nonnegative_int(value: Any, field_name: str) -> int:
    """Return a non-negative integer or raise a contract validation error."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def bool_value(value: Any, field_name: str) -> bool:
    """Return a boolean or raise a contract validation error."""

    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    """Return a tuple of non-empty strings from a sequence value."""

    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings")
    items = tuple(value)
    for item in items:
        require_non_empty(item, f"{field_name} item")
    return items


def require_type(value: Any, field_name: str, expected_type: type) -> None:
    """Validate a nested typed result object."""

    if not isinstance(value, expected_type):
        raise ValueError(f"{field_name} must be {expected_type.__name__}")
