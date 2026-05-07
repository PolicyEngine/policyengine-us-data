"""Private validation and serialization helpers for stage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType
from typing import Any

from .constants import CONTRACT_SCHEMA_VERSION


def validate_schema_version(schema_version: str, owner: str) -> None:
    if schema_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"{owner} schema_version must be {CONTRACT_SCHEMA_VERSION!r}")


def require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def required_string(data: Mapping[str, Any], field_name: str) -> str:
    return require_non_empty(data.get(field_name), field_name)


def optional_string(data: Mapping[str, Any], field_name: str) -> str | None:
    return optional_string_value(data.get(field_name), field_name)


def optional_string_value(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return value


def schema_version(data: Mapping[str, Any]) -> str:
    value = data.get("schema_version", CONTRACT_SCHEMA_VERSION)
    if not isinstance(value, str) or not value:
        raise ValueError("schema_version must be a non-empty string")
    return value


def mapping_value(data: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = data.get(field_name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def optional_mapping_value(
    data: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any] | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    return value


def validate_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def validate_optional_int(value: Any, field_name: str) -> None:
    if value is not None:
        validate_int(value, field_name)


def validate_optional_float(value: Any, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def int_value(
    data: Mapping[str, Any],
    field_name: str,
    default: int,
) -> int:
    value = data.get(field_name, default)
    validate_int(value, field_name)
    return value


def optional_int_value(
    data: Mapping[str, Any],
    field_name: str,
) -> int | None:
    value = data.get(field_name)
    validate_optional_int(value, field_name)
    return value


def optional_float_value(
    data: Mapping[str, Any],
    field_name: str,
) -> float | None:
    value = data.get(field_name)
    validate_optional_float(value, field_name)
    return float(value) if value is not None else None


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, tuple | list):
        return tuple(freeze_value(item) for item in value)
    return value


def freeze_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return MappingProxyType(
        {str(key): freeze_value(item) for key, item in value.items()}
    )


def freeze_sequence(
    value: Any,
    field_name: str,
    item_type: type | tuple[type, ...],
) -> tuple[Any, ...]:
    if not isinstance(value, tuple | list):
        raise ValueError(f"{field_name} must be a tuple or list")
    items = tuple(value)
    for item in items:
        if not isinstance(item, item_type):
            if isinstance(item_type, tuple):
                expected = " or ".join(kind.__name__ for kind in item_type)
            else:
                expected = item_type.__name__
            raise ValueError(f"{field_name} entries must be {expected}")
    return items


def validate_optional_instance(
    value: Any,
    field_name: str,
    expected_type: type,
) -> None:
    if value is not None and not isinstance(value, expected_type):
        raise ValueError(f"{field_name} must be {expected_type.__name__}")


def jsonable_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, tuple | list):
        return [jsonable_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    return value
