"""Typed request contracts for local H5 publication.

This module defines the request values introduced when the worker
boundary becomes request-aware. Later contract modules should land only
when runtime code starts using them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

AreaType = Literal["national", "state", "district", "city", "custom"]
FilterOp = Literal["eq", "in"]


def _jsonable_request_value(value: Any) -> Any:
    """Convert request values into JSON-serializable primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable_request_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_request_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable_request_value(item) for key, item in value.items()}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class AreaFilter:
    """A single geography predicate used to select rows for one output area."""

    geography_field: str
    op: FilterOp
    value: str | int | tuple[str | int, ...]

    def __post_init__(self) -> None:
        if not self.geography_field:
            raise ValueError("geography_field must be non-empty")
        if self.op == "in" and not isinstance(self.value, tuple):
            raise ValueError("AreaFilter value must be a tuple when op='in'")
        if self.op == "eq" and isinstance(self.value, tuple):
            raise ValueError("AreaFilter value must not be a tuple when op='eq'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "geography_field": self.geography_field,
            "op": self.op,
            "value": _jsonable_request_value(self.value),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AreaFilter":
        value = data["value"]
        if data["op"] == "in":
            value = tuple(value)
        return cls(
            geography_field=str(data["geography_field"]),
            op=data["op"],
            value=value,
        )


@dataclass(frozen=True)
class AreaBuildRequest:
    """A complete request describing one local or national H5 to build."""

    area_type: AreaType
    area_id: str
    display_name: str
    output_relative_path: str
    filters: tuple[AreaFilter, ...] = ()
    validation_geo_level: str | None = None
    validation_geographic_ids: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.area_id:
            raise ValueError("area_id must be non-empty")
        if not self.display_name:
            raise ValueError("display_name must be non-empty")
        if not self.output_relative_path:
            raise ValueError("output_relative_path must be non-empty")
        if self.validation_geographic_ids and self.validation_geo_level is None:
            raise ValueError(
                "validation_geo_level must be set when validation_geographic_ids "
                "are provided"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_type": self.area_type,
            "area_id": self.area_id,
            "display_name": self.display_name,
            "output_relative_path": self.output_relative_path,
            "filters": [_jsonable_request_value(item) for item in self.filters],
            "validation_geo_level": self.validation_geo_level,
            "validation_geographic_ids": list(self.validation_geographic_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AreaBuildRequest":
        return cls(
            area_type=data["area_type"],
            area_id=str(data["area_id"]),
            display_name=str(data["display_name"]),
            output_relative_path=str(data["output_relative_path"]),
            filters=tuple(
                AreaFilter.from_dict(item) for item in data.get("filters", ())
            ),
            validation_geo_level=data.get("validation_geo_level"),
            validation_geographic_ids=tuple(
                str(item) for item in data.get("validation_geographic_ids", ())
            ),
            metadata=dict(data.get("metadata", {})),
        )
