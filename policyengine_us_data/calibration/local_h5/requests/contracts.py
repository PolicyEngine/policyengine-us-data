"""Typed request contracts for local H5 publication.

These types describe what to build, where outputs go, and how a request
maps onto geography filters and validation identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .._contract_utils import jsonable_contract_value

AreaType = Literal["national", "state", "district", "city", "custom"]
FilterOp = Literal["eq", "in"]


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
            "value": jsonable_contract_value(self.value),
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

    @classmethod
    def national(cls, area_id: str = "US") -> "AreaBuildRequest":
        return cls(
            area_type="national",
            area_id=area_id,
            display_name=area_id,
            output_relative_path="national/US.h5",
            validation_geo_level="national",
            validation_geographic_ids=(area_id,),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "area_type": self.area_type,
            "area_id": self.area_id,
            "display_name": self.display_name,
            "output_relative_path": self.output_relative_path,
            "filters": [jsonable_contract_value(item) for item in self.filters],
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
