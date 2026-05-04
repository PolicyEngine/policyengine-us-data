"""Typed request contracts for local H5 publication.

This module defines the request values introduced when the worker
boundary becomes request-aware. Later contract modules should land only
when runtime code starts using them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from policyengine_us_data.pipeline_metadata import pipeline_node

AreaType = Literal["national", "state", "district", "city", "custom"]
FilterOp = Literal["eq", "in"]
AreaFilterValue = str | int | tuple[str | int, ...]
SerializedAreaFilter = dict[str, Any]
SerializedAreaBuildRequest = dict[str, Any]

__all__ = [
    "AreaBuildRequest",
    "AreaFilter",
    "AreaFilterValue",
    "AreaType",
    "FilterOp",
    "SerializedAreaBuildRequest",
    "SerializedAreaFilter",
]


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


def _validate_output_relative_path(output_relative_path: str) -> None:
    """Validate that a request output path stays within its worker output dir."""

    output_path = PurePosixPath(output_relative_path)
    if output_path.is_absolute():
        raise ValueError("output_relative_path must be relative")
    if ".." in output_path.parts:
        raise ValueError(
            "output_relative_path must not contain parent-directory traversal"
        )


@pipeline_node(
    id="local_h5_area_filter",
    label="AreaFilter",
    node_type="library",
    description="Typed geography predicate for local H5 output selection.",
    source_file="policyengine_us_data/build_outputs/requests.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/calibration/test_local_h5_requests.py"
    ],
)
@dataclass(frozen=True)
class AreaFilter:
    """Predicate used to select calibrated clones for one H5 output.

    `AreaFilter` is intentionally small so it can move across process
    boundaries as JSON. It describes one boolean condition over a geography
    vector, such as selecting all clones in a set of congressional districts or
    counties.

    Attributes:
        geography_field: Name of the geography vector to inspect. Current
            callers use fields from `GeographyAssignment`, such as `cd_geoid` or
            `county_fips`.
        op: Predicate operator. Use `"eq"` for one scalar value and `"in"` for
            membership in a tuple of values.
        value: Scalar comparison value for `"eq"` or tuple of allowed values for
            `"in"`.
    """

    geography_field: str
    op: FilterOp
    value: AreaFilterValue

    def __post_init__(self) -> None:
        if not self.geography_field:
            raise ValueError("geography_field must be non-empty")
        if self.op == "in" and not isinstance(self.value, tuple):
            raise ValueError("AreaFilter value must be a tuple when op='in'")
        if self.op == "eq" and isinstance(self.value, tuple):
            raise ValueError("AreaFilter value must not be a tuple when op='eq'")

    def to_dict(self) -> SerializedAreaFilter:
        """Serialize the filter to JSON-compatible primitives.

        Returns:
            A dictionary suitable for JSON encoding and worker handoff.
        """

        return {
            "geography_field": self.geography_field,
            "op": self.op,
            "value": _jsonable_request_value(self.value),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AreaFilter":
        """Rebuild an `AreaFilter` from serialized request data.

        Args:
            data: Mapping produced by `to_dict`.

        Returns:
            A validated `AreaFilter` instance.
        """

        value = data["value"]
        if data["op"] == "in":
            value = tuple(value)
        return cls(
            geography_field=str(data["geography_field"]),
            op=data["op"],
            value=value,
        )


@pipeline_node(
    id="local_h5_area_request",
    label="AreaBuildRequest",
    node_type="library",
    description="Typed request contract for one national, state, district, city, or custom H5 output.",
    source_file="policyengine_us_data/build_outputs/requests.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=[
        "uv run pytest tests/unit/calibration/test_local_h5_requests.py"
    ],
)
@dataclass(frozen=True)
class AreaBuildRequest:
    """Complete request for one local-area or national H5 file.

    The request is the stable library-level contract between orchestration code
    and H5 worker code. It names the output, carries the geography filters used
    to zero irrelevant clone weights, and records the target geography IDs used
    by validation.

    Attributes:
        area_type: Output family, such as `"state"`, `"district"`, `"city"`, or
            `"national"`.
        area_id: Stable identifier for the requested area, for example `"CA"`,
            `"NY-12"`, `"NYC"`, or `"US"`.
        display_name: Human-readable label used in logs and validation output.
        output_relative_path: Path below the worker output directory where the
            H5 should be written.
        filters: Geography predicates that select clones for this area.
        validation_geo_level: Target table geography level used to validate the
            output, when validation is available.
        validation_geographic_ids: Target geography IDs expected during
            validation.
        metadata: Additional non-control metadata for downstream tooling.
    """

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
        _validate_output_relative_path(self.output_relative_path)
        if self.validation_geographic_ids and self.validation_geo_level is None:
            raise ValueError(
                "validation_geo_level must be set when validation_geographic_ids "
                "are provided"
            )

    def to_dict(self) -> SerializedAreaBuildRequest:
        """Serialize the request to JSON-compatible primitives.

        Returns:
            A dictionary suitable for Modal worker payloads or manifests.
        """

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
        """Rebuild an `AreaBuildRequest` from serialized request data.

        Args:
            data: Mapping produced by `to_dict`.

        Returns:
            A validated `AreaBuildRequest` instance.
        """

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
