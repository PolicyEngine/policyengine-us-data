"""Clone-selection seam for local H5 publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from policyengine_us_data.pipeline_metadata import pipeline_node

from .requests import AreaFilter
from .weights import CloneWeightMatrix

__all__ = ["AreaSelector", "CloneSelection"]


@pipeline_node(
    id="local_h5_area_selector",
    label="AreaSelector",
    node_type="library",
    description="Select active clone-household rows for one local H5 output.",
    source_file="policyengine_us_data/build_outputs/selection.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_selection.py"],
)
class AreaSelector:
    """Apply request geography filters to clone-level calibration weights."""

    def select(
        self,
        *,
        weights: CloneWeightMatrix,
        geography: Any,
        filters: tuple[AreaFilter, ...] = (),
    ) -> "CloneSelection":
        """Return active clone rows after applying area filters.

        Args:
            weights: Structured clone-level weights.
            geography: Geography assignment with vectors aligned to weights.
            filters: Geography filters from an `AreaBuildRequest`.

        Returns:
            A `CloneSelection` with only positive-weight clone rows.

        Raises:
            ValueError: If no clones remain, geography is misaligned, or selected
                clone rows have empty block GEOIDs.
        """

        weight_matrix = np.array(weights.as_matrix(), copy=True)

        for area_filter in filters:
            field_matrix = _geography_matrix(
                geography,
                area_filter.geography_field,
                weights=weights,
            )
            mask = _filter_mask(field_matrix, area_filter)
            weight_matrix[~mask] = 0

        active_clone_indices, source_household_indices = np.where(weight_matrix > 0)
        if len(active_clone_indices) == 0:
            raise ValueError(
                "No active clones after filtering. "
                f"filters={[item.to_dict() for item in filters]}"
            )

        block_geoids = _geography_matrix(
            geography,
            "block_geoid",
            weights=weights,
        )[active_clone_indices, source_household_indices]
        empty_blocks = np.asarray(block_geoids, dtype=str) == ""
        empty_count = int(np.sum(empty_blocks))
        if empty_count > 0:
            raise ValueError(f"{empty_count} active clones have empty block GEOIDs")

        cd_geoids = _geography_matrix(
            geography,
            "cd_geoid",
            weights=weights,
        )[active_clone_indices, source_household_indices]

        return CloneSelection(
            clone_indices=active_clone_indices,
            source_household_indices=source_household_indices,
            weights=weight_matrix[active_clone_indices, source_household_indices],
            block_geoids=block_geoids,
            congressional_district_geoids=cd_geoids,
            filters=tuple(filters),
            n_source_households=weights.n_records,
            n_total_clones=weights.n_clones,
        )


@pipeline_node(
    id="local_h5_clone_selection",
    label="CloneSelection",
    node_type="library",
    description="Selected clone-household rows for one local H5 output.",
    source_file="policyengine_us_data/build_outputs/selection.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_selection.py"],
)
@dataclass(frozen=True)
class CloneSelection:
    """Active clone rows selected for one H5 output."""

    clone_indices: np.ndarray
    source_household_indices: np.ndarray
    weights: np.ndarray
    block_geoids: np.ndarray
    congressional_district_geoids: np.ndarray
    filters: tuple[AreaFilter, ...]
    n_source_households: int
    n_total_clones: int

    def __post_init__(self) -> None:
        clone_indices = _readonly_1d_array(self.clone_indices, "clone_indices")
        source_household_indices = _readonly_1d_array(
            self.source_household_indices,
            "source_household_indices",
        )
        weights = _readonly_1d_array(self.weights, "weights")
        block_geoids = _readonly_1d_array(self.block_geoids, "block_geoids")
        cd_geoids = _readonly_1d_array(
            self.congressional_district_geoids,
            "congressional_district_geoids",
        )

        lengths = {
            len(clone_indices),
            len(source_household_indices),
            len(weights),
            len(block_geoids),
            len(cd_geoids),
        }
        if len(lengths) != 1:
            raise ValueError("CloneSelection arrays must have matching lengths")
        if len(clone_indices) == 0:
            raise ValueError("CloneSelection must contain at least one clone row")
        if not np.issubdtype(clone_indices.dtype, np.integer):
            raise TypeError("clone_indices must be integer indices")
        if not np.issubdtype(source_household_indices.dtype, np.integer):
            raise TypeError("source_household_indices must be integer indices")
        if not np.issubdtype(weights.dtype, np.number):
            raise TypeError("weights must be numeric")
        if np.issubdtype(weights.dtype, np.complexfloating):
            raise TypeError("weights must be real numeric values")

        clone_indices = clone_indices.astype(np.int64, copy=True)
        clone_indices.setflags(write=False)
        source_household_indices = source_household_indices.astype(np.int64, copy=True)
        source_household_indices.setflags(write=False)

        object.__setattr__(self, "clone_indices", clone_indices)
        object.__setattr__(
            self,
            "source_household_indices",
            source_household_indices,
        )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "block_geoids", block_geoids)
        object.__setattr__(self, "congressional_district_geoids", cd_geoids)
        object.__setattr__(self, "filters", tuple(self.filters))
        object.__setattr__(self, "n_source_households", int(self.n_source_households))
        object.__setattr__(self, "n_total_clones", int(self.n_total_clones))

    @property
    def n_selected_clones(self) -> int:
        """Return the number of selected clone-household rows."""

        return int(len(self.clone_indices))


def _filter_mask(field_matrix: np.ndarray, area_filter: AreaFilter) -> np.ndarray:
    if area_filter.op == "eq":
        return field_matrix == area_filter.value
    if area_filter.op == "in":
        return np.isin(field_matrix, list(area_filter.value))
    raise ValueError(f"Unsupported area filter op: {area_filter.op}")


def _geography_matrix(
    geography: Any,
    field: str,
    *,
    weights: CloneWeightMatrix,
) -> np.ndarray:
    try:
        values = getattr(geography, field)
    except AttributeError as exc:
        raise ValueError(f"Geography is missing required field {field!r}") from exc

    vector = np.asarray(values)
    expected_length = weights.n_clones * weights.n_records
    if vector.size != expected_length:
        raise ValueError(
            f"Geography field {field!r} length {vector.size} does not equal "
            f"n_clones * n_records={expected_length}"
        )
    if field == "cd_geoid":
        vector = vector.astype(str)
    return vector.reshape(weights.n_clones, weights.n_records)


def _readonly_1d_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    normalized = np.array(array, copy=True)
    normalized.setflags(write=False)
    return normalized
