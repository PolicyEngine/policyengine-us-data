"""Pure area-selection helpers for local H5 publishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import AreaFilter
from .weights import CloneWeightMatrix


@dataclass(frozen=True)
class CloneSelection:
    active_clone_indices: np.ndarray
    active_household_indices: np.ndarray
    active_weights: np.ndarray
    active_block_geoids: np.ndarray
    active_cd_geoids: np.ndarray
    active_county_fips: np.ndarray
    active_state_fips: np.ndarray

    @property
    def n_household_clones(self) -> int:
        return int(len(self.active_household_indices))

    @property
    def is_empty(self) -> bool:
        return self.n_household_clones == 0


class AreaSelector:
    """Select active clone-household cells for an area's geography filters."""

    _SUPPORTED_FIELDS = (
        "block_geoid",
        "cd_geoid",
        "county_fips",
        "state_fips",
    )

    def select(
        self,
        weights: CloneWeightMatrix,
        geography: Any,
        *,
        filters: tuple[AreaFilter, ...] = (),
    ) -> CloneSelection:
        self._validate_geography_shape(weights, geography)

        weight_matrix = weights.as_matrix()
        shape = weight_matrix.shape

        block_matrix = self._field_matrix(geography, "block_geoid", shape)
        cd_matrix = self._field_matrix(geography, "cd_geoid", shape)
        county_matrix = self._field_matrix(geography, "county_fips", shape)
        state_matrix = self._field_matrix(geography, "state_fips", shape)

        active_mask = weight_matrix > 0
        for area_filter in filters:
            active_mask &= self._apply_filter(
                values=self._field_matrix(
                    geography,
                    area_filter.geography_field,
                    shape,
                ),
                area_filter=area_filter,
            )

        active_clone_indices, active_household_indices = np.where(active_mask)

        return CloneSelection(
            active_clone_indices=active_clone_indices.astype(np.int64),
            active_household_indices=active_household_indices.astype(np.int64),
            active_weights=weight_matrix[
                active_clone_indices, active_household_indices
            ],
            active_block_geoids=block_matrix[
                active_clone_indices, active_household_indices
            ],
            active_cd_geoids=cd_matrix[active_clone_indices, active_household_indices],
            active_county_fips=county_matrix[
                active_clone_indices, active_household_indices
            ],
            active_state_fips=state_matrix[
                active_clone_indices, active_household_indices
            ],
        )

    def _validate_geography_shape(
        self,
        weights: CloneWeightMatrix,
        geography: Any,
    ) -> None:
        if getattr(geography, "n_records", weights.n_records) != weights.n_records:
            raise ValueError(
                "Geography n_records does not match weight matrix "
                f"({getattr(geography, 'n_records', None)} != {weights.n_records})"
            )
        if getattr(geography, "n_clones", weights.n_clones) != weights.n_clones:
            raise ValueError(
                "Geography n_clones does not match weight matrix "
                f"({getattr(geography, 'n_clones', None)} != {weights.n_clones})"
            )

    def _field_matrix(
        self,
        geography: Any,
        field_name: str,
        shape: tuple[int, int],
    ) -> np.ndarray:
        if field_name not in self._SUPPORTED_FIELDS:
            raise ValueError(
                f"Unsupported geography field {field_name!r}; "
                f"supported fields: {', '.join(self._SUPPORTED_FIELDS)}"
            )
        if not hasattr(geography, field_name):
            raise ValueError(f"Geography is missing field {field_name!r}")

        values = np.asarray(getattr(geography, field_name))
        expected_size = shape[0] * shape[1]
        if values.size != expected_size:
            raise ValueError(
                f"Geography field {field_name!r} has length {values.size}; "
                f"expected {expected_size}"
            )
        return values.reshape(shape)

    def _apply_filter(
        self,
        *,
        values: np.ndarray,
        area_filter: AreaFilter,
    ) -> np.ndarray:
        if area_filter.op == "eq":
            return values == area_filter.value
        if area_filter.op == "in":
            return np.isin(values, list(area_filter.value))
        raise ValueError(f"Unsupported filter op {area_filter.op!r}")
