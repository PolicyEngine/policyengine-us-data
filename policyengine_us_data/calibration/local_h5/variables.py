"""Generic variable export for local H5 publishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .reindexing import ReindexedEntities
from .source_dataset import SourceDatasetSnapshot


@dataclass(frozen=True)
class VariableExportPolicy:
    include_input_variables: bool = True
    required_variables: frozenset[str] = frozenset()
    excluded_variables: frozenset[str] = frozenset()

    def variable_names(self, source: SourceDatasetSnapshot) -> tuple[str, ...]:
        selected = set()
        if self.include_input_variables:
            selected.update(source.input_variables)
        selected.update(self.required_variables)
        selected.difference_update(self.excluded_variables)
        return tuple(sorted(selected))


@dataclass(frozen=True)
class H5Payload:
    variables: Mapping[str, Mapping[int | str, np.ndarray]]

    @property
    def dataset_count(self) -> int:
        return sum(len(periods) for periods in self.variables.values())


class VariableCloner:
    """Clone source arrays for the selected entities and periods."""

    def clone(
        self,
        source: SourceDatasetSnapshot,
        reindexed: ReindexedEntities,
        policy: VariableExportPolicy,
    ) -> H5Payload:
        provider = source.variable_provider
        clone_index_map = {
            "household": reindexed.household_source_indices,
            "person": reindexed.person_source_indices,
            **reindexed.entity_source_indices,
        }

        payload: dict[str, dict[int | str, np.ndarray]] = {}
        for variable in policy.variable_names(source):
            var_def = provider.get_variable_definition(variable)
            if var_def is None:
                continue

            entity_key = var_def.entity.key
            if entity_key not in clone_index_map:
                continue

            periods = provider.get_known_periods(variable)
            if not periods:
                continue

            clone_indices = clone_index_map[entity_key]
            var_data: dict[int | str, np.ndarray] = {}
            for period in periods:
                values = provider.get_array(variable, period)
                coerced = self._coerce_output_array(
                    variable=variable,
                    values=values,
                    value_type=var_def.value_type,
                )
                var_data[period] = coerced[clone_indices]

            if var_data:
                payload[variable] = var_data

        return H5Payload(variables=payload)

    def _coerce_output_array(
        self,
        *,
        variable: str,
        values,
        value_type,
    ) -> np.ndarray:
        if hasattr(values, "_pa_array") or hasattr(values, "_ndarray"):
            values = np.asarray(values)

        if variable == "county_fips":
            return np.asarray(values).astype("int32")

        if self._is_string_like_value_type(value_type):
            if hasattr(values, "decode_to_str"):
                return values.decode_to_str().astype("S")
            return np.asarray(values).astype("S")

        return np.asarray(values)

    def _is_string_like_value_type(self, value_type) -> bool:
        if value_type is str:
            return True
        return getattr(value_type, "__name__", None) == "Enum"
