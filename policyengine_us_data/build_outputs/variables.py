"""Variable-cloning seam for local H5 publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from policyengine_core.enums import Enum

from policyengine_us_data.calibration.formulaic_inputs import (
    drop_formulaic_spm_inputs,
)
from policyengine_us_data.datasets.puf.variable_roles import (
    PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES,
)
from policyengine_us_data.pipeline_metadata import pipeline_node

from .reindexing import ReindexedEntities
from .selection import CloneSelection
from .source_dataset import SourceDatasetSnapshot, SourceVariableMetadata

__all__ = ["VariableClonePayload", "VariableCloner", "default_variables_to_save"]

GEOGRAPHY_VARIABLES = (
    "block_geoid",
    "tract_geoid",
    "cbsa_code",
    "sldu",
    "sldl",
    "place_fips",
    "vtd",
    "puma",
    "zcta",
)


@pipeline_node(
    id="local_h5_variable_cloner",
    label="VariableCloner",
    node_type="library",
    description="Clone selected source variables into period-grouped local H5 payloads.",
    source_file="policyengine_us_data/build_outputs/variables.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_variables.py"],
)
class VariableCloner:
    """Clone source variable arrays using selected and reindexed entity rows.

    The skip/error policy intentionally mirrors the first local H5 pipeline
    implementation. Variables outside the local-H5 allowlist are skipped before
    holder access. Allowed variables are skipped when they have no known periods
    or belong to an entity that local H5s do not clone. Malformed metadata for
    an allowed variable is not skipped; it should fail so incomplete outputs are
    not produced silently.
    """

    def clone(
        self,
        *,
        source: SourceDatasetSnapshot,
        selection: CloneSelection,
        reindexed: ReindexedEntities,
        variables_to_save: set[str] | None = None,
    ) -> "VariableClonePayload":
        """Clone period-grouped variable arrays for one selected H5 output.

        Args:
            source: Source dataset snapshot with lazy variable provider.
            selection: Selected clone-household rows.
            reindexed: Output entity/source index mapping.
            variables_to_save: Optional explicit variable allowlist. When
                omitted, current local H5 inclusion rules are used.

        Returns:
            A period-grouped variable payload and cloned-period count.
        """

        _validate_reindexed_selection_alignment(
            selection=selection,
            reindexed=reindexed,
        )
        variable_allowlist = (
            set(variables_to_save)
            if variables_to_save is not None
            else default_variables_to_save(source)
        )
        indices_by_entity = _source_indices_by_entity(reindexed)

        data: dict[str, dict[Any, np.ndarray]] = {}
        variables_saved = 0

        for variable in _provider_variable_names(source.variable_provider):
            if variable not in variable_allowlist:
                continue

            metadata = source.variable_provider.get_metadata(variable)

            source_indices = indices_by_entity.get(metadata.entity_key)
            if source_indices is None:
                continue

            periods = source.variable_provider.known_periods(variable)
            if not periods:
                continue

            period_data: dict[Any, np.ndarray] = {}
            for period in periods:
                values = _provider_raw_array(source.variable_provider, variable, period)
                normalized = _normalize_values(
                    variable=variable,
                    values=values,
                    metadata=metadata,
                )
                period_data[period] = normalized[source_indices]
                variables_saved += 1

            if period_data:
                data[variable] = period_data

        return VariableClonePayload(data=data, values_saved=variables_saved)


@pipeline_node(
    id="local_h5_variable_clone_payload",
    label="VariableClonePayload",
    node_type="library",
    description="Period-grouped cloned source variables for one local H5 output.",
    source_file="policyengine_us_data/build_outputs/variables.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_variables.py"],
)
@dataclass(frozen=True)
class VariableClonePayload:
    """Cloned source variable arrays before H5-specific overrides."""

    data: Mapping[str, Mapping[Any, np.ndarray]]
    values_saved: int


def default_variables_to_save(source: SourceDatasetSnapshot) -> set[str]:
    """Return the current local H5 source-variable inclusion set."""

    variables = (
        set(source.input_variables) - PUF_REPORTED_CALCULATED_TAX_OUTPUT_VARIABLES
    )
    drop_formulaic_spm_inputs(variables)
    variables.add("county")
    variables.add("congressional_district_geoid")
    variables.update(GEOGRAPHY_VARIABLES)
    return variables


def _source_indices_by_entity(
    reindexed: ReindexedEntities,
) -> dict[str, np.ndarray]:
    indices = {
        "household": reindexed.household_source_indices,
        "person": reindexed.person_source_indices,
    }
    indices.update(reindexed.subentity_source_indices)
    return indices


def _validate_reindexed_selection_alignment(
    *,
    selection: CloneSelection,
    reindexed: ReindexedEntities,
) -> None:
    if len(reindexed.household_ids) != selection.n_selected_clones:
        raise ValueError("Reindexed household count must match selected clone count")
    if not np.array_equal(
        reindexed.household_source_indices,
        selection.source_household_indices,
    ):
        raise ValueError(
            "Reindexed household source indices must match clone selection"
        )


def _provider_variable_names(provider) -> tuple[str, ...]:
    variable_names = getattr(provider, "variable_names", None)
    if variable_names is None:
        return tuple(sorted(provider.input_variables))
    return tuple(variable_names)


def _provider_raw_array(provider, variable: str, period: Any) -> Any:
    get_raw_array = getattr(provider, "get_raw_array", None)
    if callable(get_raw_array):
        return get_raw_array(variable, period)
    return provider.get_array(variable, period)


def _normalize_values(
    *,
    variable: str,
    values: Any,
    metadata: SourceVariableMetadata,
) -> np.ndarray:
    if hasattr(values, "_pa_array") or hasattr(values, "_ndarray"):
        values = np.asarray(values)

    if metadata.value_type in (Enum, str) and variable != "county_fips":
        if hasattr(values, "decode_to_str"):
            return values.decode_to_str().astype("S")
        return np.asarray(values).astype("S")

    if variable == "county_fips":
        return np.asarray(values).astype("int32")

    return np.asarray(values)
