"""In-memory H5 payload contracts for local-area publication outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from policyengine_us_data.pipeline_metadata import pipeline_node

from .reindexing import ReindexedEntities
from .selection import CloneSelection
from .source_dataset import SourceDatasetSnapshot

__all__ = ["H5Payload", "PayloadBuildContext"]

PeriodData = Mapping[Any, np.ndarray]
PayloadData = Mapping[str, PeriodData]

HOUSEHOLD_LENGTH_VARIABLES = frozenset(
    (
        "household_id",
        "household_weight",
        "state_fips",
        "county",
        "county_fips",
        "block_geoid",
        "tract_geoid",
        "cbsa_code",
        "sldu",
        "sldl",
        "place_fips",
        "vtd",
        "puma",
        "zcta",
        "zip_code",
        "congressional_district_geoid",
    )
)


@pipeline_node(
    id="local_h5_payload",
    label="H5Payload",
    node_type="library",
    description="Validated in-memory period-grouped payload for one local H5 output.",
    source_file="policyengine_us_data/build_outputs/payload.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_payload.py"],
)
@dataclass(frozen=True)
class H5Payload:
    """Period-grouped arrays ready to write to a local-area H5 file.

    `entity_lengths` records the row count expected for each entity in this
    payload. Shape validation intentionally covers the stable structural
    variables first; formula variables that do not encode their entity in the
    variable name are left to existing source-variable and runtime checks.
    """

    data: PayloadData
    time_period: int
    entity_lengths: Mapping[str, int] = field(default_factory=dict)
    variable_entities: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_data = _normalize_payload_data(self.data)
        normalized_entity_lengths = {
            str(entity): int(length) for entity, length in self.entity_lengths.items()
        }
        object.__setattr__(self, "data", normalized_data)
        object.__setattr__(
            self,
            "entity_lengths",
            normalized_entity_lengths,
        )
        object.__setattr__(
            self,
            "variable_entities",
            {
                str(variable): str(entity)
                for variable, entity in self.variable_entities.items()
            },
        )
        self.validate_shapes()

    def validate_shapes(self) -> None:
        """Raise when structural payload arrays do not match entity lengths."""

        for variable, periods in self.data.items():
            if not periods:
                raise ValueError(f"{variable} must contain at least one period")
            explicit_entity = self.variable_entities.get(variable)
            expected_entity = _infer_entity(
                variable,
                self.entity_lengths,
                self.variable_entities,
            )
            for period, values in periods.items():
                array = np.asarray(values)
                if array.ndim == 0:
                    raise ValueError(f"{variable}[{period}] must be array-like")
                if expected_entity is None:
                    continue
                expected_length = self.entity_lengths.get(expected_entity)
                if expected_length is None:
                    if explicit_entity is not None:
                        raise ValueError(
                            f"{variable} maps to unknown entity {expected_entity!r}"
                        )
                    continue
                if len(array) != expected_length:
                    raise ValueError(
                        f"{variable}[{period}] length {len(array)} does not match "
                        f"{expected_entity} length {expected_length}"
                    )


@pipeline_node(
    id="local_h5_payload_build_context",
    label="PayloadBuildContext",
    node_type="library",
    description="Context passed to payload postprocessors during one H5 build.",
    source_file="policyengine_us_data/build_outputs/payload.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_payload.py"],
)
@dataclass(frozen=True)
class PayloadBuildContext:
    """Context available to country-specific local H5 payload postprocessors."""

    source: SourceDatasetSnapshot
    simulation: Any
    selection: CloneSelection
    reindexed: ReindexedEntities
    geography: Any
    time_period: int
    takeup_filter: tuple[str, ...] | None = None


def _normalize_payload_data(data: PayloadData) -> dict[str, dict[Any, np.ndarray]]:
    return {
        str(variable): {
            period: np.asarray(values) for period, values in periods.items()
        }
        for variable, periods in data.items()
    }


def _infer_entity(
    variable: str,
    entity_lengths: Mapping[str, int],
    variable_entities: Mapping[str, str],
) -> str | None:
    if variable in variable_entities:
        return variable_entities[variable]
    if variable in HOUSEHOLD_LENGTH_VARIABLES:
        return "household"
    if variable in ("person_id", "person_household_id"):
        return "person"
    if variable.startswith("person_") and variable.endswith("_id"):
        return "person"
    if variable.endswith("_id"):
        entity = variable.removesuffix("_id")
        if entity in entity_lengths:
            return entity
    return None
