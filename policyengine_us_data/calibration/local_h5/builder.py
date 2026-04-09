"""One-area orchestration for local H5 publishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import AreaFilter
from .reindexing import EntityReindexer, ReindexedEntities
from .selection import AreaSelector, CloneSelection
from .source_dataset import SourceDatasetSnapshot
from .us_augmentations import USAugmentationService
from .variables import H5Payload, VariableCloner, VariableExportPolicy
from .weights import CloneWeightMatrix


@dataclass(frozen=True)
class LocalAreaBuildArtifacts:
    payload: H5Payload
    selection: CloneSelection
    reindexed: ReindexedEntities
    time_period: int | str


class LocalAreaDatasetBuilder:
    """Compose the pure local-H5 build steps for one output area."""

    def __init__(
        self,
        *,
        selector: AreaSelector | None = None,
        reindexer: EntityReindexer | None = None,
        variable_cloner: VariableCloner | None = None,
        us_augmentations: USAugmentationService | None = None,
        export_policy: VariableExportPolicy | None = None,
    ) -> None:
        self.selector = selector or AreaSelector()
        self.reindexer = reindexer or EntityReindexer()
        self.variable_cloner = variable_cloner or VariableCloner()
        self.us_augmentations = us_augmentations or USAugmentationService()
        self.export_policy = export_policy or VariableExportPolicy(
            include_input_variables=True
        )

    def build(
        self,
        *,
        weights: np.ndarray,
        geography,
        source: SourceDatasetSnapshot,
        filters: tuple[AreaFilter, ...] = (),
        takeup_filter: Sequence[str] | None = None,
    ) -> LocalAreaBuildArtifacts:
        weight_matrix = CloneWeightMatrix.from_vector(weights, source.n_households)
        selection = self.selector.select(
            weight_matrix,
            geography,
            filters=filters,
        )
        self._validate_selection(selection=selection, filters=filters)

        reindexed = self.reindexer.reindex(source, selection)
        time_period = source.time_period
        cloned = self.variable_cloner.clone(
            source,
            reindexed,
            self.export_policy,
        )

        data = {
            variable: dict(periods)
            for variable, periods in cloned.variables.items()
        }
        self._inject_entity_ids(
            data=data,
            time_period=time_period,
            reindexed=reindexed,
        )
        self._inject_household_weights(
            data=data,
            time_period=time_period,
            active_weights=selection.active_weights,
        )

        self.us_augmentations.apply_all(
            data,
            time_period=time_period,
            selection=selection,
            source=source,
            reindexed=reindexed,
            takeup_filter=takeup_filter,
        )

        return LocalAreaBuildArtifacts(
            payload=H5Payload(
                variables=data,
                attrs=cloned.attrs,
            ),
            selection=selection,
            reindexed=reindexed,
            time_period=time_period,
        )

    def build_payload(self, **kwargs) -> H5Payload:
        return self.build(**kwargs).payload

    def _inject_entity_ids(
        self,
        *,
        data: dict[str, dict[int | str, np.ndarray]],
        time_period: int | str,
        reindexed: ReindexedEntities,
    ) -> None:
        data["household_id"] = {time_period: reindexed.new_household_ids}
        data["person_id"] = {time_period: reindexed.new_person_ids}
        data["person_household_id"] = {
            time_period: reindexed.new_person_household_ids,
        }
        for entity_key, entity_ids in reindexed.new_entity_ids.items():
            data[f"{entity_key}_id"] = {time_period: entity_ids}
            data[f"person_{entity_key}_id"] = {
                time_period: reindexed.new_person_entity_ids[entity_key],
            }

    def _inject_household_weights(
        self,
        *,
        data: dict[str, dict[int | str, np.ndarray]],
        time_period: int | str,
        active_weights: np.ndarray,
    ) -> None:
        data["household_weight"] = {
            time_period: active_weights.astype(np.float32),
        }

    def _validate_selection(
        self,
        *,
        selection: CloneSelection,
        filters: tuple[AreaFilter, ...],
    ) -> None:
        if selection.is_empty:
            raise ValueError(
                "No active clones after filtering. "
                f"filters={self._format_filters(filters)}"
            )

        empty_count = int(np.sum(selection.active_block_geoids == ""))
        if empty_count > 0:
            raise ValueError(f"{empty_count} active clones have empty block GEOIDs")

    def _format_filters(self, filters: tuple[AreaFilter, ...]) -> str:
        if not filters:
            return "()"
        return ", ".join(
            f"{area_filter.geography_field} {area_filter.op} {area_filter.value}"
            for area_filter in filters
        )
