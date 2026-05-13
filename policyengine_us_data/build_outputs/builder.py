"""One-area local H5 build orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TypeAlias

import numpy as np

from policyengine_us_data.pipeline_metadata import pipeline_node

from .payload import H5Payload, PayloadBuildContext
from .reindexing import EntityReindexer, ReindexedEntities
from .requests import AreaBuildRequest
from .selection import AreaSelector, CloneSelection
from .source_dataset import SourceDatasetSnapshot
from .variables import VariableCloner
from .weights import CloneWeightMatrix

__all__ = [
    "LocalAreaBuildResult",
    "LocalAreaDatasetBuilder",
    "PayloadPostProcessor",
    "PayloadPostProcessorRun",
    "PayloadPostProcessorResult",
]


class PayloadPostProcessorResult(Protocol):
    """Result contract for a payload postprocessor."""

    payload: H5Payload


PostProcessorReturn: TypeAlias = H5Payload | PayloadPostProcessorResult


class PayloadPostProcessor(Protocol):
    """Country- or product-specific processor for an H5 payload."""

    def apply(
        self,
        *,
        payload: H5Payload,
        context: PayloadBuildContext,
    ) -> PostProcessorReturn:
        """Return a processed `H5Payload` or structured result with `.payload`."""


@dataclass(frozen=True)
class PayloadPostProcessorRun:
    """Result metadata for one payload postprocessor invocation."""

    name: str
    postprocessor_type: type
    result: PostProcessorReturn


@pipeline_node(
    id="local_h5_build_result",
    label="LocalAreaBuildResult",
    node_type="library",
    description="In-memory local H5 payload and diagnostics for one area.",
    source_file="policyengine_us_data/build_outputs/builder.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_builder.py"],
)
@dataclass(frozen=True)
class LocalAreaBuildResult:
    """In-memory output from building one local H5 area."""

    payload: H5Payload
    selection: CloneSelection
    reindexed: ReindexedEntities
    variables_saved: int
    summary: Mapping[str, int | float | str]
    postprocessor_runs: tuple[PayloadPostProcessorRun, ...] = ()

    @property
    def data(self) -> Mapping[str, Mapping[Any, np.ndarray]]:
        """Payload data retained for transitional callers."""

        return self.payload.data

    @property
    def time_period(self) -> int:
        """Payload time period retained for transitional callers."""

        return self.payload.time_period

    def postprocessor_result(self, postprocessor: type | str) -> Any | None:
        """Return the result for one configured postprocessor."""

        key = (
            postprocessor if isinstance(postprocessor, str) else postprocessor.__name__
        )
        for run in self.postprocessor_runs:
            if run.name == key:
                return run.result
            if not isinstance(postprocessor, str) and issubclass(
                run.postprocessor_type,
                postprocessor,
            ):
                return run.result
        return None

    def postprocessor_results(self, postprocessor: type | str) -> tuple[Any, ...]:
        """Return every result for a configured postprocessor type or name."""

        key = (
            postprocessor if isinstance(postprocessor, str) else postprocessor.__name__
        )
        return tuple(
            run.result
            for run in self.postprocessor_runs
            if run.name == key
            or (
                not isinstance(postprocessor, str)
                and issubclass(run.postprocessor_type, postprocessor)
            )
        )


@pipeline_node(
    id="local_h5_dataset_builder",
    label="LocalAreaDatasetBuilder",
    node_type="library",
    description="Build the in-memory payload for one local-area or national H5 output.",
    source_file="policyengine_us_data/build_outputs/builder.py",
    status="current",
    stability="moving",
    pathways=["local_h5"],
    validation_commands=["uv run pytest tests/unit/build_outputs/test_builder.py"],
)
@dataclass(frozen=True)
class LocalAreaDatasetBuilder:
    """Coordinate clone selection, reindexing, variable cloning, and postprocessing."""

    selector: AreaSelector = field(default_factory=AreaSelector)
    reindexer: EntityReindexer = field(default_factory=EntityReindexer)
    variable_cloner: VariableCloner = field(default_factory=VariableCloner)
    postprocessors: tuple[PayloadPostProcessor, ...] = ()

    def build(
        self,
        *,
        source: SourceDatasetSnapshot,
        simulation: Any,
        weights: CloneWeightMatrix,
        geography: Any,
        request: AreaBuildRequest,
        takeup_filter: tuple[str, ...] | None = None,
    ) -> LocalAreaBuildResult:
        """Build one local H5 payload without writing it to disk."""

        selection = self.selector.select(
            weights=weights,
            geography=geography,
            filters=request.filters,
        )
        reindexed = self.reindexer.reindex(source=source, selection=selection)
        payload = self.variable_cloner.clone(
            source=source,
            selection=selection,
            reindexed=reindexed,
        )
        h5_payload = H5Payload(
            data=payload.data,
            time_period=int(source.time_period),
            entity_lengths=_entity_lengths(reindexed),
        )
        context = PayloadBuildContext(
            source=source,
            simulation=simulation,
            selection=selection,
            reindexed=reindexed,
            geography=geography,
            time_period=int(source.time_period),
            takeup_filter=takeup_filter,
        )
        postprocessor_runs: list[PayloadPostProcessorRun] = []
        for postprocessor in self.postprocessors:
            result = postprocessor.apply(payload=h5_payload, context=context)
            postprocessor_runs.append(
                PayloadPostProcessorRun(
                    name=type(postprocessor).__name__,
                    postprocessor_type=type(postprocessor),
                    result=result,
                )
            )
            h5_payload = _payload_from_postprocessor_result(result)

        return LocalAreaBuildResult(
            payload=h5_payload,
            selection=selection,
            reindexed=reindexed,
            variables_saved=payload.values_saved,
            summary=_build_summary(
                request=request,
                selection=selection,
                reindexed=reindexed,
                variables_saved=payload.values_saved,
            ),
            postprocessor_runs=tuple(postprocessor_runs),
        )


def _build_summary(
    *,
    request: AreaBuildRequest,
    selection: CloneSelection,
    reindexed: ReindexedEntities,
    variables_saved: int,
) -> dict[str, int | float | str]:
    summary: dict[str, int | float | str] = {
        "area_type": request.area_type,
        "area_id": request.area_id,
        "display_name": request.display_name,
        "active_clones": selection.n_selected_clones,
        "total_weight": float(np.sum(selection.weights)),
        "persons": int(len(reindexed.person_ids)),
        "variables_saved": int(variables_saved),
    }
    for entity_key, entity_source_indices in reindexed.subentity_source_indices.items():
        summary[f"{entity_key}s"] = int(len(entity_source_indices))
    return summary


def _entity_lengths(reindexed: ReindexedEntities) -> dict[str, int]:
    lengths = {
        "household": int(len(reindexed.household_ids)),
        "person": int(len(reindexed.person_ids)),
    }
    for entity_key, entity_ids in reindexed.subentity_ids.items():
        lengths[entity_key] = int(len(entity_ids))
    return lengths


def _payload_from_postprocessor_result(result: PostProcessorReturn) -> H5Payload:
    if isinstance(result, H5Payload):
        return result
    payload = getattr(result, "payload", None)
    if isinstance(payload, H5Payload):
        return payload
    raise TypeError(
        "Payload postprocessors must return H5Payload or an object exposing "
        "an H5Payload `.payload` attribute"
    )
