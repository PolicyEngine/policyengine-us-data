from pathlib import Path
from types import SimpleNamespace

import numpy as np

from policyengine_us_data.build_outputs.builder import (
    LocalAreaBuildResult,
    LocalAreaDatasetBuilder,
    PayloadPostProcessorRun,
)
from policyengine_us_data.build_outputs.payload import H5Payload
from policyengine_us_data.build_outputs.reindexing import ReindexedEntities
from policyengine_us_data.build_outputs.requests import AreaBuildRequest, AreaFilter
from policyengine_us_data.build_outputs.selection import CloneSelection
from policyengine_us_data.build_outputs.source_dataset import (
    EntityGraph,
    SourceDatasetSnapshot,
)
from policyengine_us_data.build_outputs.variables import VariableClonePayload
from policyengine_us_data.build_outputs.weights import CloneWeightMatrix
from tests.support.build_outputs.source_dataset import make_entity_graph_arrays


def _selection() -> CloneSelection:
    return CloneSelection(
        clone_indices=np.array([0]),
        source_household_indices=np.array([1]),
        weights=np.array([2.0]),
        block_geoids=np.array(["block-1"]),
        congressional_district_geoids=np.array(["3701"]),
        filters=(),
        n_source_households=2,
        n_total_clones=1,
    )


def _reindexed() -> ReindexedEntities:
    return ReindexedEntities(
        household_ids=np.array([0]),
        person_ids=np.array([0]),
        person_household_ids=np.array([0]),
        subentity_ids={"tax_unit": np.array([0])},
        person_subentity_ids={"tax_unit": np.array([0])},
        household_source_indices=np.array([1]),
        person_source_indices=np.array([2]),
        subentity_source_indices={"tax_unit": np.array([1])},
        persons_per_household_clone=np.array([1]),
        subentities_per_household_clone={"tax_unit": np.array([1])},
        person_household_clone_indices=np.array([0]),
        subentity_household_clone_indices={"tax_unit": np.array([0])},
    )


def _source() -> SourceDatasetSnapshot:
    return SourceDatasetSnapshot(
        dataset_path=Path("source.h5"),
        time_period=2024,
        entity_graph=EntityGraph(**make_entity_graph_arrays()),
        input_variables=frozenset(),
        variable_provider=SimpleNamespace(),
    )


class _Selector:
    def __init__(self, calls, selection):
        self.calls = calls
        self.selection = selection

    def select(self, **kwargs):
        self.calls.append(("select", kwargs))
        return self.selection


class _Reindexer:
    def __init__(self, calls, reindexed):
        self.calls = calls
        self.reindexed = reindexed

    def reindex(self, **kwargs):
        self.calls.append(("reindex", kwargs))
        return self.reindexed


class _VariableCloner:
    def __init__(self, calls):
        self.calls = calls

    def clone(self, **kwargs):
        self.calls.append(("clone", kwargs))
        return VariableClonePayload(
            data={"rent": {2024: np.array([200])}},
            values_saved=1,
        )


class _AugmentationService:
    def __init__(self, calls):
        self.calls = calls

    def apply(self, **kwargs):
        self.calls.append(("augment", kwargs))
        payload = kwargs["payload"]
        return SimpleNamespace(
            payload=H5Payload(
                data={
                    **payload.data,
                    "household_id": {2024: np.array([0])},
                },
                time_period=payload.time_period,
                entity_lengths=payload.entity_lengths,
            ),
        )


def test_local_area_dataset_builder_orchestrates_one_area_build_in_memory():
    calls = []
    source = _source()
    selection = _selection()
    reindexed = _reindexed()
    weights = CloneWeightMatrix.from_vector(np.array([1.0, 2.0]), n_records=2)
    request = AreaBuildRequest(
        area_type="district",
        area_id="NC-01",
        display_name="NC-01",
        output_relative_path="districts/NC-01.h5",
        filters=(AreaFilter(geography_field="cd_geoid", op="in", value=("3701",)),),
    )

    result = LocalAreaDatasetBuilder(
        selector=_Selector(calls, selection),
        reindexer=_Reindexer(calls, reindexed),
        variable_cloner=_VariableCloner(calls),
        postprocessors=(_AugmentationService(calls),),
    ).build(
        source=source,
        simulation=SimpleNamespace(),
        weights=weights,
        geography=SimpleNamespace(),
        request=request,
        takeup_filter=("takes_up_snap",),
    )

    assert [name for name, _ in calls] == ["select", "reindex", "clone", "augment"]
    assert calls[0][1]["filters"] == request.filters
    assert calls[1][1] == {"source": source, "selection": selection}
    assert calls[2][1] == {
        "source": source,
        "selection": selection,
        "reindexed": reindexed,
    }
    assert isinstance(calls[3][1]["payload"], H5Payload)
    assert calls[3][1]["context"].takeup_filter == ("takes_up_snap",)
    np.testing.assert_array_equal(result.data["household_id"][2024], np.array([0]))
    assert result.payload.entity_lengths == {"household": 1, "person": 1, "tax_unit": 1}
    assert result.variables_saved == 1
    assert result.summary["area_id"] == "NC-01"
    assert result.summary["active_clones"] == 1
    assert result.summary["tax_units"] == 1
    assert len(result.postprocessor_runs) == 1
    assert (
        result.postprocessor_result(_AugmentationService)
        is result.postprocessor_runs[0].result
    )
    assert result.postprocessor_results(_AugmentationService) == (
        result.postprocessor_runs[0].result,
    )


def test_local_area_build_result_retains_duplicate_postprocessor_runs():
    class _PostProcessor:
        pass

    first = SimpleNamespace(payload=_payload("first"), label="first")
    second = SimpleNamespace(payload=_payload("second"), label="second")

    result = LocalAreaBuildResult(
        payload=_payload("final"),
        selection=_selection(),
        reindexed=_reindexed(),
        variables_saved=0,
        summary={},
        postprocessor_runs=(
            PayloadPostProcessorRun(
                name="_PostProcessor",
                postprocessor_type=_PostProcessor,
                result=first,
            ),
            PayloadPostProcessorRun(
                name="_PostProcessor",
                postprocessor_type=_PostProcessor,
                result=second,
            ),
        ),
    )

    assert result.postprocessor_result(_PostProcessor) is first
    assert result.postprocessor_results(_PostProcessor) == (first, second)


def _payload(label: str) -> H5Payload:
    return H5Payload(
        data={label: {2024: np.array([1])}},
        time_period=2024,
        entity_lengths={"household": 1},
    )
