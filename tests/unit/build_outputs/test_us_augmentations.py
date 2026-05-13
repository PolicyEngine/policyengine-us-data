from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policyengine_us_data.build_outputs.payload import H5Payload, PayloadBuildContext
from policyengine_us_data.build_outputs.reindexing import EntityReindexer
from policyengine_us_data.build_outputs.selection import CloneSelection
from policyengine_us_data.build_outputs.source_dataset import (
    EntityGraph,
    SourceDatasetSnapshot,
)
from policyengine_us_data.build_outputs.us_augmentations import (
    USEntityPostProcessor,
    USGeographyPostProcessor,
    USTakeupPostProcessor,
    _build_reported_takeup_anchors,
    default_us_postprocessors,
)
from tests.support.build_outputs.source_dataset import make_entity_graph_arrays


class _Calculation:
    def __init__(self, values):
        self.values = np.asarray(values)


class _Simulation:
    def calculate(self, variable, period=None, map_to=None):
        values = {
            ("tax_unit_child_dependents", "tax_unit"): np.array([1, 2]),
            ("employment_income", "person"): np.array([10, 20, 30]),
            ("age_head", "tax_unit"): np.array([40, 50]),
        }[(variable, map_to)]
        return _Calculation(values)


def _source_snapshot() -> SourceDatasetSnapshot:
    return SourceDatasetSnapshot(
        dataset_path=Path("source.h5"),
        time_period=2024,
        entity_graph=EntityGraph(**make_entity_graph_arrays()),
        input_variables=frozenset(),
        variable_provider=SimpleNamespace(),
    )


def _selection() -> CloneSelection:
    return CloneSelection(
        clone_indices=np.array([0, 1]),
        source_household_indices=np.array([1, 0]),
        weights=np.array([2.0, 3.0]),
        block_geoids=np.array(["block-la", "block-nc"]),
        congressional_district_geoids=np.array(["6037", "3701"]),
        filters=(),
        n_source_households=2,
        n_total_clones=2,
    )


def _context(*, takeup_filter=("takes_up_snap_if_eligible",)) -> PayloadBuildContext:
    source = _source_snapshot()
    selection = _selection()
    reindexed = EntityReindexer().reindex(source=source, selection=selection)
    return PayloadBuildContext(
        source=source,
        simulation=_Simulation(),
        selection=selection,
        reindexed=reindexed,
        geography=SimpleNamespace(),
        time_period=2024,
        takeup_filter=takeup_filter,
    )


def _base_payload(data=None) -> H5Payload:
    return H5Payload(
        data=data or {},
        time_period=2024,
        entity_lengths={
            "household": 2,
            "person": 3,
            "tax_unit": 2,
            "spm_unit": 2,
            "family": 2,
            "marital_unit": 2,
        },
    )


def _geography_deriver(blocks):
    assert tuple(blocks) == ("block-la", "block-nc")
    return {
        "block_geoid": np.asarray(blocks),
        "state_fips": np.array([6, 37]),
        "county_index": np.array([100, 200]),
        "county_fips": np.array(["06037", "37183"]),
        "tract_geoid": np.array(["06037000100", "37183000100"]),
    }


def _entity_payload(context=None) -> H5Payload:
    context = context or _context()
    return (
        USEntityPostProcessor()
        .apply(
            payload=_base_payload(),
            context=context,
        )
        .payload
    )


def _geography_payload(context=None) -> H5Payload:
    context = context or _context()
    payload = _entity_payload(context)
    return (
        USGeographyPostProcessor(geography_deriver=_geography_deriver)
        .apply(
            payload=payload,
            context=context,
        )
        .payload
    )


def test_default_us_postprocessors_are_in_runtime_order():
    assert tuple(type(processor) for processor in default_us_postprocessors()) == (
        USEntityPostProcessor,
        USGeographyPostProcessor,
        USTakeupPostProcessor,
    )


def test_build_reported_takeup_anchors_skips_missing_period():
    data = {
        "person_tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "reported_has_subsidized_marketplace_health_coverage_at_interview": {
            2023: np.array([True, False])
        },
        "has_medicaid_health_coverage_at_interview": {2023: np.array([True, False])},
    }

    assert _build_reported_takeup_anchors(data, 2024) == {}


def test_build_reported_takeup_anchors_uses_present_period():
    data = {
        "person_tax_unit_id": {2024: np.array([1, 1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "reported_has_subsidized_marketplace_health_coverage_at_interview": {
            2024: np.array([True, False, False])
        },
        "has_medicaid_health_coverage_at_interview": {
            2024: np.array([False, True, False])
        },
    }

    anchors = _build_reported_takeup_anchors(data, 2024)

    np.testing.assert_array_equal(
        anchors["takes_up_aca_if_eligible"],
        np.array([True, False]),
    )
    np.testing.assert_array_equal(
        anchors["takes_up_medicaid_if_eligible"],
        np.array([False, True, False]),
    )


def test_build_reported_takeup_anchors_uses_subsidized_marketplace_only():
    data = {
        "person_tax_unit_id": {2024: np.array([1, 1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "has_marketplace_health_coverage_at_interview": {
            2024: np.array([True, False, True])
        },
        "reported_has_subsidized_marketplace_health_coverage_at_interview": {
            2024: np.array([False, False, True])
        },
    }

    anchors = _build_reported_takeup_anchors(data, 2024)

    np.testing.assert_array_equal(
        anchors["takes_up_aca_if_eligible"],
        np.array([False, True]),
    )


def test_us_entity_postprocessor_applies_entity_ids_and_weights():
    context = _context()

    result = USEntityPostProcessor().apply(
        payload=_base_payload({"rent": {2024: np.array([200, 100])}}),
        context=context,
    )

    np.testing.assert_array_equal(result.data["rent"][2024], np.array([200, 100]))
    np.testing.assert_array_equal(result.data["household_id"][2024], np.array([0, 1]))
    np.testing.assert_array_equal(
        result.data["person_household_id"][2024],
        np.array([0, 1, 1]),
    )
    np.testing.assert_array_equal(
        result.data["tax_unit_id"][2024],
        np.array([0, 1]),
    )
    np.testing.assert_array_equal(
        result.data["person_tax_unit_id"][2024],
        np.array([0, 1, 1]),
    )
    np.testing.assert_array_equal(
        result.data["household_weight"][2024],
        np.array([2.0, 3.0], dtype=np.float32),
    )


def test_us_geography_postprocessor_applies_geography_and_la_zip_overrides():
    result = USGeographyPostProcessor(geography_deriver=_geography_deriver).apply(
        payload=_base_payload({"rent": {2024: np.array([200, 100])}}),
        context=_context(takeup_filter=None),
    )

    np.testing.assert_array_equal(result.data["rent"][2024], np.array([200, 100]))
    np.testing.assert_array_equal(
        result.clone_geography["state_fips"],
        np.array([6, 37]),
    )
    np.testing.assert_array_equal(
        result.data["state_fips"][2024],
        np.array([6, 37], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.data["county_fips"][2024],
        np.array([6037, 37183], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.data["zip_code"][2024],
        np.array([b"90001", b"00000"]),
    )
    np.testing.assert_array_equal(
        result.data["congressional_district_geoid"][2024],
        np.array([6037, 3701], dtype=np.int32),
    )


def test_us_takeup_postprocessor_passes_takeup_contract_inputs():
    seen = {}

    def fake_takeup(**kwargs):
        seen.update(kwargs)
        return {"takes_up_snap_if_eligible": np.array([True, False])}

    def fake_sum_person_values(person_values, person_tax_unit_ids, tax_unit_ids):
        np.testing.assert_array_equal(person_values, np.array([30, 10, 20]))
        np.testing.assert_array_equal(person_tax_unit_ids, np.array([0, 1, 1]))
        np.testing.assert_array_equal(tax_unit_ids, np.array([0, 1]))
        return np.array([30, 30])

    result = USTakeupPostProcessor(
        takeup_applier=fake_takeup,
        sum_person_values_to_tax_units=fake_sum_person_values,
    ).apply(
        payload=_geography_payload(),
        context=_context(),
    )

    assert result.takeup_variables == ("takes_up_snap_if_eligible",)
    np.testing.assert_array_equal(
        result.data["takes_up_snap_if_eligible"][2024],
        np.array([True, False]),
    )
    np.testing.assert_array_equal(seen["hh_blocks"], np.array(["block-la", "block-nc"]))
    np.testing.assert_array_equal(
        seen["hh_state_fips"],
        np.array([6, 37], dtype=np.int32),
    )
    np.testing.assert_array_equal(seen["hh_ids"], np.array([20, 10]))
    np.testing.assert_array_equal(seen["hh_clone_indices"], np.array([0, 1]))
    np.testing.assert_array_equal(
        seen["entity_hh_indices"]["person"], np.array([0, 1, 1])
    )
    assert seen["entity_counts"] == {"person": 3, "tax_unit": 2, "spm_unit": 2}
    assert seen["takeup_filter"] == ["takes_up_snap_if_eligible"]
    np.testing.assert_array_equal(
        seen["voluntary_filing_inputs"]["tax_unit_child_dependents"],
        np.array([2, 1]),
    )
    np.testing.assert_array_equal(
        seen["voluntary_filing_inputs"]["tax_unit_wage_income"],
        np.array([30, 30]),
    )
    np.testing.assert_array_equal(
        seen["voluntary_filing_inputs"]["age_head"],
        np.array([50, 40]),
    )


def test_us_takeup_postprocessor_rejects_wrong_length_takeup_results():
    service = USTakeupPostProcessor(
        takeup_applier=lambda **kwargs: {"takes_up_snap_if_eligible": np.array([True])},
    )

    with pytest.raises(
        ValueError,
        match="takes_up_snap_if_eligible\\[2024\\] length 1 "
        "does not match spm_unit length 2",
    ):
        service.apply(
            payload=_geography_payload(),
            context=_context(),
        )


def test_us_takeup_postprocessor_rejects_missing_required_subentities():
    context = _context()
    reindexed = context.reindexed
    regional_only_reindexed = replace(
        reindexed,
        subentity_ids={"tax_unit": reindexed.subentity_ids["tax_unit"]},
        person_subentity_ids={"tax_unit": reindexed.person_subentity_ids["tax_unit"]},
        subentity_source_indices={
            "tax_unit": reindexed.subentity_source_indices["tax_unit"]
        },
        subentities_per_household_clone={
            "tax_unit": reindexed.subentities_per_household_clone["tax_unit"]
        },
        subentity_household_clone_indices={
            "tax_unit": reindexed.subentity_household_clone_indices["tax_unit"]
        },
    )
    context = replace(context, reindexed=regional_only_reindexed)

    with pytest.raises(
        ValueError,
        match="US take-up requires reindexed subentities: spm_unit",
    ):
        USTakeupPostProcessor(takeup_applier=lambda **kwargs: {}).apply(
            payload=_base_payload(),
            context=context,
        )


def test_us_takeup_postprocessor_requires_geography_first():
    with pytest.raises(
        ValueError,
        match="US take-up requires state_fips from USGeographyPostProcessor",
    ):
        USTakeupPostProcessor(takeup_applier=lambda **kwargs: {}).apply(
            payload=_entity_payload(),
            context=_context(),
        )


def test_us_takeup_postprocessor_rejects_unknown_takeup_results():
    service = USTakeupPostProcessor(
        takeup_applier=lambda **kwargs: {
            "takes_up_new_program": np.array([True, False])
        },
    )

    with pytest.raises(
        ValueError,
        match="Unknown take-up variable\\(s\\) returned by takeup applier: "
        "takes_up_new_program",
    ):
        service.apply(
            payload=_geography_payload(),
            context=_context(),
        )
