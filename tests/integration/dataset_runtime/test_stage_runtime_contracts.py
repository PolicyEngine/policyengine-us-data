"""Fixture-scale runtime contracts for dataset build phases 1-5."""

from __future__ import annotations

import pytest

from tests.integration.dataset_runtime.support import (
    assert_entity_graph_is_consistent,
    assert_file_pair_equal,
    assert_has_period_arrays,
    assert_runtime_core_variables,
    assert_runtime_head_flags,
    assert_runtime_matches_h5,
    build_dataset_runtime_workspace,
)

pytestmark = pytest.mark.integration


def test_phase_1_source_artifacts_are_policyengine_runtime_compatible(tmp_path):
    workspace = build_dataset_runtime_workspace(tmp_path)

    assert workspace.uprating_factors.read_text(encoding="utf-8").splitlines() == [
        "variable,year,factor",
        "employment_income,2023,1.0",
    ]

    for artifact in (workspace.acs, workspace.irs_puf):
        assert artifact.exists()
        assert_has_period_arrays(
            artifact,
            (
                "household_id",
                "person_id",
                "person_household_id",
                "household_weight",
                "employment_income",
            ),
        )
        assert_entity_graph_is_consistent(artifact)
        assert_runtime_core_variables(artifact)


def test_phase_2_cps_puf_artifacts_expose_rent_prerequisite_runtime_fields(
    tmp_path,
):
    workspace = build_dataset_runtime_workspace(tmp_path)

    for artifact in (workspace.cps, workspace.puf):
        assert_entity_graph_is_consistent(artifact)
        assert_runtime_core_variables(artifact)

    assert_runtime_head_flags(workspace.cps)


def test_phase_3_extended_cps_preserves_entity_spine_and_runtime_values(
    tmp_path,
):
    workspace = build_dataset_runtime_workspace(tmp_path)

    assert_entity_graph_is_consistent(workspace.extended_cps)
    assert_runtime_core_variables(workspace.extended_cps)
    assert_runtime_matches_h5(
        workspace.extended_cps,
        "state_fips",
        map_to="household",
    )


def test_phase_4_parallel_outputs_keep_runtime_contracts_and_shared_spine(
    tmp_path,
):
    workspace = build_dataset_runtime_workspace(tmp_path)

    for artifact in (workspace.enhanced_cps, workspace.stratified_cps):
        assert_entity_graph_is_consistent(artifact)
        assert_runtime_core_variables(artifact)
        assert_runtime_head_flags(artifact)
        assert_runtime_matches_h5(
            artifact,
            "state_fips",
            map_to="household",
        )

    assert workspace.calibration_log.read_text(encoding="utf-8").splitlines() == [
        "stage,status",
        "fixture-scale,complete",
    ]


def test_phase_5_publication_artifacts_keep_runtime_and_handoff_contracts(
    tmp_path,
):
    workspace = build_dataset_runtime_workspace(tmp_path)

    assert_file_pair_equal(
        workspace.source_imputed_cps,
        workspace.source_imputed_alias,
    )

    for artifact in (
        workspace.source_imputed_cps,
        workspace.source_imputed_alias,
        workspace.small_enhanced_cps,
        workspace.sparse_enhanced_cps,
    ):
        assert_entity_graph_is_consistent(artifact)
        assert_runtime_core_variables(artifact)

    assert_runtime_head_flags(workspace.source_imputed_alias)
