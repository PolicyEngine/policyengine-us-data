"""Unit tests for paper-l0/benchmarking/ipf_conversion.py.

These tests stay on the pure-Python side of the converter: target resolution,
closed-margin classification, exact complement derivation from authored parent
totals, and mixed-scope invariance guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "paper-l0" / "benchmarking"


@pytest.fixture(scope="module")
def ipf_conversion():
    benchmark_dir_str = str(BENCHMARK_DIR)
    if benchmark_dir_str not in sys.path:
        sys.path.insert(0, benchmark_dir_str)
    import ipf_conversion as module  # noqa: PLC0415

    return module


def _make_age_partition_targets():
    targets = pd.DataFrame(
        [
            {
                "target_id": 1,
                "stratum_id": 1,
                "variable": "person_count",
                "value": 210.0,
                "target_name": "pc_0_4_d601",
            },
            {
                "target_id": 2,
                "stratum_id": 2,
                "variable": "person_count",
                "value": 205.0,
                "target_name": "pc_5_9_d601",
            },
        ]
    )
    constraints = {
        1: [
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "601",
            },
            {"variable": "age", "operation": ">", "value": "-1"},
            {"variable": "age", "operation": "<", "value": "5"},
        ],
        2: [
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "601",
            },
            {"variable": "age", "operation": ">", "value": "4"},
            {"variable": "age", "operation": "<", "value": "10"},
        ],
    }
    unit_data = pd.DataFrame(
        {
            "unit_index": [0, 1, 2, 3],
            "household_id": [0, 0, 1, 1],
            "congressional_district_geoid": [601, 601, 601, 601],
            "age_bracket": ["0-4", "0-4", "5-9", "5-9"],
        }
    )
    return targets, constraints, unit_data


def _make_snap_subset_and_total_targets():
    targets = pd.DataFrame(
        [
            {
                "target_id": 10,
                "stratum_id": 10,
                "variable": "household_count",
                "value": 10.0,
                "target_name": "hh_total_d601",
            },
            {
                "target_id": 11,
                "stratum_id": 11,
                "variable": "household_count",
                "value": 8.0,
                "target_name": "hh_total_d602",
            },
            {
                "target_id": 12,
                "stratum_id": 12,
                "variable": "household_count",
                "value": 4.0,
                "target_name": "hh_snap_pos_d601",
            },
            {
                "target_id": 13,
                "stratum_id": 13,
                "variable": "household_count",
                "value": 3.0,
                "target_name": "hh_snap_pos_d602",
            },
        ]
    )
    constraints = {
        10: [
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "601",
            }
        ],
        11: [
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "602",
            }
        ],
        12: [
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "601",
            },
            {"variable": "snap", "operation": ">", "value": "0"},
        ],
        13: [
            {
                "variable": "congressional_district_geoid",
                "operation": "==",
                "value": "602",
            },
            {"variable": "snap", "operation": ">", "value": "0"},
        ],
    }
    unit_data = pd.DataFrame(
        {
            "unit_index": [0, 1, 2, 3],
            "household_id": [0, 1, 2, 3],
            "congressional_district_geoid": [601, 601, 602, 602],
            "snap_positive": [
                "positive",
                "non_positive",
                "positive",
                "non_positive",
            ],
        }
    )
    return targets, constraints, unit_data


def test_close_margins_full_partition_keeps_authored_cells(ipf_conversion):
    targets, constraints, unit_data = _make_age_partition_targets()
    resolved, unresolved = ipf_conversion.resolve_targets_for_testing(
        targets, constraints
    )
    assert unresolved == []

    closed_blocks, dropped = ipf_conversion.close_margins_for_testing(
        resolved=resolved,
        unit_data=unit_data,
    )
    assert dropped == []
    assert len(closed_blocks) == 1
    block = closed_blocks[0]
    assert block.closure_status == "full_partition"
    assert all(cell.is_authored for cell in block.cells)

    emitted = ipf_conversion.emit_target_rows(closed_blocks)
    assert len(emitted) == 2
    assert emitted["is_authored"].tolist() == [True, True]
    assert set(emitted["cell"]) == {
        "age_bracket=0-4|congressional_district_geoid=601",
        "age_bracket=5-9|congressional_district_geoid=601",
    }


def test_close_margins_binary_subset_derives_complement_from_parent_total(
    ipf_conversion,
):
    targets, constraints, unit_data = _make_snap_subset_and_total_targets()
    resolved, unresolved = ipf_conversion.resolve_targets_for_testing(
        targets, constraints
    )
    assert unresolved == []

    closed_blocks, dropped = ipf_conversion.close_margins_for_testing(
        resolved=resolved,
        unit_data=unit_data,
    )
    assert dropped == []
    assert len(closed_blocks) == 2
    assert {block.closure_status for block in closed_blocks} == {
        "full_partition",
        "binary_subset_with_parent_total",
    }

    emitted = ipf_conversion.emit_target_rows(closed_blocks)
    subset_rows = emitted.loc[
        emitted["closure_status"] == "binary_subset_with_parent_total"
    ].reset_index(drop=True)
    derived_rows = subset_rows.loc[~subset_rows["is_authored"]].sort_values(
        "cell"
    ).reset_index(drop=True)
    assert len(derived_rows) == 2
    assert set(derived_rows["cell"]) == {
        "congressional_district_geoid=601|snap_positive=non_positive",
        "congressional_district_geoid=602|snap_positive=non_positive",
    }
    assert derived_rows["target_value"].tolist() == [6.0, 5.0]
    assert (derived_rows["derivation_reason"] == "authored_parent_total").all()


def test_close_margins_missing_parent_total_drops_open_subset(ipf_conversion):
    targets, constraints, unit_data = _make_snap_subset_and_total_targets()
    open_subset_targets = targets.loc[targets["target_id"].isin([12, 13])].reset_index(
        drop=True
    )
    open_subset_constraints = {12: constraints[12], 13: constraints[13]}

    resolved, unresolved = ipf_conversion.resolve_targets_for_testing(
        open_subset_targets, open_subset_constraints
    )
    assert unresolved == []

    closed_blocks, dropped = ipf_conversion.close_margins_for_testing(
        resolved=resolved,
        unit_data=unit_data,
    )
    assert closed_blocks == []
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "missing_parent_total"


def test_close_margins_ambiguous_parent_total_drops_subset(ipf_conversion):
    targets, constraints, unit_data = _make_snap_subset_and_total_targets()
    ambiguous_targets = pd.concat(
        [
            targets.loc[targets["target_id"].isin([10, 11, 12, 13])],
            pd.DataFrame(
                [
                    {
                        "target_id": 14,
                        "stratum_id": 14,
                        "variable": "household_count",
                        "value": 9.0,
                        "target_name": "hh_total_d601_duplicate",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ambiguous_constraints = dict(constraints)
    ambiguous_constraints[14] = [
        {
            "variable": "congressional_district_geoid",
            "operation": "==",
            "value": "601",
        }
    ]

    resolved, unresolved = ipf_conversion.resolve_targets_for_testing(
        ambiguous_targets, ambiguous_constraints
    )
    assert unresolved == []

    closed_blocks, dropped = ipf_conversion.close_margins_for_testing(
        resolved=resolved,
        unit_data=unit_data,
    )
    assert len(closed_blocks) == 1
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "ambiguous_parent_total"


def test_margin_consistency_uses_closed_subset_totals(ipf_conversion):
    targets, constraints, unit_data = _make_snap_subset_and_total_targets()
    resolved, _ = ipf_conversion.resolve_targets_for_testing(targets, constraints)
    closed_blocks, dropped = ipf_conversion.close_margins_for_testing(
        resolved=resolved,
        unit_data=unit_data,
    )
    assert dropped == []
    issues = ipf_conversion.check_margin_consistency(closed_blocks)
    assert issues == []


def test_household_invariance_guard_drops_person_varying_household_margin(
    ipf_conversion,
):
    targets = pd.DataFrame(
        [
            {
                "target_id": 30,
                "stratum_id": 30,
                "variable": "household_count",
                "value": 1.0,
                "target_name": "hh_age_u5",
            }
        ]
    )
    constraints = {
        30: [
            {"variable": "age", "operation": ">", "value": "-1"},
            {"variable": "age", "operation": "<", "value": "5"},
        ]
    }
    resolved, unresolved = ipf_conversion.resolve_targets_for_testing(
        targets, constraints
    )
    assert unresolved == []
    blocks = ipf_conversion.assemble_margins_for_testing(resolved)
    unit_data = pd.DataFrame(
        {
            "household_id": [0, 0, 1, 1],
            "age_bracket": ["0-4", "5-9", "0-4", "5-9"],
        }
    )

    valid_blocks, dropped = ipf_conversion._validate_household_margin_invariance(
        unit_data=unit_data,
        blocks=blocks,
    )
    assert valid_blocks == []
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "non_invariant_household_constraint_variable"
    assert dropped[0]["columns"] == ["age_bracket"]


def test_emit_target_rows_rejects_unclosed_margin_blocks(ipf_conversion):
    targets, constraints, _ = _make_age_partition_targets()
    resolved, unresolved = ipf_conversion.resolve_targets_for_testing(
        targets, constraints
    )
    assert unresolved == []
    blocks = ipf_conversion.assemble_margins_for_testing(resolved)

    with pytest.raises(TypeError, match="expects closed margin blocks"):
        ipf_conversion.emit_target_rows(blocks)
