import numpy as np
import pandas as pd

from policyengine_us_data.calibration.calibration_utils import apply_op
from policyengine_us_data.calibration.unified_matrix_builder import (
    UnifiedMatrixBuilder,
    _assemble_clone_values_standalone,
    _calculate_target_values_standalone,
)


def _state_values_for_string_constraint():
    return {
        1: {
            "hh": {},
            "person": {
                "ssn_card_type": np.array(
                    [b"CITIZEN", b"", b""],
                    dtype="S24",
                ),
            },
            "reform_hh": {},
        },
        2: {
            "hh": {},
            "person": {
                "ssn_card_type": np.array(
                    [
                        b"",
                        b"NON_CITIZEN_VALID_EAD",
                        b"OTHER_NON_CITIZEN",
                    ],
                    dtype="S24",
                ),
            },
            "reform_hh": {},
        },
    }


def test_assemble_clone_values_standalone_preserves_string_constraints():
    _, person_vars, _ = _assemble_clone_values_standalone(
        state_values=_state_values_for_string_constraint(),
        clone_states=np.array([1, 2, 2]),
        person_hh_indices=np.array([0, 1, 2]),
        target_vars=set(),
        constraint_vars={"ssn_card_type"},
    )

    assert person_vars["ssn_card_type"].tolist() == [
        b"CITIZEN",
        b"NON_CITIZEN_VALID_EAD",
        b"OTHER_NON_CITIZEN",
    ]


def test_apply_op_matches_fixed_width_byte_string_constraints():
    values = np.array([b"NONE", b"CITIZEN", b"NONE"], dtype="S9")

    np.testing.assert_array_equal(
        apply_op(values, "==", "NONE"),
        np.array([True, False, True]),
    )


def test_builder_assemble_clone_values_preserves_string_constraints():
    builder = UnifiedMatrixBuilder.__new__(UnifiedMatrixBuilder)

    _, person_vars, _ = builder._assemble_clone_values(
        state_values=_state_values_for_string_constraint(),
        clone_states=np.array([1, 2, 2]),
        person_hh_indices=np.array([0, 1, 2]),
        target_vars=set(),
        constraint_vars={"ssn_card_type"},
    )

    assert person_vars["ssn_card_type"].tolist() == [
        b"CITIZEN",
        b"NON_CITIZEN_VALID_EAD",
        b"OTHER_NON_CITIZEN",
    ]


def test_person_amount_targets_filter_before_household_sum():
    entity_rel = pd.DataFrame(
        {
            "person_id": np.array([0, 1]),
            "household_id": np.array([100, 100]),
            "tax_unit_id": np.array([10, 11]),
            "spm_unit_id": np.array([20, 20]),
        }
    )

    values = _calculate_target_values_standalone(
        target_variable="total_self_employment_income",
        non_geo_constraints=[
            {
                "variable": "total_self_employment_income",
                "operation": ">",
                "value": "0",
            },
            {
                "variable": "tax_unit_is_filer",
                "operation": "==",
                "value": "1",
            },
        ],
        n_households=1,
        hh_vars={
            "total_self_employment_income": np.array([15000], dtype=np.float32),
        },
        reform_hh_vars={},
        target_entity_vars={
            "total_self_employment_income": np.array(
                [10000, 5000],
                dtype=np.float32,
            ),
        },
        person_vars={
            "total_self_employment_income": np.array(
                [10000, 5000],
                dtype=np.float32,
            ),
            "tax_unit_is_filer": np.array([1, 0], dtype=np.float32),
        },
        entity_rel=entity_rel,
        household_ids=np.array([100]),
        variable_entity_map={"total_self_employment_income": "person"},
        entity_hh_idx_map={"person": np.array([0, 0])},
        person_to_entity_idx_map={"person": np.array([0, 1])},
    )

    np.testing.assert_array_equal(values, np.array([10000], dtype=np.float32))


def test_tax_unit_amount_targets_count_each_unit_once():
    entity_rel = pd.DataFrame(
        {
            "person_id": np.array([0, 1]),
            "household_id": np.array([100, 100]),
            "tax_unit_id": np.array([10, 10]),
            "spm_unit_id": np.array([20, 20]),
        }
    )

    values = _calculate_target_values_standalone(
        target_variable="aca_ptc",
        non_geo_constraints=[
            {
                "variable": "aca_ptc",
                "operation": ">",
                "value": "0",
            }
        ],
        n_households=1,
        hh_vars={"aca_ptc": np.array([1000], dtype=np.float32)},
        reform_hh_vars={},
        target_entity_vars={"aca_ptc": np.array([1000], dtype=np.float32)},
        person_vars={"aca_ptc": np.array([1000, 1000], dtype=np.float32)},
        entity_rel=entity_rel,
        household_ids=np.array([100]),
        variable_entity_map={"aca_ptc": "tax_unit"},
        entity_hh_idx_map={"tax_unit": np.array([0])},
        person_to_entity_idx_map={"tax_unit": np.array([0, 0])},
    )

    np.testing.assert_array_equal(values, np.array([1000], dtype=np.float32))


def test_household_amount_targets_keep_household_any_semantics():
    entity_rel = pd.DataFrame(
        {
            "person_id": np.array([0, 1]),
            "household_id": np.array([100, 100]),
            "tax_unit_id": np.array([10, 10]),
            "spm_unit_id": np.array([20, 20]),
        }
    )

    values = _calculate_target_values_standalone(
        target_variable="snap",
        non_geo_constraints=[
            {
                "variable": "age",
                "operation": ">=",
                "value": "65",
            }
        ],
        n_households=1,
        hh_vars={"snap": np.array([300], dtype=np.float32)},
        reform_hh_vars={},
        target_entity_vars={},
        person_vars={"age": np.array([70, 40], dtype=np.float32)},
        entity_rel=entity_rel,
        household_ids=np.array([100]),
        variable_entity_map={"snap": "household"},
        entity_hh_idx_map={},
        person_to_entity_idx_map={},
    )

    np.testing.assert_array_equal(values, np.array([300], dtype=np.float32))


def test_person_amount_targets_are_scoped_per_household():
    entity_rel = pd.DataFrame(
        {
            "person_id": np.array([0, 1, 2, 3]),
            "household_id": np.array([100, 100, 200, 200]),
            "tax_unit_id": np.array([10, 11, 12, 13]),
            "spm_unit_id": np.array([20, 20, 21, 21]),
        }
    )

    values = _calculate_target_values_standalone(
        target_variable="total_self_employment_income",
        non_geo_constraints=[
            {
                "variable": "total_self_employment_income",
                "operation": ">",
                "value": "0",
            },
            {
                "variable": "tax_unit_is_filer",
                "operation": "==",
                "value": "1",
            },
        ],
        n_households=2,
        hh_vars={
            "total_self_employment_income": np.array(
                [15000, 7000],
                dtype=np.float32,
            ),
        },
        reform_hh_vars={},
        target_entity_vars={
            "total_self_employment_income": np.array(
                [10000, 5000, 7000, 0],
                dtype=np.float32,
            ),
        },
        person_vars={
            "total_self_employment_income": np.array(
                [10000, 5000, 7000, 0],
                dtype=np.float32,
            ),
            "tax_unit_is_filer": np.array([1, 0, 0, 1], dtype=np.float32),
        },
        entity_rel=entity_rel,
        household_ids=np.array([100, 200]),
        variable_entity_map={"total_self_employment_income": "person"},
        entity_hh_idx_map={"person": np.array([0, 0, 1, 1])},
        person_to_entity_idx_map={"person": np.array([0, 1, 2, 3])},
    )

    np.testing.assert_array_equal(
        values,
        np.array([10000, 0], dtype=np.float32),
    )


def test_spm_unit_amount_targets_count_each_unit_once():
    entity_rel = pd.DataFrame(
        {
            "person_id": np.array([0, 1]),
            "household_id": np.array([100, 100]),
            "tax_unit_id": np.array([10, 10]),
            "spm_unit_id": np.array([20, 20]),
        }
    )

    values = _calculate_target_values_standalone(
        target_variable="snap",
        non_geo_constraints=[
            {
                "variable": "snap",
                "operation": ">",
                "value": "0",
            }
        ],
        n_households=1,
        hh_vars={"snap": np.array([300], dtype=np.float32)},
        reform_hh_vars={},
        target_entity_vars={"snap": np.array([300], dtype=np.float32)},
        person_vars={"snap": np.array([300, 300], dtype=np.float32)},
        entity_rel=entity_rel,
        household_ids=np.array([100]),
        variable_entity_map={"snap": "spm_unit"},
        entity_hh_idx_map={"spm_unit": np.array([0])},
        person_to_entity_idx_map={"spm_unit": np.array([0, 0])},
    )

    np.testing.assert_array_equal(values, np.array([300], dtype=np.float32))


def test_spm_unit_count_targets_preserve_entity_counting():
    entity_rel = pd.DataFrame(
        {
            "person_id": np.array([0, 1, 2, 3]),
            "household_id": np.array([100, 100, 200, 200]),
            "tax_unit_id": np.array([10, 10, 11, 12]),
            "spm_unit_id": np.array([20, 20, 21, 22]),
        }
    )

    values = _calculate_target_values_standalone(
        target_variable="spm_unit_count",
        non_geo_constraints=[
            {
                "variable": "snap",
                "operation": ">",
                "value": "0",
            }
        ],
        n_households=2,
        hh_vars={},
        reform_hh_vars={},
        target_entity_vars={},
        person_vars={"snap": np.array([300, 300, 0, 80], dtype=np.float32)},
        entity_rel=entity_rel,
        household_ids=np.array([100, 200]),
        variable_entity_map={"spm_unit_count": "spm_unit"},
        entity_hh_idx_map={"spm_unit": np.array([0, 1, 1])},
        person_to_entity_idx_map={"spm_unit": np.array([0, 0, 1, 2])},
    )

    np.testing.assert_array_equal(values, np.array([1, 1], dtype=np.float32))
