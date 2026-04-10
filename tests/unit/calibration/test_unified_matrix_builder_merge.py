import numpy as np

from policyengine_us_data.calibration.calibration_utils import apply_op
from policyengine_us_data.calibration.unified_matrix_builder import (
    UnifiedMatrixBuilder,
    _assemble_clone_values_standalone,
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
