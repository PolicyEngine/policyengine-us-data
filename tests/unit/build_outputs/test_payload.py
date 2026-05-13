import numpy as np
import pytest

from policyengine_us_data.build_outputs.payload import H5Payload


def test_h5_payload_accepts_structural_variables_matching_entity_lengths():
    payload = H5Payload(
        data={
            "household_id": {2024: np.array([0, 1])},
            "person_id": {2024: np.array([0, 1, 2])},
            "person_tax_unit_id": {2024: np.array([0, 1, 1])},
            "tax_unit_id": {2024: np.array([0, 1])},
            "rent": {2024: np.array([100, 200])},
        },
        time_period=2024,
        entity_lengths={
            "household": 2,
            "person": 3,
            "tax_unit": 2,
        },
    )

    assert payload.time_period == 2024
    assert payload.entity_lengths["household"] == 2


def test_h5_payload_rejects_structural_variable_length_mismatch():
    with pytest.raises(
        ValueError,
        match="household_id\\[2024\\] length 1 does not match household length 2",
    ):
        H5Payload(
            data={"household_id": {2024: np.array([0])}},
            time_period=2024,
            entity_lengths={"household": 2},
        )


def test_h5_payload_rejects_explicit_variable_entity_length_mismatch():
    with pytest.raises(
        ValueError,
        match="takes_up_snap_if_eligible\\[2024\\] length 1 "
        "does not match spm_unit length 2",
    ):
        H5Payload(
            data={"takes_up_snap_if_eligible": {2024: np.array([True])}},
            time_period=2024,
            entity_lengths={"spm_unit": 2},
            variable_entities={"takes_up_snap_if_eligible": "spm_unit"},
        )


def test_h5_payload_rejects_unknown_explicit_variable_entity():
    with pytest.raises(
        ValueError,
        match="takes_up_snap_if_eligible maps to unknown entity 'spm_unit'",
    ):
        H5Payload(
            data={"takes_up_snap_if_eligible": {2024: np.array([True])}},
            time_period=2024,
            variable_entities={"takes_up_snap_if_eligible": "spm_unit"},
        )


def test_h5_payload_rejects_scalar_values():
    with pytest.raises(ValueError, match="rent\\[2024\\] must be array-like"):
        H5Payload(
            data={"rent": {2024: np.array(1)}},
            time_period=2024,
        )
