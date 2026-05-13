import h5py
import numpy as np

from policyengine_us_data.build_outputs.payload import H5Payload
from policyengine_us_data.build_outputs.writer import H5Writer


def test_h5_writer_writes_period_grouped_payload_and_verifies_counts(tmp_path):
    output_path = tmp_path / "nested" / "output.h5"
    data = {
        "household_id": {2024: np.array([0, 1], dtype=np.int32)},
        "person_id": {2024: np.array([0, 1, 2], dtype=np.int32)},
        "household_weight": {2024: np.array([2.5, 3.0], dtype=np.float32)},
        "block_geoid": {2024: np.array([b"block-1", b"block-2"])},
    }

    result = H5Writer().write(
        payload=H5Payload(
            data=data,
            time_period=2024,
            entity_lengths={"household": 2, "person": 3},
        ),
        output_path=output_path,
    )

    assert result.path == output_path
    assert result.households == 2
    assert result.persons == 3
    assert result.household_weight_sum == 5.5
    assert result.person_weight_sum is None

    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(
            file["block_geoid"]["2024"][:],
            np.array([b"block-1", b"block-2"]),
        )


def test_h5_writer_verify_handles_missing_summary_variables(tmp_path):
    output_path = tmp_path / "output.h5"
    H5Writer().write(
        payload=H5Payload(data={"rent": {2024: np.array([100])}}, time_period=2024),
        output_path=output_path,
    )

    result = H5Writer().verify(path=output_path, time_period=2024)

    assert result.households is None
    assert result.persons is None
    assert result.household_weight_sum is None
    assert result.person_weight_sum is None
