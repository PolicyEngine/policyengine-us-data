import numpy as np

from policyengine_us_data.utils.h5 import to_h5_values


class EncodedArray:
    def __init__(self, values):
        self.values = np.asarray(values)

    def decode_to_str(self):
        return self.values


def test_to_h5_values_encodes_decodable_string_arrays():
    values = to_h5_values(
        EncodedArray(["a", "b"]),
        is_string_like=True,
        variable="example",
    )

    assert values.dtype.kind == "S"
    assert values.tolist() == [b"a", b"b"]


def test_to_h5_values_encodes_plain_numpy_string_arrays():
    values = to_h5_values(
        np.asarray(["a", "b"]),
        is_string_like=True,
        variable="example",
    )

    assert values.dtype.kind == "S"
    assert values.tolist() == [b"a", b"b"]


def test_to_h5_values_preserves_county_fips_as_int32():
    values = to_h5_values(
        np.asarray(["01001", "01003"]),
        is_string_like=True,
        variable="county_fips",
    )

    assert values.dtype == np.int32
    assert values.tolist() == [1001, 1003]
