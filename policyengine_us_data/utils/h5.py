"""HDF5 serialization helpers."""

import numpy as np


def to_h5_values(values, *, is_string_like: bool, variable: str):
    """Normalize PolicyEngine holder arrays and ndarray values for HDF5."""
    if is_string_like and variable != "county_fips":
        if hasattr(values, "decode_to_str"):
            values = values.decode_to_str()
        return values.astype("S")
    if variable == "county_fips":
        return values.astype("int32")
    return np.array(values)
