import numpy as np
import pytest

from policyengine_us_data.calibration.create_stratified_cps import (
    _split_non_top_strata,
    _top_agi_floor,
)


def test_top_agi_floor_uses_first_passed_bracket():
    assert _top_agi_floor([(1_000_000, 2_000_000, 10)]) == pytest.approx(1_000_000.0)


def test_split_non_top_strata_uses_custom_top_floor_without_gap():
    agi = np.array([100_000.0, 400_000.0, 600_000.0, 900_000.0, 1_200_000.0])

    non_top_mask, bottom_mask, middle_mask, bottom_25_threshold = _split_non_top_strata(
        agi,
        1_000_000.0,
    )

    np.testing.assert_array_equal(
        non_top_mask,
        np.array([True, True, True, True, False]),
    )
    np.testing.assert_array_equal(
        bottom_mask,
        np.array([True, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        middle_mask,
        np.array([False, True, True, True, False]),
    )
    assert bottom_25_threshold == pytest.approx(325_000.0)
