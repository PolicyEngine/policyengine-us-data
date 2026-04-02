import numpy as np

from policyengine_us_data.calibration.publish_local_area import (
    _build_reported_takeup_anchors,
)


def test_build_reported_takeup_anchors_skips_missing_period():
    data = {
        "person_tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "has_marketplace_health_coverage_at_interview": {2023: np.array([True, False])},
        "has_medicaid_health_coverage_at_interview": {2023: np.array([True, False])},
    }

    assert _build_reported_takeup_anchors(data, 2024) == {}


def test_build_reported_takeup_anchors_uses_present_period():
    data = {
        "person_tax_unit_id": {2024: np.array([1, 1, 2], dtype=np.int64)},
        "tax_unit_id": {2024: np.array([1, 2], dtype=np.int64)},
        "has_marketplace_health_coverage_at_interview": {
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
