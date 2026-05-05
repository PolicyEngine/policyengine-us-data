import numpy as np

from policyengine_us_data.datasets.cps.enhanced_cps import (
    _get_base_aca_takeup,
    _set_period_array,
    create_aca_2025_takeup_override,
)


def test_get_base_aca_takeup_uses_stored_values():
    data = {
        "takes_up_aca_if_eligible": {
            2024: np.array([True, False, True], dtype=bool),
        }
    }

    result = _get_base_aca_takeup(data=data, base_year=2024, tax_unit_count=3)

    np.testing.assert_array_equal(
        result,
        np.array([True, False, True], dtype=bool),
    )


def test_get_base_aca_takeup_defaults_to_true_when_missing():
    result = _get_base_aca_takeup(data={}, base_year=2024, tax_unit_count=4)

    np.testing.assert_array_equal(result, np.ones(4, dtype=bool))


def test_set_period_array_creates_missing_variable_entry():
    data = {}
    values = np.array([True, False], dtype=bool)

    _set_period_array(data, "takes_up_aca_if_eligible", 2025, values)

    np.testing.assert_array_equal(data["takes_up_aca_if_eligible"][2025], values)


def test_create_aca_2025_takeup_override_matches_state_targets():
    result = create_aca_2025_takeup_override(
        base_takeup=np.array([True, True, False, False], dtype=bool),
        person_enrolled_if_takeup=np.ones(4, dtype=bool),
        person_weights=np.array([5.0, 5.0, 7.0, 3.0], dtype=np.float64),
        person_tax_unit_ids=np.array([10, 11, 12, 13], dtype=np.int64),
        tax_unit_ids=np.array([10, 11, 12, 13], dtype=np.int64),
        person_state_codes=np.array(["NY", "NY", "FL", "FL"]),
        target_people_by_state={"NY": 5.0, "FL": 10.0},
    )

    np.testing.assert_allclose(
        [
            np.array([5.0, 5.0])[result[:2]].sum(),
            np.array([7.0, 3.0])[result[2:]].sum(),
        ],
        [5.0, 10.0],
    )


def test_create_aca_2025_takeup_override_uses_state_spending_targets():
    result = create_aca_2025_takeup_override(
        base_takeup=np.array([True, True, False], dtype=bool),
        person_enrolled_if_takeup=np.ones(3, dtype=bool),
        person_weights=np.array([5.0, 5.0, 5.0], dtype=np.float64),
        person_tax_unit_ids=np.array([10, 11, 12], dtype=np.int64),
        tax_unit_ids=np.array([10, 11, 12], dtype=np.int64),
        person_state_codes=np.array(["NY", "NY", "NY"]),
        target_people_by_state={"NY": 5.0},
        tax_unit_aca_ptc=np.array([20.0, 100.0, 60.0], dtype=np.float64),
        tax_unit_weights=np.array([5.0, 5.0, 5.0], dtype=np.float64),
        target_spending_by_state={"NY": 500.0},
    )

    np.testing.assert_array_equal(
        result,
        np.array([False, True, False], dtype=bool),
    )
