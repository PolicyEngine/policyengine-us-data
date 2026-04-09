import numpy as np

from policyengine_us_data.datasets.cps.enhanced_cps import (
    _get_base_aca_takeup,
    _set_period_array,
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
