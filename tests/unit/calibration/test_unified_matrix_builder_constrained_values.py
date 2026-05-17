import numpy as np
import pandas as pd

from policyengine_us_data.calibration.unified_matrix_builder import (
    _calculate_target_values_standalone,
)


ENTITY_REL = pd.DataFrame(
    {
        "person_id": [1, 2, 3],
        "household_id": [10, 10, 20],
        "tax_unit_id": [100, 101, 102],
        "spm_unit_id": [1000, 1001, 1002],
    }
)
HOUSEHOLD_IDS = np.array([10, 20])
ENTITY_HH_IDX_MAP = {"person": np.array([0, 0, 1])}
PERSON_TO_ENTITY_IDX_MAP = {"person": np.array([0, 1, 2])}
VARIABLE_ENTITY_MAP = {
    "irs_employment_income": "person",
    "taxable_interest_income": "person",
    "dividend_income": "person",
    "total_self_employment_income": "person",
    "farm_operations_income": "person",
    "partnership_s_corp_income": "person",
}
FILER_CONSTRAINT = [
    {
        "variable": "tax_unit_is_filer",
        "operation": "==",
        "value": "1",
    }
]


def _calculate(target_variable):
    return _calculate_target_values_standalone(
        target_variable=target_variable,
        non_geo_constraints=FILER_CONSTRAINT,
        n_households=2,
        hh_vars={},
        reform_hh_vars={},
        target_entity_vars={
            "irs_employment_income": np.array([100, 200, 300]),
            "taxable_interest_income": np.array([10, 20, 30]),
            "dividend_income": np.array([1, 2, 3]),
            "total_self_employment_income": np.array([1000, 2000, 3000]),
            "farm_operations_income": np.array([100, 200, 300]),
            "partnership_s_corp_income": np.array([10, 20, 30]),
        },
        person_vars={"tax_unit_is_filer": np.array([1, 0, 1])},
        entity_rel=ENTITY_REL,
        household_ids=HOUSEHOLD_IDS,
        variable_entity_map=VARIABLE_ENTITY_MAP,
        entity_hh_idx_map=ENTITY_HH_IDX_MAP,
        person_to_entity_idx_map=PERSON_TO_ENTITY_IDX_MAP,
    )


def test_constrained_person_value_target_filters_before_household_mapping():
    values = _calculate("irs_employment_income")

    np.testing.assert_array_equal(values, np.array([100, 300]))


def test_constrained_additive_value_target_filters_before_household_mapping():
    values = _calculate("taxable_interest_income+dividend_income")

    np.testing.assert_array_equal(values, np.array([11, 33]))


def test_constrained_three_part_additive_target():
    values = _calculate(
        "total_self_employment_income+farm_operations_income+partnership_s_corp_income"
    )

    np.testing.assert_array_equal(values, np.array([1110, 3330]))
