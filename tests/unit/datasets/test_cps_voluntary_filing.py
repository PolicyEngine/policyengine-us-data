import numpy as np

from policyengine_us_data.datasets.cps.cps import (
    _sum_person_values_to_tax_units,
    _voluntary_filing_age_bin,
    _voluntary_filing_children_bin,
    _voluntary_filing_rate_by_tax_unit,
    _voluntary_filing_wage_income_bin,
)


def test_sum_person_values_to_tax_units_aggregates_wages():
    result = _sum_person_values_to_tax_units(
        person_values=np.array([10_000, 5_000, 2_500, 7_500], dtype=np.float32),
        person_tax_unit_ids=np.array([101, 101, 102, 103]),
        tax_unit_ids=np.array([101, 102, 103]),
    )

    np.testing.assert_allclose(result, np.array([15_000, 2_500, 7_500]))


def test_voluntary_filing_bins_map_expected_categories():
    np.testing.assert_array_equal(
        _voluntary_filing_children_bin(np.array([0, 1, 3])),
        np.array(["no_children", "with_children", "with_children"]),
    )
    np.testing.assert_array_equal(
        _voluntary_filing_wage_income_bin(
            np.array([0, 1, 14_999, 15_000, 29_999, 30_000], dtype=np.float32)
        ),
        np.array(["zero", "low", "low", "medium", "medium", "high"]),
    )
    np.testing.assert_array_equal(
        _voluntary_filing_age_bin(np.array([24, 64, 65, 80])),
        np.array(["under_65", "under_65", "age_65_plus", "age_65_plus"]),
    )


def test_voluntary_filing_rate_lookup_uses_all_three_dimensions():
    rates = {
        "no_children": {
            "zero": {"under_65": 0.2, "age_65_plus": 0.05},
            "low": {"under_65": 0.24, "age_65_plus": 0.04},
            "medium": {"under_65": 0.0, "age_65_plus": 0.0},
            "high": {"under_65": 0.0, "age_65_plus": 0.005},
        },
        "with_children": {
            "zero": {"under_65": 0.5, "age_65_plus": 0.075},
            "low": {"under_65": 0.6, "age_65_plus": 0.06},
            "medium": {"under_65": 0.0, "age_65_plus": 0.0},
            "high": {"under_65": 0.025, "age_65_plus": 0.0037},
        },
    }

    result = _voluntary_filing_rate_by_tax_unit(
        rates,
        children_bin=np.array(["no_children", "with_children", "with_children"]),
        wage_income_bin=np.array(["zero", "low", "high"]),
        age_bin=np.array(["under_65", "under_65", "age_65_plus"]),
    )

    np.testing.assert_allclose(result, np.array([0.2, 0.6, 0.0037]))
