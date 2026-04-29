from policyengine_us_data.datasets.puf.puf import (
    MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS,
)
from policyengine_us_data.utils import policyengine as policyengine_utils


def test_puf_medical_breakdown_still_sums_to_one():
    assert sum(MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS.values()) == 1.0


def test_supports_medicare_enrollment_input_allows_partial_support(monkeypatch):
    monkeypatch.setattr(
        policyengine_utils,
        "has_policyengine_us_variables",
        lambda *variables: variables == ("medicare_enrolled",),
    )

    assert policyengine_utils.supports_medicare_enrollment_input() is True


def test_medicare_part_b_premium_variable_name_prefers_clean_name(monkeypatch):
    monkeypatch.setattr(
        policyengine_utils,
        "has_policyengine_us_variables",
        lambda *variables: variables == ("medicare_part_b_premium",),
    )

    assert (
        policyengine_utils.medicare_part_b_premium_variable_name()
        == "medicare_part_b_premium"
    )


def test_medicare_part_b_premium_variable_name_falls_back(monkeypatch):
    monkeypatch.setattr(
        policyengine_utils,
        "has_policyengine_us_variables",
        lambda *variables: False,
    )

    assert (
        policyengine_utils.medicare_part_b_premium_variable_name()
        == "medicare_part_b_premiums"
    )
