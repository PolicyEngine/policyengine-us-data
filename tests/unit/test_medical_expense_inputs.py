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
