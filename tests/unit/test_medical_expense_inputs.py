from policyengine_us_data.datasets.puf.puf import (
    MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS,
)


def test_puf_medical_breakdown_still_sums_to_one():
    assert sum(MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS.values()) == 1.0
