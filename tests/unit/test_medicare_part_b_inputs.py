from policyengine_us_data.datasets.cps.extended_cps import CPS_ONLY_IMPUTED_VARIABLES
from policyengine_us_data.datasets.puf.puf import MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS


def test_medicare_part_b_not_qrf_imputed_for_clone_half():
    assert "medicare_part_b_premiums" not in set(CPS_ONLY_IMPUTED_VARIABLES)


def test_medicare_part_b_not_allocated_from_generic_puf_medical_expenses():
    assert "medicare_part_b_premiums" not in MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS
