from policyengine_us_data.datasets.cps.extended_cps import (
    CPS_ONLY_IMPUTED_VARIABLES,
    supports_modeled_medicare_part_b_inputs,
)
from policyengine_us_data.datasets.puf.puf import MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS


def test_medicare_part_b_clone_imputation_matches_installed_model_support():
    assert ("medicare_part_b_premiums" in set(CPS_ONLY_IMPUTED_VARIABLES)) is (
        not supports_modeled_medicare_part_b_inputs()
    )


def test_puf_medical_breakdown_still_sums_to_one():
    assert sum(MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS.values()) == 1.0
