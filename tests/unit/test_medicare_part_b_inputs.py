from policyengine_us_data.datasets.cps.extended_cps import (
    CPS_ONLY_IMPUTED_VARIABLES,
    supports_modeled_medicare_part_b_inputs,
)
from policyengine_us_data.datasets.puf.puf import MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS
from policyengine_us_data.utils import policyengine as policyengine_utils


class _Variable:
    adds = None
    defined_for = None
    formulas = None
    subtracts = None

    def __init__(self, *, is_input=True, defined_for=None, formulas=None):
        self._is_input = is_input
        self.defined_for = defined_for
        self.formulas = formulas

    def is_input_variable(self):
        return self._is_input


def test_medicare_part_b_clone_imputation_matches_installed_model_support():
    assert ("medicare_part_b_premiums" in set(CPS_ONLY_IMPUTED_VARIABLES)) is (
        not supports_modeled_medicare_part_b_inputs()
    )


def test_puf_medical_breakdown_still_sums_to_one():
    assert sum(MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS.values()) == 1.0


def test_supports_medicare_enrollment_input_allows_partial_support(monkeypatch):
    monkeypatch.setattr(
        policyengine_utils,
        "has_policyengine_us_variables",
        lambda *variables: variables == ("medicare_enrolled",),
    )
    monkeypatch.setattr(
        policyengine_utils,
        "has_policyengine_us_pure_input_variables",
        lambda *variables: variables == ("medicare_enrolled",),
    )

    assert policyengine_utils.supports_medicare_enrollment_input() is True
    assert policyengine_utils.supports_modeled_medicare_part_b_inputs() is False


def test_pure_input_support_rejects_formula_and_conditional_variables(monkeypatch):
    monkeypatch.setattr(
        policyengine_utils,
        "_policyengine_us_variables",
        lambda: {
            "pure_input": _Variable(),
            "formula_input": _Variable(formulas={"2025": object()}),
            "conditional_input": _Variable(defined_for="is_eligible"),
            "formula_variable": _Variable(is_input=False),
        },
    )

    assert policyengine_utils.has_policyengine_us_pure_input_variables("pure_input")
    assert not policyengine_utils.has_policyengine_us_pure_input_variables(
        "formula_input"
    )
    assert not policyengine_utils.has_policyengine_us_pure_input_variables(
        "conditional_input"
    )
    assert not policyengine_utils.has_policyengine_us_pure_input_variables(
        "formula_variable"
    )
