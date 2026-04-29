import numpy as np
import pandas as pd

from policyengine_us_data.datasets.puf import puf as puf_module
from policyengine_us_data.datasets.puf.puf import (
    _lifetime_learning_credit_student_from_puf,
    _person_financial_value_from_puf_row,
    _qualified_tuition_expenses_from_puf,
    _with_lifetime_learning_credit_inputs,
)


def test_qualified_tuition_expenses_prefer_form_8863_llc_expenses():
    puf = pd.DataFrame(
        {
            "E03230": [1_000.0, 3_000.0, 0.0],
            "E87530": [2_000.0, 1_500.0, 4_000.0],
        }
    )

    result = _qualified_tuition_expenses_from_puf(puf)

    assert result.tolist() == [2_000.0, 3_000.0, 4_000.0]


def test_lifetime_learning_credit_student_uses_form_8863_when_available():
    puf = pd.DataFrame(
        {
            "E03230": [1_000.0, 0.0],
            "E87530": [0.0, 2_000.0],
        }
    )

    result = _lifetime_learning_credit_student_from_puf(puf)

    assert result.tolist() == [False, True]


def test_puf_arrays_add_lifetime_learning_credit_inputs(monkeypatch):
    monkeypatch.setattr(
        puf_module,
        "has_policyengine_us_variables",
        lambda *variables: True,
    )
    arrays = {"qualified_tuition_expenses": np.array([0.0, 1_000.0])}

    result = _with_lifetime_learning_credit_inputs(arrays)

    for variable in puf_module.PUF_LLC_ELIGIBILITY_INPUTS:
        np.testing.assert_array_equal(result[variable], np.array([False, True]))


def test_person_financial_value_keeps_llc_inputs_boolean():
    row = pd.Series(
        {
            "qualified_tuition_expenses": 1_000.0,
            "attends_eligible_educational_institution_for_lifetime_learning_credit": True,
            "employment_income": 100.0,
        }
    )

    assert (
        _person_financial_value_from_puf_row(
            "attends_eligible_educational_institution_for_lifetime_learning_credit",
            row,
            0.25,
        )
        is True
    )
    assert (
        _person_financial_value_from_puf_row(
            "attends_eligible_educational_institution_for_lifetime_learning_credit",
            row,
            0,
        )
        is False
    )
    assert _person_financial_value_from_puf_row("employment_income", row, 0.25) == 25
