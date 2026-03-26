import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.mortgage_interest import (
    STRUCTURAL_MORTGAGE_VARIABLES,
    convert_mortgage_interest_to_structural_inputs,
    impute_tax_unit_mortgage_balance_hints,
)
from policyengine_us_data.utils.policyengine import has_policyengine_us_variables

TIME_PERIOD = 2024
HAS_STRUCTURAL_MORTGAGE_INPUTS = has_policyengine_us_variables(
    *STRUCTURAL_MORTGAGE_VARIABLES
)


def _at_time_period(values, dtype=None):
    return {TIME_PERIOD: np.array(values, dtype=dtype)}


def _time_period_variables(**variables):
    return {name: _at_time_period(values) for name, values in variables.items()}


def _head_and_spouse_flags(person_tax_unit_ids):
    first_seen = {}
    heads = np.zeros(len(person_tax_unit_ids), dtype=bool)
    spouses = np.zeros(len(person_tax_unit_ids), dtype=bool)

    for idx, tax_unit_id in enumerate(person_tax_unit_ids):
        occurrence = first_seen.get(int(tax_unit_id), 0)
        if occurrence == 0:
            heads[idx] = True
        elif occurrence == 1:
            spouses[idx] = True
        first_seen[int(tax_unit_id)] = occurrence + 1

    return heads, spouses


def _base_dataset_dict(
    *,
    person_tax_unit_ids,
    ages,
    deductible_mortgage_interest=None,
    interest_deduction=None,
    filing_status=None,
):
    person_tax_unit_ids = np.array(person_tax_unit_ids, dtype=np.int32)
    tax_unit_ids = np.unique(person_tax_unit_ids)
    n_people = len(person_tax_unit_ids)
    person_ids = np.arange(1, n_people + 1, dtype=np.int32)
    heads, spouses = _head_and_spouse_flags(person_tax_unit_ids)

    data = {
        "person_id": _at_time_period(person_ids),
        "tax_unit_id": _at_time_period(tax_unit_ids),
        "marital_unit_id": _at_time_period(tax_unit_ids),
        "spm_unit_id": _at_time_period(tax_unit_ids),
        "family_id": _at_time_period(tax_unit_ids),
        "household_id": _at_time_period(tax_unit_ids),
        "person_tax_unit_id": _at_time_period(person_tax_unit_ids),
        "person_marital_unit_id": _at_time_period(person_tax_unit_ids),
        "person_spm_unit_id": _at_time_period(person_tax_unit_ids),
        "person_family_id": _at_time_period(person_tax_unit_ids),
        "person_household_id": _at_time_period(person_tax_unit_ids),
        "is_tax_unit_head": _at_time_period(heads),
        "is_tax_unit_spouse": _at_time_period(spouses),
        "age": _at_time_period(ages),
    }

    if filing_status is not None:
        data["filing_status"] = _at_time_period(filing_status)
    if deductible_mortgage_interest is not None:
        data["deductible_mortgage_interest"] = _at_time_period(
            deductible_mortgage_interest,
            dtype=np.float32,
        )
    if interest_deduction is not None:
        data["interest_deduction"] = _at_time_period(
            interest_deduction,
            dtype=np.float32,
        )

    return data


def _mock_scf_dataset():
    return {
        "age": np.array([45, 55]),
        "is_female": np.array([0, 1]),
        "cps_race": np.array([1, 2]),
        "is_married": np.array([1, 0]),
        "own_children_in_household": np.array([1, 0]),
        "employment_income": np.array([80_000, 40_000]),
        "interest_dividend_income": np.array([2_000, 1_000]),
        "social_security_pension_income": np.array([0, 5_000]),
        "nh_mort": np.array([250_000, 0]),
        "heloc": np.array([25_000, 0]),
        "houses": np.array([500_000, 350_000]),
        "wgt": np.array([1, 1]),
    }


def _current_law_cap(filing_status: bytes, origination_year: int) -> float:
    is_separate = b"SEPARATE" in filing_status
    if origination_year <= 2017:
        return 500_000.0 if is_separate else 1_000_000.0
    return 375_000.0 if is_separate else 750_000.0


@pytest.mark.skipif(
    not HAS_STRUCTURAL_MORTGAGE_INPUTS,
    reason="Installed policyengine-us does not yet expose structural MID inputs.",
)
def test_structural_mortgage_conversion_preserves_current_law_interest_deduction():
    data = _base_dataset_dict(
        person_tax_unit_ids=[1, 1],
        ages=[55, 53],
        deductible_mortgage_interest=[6_000.0, 0.0],
        interest_deduction=[7_000.0],
        filing_status=[b"JOINT"],
    )
    converted = convert_mortgage_interest_to_structural_inputs(data, TIME_PERIOD)

    assert "deductible_mortgage_interest" not in converted
    assert "interest_deduction" not in converted
    assert converted["first_home_mortgage_balance"][TIME_PERIOD][0] > 0
    assert converted["first_home_mortgage_interest"][TIME_PERIOD][0] >= 6_000
    assert converted["first_home_mortgage_origination_year"][TIME_PERIOD][0] > 0
    assert converted["investment_interest_expense"][TIME_PERIOD].sum() == pytest.approx(
        1_000.0
    )
    cap = _current_law_cap(
        converted["filing_status"][TIME_PERIOD][0],
        int(converted["first_home_mortgage_origination_year"][TIME_PERIOD][0]),
    )
    balance = converted["first_home_mortgage_balance"][TIME_PERIOD][0]
    total_interest = converted["first_home_mortgage_interest"][TIME_PERIOD][0]
    deductible_share = min(1.0, cap / balance) if balance > 0 else 0.0

    assert total_interest * deductible_share == pytest.approx(6_000.0)
    assert converted["home_mortgage_interest"][TIME_PERIOD].sum() == pytest.approx(
        total_interest
    )
    assert (
        total_interest * deductible_share
        + converted["investment_interest_expense"][TIME_PERIOD].sum()
    ) == pytest.approx(7_000.0)


@pytest.mark.skipif(
    not HAS_STRUCTURAL_MORTGAGE_INPUTS,
    reason="Installed policyengine-us does not yet expose structural MID inputs.",
)
def test_structural_mortgage_conversion_preserves_non_mortgage_interest():
    data = _base_dataset_dict(
        person_tax_unit_ids=[1, 1],
        ages=[55, 53],
        deductible_mortgage_interest=[0.0, 0.0],
        interest_deduction=[2_500.0],
        filing_status=[b"JOINT"],
    )
    converted = convert_mortgage_interest_to_structural_inputs(data, TIME_PERIOD)

    assert converted["first_home_mortgage_balance"][TIME_PERIOD][0] == 0
    assert converted["first_home_mortgage_interest"][TIME_PERIOD][0] == 0
    assert converted["home_mortgage_interest"][TIME_PERIOD].sum() == 0
    assert converted["investment_interest_expense"][TIME_PERIOD].sum() == pytest.approx(
        2_500.0
    )


@pytest.mark.skipif(
    not HAS_STRUCTURAL_MORTGAGE_INPUTS,
    reason="Installed policyengine-us does not yet expose structural MID inputs.",
)
def test_structural_mortgage_conversion_keeps_balance_hints_for_non_itemizers():
    data = _base_dataset_dict(
        person_tax_unit_ids=[1, 1],
        ages=[55, 53],
        deductible_mortgage_interest=[0.0, 0.0],
        interest_deduction=[0.0],
        filing_status=[b"JOINT"],
    )
    data["imputed_first_home_mortgage_balance_hint"] = {
        TIME_PERIOD: np.array([250_000.0], dtype=np.float32)
    }
    data["imputed_second_home_mortgage_balance_hint"] = {
        TIME_PERIOD: np.array([25_000.0], dtype=np.float32)
    }

    converted = convert_mortgage_interest_to_structural_inputs(data, TIME_PERIOD)

    assert converted["first_home_mortgage_balance"][TIME_PERIOD][0] == pytest.approx(
        250_000.0
    )
    assert converted["second_home_mortgage_balance"][TIME_PERIOD][0] == pytest.approx(
        25_000.0
    )
    assert converted["first_home_mortgage_interest"][TIME_PERIOD][0] == 0
    assert converted["second_home_mortgage_interest"][TIME_PERIOD][0] == 0
    assert converted["first_home_mortgage_origination_year"][TIME_PERIOD][0] > 0
    assert converted["second_home_mortgage_origination_year"][TIME_PERIOD][0] >= 2018
    assert converted["home_mortgage_interest"][TIME_PERIOD].sum() == 0
    assert converted["investment_interest_expense"][TIME_PERIOD].sum() == 0


@pytest.mark.skipif(
    not HAS_STRUCTURAL_MORTGAGE_INPUTS,
    reason="Installed policyengine-us does not yet expose structural MID inputs.",
)
def test_scf_balance_hint_imputation_zeroes_non_mortgaged_owner(monkeypatch):
    import microimpute.models.qrf as qrf_module
    import policyengine_us_data.datasets.scf.scf as scf_module

    class DummyQRF:
        def fit(self, *args, **kwargs):
            return self

        def predict(self, X_test):
            return pd.DataFrame(
                {
                    "imputed_first_home_mortgage_balance_hint": X_test[
                        "mortgage_owner_status"
                    ]
                    * 100_000,
                    "imputed_second_home_mortgage_balance_hint": X_test[
                        "mortgage_owner_status"
                    ]
                    * 10_000,
                }
            )

    monkeypatch.setattr(qrf_module, "QRF", DummyQRF)
    monkeypatch.setattr(
        scf_module.SCF_2022,
        "load_dataset",
        lambda self: _mock_scf_dataset(),
    )

    data = _base_dataset_dict(
        person_tax_unit_ids=[1, 2],
        ages=[45, 55],
    )
    data |= _time_period_variables(
        is_male=[1, 0],
        cps_race=[1, 2],
        employment_income=[80_000, 40_000],
        taxable_interest_income=[1_000, 500],
        tax_exempt_interest_income=[0, 0],
        qualified_dividend_income=[500, 250],
        non_qualified_dividend_income=[0, 0],
        social_security_retirement=[0, 5_000],
        taxable_private_pension_income=[0, 0],
        tax_exempt_private_pension_income=[0, 0],
        tenure_type=[b"OWNED_WITH_MORTGAGE", b"OWNED_WITH_MORTGAGE"],
        spm_unit_tenure_type=[
            b"OWNER_WITH_MORTGAGE",
            b"OWNER_WITHOUT_MORTGAGE",
        ],
    )

    imputed = impute_tax_unit_mortgage_balance_hints(data, TIME_PERIOD)

    assert imputed["imputed_first_home_mortgage_balance_hint"][
        TIME_PERIOD
    ].tolist() == [
        200_000.0,
        0.0,
    ]
    assert imputed["imputed_second_home_mortgage_balance_hint"][
        TIME_PERIOD
    ].tolist() == [
        20_000.0,
        0.0,
    ]
