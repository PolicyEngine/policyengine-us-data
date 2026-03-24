import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.mortgage_interest import (
    convert_mortgage_interest_to_structural_inputs,
    impute_tax_unit_mortgage_balance_hints,
    supports_structural_mortgage_inputs,
)


def _base_dataset_dict(deductible_mortgage_interest, interest_deduction):
    time_period = 2024
    return {
        "person_id": {time_period: np.array([1, 2])},
        "tax_unit_id": {time_period: np.array([1])},
        "marital_unit_id": {time_period: np.array([1])},
        "spm_unit_id": {time_period: np.array([1])},
        "family_id": {time_period: np.array([1])},
        "household_id": {time_period: np.array([1])},
        "person_tax_unit_id": {time_period: np.array([1, 1])},
        "person_marital_unit_id": {time_period: np.array([1, 1])},
        "person_spm_unit_id": {time_period: np.array([1, 1])},
        "person_family_id": {time_period: np.array([1, 1])},
        "person_household_id": {time_period: np.array([1, 1])},
        "is_tax_unit_head": {time_period: np.array([True, False])},
        "is_tax_unit_spouse": {time_period: np.array([False, True])},
        "age": {time_period: np.array([55, 53])},
        "filing_status": {time_period: np.array([b"JOINT"])},
        "deductible_mortgage_interest": {
            time_period: np.array(deductible_mortgage_interest, dtype=np.float32)
        },
        "interest_deduction": {
            time_period: np.array(interest_deduction, dtype=np.float32)
        },
    }


def _current_law_cap(filing_status: bytes, origination_year: int) -> float:
    is_separate = b"SEPARATE" in filing_status
    if origination_year <= 2017:
        return 500_000.0 if is_separate else 1_000_000.0
    return 375_000.0 if is_separate else 750_000.0


@pytest.mark.skipif(
    not supports_structural_mortgage_inputs(),
    reason="Installed policyengine-us does not yet expose structural MID inputs.",
)
def test_structural_mortgage_conversion_preserves_current_law_interest_deduction():
    data = _base_dataset_dict(
        deductible_mortgage_interest=[6_000.0, 0.0],
        interest_deduction=[7_000.0],
    )
    converted = convert_mortgage_interest_to_structural_inputs(data, 2024)

    assert "deductible_mortgage_interest" not in converted
    assert "interest_deduction" not in converted
    assert converted["first_home_mortgage_balance"][2024][0] > 0
    assert converted["first_home_mortgage_interest"][2024][0] >= 6_000
    assert converted["first_home_mortgage_origination_year"][2024][0] > 0
    assert converted["investment_interest_expense"][2024].sum() == pytest.approx(
        1_000.0
    )
    cap = _current_law_cap(
        converted["filing_status"][2024][0],
        int(converted["first_home_mortgage_origination_year"][2024][0]),
    )
    balance = converted["first_home_mortgage_balance"][2024][0]
    total_interest = converted["first_home_mortgage_interest"][2024][0]
    deductible_share = min(1.0, cap / balance) if balance > 0 else 0.0

    assert total_interest * deductible_share == pytest.approx(6_000.0)
    assert converted["home_mortgage_interest"][2024].sum() == pytest.approx(
        total_interest
    )
    assert (
        total_interest * deductible_share
        + converted["investment_interest_expense"][2024].sum()
    ) == pytest.approx(7_000.0)


@pytest.mark.skipif(
    not supports_structural_mortgage_inputs(),
    reason="Installed policyengine-us does not yet expose structural MID inputs.",
)
def test_structural_mortgage_conversion_preserves_non_mortgage_interest():
    data = _base_dataset_dict(
        deductible_mortgage_interest=[0.0, 0.0],
        interest_deduction=[2_500.0],
    )
    converted = convert_mortgage_interest_to_structural_inputs(data, 2024)

    assert converted["first_home_mortgage_balance"][2024][0] == 0
    assert converted["first_home_mortgage_interest"][2024][0] == 0
    assert converted["home_mortgage_interest"][2024].sum() == 0
    assert converted["investment_interest_expense"][2024].sum() == pytest.approx(
        2_500.0
    )


@pytest.mark.skipif(
    not supports_structural_mortgage_inputs(),
    reason="Installed policyengine-us does not yet expose structural MID inputs.",
)
def test_structural_mortgage_conversion_keeps_balance_hints_for_non_itemizers():
    data = _base_dataset_dict(
        deductible_mortgage_interest=[0.0, 0.0],
        interest_deduction=[0.0],
    )
    data["imputed_first_home_mortgage_balance_hint"] = {
        2024: np.array([250_000.0], dtype=np.float32)
    }
    data["imputed_second_home_mortgage_balance_hint"] = {
        2024: np.array([25_000.0], dtype=np.float32)
    }

    converted = convert_mortgage_interest_to_structural_inputs(data, 2024)

    assert converted["first_home_mortgage_balance"][2024][0] == pytest.approx(250_000.0)
    assert converted["second_home_mortgage_balance"][2024][0] == pytest.approx(25_000.0)
    assert converted["first_home_mortgage_interest"][2024][0] == 0
    assert converted["second_home_mortgage_interest"][2024][0] == 0
    assert converted["first_home_mortgage_origination_year"][2024][0] > 0
    assert converted["second_home_mortgage_origination_year"][2024][0] >= 2018
    assert converted["home_mortgage_interest"][2024].sum() == 0
    assert converted["investment_interest_expense"][2024].sum() == 0


@pytest.mark.skipif(
    not supports_structural_mortgage_inputs(),
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
        lambda self: {
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
        },
    )

    data = {
        "person_id": {2024: np.array([1, 2])},
        "tax_unit_id": {2024: np.array([1, 2])},
        "marital_unit_id": {2024: np.array([1, 2])},
        "spm_unit_id": {2024: np.array([1, 2])},
        "family_id": {2024: np.array([1, 2])},
        "household_id": {2024: np.array([1, 2])},
        "person_tax_unit_id": {2024: np.array([1, 2])},
        "person_marital_unit_id": {2024: np.array([1, 2])},
        "person_spm_unit_id": {2024: np.array([1, 2])},
        "person_family_id": {2024: np.array([1, 2])},
        "person_household_id": {2024: np.array([1, 2])},
        "is_tax_unit_head": {2024: np.array([True, True])},
        "is_tax_unit_spouse": {2024: np.array([False, False])},
        "age": {2024: np.array([45, 55])},
        "is_male": {2024: np.array([1, 0])},
        "cps_race": {2024: np.array([1, 2])},
        "employment_income": {2024: np.array([80_000, 40_000])},
        "taxable_interest_income": {2024: np.array([1_000, 500])},
        "tax_exempt_interest_income": {2024: np.array([0, 0])},
        "qualified_dividend_income": {2024: np.array([500, 250])},
        "non_qualified_dividend_income": {2024: np.array([0, 0])},
        "social_security_retirement": {2024: np.array([0, 5_000])},
        "taxable_private_pension_income": {2024: np.array([0, 0])},
        "tax_exempt_private_pension_income": {2024: np.array([0, 0])},
        "tenure_type": {
            2024: np.array([b"OWNED_WITH_MORTGAGE", b"OWNED_WITH_MORTGAGE"])
        },
        "spm_unit_tenure_type": {
            2024: np.array([b"OWNER_WITH_MORTGAGE", b"OWNER_WITHOUT_MORTGAGE"])
        },
    }

    imputed = impute_tax_unit_mortgage_balance_hints(data, 2024)

    assert imputed["imputed_first_home_mortgage_balance_hint"][2024].tolist() == [
        200_000.0,
        0.0,
    ]
    assert imputed["imputed_second_home_mortgage_balance_hint"][2024].tolist() == [
        20_000.0,
        0.0,
    ]
