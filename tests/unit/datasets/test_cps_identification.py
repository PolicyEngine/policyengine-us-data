import numpy as np
import pandas as pd

from policyengine_us_data.utils.identification import (
    _derive_has_tin_from_identification_inputs,
    _derive_has_valid_ssn_from_ssn_card_type_codes,
    _derive_taxpayer_id_type_from_identification_flags,
    _high_confidence_tin_evidence,
    _impute_has_tin,
    _impute_has_valid_ssn,
    _proxy_tax_unit_filers,
    _store_identification_variables,
)


def _person_fixture(**overrides):
    n = max((len(value) for value in overrides.values()), default=4)
    defaults = {
        "SS_YN": np.zeros(n, dtype=int),
        "RESNSS1": np.zeros(n, dtype=int),
        "RESNSS2": np.zeros(n, dtype=int),
        "MCARE": np.zeros(n, dtype=int),
        "PEN_SC1": np.zeros(n, dtype=int),
        "PEN_SC2": np.zeros(n, dtype=int),
        "PEIO1COW": np.zeros(n, dtype=int),
        "A_MJOCC": np.zeros(n, dtype=int),
        "MIL": np.zeros(n, dtype=int),
        "PEAFEVER": np.zeros(n, dtype=int),
        "CHAMPVA": np.zeros(n, dtype=int),
        "SSI_YN": np.zeros(n, dtype=int),
        "WSAL_VAL": np.zeros(n, dtype=int),
        "SEMP_VAL": np.zeros(n, dtype=int),
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


def _cps_fixture(
    *,
    age,
    tax_unit_ids,
    weights=None,
    employment_income=None,
    self_employment_income=None,
    prior_employment_income=None,
    prior_self_employment_income=None,
):
    n = len(age)
    weights = np.ones(n) if weights is None else np.asarray(weights)
    household_ids = np.arange(n)
    return {
        "age": np.asarray(age),
        "person_tax_unit_id": np.asarray(tax_unit_ids),
        "person_household_id": household_ids,
        "household_id": household_ids,
        "household_weight": weights,
        "employment_income": (
            np.zeros(n) if employment_income is None else np.asarray(employment_income)
        ),
        "self_employment_income": (
            np.zeros(n)
            if self_employment_income is None
            else np.asarray(self_employment_income)
        ),
        "employment_income_last_year": (
            np.zeros(n)
            if prior_employment_income is None
            else np.asarray(prior_employment_income)
        ),
        "self_employment_income_last_year": (
            np.zeros(n)
            if prior_self_employment_income is None
            else np.asarray(prior_self_employment_income)
        ),
    }


def test_derive_has_valid_ssn_from_ssn_card_type_codes():
    result = _derive_has_valid_ssn_from_ssn_card_type_codes(
        np.array([0, 1, 2, 3]),
    )

    np.testing.assert_array_equal(
        result,
        np.array([False, True, False, False], dtype=bool),
    )


def test_impute_has_valid_ssn_does_not_treat_ead_proxy_as_direct_evidence():
    result = _impute_has_valid_ssn(
        ssn_card_type=np.array([0, 1, 2, 3]),
    )

    np.testing.assert_array_equal(result, np.array([False, True, False, False]))


def test_derive_taxpayer_id_type_from_identification_flags():
    result = _derive_taxpayer_id_type_from_identification_flags(
        has_valid_ssn=np.array([False, True, False]),
        has_tin=np.array([False, True, True]),
    )

    assert result.tolist() == ["NONE", "VALID_SSN", "OTHER_TIN"]


def test_high_confidence_admin_signal_gets_tin():
    person = _person_fixture(SS_YN=np.array([1, 0]), MCARE=np.array([0, 1]))

    result = _high_confidence_tin_evidence(person)

    np.testing.assert_array_equal(result, np.array([True, True]))


def test_medicaid_only_is_not_high_confidence_tin_evidence():
    person = _person_fixture()
    person["CAID"] = np.array([1, 0, 0, 0])

    result = _high_confidence_tin_evidence(person)

    np.testing.assert_array_equal(result, np.zeros(4, dtype=bool))


def test_derive_has_tin_from_identification_inputs_is_conservative():
    person = _person_fixture(SS_YN=np.zeros(5, dtype=int))
    result = _derive_has_tin_from_identification_inputs(
        person=person,
        ssn_card_type=np.array([0, 1, 2, 3, 0]),
        has_itin_number=np.array([False, False, False, False, True]),
    )

    np.testing.assert_array_equal(
        result,
        np.array([False, True, False, False, True], dtype=bool),
    )


def test_other_non_citizen_with_admin_signal_gets_tin():
    person = _person_fixture(SS_YN=np.array([1]))
    result = _derive_has_tin_from_identification_inputs(
        person=person,
        ssn_card_type=np.array([3]),
    )

    np.testing.assert_array_equal(result, np.array([True]))


def test_other_non_citizen_without_evidence_does_not_get_tin():
    person = _person_fixture()
    cps = _cps_fixture(age=[40], tax_unit_ids=[1])

    result = _impute_has_tin(
        cps,
        person.iloc[:1],
        ssn_card_type=np.array([3]),
        time_period=2024,
        non_ssn_filer_tin_target=0,
    )

    np.testing.assert_array_equal(result, np.array([False]))


def test_tin_target_does_not_select_other_non_citizen_without_evidence():
    person = _person_fixture()
    cps = _cps_fixture(
        age=[40],
        tax_unit_ids=[1],
        self_employment_income=[5_000],
    )

    result = _impute_has_tin(
        cps,
        person.iloc[:1],
        ssn_card_type=np.array([3]),
        time_period=2024,
        non_ssn_filer_tin_target=1,
    )

    np.testing.assert_array_equal(result, np.array([False]))


def test_proxy_tax_unit_filers_selects_two_oldest_adults():
    result = _proxy_tax_unit_filers(
        person_tax_unit_ids=np.array([1, 1, 1, 2, 2]),
        age=np.array([16, 40, 38, 12, 50]),
    )

    np.testing.assert_array_equal(result, np.array([False, True, True, False, True]))


def test_impute_has_tin_targets_likely_itin_filer_unit_and_minor_children():
    person = _person_fixture(
        SS_YN=np.zeros(4, dtype=int),
        MCARE=np.zeros(4, dtype=int),
    )
    cps = _cps_fixture(
        age=[40, 8, 40, 8],
        tax_unit_ids=[1, 1, 2, 2],
        self_employment_income=[5_000, 0, 0, 0],
    )

    result = _impute_has_tin(
        cps,
        person,
        ssn_card_type=np.array([0, 0, 0, 0]),
        time_period=2024,
        non_ssn_filer_tin_target=1,
    )

    np.testing.assert_array_equal(result, np.array([True, True, False, False]))


def test_store_identification_variables_writes_id_primitives():
    cps = {}
    person = _person_fixture(SS_YN=np.zeros(4, dtype=int))
    cps = _cps_fixture(
        age=[40, 40, 40, 40],
        tax_unit_ids=[1, 2, 3, 4],
    )

    _store_identification_variables(
        cps,
        person,
        np.array([0, 1, 2, 3]),
        time_period=2023,
    )

    assert cps["ssn_card_type"].tolist() == [
        b"NONE",
        b"CITIZEN",
        b"NON_CITIZEN_VALID_EAD",
        b"OTHER_NON_CITIZEN",
    ]
    assert cps["taxpayer_id_type"].tolist() == [
        b"NONE",
        b"VALID_SSN",
        b"NONE",
        b"NONE",
    ]
    np.testing.assert_array_equal(
        cps["has_tin"],
        np.array([False, True, False, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        cps["has_valid_ssn"],
        np.array([False, True, False, False], dtype=bool),
    )
    np.testing.assert_array_equal(cps["has_itin"], cps["has_tin"])
