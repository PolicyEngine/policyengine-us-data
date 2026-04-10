import numpy as np
import pandas as pd

from policyengine_us_data.utils.identification import (
    _derive_has_tin_from_identification_inputs,
    _derive_has_valid_ssn_from_ssn_card_type_codes,
    _derive_taxpayer_id_type_from_identification_flags,
    _high_confidence_tin_evidence,
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
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


def test_derive_has_valid_ssn_from_ssn_card_type_codes():
    result = _derive_has_valid_ssn_from_ssn_card_type_codes(
        np.array([0, 1, 2, 3]),
    )

    np.testing.assert_array_equal(
        result,
        np.array([False, True, False, False], dtype=bool),
    )


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


def test_store_identification_variables_writes_id_primitives():
    cps = {}
    person = _person_fixture(SS_YN=np.zeros(5, dtype=int))
    has_itin = np.array([False, False, False, False, True])

    _store_identification_variables(
        cps,
        person,
        np.array([0, 1, 2, 3, 0]),
        has_itin,
    )

    assert cps["ssn_card_type"].tolist() == [
        b"NONE",
        b"CITIZEN",
        b"NON_CITIZEN_VALID_EAD",
        b"OTHER_NON_CITIZEN",
        b"NONE",
    ]
    assert cps["taxpayer_id_type"].tolist() == [
        b"NONE",
        b"VALID_SSN",
        b"NONE",
        b"NONE",
        b"OTHER_TIN",
    ]
    np.testing.assert_array_equal(
        cps["has_tin"],
        np.array([False, True, False, False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        cps["has_valid_ssn"],
        np.array([False, True, False, False, False], dtype=bool),
    )
    np.testing.assert_array_equal(cps["has_itin"], cps["has_tin"])
