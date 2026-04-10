import numpy as np

from policyengine_us_data.utils.identification import (
    _derive_has_valid_ssn_from_ssn_card_type_codes,
    _derive_has_tin_from_ssn_card_type_codes,
    _derive_taxpayer_id_type_from_ssn_card_type_codes,
    _store_identification_variables,
)


def test_derive_taxpayer_id_type_from_ssn_card_type_codes():
    result = _derive_taxpayer_id_type_from_ssn_card_type_codes(np.array([0, 1, 2, 3]))

    assert result.tolist() == [
        "NONE",
        "VALID_SSN",
        "VALID_SSN",
        "OTHER_TIN",
    ]


def test_derive_has_tin_from_ssn_card_type_codes():
    result = _derive_has_tin_from_ssn_card_type_codes(np.array([0, 1, 2, 3]))

    np.testing.assert_array_equal(
        result,
        np.array([False, True, True, True], dtype=bool),
    )


def test_derive_has_valid_ssn_from_ssn_card_type_codes():
    result = _derive_has_valid_ssn_from_ssn_card_type_codes(np.array([0, 1, 2, 3]))

    np.testing.assert_array_equal(
        result,
        np.array([False, True, True, False], dtype=bool),
    )


def test_store_identification_variables_writes_id_primitives():
    cps = {}

    _store_identification_variables(cps, np.array([0, 1, 2, 3]))

    assert cps["ssn_card_type"].tolist() == [
        b"NONE",
        b"CITIZEN",
        b"NON_CITIZEN_VALID_EAD",
        b"OTHER_NON_CITIZEN",
    ]
    assert cps["taxpayer_id_type"].tolist() == [
        b"NONE",
        b"VALID_SSN",
        b"VALID_SSN",
        b"OTHER_TIN",
    ]
    np.testing.assert_array_equal(
        cps["has_tin"],
        np.array([False, True, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        cps["has_valid_ssn"],
        np.array([False, True, True, False], dtype=bool),
    )
    np.testing.assert_array_equal(cps["has_itin"], cps["has_tin"])
