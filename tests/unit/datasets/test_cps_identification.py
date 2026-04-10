import numpy as np

from policyengine_us_data.utils.identification import (
    _derive_has_tin_from_ssn_card_type_codes,
    _store_identification_variables,
)


def test_derive_has_tin_from_ssn_card_type_codes():
    result = _derive_has_tin_from_ssn_card_type_codes(np.array([0, 1, 2, 3]))

    np.testing.assert_array_equal(
        result,
        np.array([False, True, True, True], dtype=bool),
    )


def test_derive_has_tin_with_itin():
    ssn_codes = np.array([0, 0, 1, 2, 3])
    has_itin = np.array([True, False, False, False, False])

    result = _derive_has_tin_from_ssn_card_type_codes(ssn_codes, has_itin)

    # code-0 with ITIN → True, code-0 without ITIN → False, others → True
    np.testing.assert_array_equal(
        result,
        np.array([True, False, True, True, True], dtype=bool),
    )


def test_store_identification_variables_writes_has_tin_and_alias():
    cps = {}

    _store_identification_variables(cps, np.array([0, 1, 2, 3]))

    assert cps["ssn_card_type"].tolist() == [
        b"NONE",
        b"CITIZEN",
        b"NON_CITIZEN_VALID_EAD",
        b"OTHER_NON_CITIZEN",
    ]
    np.testing.assert_array_equal(
        cps["has_tin"],
        np.array([False, True, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(cps["has_itin"], cps["has_tin"])


def test_store_identification_variables_with_itin():
    cps = {}
    ssn_codes = np.array([0, 0, 1, 2, 3])
    has_itin = np.array([True, False, False, False, False])

    _store_identification_variables(cps, ssn_codes, has_itin)

    # code-0 with ITIN gets has_tin=True
    assert cps["has_tin"][0] == True  # noqa: E712
    # code-0 without ITIN gets has_tin=False
    assert cps["has_tin"][1] == False  # noqa: E712
    # SSN holders still have has_tin=True
    assert cps["has_tin"][2] == True  # noqa: E712
    # alias still matches
    np.testing.assert_array_equal(cps["has_itin"], cps["has_tin"])
