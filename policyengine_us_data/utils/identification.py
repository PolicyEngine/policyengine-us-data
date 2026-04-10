import numpy as np
import pandas as pd


SSN_CARD_TYPE_CODE_TO_STR = {
    0: "NONE",
    1: "CITIZEN",
    2: "NON_CITIZEN_VALID_EAD",
    3: "OTHER_NON_CITIZEN",
}


def _derive_taxpayer_id_type_from_ssn_card_type_codes(
    ssn_card_type: np.ndarray,
) -> np.ndarray:
    """Return statute-facing taxpayer ID classes from CPS ID status codes."""
    ssn_card_type = np.asarray(ssn_card_type)
    return np.where(
        ssn_card_type == 0,
        "NONE",
        np.where(
            np.isin(ssn_card_type, [1, 2]),
            "VALID_SSN",
            "OTHER_TIN",
        ),
    )


def _derive_has_tin_from_ssn_card_type_codes(ssn_card_type: np.ndarray) -> np.ndarray:
    """Return whether a person has any taxpayer ID from CPS ID status codes."""
    return _derive_taxpayer_id_type_from_ssn_card_type_codes(ssn_card_type) != "NONE"


def _derive_has_valid_ssn_from_ssn_card_type_codes(
    ssn_card_type: np.ndarray,
) -> np.ndarray:
    """Return whether a person has a valid SSN for SSN-gated federal tax rules."""
    return (
        _derive_taxpayer_id_type_from_ssn_card_type_codes(ssn_card_type) == "VALID_SSN"
    )


def _store_identification_variables(cps: dict, ssn_card_type: np.ndarray) -> None:
    """Persist identification inputs used by PolicyEngine US."""
    taxpayer_id_type = _derive_taxpayer_id_type_from_ssn_card_type_codes(ssn_card_type)
    has_tin = _derive_has_tin_from_ssn_card_type_codes(ssn_card_type)
    has_valid_ssn = _derive_has_valid_ssn_from_ssn_card_type_codes(ssn_card_type)
    cps["ssn_card_type"] = (
        pd.Series(ssn_card_type).map(SSN_CARD_TYPE_CODE_TO_STR).astype("S").values
    )
    cps["taxpayer_id_type"] = pd.Series(taxpayer_id_type).astype("S").values
    cps["has_tin"] = has_tin
    cps["has_valid_ssn"] = has_valid_ssn
    # Temporary compatibility alias while policyengine-us users migrate.
    cps["has_itin"] = has_tin
