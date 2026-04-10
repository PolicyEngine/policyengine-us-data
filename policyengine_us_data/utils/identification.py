import numpy as np
import pandas as pd


SSN_CARD_TYPE_CODE_TO_STR = {
    0: "NONE",
    1: "CITIZEN",
    2: "NON_CITIZEN_VALID_EAD",
    3: "OTHER_NON_CITIZEN",
}


def _derive_has_tin_from_ssn_card_type_codes(
    ssn_card_type: np.ndarray,
    has_itin_number: np.ndarray | None = None,
) -> np.ndarray:
    """Return whether a person has any taxpayer ID from CPS ID status codes."""
    has_ssn = np.asarray(ssn_card_type) != 0
    if has_itin_number is not None:
        return has_ssn | np.asarray(has_itin_number)
    return has_ssn


def _store_identification_variables(
    cps: dict,
    ssn_card_type: np.ndarray,
    has_itin_number: np.ndarray | None = None,
) -> None:
    """Persist identification inputs used by PolicyEngine US."""
    has_tin = _derive_has_tin_from_ssn_card_type_codes(ssn_card_type, has_itin_number)
    cps["ssn_card_type"] = (
        pd.Series(ssn_card_type).map(SSN_CARD_TYPE_CODE_TO_STR).astype("S").values
    )
    cps["has_tin"] = has_tin
    # Temporary compatibility alias while policyengine-us users migrate.
    cps["has_itin"] = has_tin
