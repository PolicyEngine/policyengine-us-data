import numpy as np
import pandas as pd


SSN_CARD_TYPE_CODE_TO_STR = {
    0: "NONE",
    1: "CITIZEN",
    2: "NON_CITIZEN_VALID_EAD",
    3: "OTHER_NON_CITIZEN",
}


def _derive_has_tin_from_ssn_card_type_codes(ssn_card_type: np.ndarray) -> np.ndarray:
    """Return whether a person has any taxpayer ID from CPS ID status codes."""
    return np.asarray(ssn_card_type) != 0


def _store_identification_variables(cps: dict, ssn_card_type: np.ndarray) -> None:
    """Persist identification inputs used by PolicyEngine US."""
    has_tin = _derive_has_tin_from_ssn_card_type_codes(ssn_card_type)
    cps["ssn_card_type"] = (
        pd.Series(ssn_card_type).map(SSN_CARD_TYPE_CODE_TO_STR).astype("S").values
    )
    cps["has_tin"] = has_tin
    # Temporary compatibility alias while policyengine-us users migrate.
    cps["has_itin"] = has_tin
