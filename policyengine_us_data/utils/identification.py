import numpy as np
import pandas as pd


SSN_CARD_TYPE_CODE_TO_STR = {
    0: "NONE",
    1: "CITIZEN",
    2: "NON_CITIZEN_VALID_EAD",
    3: "OTHER_NON_CITIZEN",
}


def _derive_has_valid_ssn_from_ssn_card_type_codes(
    ssn_card_type: np.ndarray,
) -> np.ndarray:
    """Return direct valid-SSN evidence from CPS ID status codes."""
    ssn_card_type = np.asarray(ssn_card_type)
    return ssn_card_type == 1


def _derive_taxpayer_id_type_from_identification_flags(
    has_valid_ssn: np.ndarray,
    has_tin: np.ndarray,
) -> np.ndarray:
    """Return statute-facing taxpayer ID classes from ID flags."""
    return np.where(
        has_valid_ssn,
        "VALID_SSN",
        np.where(has_tin, "OTHER_TIN", "NONE"),
    )


def _high_confidence_tin_evidence(person: pd.DataFrame) -> np.ndarray:
    """Return admin-linked signals that strongly imply TIN possession."""
    social_security = (
        (person.SS_YN == 1)
        | np.isin(person.RESNSS1, [1, 2, 3, 4, 5, 6, 7])
        | np.isin(person.RESNSS2, [1, 2, 3, 4, 5, 6, 7])
    )
    medicare = person.MCARE == 1
    federal_pension = np.isin(person.PEN_SC1, [3]) | np.isin(person.PEN_SC2, [3])
    government_worker = np.isin(person.PEIO1COW, [1, 2, 3]) | (person.A_MJOCC == 11)
    military_link = (person.MIL == 1) | (person.PEAFEVER == 1) | (person.CHAMPVA == 1)
    ssi = person.SSI_YN == 1
    return (
        social_security
        | medicare
        | federal_pension
        | government_worker
        | military_link
        | ssi
    ).to_numpy(dtype=bool)


def _derive_has_tin_from_identification_inputs(
    person: pd.DataFrame,
    ssn_card_type: np.ndarray,
    has_itin_number: np.ndarray | None = None,
) -> np.ndarray:
    """Return broad TIN possession without treating proxy codes as direct IDs."""
    has_valid_ssn = _derive_has_valid_ssn_from_ssn_card_type_codes(ssn_card_type)
    has_tin = has_valid_ssn.copy()
    has_tin |= ~has_valid_ssn & _high_confidence_tin_evidence(person)
    if has_itin_number is not None:
        has_tin |= np.asarray(has_itin_number, dtype=bool)
    return has_tin


def _store_identification_variables(
    cps: dict,
    person: pd.DataFrame,
    ssn_card_type: np.ndarray,
    has_itin_number: np.ndarray | None = None,
) -> None:
    """Persist identification inputs used by PolicyEngine US."""
    has_valid_ssn = _derive_has_valid_ssn_from_ssn_card_type_codes(ssn_card_type)
    has_tin = _derive_has_tin_from_identification_inputs(
        person=person,
        ssn_card_type=ssn_card_type,
        has_itin_number=has_itin_number,
    )
    taxpayer_id_type = _derive_taxpayer_id_type_from_identification_flags(
        has_valid_ssn=has_valid_ssn,
        has_tin=has_tin,
    )
    cps["ssn_card_type"] = (
        pd.Series(ssn_card_type).map(SSN_CARD_TYPE_CODE_TO_STR).astype("S").values
    )
    cps["taxpayer_id_type"] = pd.Series(taxpayer_id_type).astype("S").values
    cps["has_tin"] = has_tin
    cps["has_valid_ssn"] = has_valid_ssn
    # Temporary compatibility alias while policyengine-us users migrate.
    cps["has_itin"] = has_tin
