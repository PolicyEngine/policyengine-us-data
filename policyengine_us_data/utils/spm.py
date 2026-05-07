"""SPM geographic adjustment utilities aligned with Census rent data."""

import numpy as np

TENURE_HOUSING_SHARES = {
    "owner_with_mortgage": 0.434,
    "owner_without_mortgage": 0.323,
    "renter": 0.443,
}

TENURE_CODE_MAP = {
    1: "owner_with_mortgage",
    2: "owner_without_mortgage",
    3: "renter",
}

SPM_TENURE_TYPE_TO_REFERENCE_KEY = {
    "OWNER_WITH_MORTGAGE": "owner_with_mortgage",
    "OWNER_WITHOUT_MORTGAGE": "owner_without_mortgage",
    "RENTER": "renter",
}


def calculate_geoadj_from_rent(
    local_rent,
    national_rent: float,
    tenure: str = "renter",
):
    share = TENURE_HOUSING_SHARES[tenure]
    rent_ratio = np.asarray(local_rent, dtype=float) / float(national_rent)
    return rent_ratio * share + (1.0 - share)


def reference_key_for_spm_tenure_type(tenure_type) -> str:
    if isinstance(tenure_type, (bytes, np.bytes_)):
        tenure_type = tenure_type.decode()
    reference_key = SPM_TENURE_TYPE_TO_REFERENCE_KEY.get(str(tenure_type).upper())
    if reference_key is None:
        raise ValueError(f"Unsupported spm_unit_tenure_type: {tenure_type!r}")
    return reference_key


def geoadj_for_tenure(geoadj_values, tenure_type) -> float:
    """Return a tenure-specific geoadj, accepting legacy scalar lookups."""
    if not isinstance(geoadj_values, dict):
        return float(geoadj_values)
    reference_key = (
        tenure_type
        if tenure_type in TENURE_HOUSING_SHARES
        else reference_key_for_spm_tenure_type(tenure_type)
    )
    return float(geoadj_values.get(reference_key, 1.0))
