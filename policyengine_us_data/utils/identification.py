import numpy as np
import pandas as pd


NON_SSN_FILER_TIN_TARGET_BY_YEAR = {
    # Latest available public IRS/TAS figure: about 3.8M TY 2023 returns
    # included an ITIN. Use it as a recent proxy for non-SSN filer TINs.
    2024: 3.8e6,
}

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


def _impute_has_valid_ssn(ssn_card_type: np.ndarray) -> np.ndarray:
    """Impute valid SSNs without treating EAD or documented-status proxies as IDs."""
    return _derive_has_valid_ssn_from_ssn_card_type_codes(ssn_card_type)


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


def _person_weights(cps: dict) -> np.ndarray:
    """Return person weights from household IDs and weights."""
    household_to_weight = dict(zip(cps["household_id"], cps["household_weight"]))
    return np.array(
        [
            household_to_weight.get(household_id, 0)
            for household_id in cps["person_household_id"]
        ],
        dtype=float,
    )


def _proxy_tax_unit_filers(
    person_tax_unit_ids: np.ndarray,
    age: np.ndarray,
) -> np.ndarray:
    """Proxy tax-unit head/spouse as the two oldest adults in each tax unit."""
    person_tax_unit_ids = np.asarray(person_tax_unit_ids)
    age = np.asarray(age)
    adult = age >= 18
    ranks = pd.Series(np.inf, index=np.arange(len(age)), dtype=float)
    if adult.any():
        adults = pd.DataFrame(
            {
                "tax_unit_id": person_tax_unit_ids[adult],
                "age": age[adult],
            },
            index=np.flatnonzero(adult),
        )
        ranks.loc[adults.index] = adults.groupby("tax_unit_id")["age"].rank(
            method="first",
            ascending=False,
        )
    return adult & (ranks.to_numpy() <= 2)


def _aggregate_by_tax_unit(
    values: np.ndarray,
    tax_unit_index: np.ndarray,
    n_tax_units: int,
) -> np.ndarray:
    total = np.zeros(n_tax_units, dtype=float)
    np.add.at(total, tax_unit_index, values)
    return total


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


def _impute_has_tin(
    cps: dict,
    person: pd.DataFrame,
    ssn_card_type: np.ndarray,
    time_period: int,
    non_ssn_filer_tin_target: float | None = None,
    has_valid_ssn: np.ndarray | None = None,
) -> np.ndarray:
    """Impute broad TIN possession without treating legal-status proxies as IDs."""
    ssn_card_type = np.asarray(ssn_card_type)
    if has_valid_ssn is None:
        has_valid_ssn = _impute_has_valid_ssn(ssn_card_type)
    has_tin = has_valid_ssn.copy()

    high_confidence_tin = ~has_valid_ssn & _high_confidence_tin_evidence(person)
    has_tin |= high_confidence_tin

    target = non_ssn_filer_tin_target
    if target is None:
        target = NON_SSN_FILER_TIN_TARGET_BY_YEAR.get(time_period)
    if target is None or target <= 0:
        return has_tin

    age = np.asarray(cps["age"])
    person_tax_unit_ids = np.asarray(cps["person_tax_unit_id"])
    tax_unit_ids, person_tax_unit_index = np.unique(
        person_tax_unit_ids,
        return_inverse=True,
    )
    n_tax_units = len(tax_unit_ids)
    person_weights = _person_weights(cps)
    tax_unit_weights = np.zeros(n_tax_units, dtype=float)
    np.maximum.at(tax_unit_weights, person_tax_unit_index, person_weights)

    proxy_filer = _proxy_tax_unit_filers(person_tax_unit_ids, age)
    non_ssn_proxy_filer = proxy_filer & ~has_valid_ssn

    current_non_ssn_tin_units = np.zeros(n_tax_units, dtype=bool)
    np.logical_or.at(
        current_non_ssn_tin_units,
        person_tax_unit_index,
        non_ssn_proxy_filer & has_tin,
    )
    current_weighted_units = tax_unit_weights[current_non_ssn_tin_units].sum()
    additional_target = target - current_weighted_units
    if additional_target <= 0:
        return has_tin

    employment_income = np.asarray(cps.get("employment_income", np.zeros(len(age))))
    self_employment_income = np.asarray(
        cps.get("self_employment_income", np.zeros(len(age)))
    )
    prior_year_income = np.asarray(
        cps.get("employment_income_last_year", np.zeros(len(age)))
    ) + np.asarray(cps.get("self_employment_income_last_year", np.zeros(len(age))))

    has_filing_income = (
        (employment_income > 0) | (self_employment_income > 0) | (prior_year_income > 0)
    )
    candidate_person = (
        non_ssn_proxy_filer & ~has_tin & (ssn_card_type == 0) & has_filing_income
    )
    candidate_units = np.zeros(n_tax_units, dtype=bool)
    np.logical_or.at(candidate_units, person_tax_unit_index, candidate_person)
    if not candidate_units.any():
        return has_tin

    unit_employment_income = _aggregate_by_tax_unit(
        np.maximum(employment_income, 0),
        person_tax_unit_index,
        n_tax_units,
    )
    unit_self_employment_income = _aggregate_by_tax_unit(
        np.maximum(self_employment_income, 0),
        person_tax_unit_index,
        n_tax_units,
    )
    unit_prior_year_income = _aggregate_by_tax_unit(
        np.maximum(prior_year_income, 0),
        person_tax_unit_index,
        n_tax_units,
    )
    unit_non_ssn_filer_count = _aggregate_by_tax_unit(
        candidate_person.astype(float),
        person_tax_unit_index,
        n_tax_units,
    )
    unit_has_minor = np.zeros(n_tax_units, dtype=bool)
    np.logical_or.at(unit_has_minor, person_tax_unit_index, age < 18)

    score = (
        4.0 * (unit_self_employment_income > 0)
        + 2.0 * (unit_employment_income > 0)
        + 1.0 * (unit_prior_year_income > 0)
        + 1.0 * unit_has_minor
        + 0.5 * (unit_non_ssn_filer_count > 1)
    )

    candidate_idx = np.flatnonzero(candidate_units)
    rng = np.random.default_rng(seed=17_000 + int(time_period))
    priority = score[candidate_idx] + rng.random(len(candidate_idx)) * 0.01
    ordered_idx = candidate_idx[np.argsort(-priority)]

    selected_units = np.zeros(n_tax_units, dtype=bool)
    cumulative_weight = 0.0
    for tax_unit_index in ordered_idx:
        if cumulative_weight >= additional_target:
            break
        selected_units[tax_unit_index] = True
        cumulative_weight += tax_unit_weights[tax_unit_index]

    selected_person_unit = selected_units[person_tax_unit_index]
    selected_non_ssn_filers = selected_person_unit & non_ssn_proxy_filer
    selected_minor_dependents = selected_person_unit & ~proxy_filer & (age < 18)
    has_tin |= selected_non_ssn_filers | (selected_minor_dependents & ~has_valid_ssn)
    return has_tin


def _store_identification_variables(
    cps: dict,
    person: pd.DataFrame,
    ssn_card_type: np.ndarray,
    time_period: int,
) -> None:
    """Persist identification inputs used by PolicyEngine US."""
    has_valid_ssn = _impute_has_valid_ssn(ssn_card_type)
    has_tin = _impute_has_tin(
        cps,
        person,
        ssn_card_type,
        time_period,
        has_valid_ssn=has_valid_ssn,
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
