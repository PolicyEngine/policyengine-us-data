from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


STRUCTURAL_MORTGAGE_VARIABLES = (
    "first_home_mortgage_balance",
    "second_home_mortgage_balance",
    "first_home_mortgage_interest",
    "second_home_mortgage_interest",
    "first_home_mortgage_origination_year",
    "second_home_mortgage_origination_year",
)

MORTGAGE_HINT_VARIABLES = (
    "imputed_first_home_mortgage_balance_hint",
    "imputed_second_home_mortgage_balance_hint",
)

MORTGAGE_IMPUTATION_PREDICTORS = [
    "age",
    "is_female",
    "cps_race",
    "is_married",
    "own_children_in_household",
    "employment_income",
    "interest_dividend_income",
    "social_security_pension_income",
    "mortgage_owner_status",
]


def impute_tax_unit_mortgage_balance_hints(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Impute tax-unit mortgage balance hints from SCF data.

    The output variables are not policyengine-us inputs. They are auxiliary
    data-layer hints that let the structural MID conversion reuse an SCF-like
    mortgage balance distribution without forcing the baseline to use mortgage
    interest for non-itemizers.

    The second hint is a generic secondary acquisition-debt slot. In the
    public SCF, HELOC balances are the best observable proxy for that slot even
    though the downstream ``second_home_mortgage_*`` variables in
    policyengine-us are named around a second home.
    """
    receiver = _build_tax_unit_mortgage_receiver(data, time_period)
    if receiver.empty:
        return data

    from microimpute.models.qrf import QRF
    from policyengine_us_data.datasets.scf.scf import SCF_2022

    scf = pd.DataFrame(SCF_2022().load_dataset())
    donor = _build_scf_mortgage_donor(scf)
    if donor.empty:
        return data

    qrf = QRF()
    donor_sample = donor.sample(frac=0.5, random_state=42).reset_index(drop=True)
    fitted = qrf.fit(
        X_train=donor_sample,
        predictors=MORTGAGE_IMPUTATION_PREDICTORS,
        imputed_variables=list(MORTGAGE_HINT_VARIABLES),
        weight_col="wgt",
        tune_hyperparameters=False,
    )
    predictions = fitted.predict(X_test=receiver[MORTGAGE_IMPUTATION_PREDICTORS])

    owner_with_mortgage = receiver["mortgage_owner_status"].values == 2
    first_hint = np.where(
        owner_with_mortgage,
        np.maximum(
            predictions["imputed_first_home_mortgage_balance_hint"].values,
            0,
        ),
        0,
    ).astype(np.float32)
    second_hint = np.where(
        owner_with_mortgage,
        np.maximum(
            predictions["imputed_second_home_mortgage_balance_hint"].values,
            0,
        ),
        0,
    ).astype(np.float32)

    swap_mask = (first_hint == 0) & (second_hint > 0)
    first_hint[swap_mask] = second_hint[swap_mask]
    second_hint[swap_mask] = 0

    data["imputed_first_home_mortgage_balance_hint"] = {time_period: first_hint}
    data["imputed_second_home_mortgage_balance_hint"] = {time_period: second_hint}
    return data


def convert_mortgage_interest_to_structural_inputs(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Replace formula-level mortgage inputs with structural mortgage data.

    The current us-data calibration pipeline imputes a person-level
    ``deductible_mortgage_interest`` and a tax-unit-level
    ``interest_deduction``. That short-circuits structural MID reforms in
    policyengine-us, so this converts those imputed amounts into:

    * tax-unit mortgage balances, interest, and origination years
    * person-level ``home_mortgage_interest`` for within-tax-unit allocation
    * person-level ``investment_interest_expense`` for the residual non-mortgage
      interest share of ``interest_deduction``

    The conversion is intentionally conservative:
    * current-law deductible mortgage interest is preserved exactly
    * current-law total interest deduction is preserved exactly
    * SCF-imputed first-lien and HELOC splits are preserved when available
    * weak balance hints are lifted to a conservative lower bound implied by
      the observed deductible mortgage interest
    * the origination year is heuristic, because the current public pipeline
      does not carry a mortgage-vintage input

    The structural model has two mortgage slots. In public data, we use those
    slots for "first-lien" and "secondary acquisition debt" rather than trying
    to identify literal primary-residence versus second-home mortgages.
    """
    tp = time_period
    person_ids = data.get("person_id", {}).get(tp)
    tax_unit_ids = data.get("tax_unit_id", {}).get(tp)
    person_tax_unit_ids = data.get("person_tax_unit_id", {}).get(tp)
    filing_status = data.get("filing_status", {}).get(tp)

    if (
        person_ids is None
        or tax_unit_ids is None
        or person_tax_unit_ids is None
        or filing_status is None
    ):
        return data

    n_persons = len(person_ids)
    n_tax_units = len(tax_unit_ids)
    tax_unit_index = {
        int(tax_unit_id): idx for idx, tax_unit_id in enumerate(tax_unit_ids)
    }
    person_tax_unit_idx = np.array(
        [tax_unit_index[int(tax_unit_id)] for tax_unit_id in person_tax_unit_ids],
        dtype=np.int32,
    )

    person_deductible = _get_person_mortgage_interest_target(data, tp, n_persons)
    tax_unit_deductible = np.zeros(n_tax_units, dtype=np.float32)
    np.add.at(tax_unit_deductible, person_tax_unit_idx, person_deductible)
    (
        first_balance_hint,
        second_balance_hint,
    ) = _get_tax_unit_mortgage_balance_hints(data, tp, n_tax_units)
    hinted_total_balance = np.maximum(first_balance_hint + second_balance_hint, 0)
    balance_floor = _interest_implied_balance_floor(tax_unit_deductible, tp)

    total_interest_deduction = _get_tax_unit_interest_deduction_target(
        data,
        tp,
        tax_unit_deductible,
    )

    fallback_person_share = _filer_share(data, tp, person_tax_unit_idx, n_tax_units)
    person_share = _normalize_person_share(
        person_deductible,
        person_tax_unit_idx,
        n_tax_units,
        fallback_person_share,
    )

    tax_unit_age = _tax_unit_age(data, tp, person_tax_unit_idx, n_tax_units)
    filing_status_str = np.array(
        [_decode_filing_status(value) for value in filing_status]
    )

    post_cap = np.array(
        [_post_tcja_cap(status) for status in filing_status_str],
        dtype=np.float32,
    )
    pre_cap = np.array(
        [_pre_tcja_cap(status) for status in filing_status_str],
        dtype=np.float32,
    )

    has_mortgage = tax_unit_deductible > 0
    hinted_balance = np.maximum(hinted_total_balance, balance_floor)
    balance, origination_year = _estimate_mortgage_balance_and_year(
        tax_unit_ids,
        tax_unit_deductible,
        post_cap,
        tax_unit_age,
        tp,
        hinted_balance,
    )
    use_balance_hint = hinted_total_balance > 0
    first_balance = np.where(use_balance_hint, first_balance_hint, balance).astype(
        np.float32
    )
    second_balance = np.where(use_balance_hint, second_balance_hint, 0).astype(
        np.float32
    )
    first_balance, second_balance = _apply_interest_implied_balance_floor(
        first_balance,
        second_balance,
        balance_floor,
    )

    swap_mask = (first_balance == 0) & (second_balance > 0)
    first_balance[swap_mask] = second_balance[swap_mask]
    second_balance[swap_mask] = 0
    total_balance = first_balance + second_balance

    applicable_cap = np.where(origination_year <= 2017, pre_cap, post_cap)
    deductible_share = np.ones(n_tax_units, dtype=np.float32)
    capped_mask = has_mortgage & (total_balance > applicable_cap)
    deductible_share[capped_mask] = (
        applicable_cap[capped_mask] / total_balance[capped_mask]
    )

    total_mortgage_interest = np.zeros(n_tax_units, dtype=np.float32)
    positive_share = has_mortgage & (deductible_share > 0)
    total_mortgage_interest[positive_share] = (
        tax_unit_deductible[positive_share] / deductible_share[positive_share]
    )
    first_interest, second_interest = _split_interest_by_balance(
        total_mortgage_interest,
        first_balance,
        second_balance,
    )
    second_origination_year = np.where(
        second_balance > 0,
        # The public data's second slot is mainly a HELOC/secondary-debt proxy,
        # so treat it as post-TCJA unless a richer vintage input becomes
        # available.
        np.maximum(2018, origination_year),
        0,
    ).astype(np.int32)

    investment_interest = np.maximum(
        total_interest_deduction - tax_unit_deductible,
        0,
    ).astype(np.float32)

    person_home_mortgage_interest = (
        total_mortgage_interest[person_tax_unit_idx] * person_share
    ).astype(np.float32)
    person_investment_interest = (
        investment_interest[person_tax_unit_idx] * fallback_person_share
    ).astype(np.float32)

    data["first_home_mortgage_balance"] = {tp: first_balance}
    data["second_home_mortgage_balance"] = {tp: second_balance}
    data["first_home_mortgage_interest"] = {tp: first_interest}
    data["second_home_mortgage_interest"] = {tp: second_interest}
    data["first_home_mortgage_origination_year"] = {
        tp: origination_year.astype(np.int32)
    }
    data["second_home_mortgage_origination_year"] = {tp: second_origination_year}
    data["home_mortgage_interest"] = {tp: person_home_mortgage_interest}
    data["investment_interest_expense"] = {tp: person_investment_interest}

    data.pop("deductible_mortgage_interest", None)
    data.pop("interest_deduction", None)
    return data


def _get_person_mortgage_interest_target(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    n_persons: int,
) -> np.ndarray:
    if "deductible_mortgage_interest" in data:
        values = np.asarray(
            data["deductible_mortgage_interest"][time_period],
            dtype=np.float32,
        )
        return np.maximum(values, 0)
    return np.zeros(n_persons, dtype=np.float32)


def _get_tax_unit_interest_deduction_target(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    tax_unit_deductible: np.ndarray,
) -> np.ndarray:
    if "interest_deduction" not in data:
        return tax_unit_deductible.astype(np.float32)
    values = np.asarray(data["interest_deduction"][time_period], dtype=np.float32)
    return np.maximum(values, tax_unit_deductible).astype(np.float32)


def _get_tax_unit_mortgage_balance_hints(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    n_tax_units: int,
) -> tuple[np.ndarray, np.ndarray]:
    first_hint = np.asarray(
        data.get("imputed_first_home_mortgage_balance_hint", {}).get(
            time_period, np.zeros(n_tax_units)
        ),
        dtype=np.float32,
    )
    second_hint = np.asarray(
        data.get("imputed_second_home_mortgage_balance_hint", {}).get(
            time_period, np.zeros(n_tax_units)
        ),
        dtype=np.float32,
    )
    return np.maximum(first_hint, 0), np.maximum(second_hint, 0)


def _build_tax_unit_mortgage_receiver(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
) -> pd.DataFrame:
    tax_unit_ids = data.get("tax_unit_id", {}).get(time_period)
    person_tax_unit_ids = data.get("person_tax_unit_id", {}).get(time_period)
    is_head = data.get("is_tax_unit_head", {}).get(time_period)
    if tax_unit_ids is None or person_tax_unit_ids is None or is_head is None:
        return pd.DataFrame()

    tax_unit_ids = np.asarray(tax_unit_ids)
    person_tax_unit_ids = np.asarray(person_tax_unit_ids)
    is_head = np.asarray(is_head, dtype=bool)
    n_tax_units = len(tax_unit_ids)
    tax_unit_index = {
        int(tax_unit_id): idx for idx, tax_unit_id in enumerate(tax_unit_ids)
    }
    person_tax_unit_idx = np.array(
        [tax_unit_index[int(tax_unit_id)] for tax_unit_id in person_tax_unit_ids],
        dtype=np.int32,
    )

    head_index = np.full(n_tax_units, -1, dtype=np.int32)
    head_positions = np.flatnonzero(is_head)
    if head_positions.size > 0:
        head_index[person_tax_unit_idx[head_positions]] = head_positions

    missing_head = head_index < 0
    if np.any(missing_head):
        first_person = np.full(n_tax_units, -1, dtype=np.int32)
        for person_idx, tax_unit_idx in enumerate(person_tax_unit_idx):
            if first_person[tax_unit_idx] < 0:
                first_person[tax_unit_idx] = person_idx
        head_index[missing_head] = first_person[missing_head]

    receiver = pd.DataFrame(
        {
            "tax_unit_id": tax_unit_ids,
            "head_index": head_index,
        }
    )
    head_take = head_index.clip(min=0)

    receiver["age"] = _take_person_values(data, time_period, "age", head_take)
    is_male = _take_person_values(data, time_period, "is_male", head_take)
    receiver["is_female"] = (1 - is_male).astype(np.float32)
    receiver["cps_race"] = _take_person_values(
        data, time_period, "cps_race", head_take
    ).astype(np.float32)
    receiver["own_children_in_household"] = _take_person_values(
        data, time_period, "own_children_in_household", head_take
    )
    receiver["mortgage_owner_status"] = _tax_unit_mortgage_owner_status(
        data,
        time_period,
        head_take,
    )

    spouse_count = np.zeros(n_tax_units, dtype=np.float32)
    spouse = np.asarray(
        data.get("is_tax_unit_spouse", {}).get(
            time_period, np.zeros(len(person_tax_unit_idx))
        ),
        dtype=np.float32,
    )
    np.add.at(spouse_count, person_tax_unit_idx, spouse)
    receiver["is_married"] = (spouse_count > 0).astype(np.float32)

    receiver["employment_income"] = _sum_person_values_to_tax_unit(
        data,
        time_period,
        person_tax_unit_idx,
        n_tax_units,
        ["employment_income"],
    )
    receiver["interest_dividend_income"] = _sum_person_values_to_tax_unit(
        data,
        time_period,
        person_tax_unit_idx,
        n_tax_units,
        [
            "taxable_interest_income",
            "tax_exempt_interest_income",
            "qualified_dividend_income",
            "non_qualified_dividend_income",
        ],
    )
    receiver["social_security_pension_income"] = _sum_person_values_to_tax_unit(
        data,
        time_period,
        person_tax_unit_idx,
        n_tax_units,
        [
            "social_security_retirement",
            "taxable_private_pension_income",
            "tax_exempt_private_pension_income",
        ],
    )
    return receiver[MORTGAGE_IMPUTATION_PREDICTORS]


def _build_scf_mortgage_donor(scf: pd.DataFrame) -> pd.DataFrame:
    donor = pd.DataFrame()
    donor["age"] = _frame_column(scf, "age")
    donor["is_female"] = _frame_column(scf, "is_female")
    donor["cps_race"] = _frame_column(scf, "cps_race")
    donor["is_married"] = _frame_column(scf, "is_married")
    donor["own_children_in_household"] = _frame_column(scf, "own_children_in_household")
    donor["employment_income"] = _frame_column(scf, "employment_income")
    donor["interest_dividend_income"] = _frame_column(scf, "interest_dividend_income")
    donor["social_security_pension_income"] = _frame_column(
        scf, "social_security_pension_income"
    )

    total_mortgage = np.maximum(
        np.asarray(scf.get("nh_mort", 0), dtype=np.float32),
        np.asarray(scf.get("mortgage_debt", 0), dtype=np.float32),
    )
    heloc = np.minimum(
        np.maximum(np.asarray(scf.get("heloc", 0), dtype=np.float32), 0),
        total_mortgage,
    )
    owns_home = np.asarray(scf.get("houses", 0), dtype=np.float32) > 0
    has_mortgage = total_mortgage > 0

    donor["mortgage_owner_status"] = np.where(
        has_mortgage,
        2,
        np.where(owns_home, 1, 0),
    ).astype(np.float32)
    # The second slot is not a literal second-home mortgage in SCF. We use
    # HELOC balances as the best public proxy for secondary acquisition debt.
    donor["imputed_first_home_mortgage_balance_hint"] = np.maximum(
        total_mortgage - heloc,
        0,
    ).astype(np.float32)
    donor["imputed_second_home_mortgage_balance_hint"] = heloc.astype(np.float32)
    donor["wgt"] = _frame_column(scf, "wgt", default=1)
    return donor[
        MORTGAGE_IMPUTATION_PREDICTORS + list(MORTGAGE_HINT_VARIABLES) + ["wgt"]
    ].dropna()


def _take_person_values(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    variable: str,
    head_take: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        data.get(variable, {}).get(time_period, np.zeros(head_take.size)),
        dtype=np.float32,
    )
    if values.size == 0:
        return np.zeros(head_take.size, dtype=np.float32)
    return values[head_take].astype(np.float32)


def _frame_column(
    frame: pd.DataFrame,
    column: str,
    default: float = 0,
) -> np.ndarray:
    if column in frame:
        return np.asarray(frame[column], dtype=np.float32)
    return np.full(len(frame), default, dtype=np.float32)


def _sum_person_values_to_tax_unit(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    person_tax_unit_idx: np.ndarray,
    n_tax_units: int,
    variables: list[str],
) -> np.ndarray:
    total = np.zeros(n_tax_units, dtype=np.float32)
    for variable in variables:
        if variable not in data:
            continue
        values = np.asarray(data[variable][time_period], dtype=np.float32)
        np.add.at(total, person_tax_unit_idx, values)
    return total


def _tax_unit_mortgage_owner_status(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    head_take: np.ndarray,
) -> np.ndarray:
    household_status = np.zeros(head_take.size, dtype=np.float32)
    household_tenure = data.get("tenure_type", {}).get(time_period)
    person_household_id = data.get("person_household_id", {}).get(time_period)
    household_ids = data.get("household_id", {}).get(time_period)
    if (
        household_tenure is not None
        and person_household_id is not None
        and household_ids is not None
    ):
        household_map = {
            int(household_id): _decode_owner_status(value)
            for household_id, value in zip(household_ids, household_tenure)
        }
        household_status = np.array(
            [
                household_map.get(int(household_id), 0)
                for household_id in np.asarray(person_household_id)[head_take]
            ],
            dtype=np.float32,
        )

    spm_status = np.zeros(head_take.size, dtype=np.float32)
    spm_tenure = data.get("spm_unit_tenure_type", {}).get(time_period)
    person_spm_unit_id = data.get("person_spm_unit_id", {}).get(time_period)
    spm_unit_ids = data.get("spm_unit_id", {}).get(time_period)
    if (
        spm_tenure is not None
        and person_spm_unit_id is not None
        and spm_unit_ids is not None
    ):
        spm_map = {
            int(spm_unit_id): _decode_owner_status(value)
            for spm_unit_id, value in zip(spm_unit_ids, spm_tenure)
        }
        spm_status = np.array(
            [
                spm_map.get(int(spm_unit_id), 0)
                for spm_unit_id in np.asarray(person_spm_unit_id)[head_take]
            ],
            dtype=np.float32,
        )

    return np.where(spm_status > 0, spm_status, household_status).astype(np.float32)


def _decode_owner_status(value) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).upper()
    if "OWNER_WITH_MORTGAGE" in value or "OWNED_WITH_MORTGAGE" in value:
        return 2
    if "OWNER_WITHOUT_MORTGAGE" in value:
        return 1
    return 0


def _filer_share(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    person_tax_unit_idx: np.ndarray,
    n_tax_units: int,
) -> np.ndarray:
    is_head = np.asarray(
        data.get("is_tax_unit_head", {}).get(
            time_period, np.zeros(len(person_tax_unit_idx), dtype=bool)
        ),
        dtype=bool,
    )
    is_spouse = np.asarray(
        data.get("is_tax_unit_spouse", {}).get(
            time_period, np.zeros(len(person_tax_unit_idx), dtype=bool)
        ),
        dtype=bool,
    )
    filer_mask = (is_head | is_spouse).astype(np.float32)
    filer_count = np.zeros(n_tax_units, dtype=np.float32)
    np.add.at(filer_count, person_tax_unit_idx, filer_mask)

    share = np.zeros(len(person_tax_unit_idx), dtype=np.float32)
    positive_filers = filer_count[person_tax_unit_idx] > 0
    share[positive_filers] = (
        filer_mask[positive_filers] / filer_count[person_tax_unit_idx][positive_filers]
    )

    no_filer_mask = filer_count[person_tax_unit_idx] == 0
    if np.any(no_filer_mask):
        share[no_filer_mask] = _equal_person_share(
            person_tax_unit_idx[no_filer_mask],
            n_tax_units,
        )

    return share


def _normalize_person_share(
    person_values: np.ndarray,
    person_tax_unit_idx: np.ndarray,
    n_tax_units: int,
    fallback_share: np.ndarray,
) -> np.ndarray:
    tax_unit_totals = np.zeros(n_tax_units, dtype=np.float32)
    np.add.at(tax_unit_totals, person_tax_unit_idx, person_values)
    share = np.zeros_like(person_values, dtype=np.float32)
    positive = tax_unit_totals[person_tax_unit_idx] > 0
    share[positive] = (
        person_values[positive] / tax_unit_totals[person_tax_unit_idx][positive]
    )
    share[~positive] = fallback_share[~positive]
    return share


def _equal_person_share(
    person_tax_unit_idx: np.ndarray,
    n_tax_units: int,
) -> np.ndarray:
    counts = np.zeros(n_tax_units, dtype=np.float32)
    np.add.at(counts, person_tax_unit_idx, 1)
    return (1 / counts[person_tax_unit_idx]).astype(np.float32)


def _tax_unit_age(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    person_tax_unit_idx: np.ndarray,
    n_tax_units: int,
) -> np.ndarray:
    ages = np.asarray(
        data.get("age", {}).get(time_period, np.zeros(len(person_tax_unit_idx))),
        dtype=np.float32,
    )
    is_head = np.asarray(
        data.get("is_tax_unit_head", {}).get(
            time_period, np.zeros(len(person_tax_unit_idx), dtype=bool)
        ),
        dtype=bool,
    )
    is_spouse = np.asarray(
        data.get("is_tax_unit_spouse", {}).get(
            time_period, np.zeros(len(person_tax_unit_idx), dtype=bool)
        ),
        dtype=bool,
    )
    filer_ages = np.where(is_head | is_spouse, ages, 0)
    tax_unit_age = np.zeros(n_tax_units, dtype=np.float32)
    np.maximum.at(tax_unit_age, person_tax_unit_idx, filer_ages)

    missing_age = tax_unit_age == 0
    if np.any(missing_age):
        any_age = np.zeros(n_tax_units, dtype=np.float32)
        np.maximum.at(any_age, person_tax_unit_idx, ages)
        tax_unit_age[missing_age] = any_age[missing_age]

    tax_unit_age[missing_age & (tax_unit_age == 0)] = 45
    return tax_unit_age


def _estimate_mortgage_balance_and_year(
    tax_unit_ids: np.ndarray,
    deductible_mortgage_interest: np.ndarray,
    post_cap: np.ndarray,
    tax_unit_age: np.ndarray,
    time_period: int,
    hinted_balance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    balance = np.zeros_like(deductible_mortgage_interest, dtype=np.float32)
    year = np.zeros_like(deductible_mortgage_interest, dtype=np.int32)
    has_mortgage = (deductible_mortgage_interest > 0) | (hinted_balance > 0)
    if not np.any(has_mortgage):
        return balance, year

    older_draw = _stable_uniform(tax_unit_ids, salt=17)
    year_draw = _stable_uniform(tax_unit_ids, salt=31)

    pre_probability = np.clip(
        0.10 + 0.012 * np.maximum(tax_unit_age - 30, 0),
        0.10,
        0.85,
    )

    provisional_rate = 0.045
    provisional_balance = np.where(
        hinted_balance > 0,
        hinted_balance,
        deductible_mortgage_interest / provisional_rate,
    )
    pre_probability += 0.20 * (provisional_balance > post_cap)
    pre_probability = np.clip(pre_probability, 0.10, 0.90)

    if time_period <= 2017:
        is_pre_tcja = has_mortgage
    else:
        is_pre_tcja = has_mortgage & (older_draw < pre_probability)

    pre_span = 13  # 2005-2017 inclusive
    year[is_pre_tcja] = 2005 + np.floor(year_draw[is_pre_tcja] * pre_span).astype(
        np.int32
    )

    post_mask = has_mortgage & ~is_pre_tcja
    post_start = 2018 if time_period >= 2018 else time_period
    post_span = max(1, time_period - post_start + 1)
    year[post_mask] = post_start + np.floor(year_draw[post_mask] * post_span).astype(
        np.int32
    )

    rate = _mortgage_rate(year)
    balance[has_mortgage] = np.where(
        hinted_balance[has_mortgage] > 0,
        hinted_balance[has_mortgage],
        deductible_mortgage_interest[has_mortgage] / rate[has_mortgage],
    )
    return balance, year


def _interest_implied_balance_floor(
    deductible_mortgage_interest: np.ndarray,
    time_period: int,
) -> np.ndarray:
    """Conservative balance lower bound implied by deductible interest.

    Uses the current-period market mortgage rate as the denominator, so the
    inferred balance is a lower bound rather than an aggressive reconstruction
    of total acquisition debt.
    """
    current_market_rate = float(
        _mortgage_rate(np.array([time_period], dtype=np.int32))[0]
    )
    if current_market_rate <= 0:
        return np.zeros_like(deductible_mortgage_interest, dtype=np.float32)
    return np.where(
        deductible_mortgage_interest > 0,
        deductible_mortgage_interest / current_market_rate,
        0,
    ).astype(np.float32)


def _apply_interest_implied_balance_floor(
    first_balance: np.ndarray,
    second_balance: np.ndarray,
    balance_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prevent donor balance hints from understating observed mortgage interest."""
    first_balance = np.asarray(first_balance, dtype=np.float32).copy()
    second_balance = np.asarray(second_balance, dtype=np.float32).copy()
    balance_floor = np.maximum(np.asarray(balance_floor, dtype=np.float32), 0)

    total_balance = first_balance + second_balance
    needs_floor = balance_floor > total_balance
    with_existing_split = needs_floor & (total_balance > 0)

    scale = np.ones_like(total_balance, dtype=np.float32)
    scale[with_existing_split] = (
        balance_floor[with_existing_split] / total_balance[with_existing_split]
    )
    first_balance[with_existing_split] *= scale[with_existing_split]
    second_balance[with_existing_split] *= scale[with_existing_split]

    no_existing_balance = needs_floor & (total_balance == 0)
    first_balance[no_existing_balance] = balance_floor[no_existing_balance]
    second_balance[no_existing_balance] = 0

    return first_balance.astype(np.float32), second_balance.astype(np.float32)


def _split_interest_by_balance(
    total_interest: np.ndarray,
    first_balance: np.ndarray,
    second_balance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_balance = first_balance + second_balance
    first_interest = np.zeros_like(total_interest, dtype=np.float32)
    second_interest = np.zeros_like(total_interest, dtype=np.float32)

    with_second = total_balance > 0
    first_interest[with_second] = (
        total_interest[with_second]
        * first_balance[with_second]
        / total_balance[with_second]
    )
    second_interest[with_second] = (
        total_interest[with_second] - first_interest[with_second]
    )

    no_second = second_balance == 0
    first_interest[no_second] = total_interest[no_second]
    second_interest[no_second] = 0
    return first_interest.astype(np.float32), second_interest.astype(np.float32)


def _mortgage_rate(origination_year: np.ndarray) -> np.ndarray:
    year = np.asarray(origination_year, dtype=np.int32)
    rate = np.full(year.shape, 0.045, dtype=np.float32)
    rate[year <= 2017] = 0.040
    rate[(year >= 2018) & (year <= 2019)] = 0.045
    rate[(year >= 2020) & (year <= 2021)] = 0.035
    rate[year == 2022] = 0.0525
    rate[year >= 2023] = 0.0675
    return rate


def _stable_uniform(ids: np.ndarray, salt: int) -> np.ndarray:
    values = np.asarray(ids, dtype=np.uint64)
    hashed = values * np.uint64(1_103_515_245 + salt) + np.uint64(12_345 + salt)
    return ((hashed % np.uint64(2**31)).astype(np.float64) / float(2**31)).astype(
        np.float32
    )


def _decode_filing_status(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").upper()
    return str(value).upper()


def _post_tcja_cap(status: str) -> float:
    if "SEPARATE" in status:
        return 375_000.0
    return 750_000.0


def _pre_tcja_cap(status: str) -> float:
    if "SEPARATE" in status:
        return 500_000.0
    return 1_000_000.0
