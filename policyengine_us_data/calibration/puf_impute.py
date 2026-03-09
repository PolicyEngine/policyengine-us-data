"""PUF clone and QRF imputation for calibration pipeline.

Doubles CPS records: one half keeps original values, the other half
gets PUF tax variables imputed via Quantile Random Forest.

Usage within the calibration pipeline:
    1. Load raw CPS dataset
    2. Clone 10x and assign geography
    3. Call puf_clone_dataset() to double records and impute PUF vars
    4. Save expanded dataset for matrix building
"""

import gc
import logging
from importlib.resources import files
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from policyengine_us_data.utils.retirement_limits import (
    get_retirement_limits,
)

logger = logging.getLogger(__name__)

PUF_SUBSAMPLE_TARGET = 20_000
PUF_TOP_PERCENTILE = 99.5

DEMOGRAPHIC_PREDICTORS = [
    "age",
    "is_male",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
]

IMPUTED_VARIABLES = [
    "employment_income",
    "partnership_s_corp_income",
    "social_security",
    "taxable_pension_income",
    "interest_deduction",
    "tax_exempt_pension_income",
    "long_term_capital_gains",
    "unreimbursed_business_employee_expenses",
    "pre_tax_contributions",
    "taxable_ira_distributions",
    "self_employment_income",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",
    "short_term_capital_gains",
    "qualified_dividend_income",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "taxable_unemployment_compensation",
    "taxable_interest_income",
    "domestic_production_ald",
    "self_employed_health_insurance_ald",
    "rental_income",
    "non_qualified_dividend_income",
    "cdcc_relevant_expenses",
    "tax_exempt_interest_income",
    "salt_refund_income",
    "foreign_tax_credit",
    "estate_income",
    "charitable_non_cash_donations",
    "american_opportunity_credit",
    "miscellaneous_income",
    "alimony_expense",
    "farm_income",
    "partnership_se_income",
    "alimony_income",
    "health_savings_account_ald",
    "non_sch_d_capital_gains",
    "general_business_credit",
    "energy_efficient_home_improvement_credit",
    "amt_foreign_tax_credit",
    "excess_withheld_payroll_tax",
    "savers_credit",
    "student_loan_interest",
    "investment_income_elected_form_4952",
    "early_withdrawal_penalty",
    "prior_year_minimum_tax_credit",
    "farm_rent_income",
    "qualified_tuition_expenses",
    "educator_expense",
    "long_term_capital_gains_on_collectibles",
    "other_credits",
    "casualty_loss",
    "unreported_payroll_tax",
    "recapture_of_investment_credit",
    "deductible_mortgage_interest",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
    "farm_operations_income",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
    "self_employment_income_would_be_qualified",
]

SS_SUBCOMPONENTS = [
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
]

OVERRIDDEN_IMPUTED_VARIABLES = [
    "partnership_s_corp_income",
    "interest_deduction",
    "unreimbursed_business_employee_expenses",
    "pre_tax_contributions",
    "w2_wages_from_qualified_business",
    "unadjusted_basis_qualified_property",
    "business_is_sstb",
    "charitable_cash_donations",
    "self_employed_pension_contribution_ald",
    "unrecaptured_section_1250_gain",
    "taxable_unemployment_compensation",
    "domestic_production_ald",
    "self_employed_health_insurance_ald",
    "cdcc_relevant_expenses",
    "salt_refund_income",
    "foreign_tax_credit",
    "estate_income",
    "charitable_non_cash_donations",
    "american_opportunity_credit",
    "miscellaneous_income",
    "alimony_expense",
    "health_savings_account_ald",
    "non_sch_d_capital_gains",
    "general_business_credit",
    "energy_efficient_home_improvement_credit",
    "amt_foreign_tax_credit",
    "excess_withheld_payroll_tax",
    "savers_credit",
    "student_loan_interest",
    "investment_income_elected_form_4952",
    "early_withdrawal_penalty",
    "prior_year_minimum_tax_credit",
    "farm_rent_income",
    "qualified_tuition_expenses",
    "educator_expense",
    "long_term_capital_gains_on_collectibles",
    "other_credits",
    "casualty_loss",
    "unreported_payroll_tax",
    "recapture_of_investment_credit",
    "deductible_mortgage_interest",
    "qualified_reit_and_ptp_income",
    "qualified_bdc_income",
    "farm_operations_income",
    "estate_income_would_be_qualified",
    "farm_operations_income_would_be_qualified",
    "farm_rent_income_would_be_qualified",
    "partnership_s_corp_income_would_be_qualified",
    "rental_income_would_be_qualified",
]

CPS_RETIREMENT_VARIABLES = [
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "self_employed_pension_contributions",
]

RETIREMENT_DEMOGRAPHIC_PREDICTORS = [
    "age",
    "is_male",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
]

# Income predictors sourced from PUF imputations on the test side.
RETIREMENT_INCOME_PREDICTORS = [
    "employment_income",
    "self_employment_income",
    "taxable_interest_income",
    "qualified_dividend_income",
    "taxable_pension_income",
    "social_security",
]

RETIREMENT_PREDICTORS = RETIREMENT_DEMOGRAPHIC_PREDICTORS + RETIREMENT_INCOME_PREDICTORS


def _get_retirement_limits(year: int) -> dict:
    """Return contribution limits for the given tax year.

    Merges 401k/IRA limits from policyengine-us parameters
    (via get_retirement_limits) with SE pension params from
    imputation_parameters.yaml.
    """
    limits = dict(get_retirement_limits(year))
    yaml_path = (
        files("policyengine_us_data")
        / "datasets"
        / "cps"
        / "imputation_parameters.yaml"
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    limits["se_pension_rate"] = params["se_pension_contribution_rate"]
    se_dollar_limits = params["se_pension_contribution_dollar_limit"]
    clamped = max(min(year, max(se_dollar_limits)), min(se_dollar_limits))
    limits["se_pension_dollar_limit"] = se_dollar_limits[clamped]
    return limits


MINIMUM_RETIREMENT_AGE = 62

SS_SPLIT_PREDICTORS = [
    "age",
    "is_male",
    "tax_unit_is_joint",
    "is_tax_unit_head",
    "is_tax_unit_dependent",
]

MIN_QRF_TRAINING_RECORDS = 100


def _qrf_ss_shares(
    data: Dict[str, Dict[int, np.ndarray]],
    n_cps: int,
    time_period: int,
    puf_has_ss: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Predict SS sub-component shares via QRF.

    Trains on CPS records that have SS > 0 (where the reason-code
    split is known), then predicts shares for all PUF records with
    positive SS. The CPS-PUF link is statistical (not identity-based),
    so the QRF gives a better expected prediction than using the
    paired CPS record's split.

    Args:
        data: Dataset dict.
        n_cps: Records in CPS half.
        time_period: Tax year.
        puf_has_ss: Boolean mask (length n_cps) — True where the
            PUF half has positive social_security.

    Returns:
        Dict mapping sub-component name to predicted share arrays
        (length = puf_has_ss.sum()), or None if training data is
        insufficient.
    """
    from microimpute.models.qrf import QRF

    cps_ss = data["social_security"][time_period][:n_cps]
    has_ss = cps_ss > 0

    if has_ss.sum() < MIN_QRF_TRAINING_RECORDS:
        return None

    # Build training features from available predictors.
    predictors = []
    train_cols = {}
    test_cols = {}
    for pred in SS_SPLIT_PREDICTORS:
        if pred not in data:
            continue
        vals = data[pred][time_period][:n_cps]
        train_cols[pred] = vals[has_ss].astype(np.float32)
        test_cols[pred] = vals[puf_has_ss].astype(np.float32)
        predictors.append(pred)

    if not predictors:
        return None

    X_train = pd.DataFrame(train_cols)
    X_test = pd.DataFrame(test_cols)

    # Training targets: share going to each sub-component (0 or 1).
    share_vars = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for sub in SS_SUBCOMPONENTS:
            if sub not in data:
                continue
            sub_vals = data[sub][time_period][:n_cps][has_ss]
            share_name = sub + "_share"
            X_train[share_name] = np.where(
                cps_ss[has_ss] > 0,
                sub_vals / cps_ss[has_ss],
                0.0,
            )
            share_vars.append(share_name)

    if not share_vars:
        return None

    qrf = QRF(log_level="WARNING", memory_efficient=True)
    try:
        fitted = qrf.fit(
            X_train=X_train[predictors + share_vars],
            predictors=predictors,
            imputed_variables=share_vars,
            n_jobs=1,
        )
        predictions = fitted.predict(X_test=X_test)
    except Exception:
        logger.warning(
            "QRF SS split failed, falling back to heuristic",
            exc_info=True,
        )
        return None

    # Clip to [0, 1] and normalise so shares sum to 1.
    shares = {}
    total = np.zeros(len(X_test))
    for sub in SS_SUBCOMPONENTS:
        key = sub + "_share"
        if key in predictions.columns:
            s = np.clip(predictions[key].values, 0, 1)
            shares[sub] = s
            total += s

    for sub in shares:
        shares[sub] = np.where(total > 0, shares[sub] / total, 0.0)

    del fitted, predictions
    gc.collect()

    logger.info(
        "QRF SS split: predicted shares for %d PUF records",
        puf_has_ss.sum(),
    )
    return shares


def _age_heuristic_ss_shares(
    data: Dict[str, Dict[int, np.ndarray]],
    n_cps: int,
    time_period: int,
    puf_has_ss: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Fallback: assign SS type by age threshold.

    Age >= 62 -> retirement, < 62 -> disability.
    If age is unavailable, all go to retirement.
    """
    n_pred = puf_has_ss.sum()
    shares = {sub: np.zeros(n_pred) for sub in SS_SUBCOMPONENTS}

    age = None
    if "age" in data:
        age = data["age"][time_period][:n_cps][puf_has_ss]

    if age is not None:
        is_old = age >= MINIMUM_RETIREMENT_AGE
        if "social_security_retirement" in shares:
            shares["social_security_retirement"] = is_old.astype(np.float64)
        if "social_security_disability" in shares:
            shares["social_security_disability"] = (~is_old).astype(np.float64)
    else:
        if "social_security_retirement" in shares:
            shares["social_security_retirement"] = np.ones(n_pred)

    return shares


def reconcile_ss_subcomponents(
    data: Dict[str, Dict[int, np.ndarray]],
    n_cps: int,
    time_period: int,
) -> None:
    """Predict SS sub-components for PUF half from demographics.

    The CPS-PUF link is statistical (not identity-based), so the
    paired CPS record's sub-component split is just one noisy draw.
    A QRF trained on all CPS SS recipients gives a better expected
    prediction by pooling across the full training set.

    For all PUF records with positive social_security, this function
    predicts shares via QRF (falling back to an age heuristic) and
    scales them to match the imputed total. PUF records with zero
    SS get all sub-components cleared to zero.

    Modifies ``data`` in place. Only the PUF half (indices
    n_cps .. 2*n_cps) is changed.

    Args:
        data: Dataset dict {variable: {time_period: array}}.
        n_cps: Number of records in the CPS half.
        time_period: Tax year key into data dicts.
    """
    if "social_security" not in data:
        return

    puf_ss = data["social_security"][time_period][n_cps:]
    puf_has_ss = puf_ss > 0

    # Predict shares for all PUF records with SS > 0.
    shares = None
    if puf_has_ss.any():
        shares = _qrf_ss_shares(data, n_cps, time_period, puf_has_ss)
        if shares is None:
            shares = _age_heuristic_ss_shares(data, n_cps, time_period, puf_has_ss)

    for sub in SS_SUBCOMPONENTS:
        if sub not in data:
            continue
        arr = data[sub][time_period]

        new_puf = np.zeros(n_cps)
        if puf_has_ss.any() and shares is not None:
            share = shares.get(sub, np.zeros(puf_has_ss.sum()))
            new_puf[puf_has_ss] = puf_ss[puf_has_ss] * share

        arr[n_cps:] = new_puf.astype(arr.dtype)
        data[sub][time_period] = arr


def puf_clone_dataset(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int = 2024,
    puf_dataset=None,
    skip_qrf: bool = False,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Clone CPS data 2x and impute PUF variables on one half.

    The first half keeps CPS values (with OVERRIDDEN vars QRF'd).
    The second half gets full PUF QRF imputation. The second half
    has household weights set to zero.

    Args:
        data: CPS dataset dict {variable: {time_period: array}}.
        state_fips: State FIPS per household, shape (n_households,).
        time_period: Tax year.
        puf_dataset: PUF dataset class or path for QRF training.
            If None, skips QRF (same as skip_qrf=True).
        skip_qrf: If True, skip QRF imputation (for testing).
        dataset_path: Path to CPS h5 file (needed for QRF to
            compute demographic predictors via Microsimulation).

    Returns:
        New data dict with doubled records.
    """
    household_ids = data["household_id"][time_period]
    n_households = len(household_ids)
    person_count = len(data["person_id"][time_period])

    logger.info(
        "PUF clone: %d households, %d persons",
        n_households,
        person_count,
    )

    y_full = None
    y_override = None
    if not skip_qrf and puf_dataset is not None:
        y_full, y_override = _run_qrf_imputation(
            data,
            time_period,
            puf_dataset,
            dataset_path=dataset_path,
        )

    cps_sim = None
    tbs = None
    if (y_full or y_override) and dataset_path is not None:
        from policyengine_us import Microsimulation

        cps_sim = Microsimulation(dataset=dataset_path)
        tbs = cps_sim.tax_benefit_system

    def _map_to_entity(pred_values, variable_name):
        if cps_sim is None or tbs is None:
            return pred_values
        var_meta = tbs.variables.get(variable_name)
        if var_meta is None:
            return pred_values
        entity = var_meta.entity.key
        if entity != "person":
            return cps_sim.populations[entity].value_from_first_person(pred_values)
        return pred_values

    # Impute weeks_unemployed for PUF half
    puf_weeks = None
    if y_full is not None and dataset_path is not None:
        puf_weeks = _impute_weeks_unemployed(data, y_full, time_period, dataset_path)

    # Impute retirement contributions for PUF half
    puf_retirement = None
    if y_full is not None and dataset_path is not None:
        puf_retirement = _impute_retirement_contributions(
            data, y_full, time_period, dataset_path
        )

    new_data = {}
    for variable, time_dict in data.items():
        values = time_dict[time_period]

        if variable in OVERRIDDEN_IMPUTED_VARIABLES and y_override:
            pred = _map_to_entity(y_override[variable], variable)
            new_data[variable] = {time_period: np.concatenate([pred, pred])}
        elif variable in IMPUTED_VARIABLES and y_full:
            pred = _map_to_entity(y_full[variable], variable)
            new_data[variable] = {time_period: np.concatenate([values, pred])}
        elif "_id" in variable:
            new_data[variable] = {
                time_period: np.concatenate([values, values + values.max()])
            }
        elif "_weight" in variable:
            new_data[variable] = {time_period: np.concatenate([values, values * 0])}
        elif variable == "weeks_unemployed" and puf_weeks is not None:
            new_data[variable] = {time_period: np.concatenate([values, puf_weeks])}
        elif variable in CPS_RETIREMENT_VARIABLES and puf_retirement is not None:
            puf_vals = puf_retirement[variable]
            new_data[variable] = {time_period: np.concatenate([values, puf_vals])}
        else:
            new_data[variable] = {time_period: np.concatenate([values, values])}

    new_data["state_fips"] = {
        time_period: np.concatenate([state_fips, state_fips]).astype(np.int32)
    }

    if y_full:
        for var in IMPUTED_VARIABLES:
            if var not in data:
                pred = _map_to_entity(y_full[var], var)
                orig = np.zeros_like(pred)
                new_data[var] = {time_period: np.concatenate([orig, pred])}

    if cps_sim is not None:
        del cps_sim

    # Ensure SS sub-components match the (possibly imputed) total.
    reconcile_ss_subcomponents(new_data, person_count, time_period)

    logger.info(
        "PUF clone complete: %d -> %d households",
        n_households,
        n_households * 2,
    )
    return new_data


def _impute_weeks_unemployed(
    data: Dict[str, Dict[int, np.ndarray]],
    puf_imputations: Dict[str, np.ndarray],
    time_period: int,
    dataset_path: str,
) -> np.ndarray:
    """Impute weeks_unemployed for the PUF half using QRF.

    Uses CPS as training data and imputed PUF demographics as
    test data, preserving the joint distribution of weeks with
    unemployment compensation.

    Args:
        data: CPS data dict.
        puf_imputations: Dict of PUF-imputed variable arrays.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.

    Returns:
        Array of imputed weeks for PUF half.
    """
    from microimpute.models.qrf import QRF
    from policyengine_us import Microsimulation

    cps_sim = Microsimulation(dataset=dataset_path)
    try:
        cps_weeks = cps_sim.calculate("weeks_unemployed").values
    except (ValueError, KeyError):
        logger.warning("weeks_unemployed not in CPS, returning zeros")
        n_persons = len(data["person_id"][time_period])
        del cps_sim
        return np.zeros(n_persons)

    WEEKS_PREDICTORS = [
        "age",
        "is_male",
        "tax_unit_is_joint",
        "is_tax_unit_head",
        "is_tax_unit_spouse",
        "is_tax_unit_dependent",
    ]

    X_train = cps_sim.calculate_dataframe(WEEKS_PREDICTORS)
    X_train["weeks_unemployed"] = cps_weeks

    if "taxable_unemployment_compensation" in puf_imputations:
        cps_uc = cps_sim.calculate("unemployment_compensation").values
        X_train["unemployment_compensation"] = cps_uc
        WEEKS_PREDICTORS = WEEKS_PREDICTORS + ["unemployment_compensation"]

    X_test = cps_sim.calculate_dataframe(
        [p for p in WEEKS_PREDICTORS if p != "unemployment_compensation"]
    )

    if "taxable_unemployment_compensation" in puf_imputations:
        X_test["unemployment_compensation"] = puf_imputations[
            "taxable_unemployment_compensation"
        ]

    del cps_sim

    qrf = QRF(
        log_level="INFO",
        memory_efficient=True,
        max_train_samples=5000,
    )
    predictions = qrf.fit_predict(
        X_train=X_train,
        X_test=X_test,
        predictors=WEEKS_PREDICTORS,
        imputed_variables=["weeks_unemployed"],
        n_jobs=1,
    )
    imputed_weeks = predictions["weeks_unemployed"].values

    imputed_weeks = np.clip(imputed_weeks, 0, 52)
    if "unemployment_compensation" in X_test.columns:
        imputed_weeks = np.where(
            X_test["unemployment_compensation"].values > 0,
            imputed_weeks,
            0,
        )

    logger.info(
        "Imputed weeks_unemployed for PUF: %d with weeks > 0, mean = %.1f",
        (imputed_weeks > 0).sum(),
        (imputed_weeks[imputed_weeks > 0].mean() if (imputed_weeks > 0).any() else 0),
    )

    return imputed_weeks


def _impute_retirement_contributions(
    data: Dict[str, Dict[int, np.ndarray]],
    puf_imputations: Dict[str, np.ndarray],
    time_period: int,
    dataset_path: str,
) -> Dict[str, np.ndarray]:
    """Impute retirement contributions for the PUF half using QRF.

    Trains on CPS data (which has realistic income-to-contribution
    relationships) and predicts onto PUF clone records using
    PUF-imputed income as input features.

    Note: ``pre_tax_contributions`` is separately imputed from PUF
    via OVERRIDDEN_IMPUTED_VARIABLES.  In PolicyEngine it is a
    formula (``adds`` of traditional_401k + traditional_403b + …),
    so the stored value is only used when the formula is bypassed.
    A future improvement could reconcile or drop the stored
    pre_tax_contributions in favour of the formula sum.

    Args:
        data: CPS data dict.
        puf_imputations: Dict of PUF-imputed variable arrays.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.

    Returns:
        Dict mapping retirement variable names to imputed arrays.
        Returns all-zeros on QRF failure.
    """
    from microimpute.models.qrf import QRF
    from policyengine_us import Microsimulation

    cps_sim = Microsimulation(dataset=dataset_path)

    # Build training data from CPS (has realistic relationships)
    train_cols = RETIREMENT_PREDICTORS + CPS_RETIREMENT_VARIABLES
    try:
        X_train = cps_sim.calculate_dataframe(train_cols)
    except (ValueError, KeyError) as e:
        logger.warning("Could not build retirement training data: %s", e)
        n_persons = len(data["person_id"][time_period])
        del cps_sim
        return {var: np.zeros(n_persons) for var in CPS_RETIREMENT_VARIABLES}

    # Build test data: demographics from CPS sim, income from PUF
    X_test = cps_sim.calculate_dataframe(RETIREMENT_DEMOGRAPHIC_PREDICTORS)
    for income_var in RETIREMENT_INCOME_PREDICTORS:
        if income_var in puf_imputations:
            X_test[income_var] = puf_imputations[income_var]
        else:
            X_test[income_var] = cps_sim.calculate(income_var).values

    del cps_sim

    qrf = QRF(
        log_level="INFO",
        memory_efficient=True,
        max_train_samples=5000,
    )
    try:
        predictions = qrf.fit_predict(
            X_train=X_train,
            X_test=X_test,
            predictors=RETIREMENT_PREDICTORS,
            imputed_variables=CPS_RETIREMENT_VARIABLES,
            n_jobs=1,
        )
    except Exception:
        logger.warning(
            "QRF retirement imputation failed, returning zeros",
            exc_info=True,
        )
        n_persons = len(data["person_id"][time_period])
        return {var: np.zeros(n_persons) for var in CPS_RETIREMENT_VARIABLES}

    # Extract results and apply constraints
    limits = _get_retirement_limits(time_period)
    age = X_test["age"].values
    catch_up_eligible = age >= 50
    limit_401k = limits["401k"] + catch_up_eligible * limits["401k_catch_up"]
    limit_ira = limits["ira"] + catch_up_eligible * limits["ira_catch_up"]
    se_pension_cap = np.minimum(
        X_test["self_employment_income"].values * limits["se_pension_rate"],
        limits["se_pension_dollar_limit"],
    )

    emp_income = X_test["employment_income"].values
    se_income = X_test["self_employment_income"].values

    result = {}
    for var in CPS_RETIREMENT_VARIABLES:
        vals = predictions[var].values

        # Non-negativity
        vals = np.maximum(vals, 0)

        # Cap 401k at year-specific limit
        if "401k" in var:
            vals = np.minimum(vals, limit_401k)
            # Zero out for records with no employment income
            vals = np.where(emp_income > 0, vals, 0)

        # Cap IRA at year-specific limit
        if "ira" in var:
            vals = np.minimum(vals, limit_ira)

        # Cap SE pension at min(25% of SE income, dollar limit)
        if var == "self_employed_pension_contributions":
            vals = np.minimum(vals, se_pension_cap)
            vals = np.where(se_income > 0, vals, 0)

        result[var] = vals

    logger.info(
        "Imputed retirement contributions for PUF: "
        "401k mean=$%.0f, IRA mean=$%.0f, SE pension mean=$%.0f",
        result["traditional_401k_contributions"].mean()
        + result["roth_401k_contributions"].mean(),
        result["traditional_ira_contributions"].mean()
        + result["roth_ira_contributions"].mean(),
        result["self_employed_pension_contributions"].mean(),
    )

    return result


def _run_qrf_imputation(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    puf_dataset,
    dataset_path: Optional[str] = None,
) -> tuple:
    """Run QRF imputation for PUF variables.

    Stratified-subsamples PUF records (top 0.5% by AGI kept,
    rest randomly sampled to ~20K total), trains QRF, and
    predicts on CPS data.

    Args:
        data: CPS data dict.
        time_period: Tax year.
        puf_dataset: PUF dataset class or path.
        dataset_path: Path to CPS h5 for computing
            demographic predictors via Microsimulation.

    Returns:
        Tuple of (y_full_imputations, y_override_imputations)
        as dicts of {variable: np.ndarray}.
    """
    from microimpute.models.qrf import QRF
    from policyengine_us import Microsimulation

    logger.info("Running QRF imputation")

    puf_sim = Microsimulation(dataset=puf_dataset)

    puf_agi = puf_sim.calculate("adjusted_gross_income", map_to="person").values

    X_train_full = puf_sim.calculate_dataframe(
        DEMOGRAPHIC_PREDICTORS + IMPUTED_VARIABLES
    )

    X_train_override = puf_sim.calculate_dataframe(
        DEMOGRAPHIC_PREDICTORS + OVERRIDDEN_IMPUTED_VARIABLES
    )

    del puf_sim

    sub_idx = _stratified_subsample_index(puf_agi)
    logger.info(
        "Stratified PUF subsample: %d -> %d records "
        "(top %.1f%% preserved, AGI threshold $%,.0f)",
        len(puf_agi),
        len(sub_idx),
        100 - PUF_TOP_PERCENTILE,
        np.percentile(puf_agi, PUF_TOP_PERCENTILE),
    )
    X_train_full = X_train_full.iloc[sub_idx].reset_index(drop=True)
    X_train_override = X_train_override.iloc[sub_idx].reset_index(drop=True)

    if dataset_path is not None:
        cps_sim = Microsimulation(dataset=dataset_path)
        X_test = cps_sim.calculate_dataframe(DEMOGRAPHIC_PREDICTORS)
        del cps_sim
    else:
        X_test = pd.DataFrame()
        for pred in DEMOGRAPHIC_PREDICTORS:
            if pred in data:
                X_test[pred] = data[pred][time_period].astype(np.float32)

    logger.info("Imputing %d PUF variables (full)", len(IMPUTED_VARIABLES))
    y_full = _sequential_qrf(
        X_train_full, X_test, DEMOGRAPHIC_PREDICTORS, IMPUTED_VARIABLES
    )

    logger.info(
        "Imputing %d PUF variables (override)",
        len(OVERRIDDEN_IMPUTED_VARIABLES),
    )
    y_override = _sequential_qrf(
        X_train_override,
        X_test,
        DEMOGRAPHIC_PREDICTORS,
        OVERRIDDEN_IMPUTED_VARIABLES,
    )

    return y_full, y_override


def _stratified_subsample_index(
    income: np.ndarray,
    target_n: int = PUF_SUBSAMPLE_TARGET,
    top_pct: float = PUF_TOP_PERCENTILE,
    seed: int = 42,
) -> np.ndarray:
    """Return indices for stratified subsample preserving top earners.

    Keeps ALL records above the top_pct percentile of income,
    then randomly samples the rest to reach target_n total.
    """
    n = len(income)
    if n <= target_n:
        return np.arange(n)

    threshold = np.percentile(income, top_pct)
    top_idx = np.where(income >= threshold)[0]
    bottom_idx = np.where(income < threshold)[0]

    remaining_quota = max(0, target_n - len(top_idx))
    rng = np.random.default_rng(seed=seed)
    if remaining_quota >= len(bottom_idx):
        selected_bottom = bottom_idx
    else:
        selected_bottom = rng.choice(bottom_idx, size=remaining_quota, replace=False)

    selected = np.concatenate([top_idx, selected_bottom])
    selected.sort()
    return selected


def _sequential_qrf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    predictors: List[str],
    output_vars: List[str],
) -> Dict[str, np.ndarray]:
    """Run a single sequential QRF preserving covariance.

    Uses microimpute's fit_predict() which handles missing variable
    detection, gc cleanup, and zero-fill internally. Each variable
    is conditioned on all previously imputed variables, preserving
    the full joint distribution.

    Args:
        X_train: Training data with predictors + output vars.
        X_test: Test data with predictors only.
        predictors: Predictor column names.
        output_vars: Output variable names to impute.

    Returns:
        Dict mapping variable name to imputed values.
    """
    from microimpute.models.qrf import QRF

    qrf = QRF(
        log_level="INFO",
        memory_efficient=True,
    )
    predictions = qrf.fit_predict(
        X_train=X_train,
        X_test=X_test,
        predictors=predictors,
        imputed_variables=output_vars,
        n_jobs=1,
    )

    result = {var: predictions[var].values for var in predictions.columns}
    missing = set(output_vars) - set(result)
    if missing:
        raise ValueError(
            f"{len(missing)} variables requested but not returned "
            f"by fit_predict(): {sorted(missing)[:10]}"
        )
    return result
