"""PUF clone and QRF imputation for calibration pipeline.

Doubles CPS records: one half keeps original values, the other half
gets PUF tax variables imputed via Quantile Random Forest.  Geography
(state_fips) is included as a QRF predictor so imputations vary by
state.

Usage within the calibration pipeline:
    1. Load raw CPS dataset
    2. Clone 10x and assign geography
    3. Call puf_clone_dataset() to double records and impute PUF vars
    4. Save expanded dataset for matrix building
"""

import gc
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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
    "state_fips",
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
    "traditional_ira_contributions",
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
            state_fips,
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
            return cps_sim.populations[entity].value_from_first_person(
                pred_values
            )
        return pred_values

    # Impute weeks_unemployed for PUF half
    puf_weeks = None
    if y_full is not None and dataset_path is not None:
        puf_weeks = _impute_weeks_unemployed(
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
        elif variable == "person_id":
            new_data[variable] = {
                time_period: np.concatenate([values, values + values.max()])
            }
        elif "_id" in variable:
            new_data[variable] = {
                time_period: np.concatenate([values, values + values.max()])
            }
        elif "_weight" in variable:
            new_data[variable] = {
                time_period: np.concatenate([values, values * 0])
            }
        elif variable == "weeks_unemployed" and puf_weeks is not None:
            new_data[variable] = {
                time_period: np.concatenate([values, puf_weeks])
            }
        else:
            new_data[variable] = {
                time_period: np.concatenate([values, values])
            }

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

    qrf = QRF(log_level="INFO", memory_efficient=True)
    sample_size = min(5000, len(X_train))
    if len(X_train) > sample_size:
        X_train_sampled = X_train.sample(n=sample_size, random_state=42)
    else:
        X_train_sampled = X_train

    fitted = qrf.fit(
        X_train=X_train_sampled,
        predictors=WEEKS_PREDICTORS,
        imputed_variables=["weeks_unemployed"],
        n_jobs=1,
    )
    predictions = fitted.predict(X_test=X_test)
    imputed_weeks = predictions["weeks_unemployed"].values

    imputed_weeks = np.clip(imputed_weeks, 0, 52)
    if "unemployment_compensation" in X_test.columns:
        imputed_weeks = np.where(
            X_test["unemployment_compensation"].values > 0,
            imputed_weeks,
            0,
        )

    logger.info(
        "Imputed weeks_unemployed for PUF: " "%d with weeks > 0, mean = %.1f",
        (imputed_weeks > 0).sum(),
        (
            imputed_weeks[imputed_weeks > 0].mean()
            if (imputed_weeks > 0).any()
            else 0
        ),
    )

    del fitted, predictions
    gc.collect()
    return imputed_weeks


def _run_qrf_imputation(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
    puf_dataset,
    dataset_path: Optional[str] = None,
) -> tuple:
    """Run QRF imputation for PUF variables.

    Stratified-subsamples PUF records (top 0.5% by AGI kept,
    rest randomly sampled to ~20K total) with random state
    assignment, trains QRF, and predicts on CPS data.

    Args:
        data: CPS data dict.
        state_fips: State FIPS per household.
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

    logger.info("Running QRF imputation with state predictor")

    puf_sim = Microsimulation(dataset=puf_dataset)

    from policyengine_us_data.calibration.clone_and_assign import (
        load_global_block_distribution,
    )

    _, _, puf_states, block_probs = load_global_block_distribution()
    rng = np.random.default_rng(seed=99)
    n_puf = len(puf_sim.calculate("person_id", map_to="person").values)
    puf_state_indices = rng.choice(len(puf_states), size=n_puf, p=block_probs)
    puf_state_fips = puf_states[puf_state_indices]

    puf_agi = puf_sim.calculate(
        "adjusted_gross_income", map_to="person"
    ).values

    demo_preds = [p for p in DEMOGRAPHIC_PREDICTORS if p != "state_fips"]

    X_train_full = puf_sim.calculate_dataframe(demo_preds + IMPUTED_VARIABLES)
    X_train_full["state_fips"] = puf_state_fips.astype(np.float32)

    X_train_override = puf_sim.calculate_dataframe(
        demo_preds + OVERRIDDEN_IMPUTED_VARIABLES
    )
    X_train_override["state_fips"] = puf_state_fips.astype(np.float32)

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

    n_hh = len(data["household_id"][time_period])
    person_count = len(data["person_id"][time_period])

    if dataset_path is not None:
        cps_sim = Microsimulation(dataset=dataset_path)
        X_test = cps_sim.calculate_dataframe(demo_preds)
        del cps_sim
    else:
        X_test = pd.DataFrame()
        for pred in demo_preds:
            if pred in data:
                X_test[pred] = data[pred][time_period].astype(np.float32)

    hh_ids_person = data.get("person_household_id", {}).get(time_period)
    if hh_ids_person is not None:
        hh_ids = data["household_id"][time_period]
        hh_to_idx = {int(hh_id): i for i, hh_id in enumerate(hh_ids)}
        person_states = np.array(
            [state_fips[hh_to_idx[int(hh_id)]] for hh_id in hh_ids_person]
        )
    else:
        person_states = np.repeat(
            state_fips,
            person_count // n_hh,
        )
    X_test["state_fips"] = person_states.astype(np.float32)

    predictors = DEMOGRAPHIC_PREDICTORS

    logger.info("Imputing %d PUF variables (full)", len(IMPUTED_VARIABLES))
    y_full = _batch_qrf(X_train_full, X_test, predictors, IMPUTED_VARIABLES)

    logger.info(
        "Imputing %d PUF variables (override)",
        len(OVERRIDDEN_IMPUTED_VARIABLES),
    )
    y_override = _batch_qrf(
        X_train_override,
        X_test,
        predictors,
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
        selected_bottom = rng.choice(
            bottom_idx, size=remaining_quota, replace=False
        )

    selected = np.concatenate([top_idx, selected_bottom])
    selected.sort()
    return selected


def _batch_qrf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    predictors: List[str],
    output_vars: List[str],
    batch_size: int = 10,
) -> Dict[str, np.ndarray]:
    """Run QRF in batches to control memory.

    Args:
        X_train: Training data with predictors + output vars.
        X_test: Test data with predictors only.
        predictors: Predictor column names.
        output_vars: Output variable names to impute.
        batch_size: Variables per batch.

    Returns:
        Dict mapping variable name to imputed values.
    """
    from microimpute.models.qrf import QRF

    available = [c for c in output_vars if c in X_train.columns]
    missing = [c for c in output_vars if c not in X_train.columns]

    if missing:
        logger.warning(
            "%d variables missing from training: %s",
            len(missing),
            missing[:5],
        )

    result = {}

    for batch_start in range(0, len(available), batch_size):
        batch_end = min(batch_start + batch_size, len(available))
        batch_vars = available[batch_start:batch_end]

        gc.collect()

        qrf = QRF(
            log_level="INFO",
            memory_efficient=True,
            batch_size=10,
            cleanup_interval=5,
        )

        batch_X_train = X_train[predictors + batch_vars].copy()

        fitted = qrf.fit(
            X_train=batch_X_train,
            predictors=predictors,
            imputed_variables=batch_vars,
            n_jobs=1,
        )

        predictions = fitted.predict(X_test=X_test)

        for var in batch_vars:
            result[var] = predictions[var].values

        del fitted, predictions, batch_X_train
        gc.collect()

    n_test = len(X_test)
    for var in missing:
        result[var] = np.zeros(n_test)

    return result
