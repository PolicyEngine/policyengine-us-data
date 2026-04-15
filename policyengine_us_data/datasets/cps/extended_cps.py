import logging
import time
from typing import Type

import numpy as np
import pandas as pd
from policyengine_core.data import Dataset

from policyengine_us_data.datasets.cps.cps import CPS, CPS_2024, CPS_2024_Full
from policyengine_us_data.datasets.org import (
    ORG_IMPUTED_VARIABLES,
    apply_org_domain_constraints,
)
from policyengine_us_data.datasets.puf import PUF, PUF_2024
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.mortgage_interest import (
    STRUCTURAL_MORTGAGE_VARIABLES,
    convert_mortgage_interest_to_structural_inputs,
    impute_tax_unit_mortgage_balance_hints,
)
from policyengine_us_data.utils.policyengine import has_policyengine_us_variables
from policyengine_us_data.utils.policyengine import (
    supports_modeled_medicare_part_b_inputs,
)
from policyengine_us_data.utils.retirement_limits import (
    get_retirement_limits,
    get_se_pension_limits,
)

logger = logging.getLogger(__name__)


def _supports_structural_mortgage_inputs() -> bool:
    return has_policyengine_us_variables(*STRUCTURAL_MORTGAGE_VARIABLES)


# CPS-only categorical features to donor-impute onto the PUF clone half.
# These drive subgroup analysis and occupation-based logic, so naive donor
# duplication dilutes the relationship between the clone's PUF-imputed
# income and its CPS-side demographic/occupation labels.
CPS_CLONE_FEATURE_VARIABLES = [
    "is_male",
    "cps_race",
    "is_hispanic",
    "detailed_occupation_recode",
]
if has_policyengine_us_variables("treasury_tipped_occupation_code"):
    CPS_CLONE_FEATURE_VARIABLES.append("treasury_tipped_occupation_code")

# Predictors used to rematch CPS features onto the PUF clone half.
# These are all available on the CPS half and on the doubled extended CPS.
CPS_CLONE_FEATURE_PREDICTORS = [
    "age",
    "state_fips",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
    "employment_income",
    "self_employment_income",
    "social_security",
]

_OVERTIME_OCCUPATION_CODES = {
    "has_never_worked": 53,
    "is_military": 52,
    "is_computer_scientist": 8,
    "is_farmer_fisher": 41,
}
_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_CODES = np.array(
    [
        1,
        2,
        3,
        5,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        24,
        25,
        27,
        28,
        29,
        30,
        32,
        33,
        34,
    ],
    dtype=np.int16,
)

# CPS-only variables that should be QRF-imputed for the PUF clone half
# instead of naively duplicated from the CPS donor. Most demographics,
# IDs, weights, and random seeds are fine to duplicate; the categorical
# clone features above are rematched separately.
CPS_ONLY_IMPUTED_VARIABLES = [
    # Retirement distributions
    "taxable_401k_distributions",
    "tax_exempt_401k_distributions",
    "taxable_403b_distributions",
    "tax_exempt_403b_distributions",
    "keogh_distributions",
    "taxable_sep_distributions",
    "tax_exempt_sep_distributions",
    # Retirement contributions
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "self_employed_pension_contributions",
    # Social Security sub-components
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
    # Transfer income
    "unemployment_compensation",
    "tanf_reported",
    "ssi_reported",
    "child_support_received",
    "veterans_benefits",
    "workers_compensation",
    "disability_benefits",
    "strike_benefits",
    "receives_wic",
    # SPM variables
    "spm_unit_total_income_reported",
    "snap_reported",
    "spm_unit_capped_housing_subsidy_reported",
    "free_school_meals_reported",
    "spm_unit_energy_subsidy_reported",
    "spm_unit_wic_reported",
    "spm_unit_broadband_subsidy_reported",
    "spm_unit_payroll_tax_reported",
    "spm_unit_federal_tax_reported",
    "spm_unit_state_tax_reported",
    "spm_unit_spm_threshold",
    "spm_unit_net_income_reported",
    "spm_unit_pre_subsidy_childcare_expenses",
    # Medical expenses
    "employer_sponsored_insurance_premiums",
    "health_insurance_premiums_without_medicare_part_b",
    "over_the_counter_health_expenses",
    "other_medical_expenses",
    "child_support_expense",
    # Hours/employment
    "weekly_hours_worked",
    "hours_worked_last_week",
    # ORG labor-market variables
    "hourly_wage",
    "is_paid_hourly",
    "is_union_member_or_covered",
    # Previous year income
    "employment_income_last_year",
    "self_employment_income_last_year",
]

if not supports_modeled_medicare_part_b_inputs():
    CPS_ONLY_IMPUTED_VARIABLES.append("medicare_part_b_premiums")

# Set for O(1) lookup in the splice loop.
_CPS_ONLY_SET = set(CPS_ONLY_IMPUTED_VARIABLES)

# Predictors used for the second-stage CPS-only imputation: demographics
# plus key income variables that were already imputed from PUF data.
CPS_STAGE2_DEMOGRAPHIC_PREDICTORS = [
    "age",
    "is_male",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
]

CPS_STAGE2_INCOME_PREDICTORS = [
    "employment_income",
    "self_employment_income",
    "social_security",
]


def _clone_half_person_values(data: dict, variable: str, time_period: int):
    """Return clone-half values for ``variable`` mapped to person rows."""
    if variable not in data:
        return None

    values = data[variable][time_period]
    n_persons = len(data["person_id"][time_period])
    n_persons_half = n_persons // 2
    if len(values) == n_persons:
        return np.asarray(values[n_persons_half:])

    entity_mappings = [
        ("household_id", "person_household_id"),
        ("tax_unit_id", "person_tax_unit_id"),
        ("spm_unit_id", "person_spm_unit_id"),
        ("family_id", "person_family_id"),
    ]
    for entity_id_var, person_entity_id_var in entity_mappings:
        if entity_id_var not in data or person_entity_id_var not in data:
            continue
        entity_ids = data[entity_id_var][time_period]
        if len(values) != len(entity_ids):
            continue
        entity_half = len(entity_ids) // 2
        clone_entity_ids = entity_ids[entity_half:]
        clone_person_entity_ids = data[person_entity_id_var][time_period][
            n_persons_half:
        ]
        value_map = dict(zip(clone_entity_ids, values[entity_half:]))
        return np.array([value_map[idx] for idx in clone_person_entity_ids])

    return None


def _build_clone_test_frame(
    cps_sim,
    data: dict,
    time_period: int,
    predictors: list[str],
) -> pd.DataFrame:
    """Build clone-half predictor data with available doubled-dataset overrides."""
    X_test = cps_sim.calculate_dataframe(predictors).copy()
    for predictor in predictors:
        clone_values = _clone_half_person_values(data, predictor, time_period)
        if clone_values is not None and len(clone_values) == len(X_test):
            X_test[predictor] = clone_values
    return X_test[predictors]


def _prepare_knn_matrix(
    df: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> np.ndarray:
    """Normalise mixed-scale donor-matching predictors for kNN."""
    X = df.astype(float).copy()
    for income_var in CPS_STAGE2_INCOME_PREDICTORS:
        if income_var in X:
            X[income_var] = np.arcsinh(X[income_var])

    ref = X if reference is None else reference.astype(float).copy()
    for income_var in CPS_STAGE2_INCOME_PREDICTORS:
        if income_var in ref:
            ref[income_var] = np.arcsinh(ref[income_var])

    means = ref.mean()
    stds = ref.std(ddof=0).replace(0, 1)
    normalised = (X - means) / stds
    return np.nan_to_num(normalised.to_numpy(dtype=np.float32), nan=0.0)


def _derive_overtime_occupation_inputs(
    occupation_codes: np.ndarray,
) -> pd.DataFrame:
    """Derive occupation-based overtime-exemption inputs from POCCU2."""
    occupation_codes = np.rint(occupation_codes).astype(np.int16, copy=False)
    derived = {
        name: occupation_codes == code
        for name, code in _OVERTIME_OCCUPATION_CODES.items()
    }
    derived["is_executive_administrative_professional"] = np.isin(
        occupation_codes,
        _EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_CODES,
    )
    return pd.DataFrame(derived)


def _impute_clone_cps_features(
    data: dict,
    time_period: int,
    dataset_path: str,
) -> pd.DataFrame:
    """Rematch CPS demographic/occupation features for the clone half."""
    from policyengine_us import Microsimulation
    from sklearn.neighbors import NearestNeighbors

    cps_sim = Microsimulation(dataset=dataset_path)
    X_train = cps_sim.calculate_dataframe(
        CPS_CLONE_FEATURE_PREDICTORS + CPS_CLONE_FEATURE_VARIABLES
    )
    available_outputs = [
        variable
        for variable in CPS_CLONE_FEATURE_VARIABLES
        if variable in X_train.columns
    ]
    if not available_outputs:
        n_half = len(data["person_id"][time_period]) // 2
        return pd.DataFrame(index=np.arange(n_half))

    X_test = _build_clone_test_frame(
        cps_sim,
        data,
        time_period,
        CPS_CLONE_FEATURE_PREDICTORS,
    )
    del cps_sim

    train_roles = (
        X_train[["is_tax_unit_head", "is_tax_unit_spouse", "is_tax_unit_dependent"]]
        .round()
        .astype(int)
        .apply(tuple, axis=1)
    )
    test_roles = (
        X_test[["is_tax_unit_head", "is_tax_unit_spouse", "is_tax_unit_dependent"]]
        .round()
        .astype(int)
        .apply(tuple, axis=1)
    )

    predictions = pd.DataFrame(index=X_test.index, columns=available_outputs)
    for role in test_roles.unique():
        test_mask = test_roles == role
        train_mask = train_roles == role
        if not train_mask.any():
            train_mask = pd.Series(True, index=X_train.index)

        train_predictors = X_train.loc[train_mask, CPS_CLONE_FEATURE_PREDICTORS]
        test_predictors = X_test.loc[test_mask, CPS_CLONE_FEATURE_PREDICTORS]
        train_matrix = _prepare_knn_matrix(train_predictors)
        test_matrix = _prepare_knn_matrix(test_predictors, reference=train_predictors)

        matcher = NearestNeighbors(n_neighbors=1)
        matcher.fit(train_matrix)
        donor_indices = matcher.kneighbors(
            test_matrix,
            return_distance=False,
        ).ravel()
        donor_outputs = (
            X_train.loc[train_mask, available_outputs]
            .iloc[donor_indices]
            .reset_index(drop=True)
        )
        predictions.loc[test_mask, available_outputs] = donor_outputs.to_numpy()

    if "detailed_occupation_recode" in predictions:
        occupation_codes = (
            predictions["detailed_occupation_recode"].astype(float).to_numpy()
        )
        for column, values in _derive_overtime_occupation_inputs(
            occupation_codes
        ).items():
            predictions[column] = values

    return predictions


def _splice_clone_feature_predictions(
    data: dict,
    predictions: pd.DataFrame,
    time_period: int,
) -> dict:
    """Replace clone-half person-level feature variables with donor matches."""
    n_half = len(data["person_id"][time_period]) // 2
    for variable in predictions.columns:
        if variable not in data:
            continue
        values = data[variable][time_period]
        new_values = np.array(values, copy=True)
        pred_values = predictions[variable].to_numpy()
        if np.issubdtype(new_values.dtype, np.bool_):
            pred_values = pred_values.astype(bool, copy=False)
        else:
            pred_values = pred_values.astype(new_values.dtype, copy=False)
        new_values[n_half:] = pred_values
        data[variable] = {time_period: new_values}
    return data


def _impute_cps_only_variables(
    data: dict,
    time_period: int,
    dataset_path: str,
) -> pd.DataFrame:
    """Second-stage QRF: train on CPS, predict for PUF clones.

    For the PUF clone half of the extended CPS we need plausible values
    of CPS-only variables (retirement distributions, transfers, hours,
    SPM components, etc.) that are consistent with the clone's
    PUF-imputed income -- not just naively copied from the CPS donor.

    We train a QRF on CPS person-level data where:
      * predictors = demographics + key income variables
      * outputs    = CPS-only variables listed in
                     ``CPS_ONLY_IMPUTED_VARIABLES``

    For PUF clone prediction we use the PUF-imputed income values
    from the second half of ``data`` (the clone half, which already
    has PUF-imputed income from stage 1).

    Uses ``fit_predict()`` with ``max_train_samples`` instead of
    manual sampling + separate fit/predict.

    Args:
        data: Extended dataset dict after ``puf_clone_dataset()`` --
            already doubled, with PUF-imputed income in the second half.
        time_period: Tax year.
        dataset_path: Path to the CPS h5 file for Microsimulation.

    Returns:
        DataFrame with one column per CPS-only variable, containing
        predicted values for the PUF clone half (person-level).
    """
    from microimpute.models.qrf import QRF
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    all_predictors = CPS_STAGE2_DEMOGRAPHIC_PREDICTORS + CPS_STAGE2_INCOME_PREDICTORS

    # Filter to variables that exist in the current policyengine-us.
    tbs = CountryTaxBenefitSystem()
    valid_outputs = [v for v in CPS_ONLY_IMPUTED_VARIABLES if v in tbs.variables]
    skipped = set(CPS_ONLY_IMPUTED_VARIABLES) - set(valid_outputs)
    if skipped:
        logger.warning(
            "CPS-only imputation: %d variables not in tax-benefit system: %s",
            len(skipped),
            sorted(skipped),
        )

    # Load original (non-doubled) CPS for training data.
    cps_sim = Microsimulation(dataset=dataset_path)
    X_train = cps_sim.calculate_dataframe(all_predictors + valid_outputs)

    available_outputs = [col for col in valid_outputs if col in X_train.columns]
    missing_outputs = [col for col in valid_outputs if col not in X_train.columns]
    if missing_outputs:
        logger.warning(
            "CPS-only imputation: %d variables not found in CPS: %s",
            len(missing_outputs),
            missing_outputs,
        )

    # Build PUF clone test data from the clone half itself, falling back to
    # the CPS sim for formula variables that are not stored in the dataset.
    X_test = _build_clone_test_frame(
        cps_sim,
        data,
        time_period,
        all_predictors,
    )
    del cps_sim

    logger.info(
        "Stage-2 CPS-only imputation: %d outputs, "
        "training on %d CPS persons, predicting for %d PUF clones",
        len(available_outputs),
        len(X_train),
        len(X_test),
    )
    total_start = time.time()

    qrf = QRF(
        log_level="INFO",
        memory_efficient=True,
        max_train_samples=5000,
    )
    predictions = qrf.fit_predict(
        X_train=X_train[all_predictors + available_outputs],
        X_test=X_test[all_predictors],
        predictors=all_predictors,
        imputed_variables=available_outputs,
        n_jobs=1,
    )

    # Add zeros for variables that weren't available in CPS.
    for var in missing_outputs:
        predictions[var] = 0

    # Apply domain constraints to retirement and SS variables.
    predictions = _apply_post_processing(predictions, X_test, time_period, data)

    logger.info(
        "Stage-2 CPS-only imputation took %.2fs total",
        time.time() - total_start,
    )
    return predictions


def apply_retirement_constraints(predictions, X_test, time_period):
    """Enforce IRS contribution limits on retirement variable predictions.

    Args:
        predictions: DataFrame of QRF predictions for retirement
            contribution variables.
        X_test: DataFrame with at least ``age``,
            ``employment_income``, and ``self_employment_income``.
        time_period: Tax year (int) for IRS limit look-up.

    Returns:
        DataFrame with constrained values (same columns).
    """
    limits = get_retirement_limits(time_period)
    se_limits = get_se_pension_limits(time_period)

    age = X_test["age"].values
    catch_up = age >= 50
    emp_income = X_test["employment_income"].values
    se_income = X_test["self_employment_income"].values

    limit_401k = limits["401k"] + catch_up * limits["401k_catch_up"]
    limit_ira = limits["ira"] + catch_up * limits["ira_catch_up"]
    se_pension_cap = np.minimum(
        se_income * se_limits["se_pension_rate"],
        se_limits["se_pension_dollar_limit"],
    )

    # Explicit mapping: variable -> (cap array, zero_mask or None).
    _CONSTRAINT_MAP = {
        "traditional_401k_contributions": (limit_401k, emp_income == 0),
        "roth_401k_contributions": (limit_401k, emp_income == 0),
        "traditional_ira_contributions": (limit_ira, None),
        "roth_ira_contributions": (limit_ira, None),
        "self_employed_pension_contributions": (
            se_pension_cap,
            se_income == 0,
        ),
    }

    result = predictions.clip(lower=0)
    for var in result.columns:
        cap, zero_mask = _CONSTRAINT_MAP.get(var, (None, None))
        if cap is not None:
            result[var] = np.minimum(result[var].values, cap)
        if zero_mask is not None:
            result.loc[zero_mask, var] = 0

    return result


def reconcile_ss_subcomponents(predictions, total_ss):
    """Normalize Social Security sub-components to sum to total.

    Args:
        predictions: DataFrame with columns for each SS
            sub-component (retirement, disability, dependents,
            survivors).
        total_ss: numpy array of total social_security per record.

    Returns:
        DataFrame with reconciled dollar values.
    """
    values = np.maximum(predictions.values, 0)
    row_sums = values.sum(axis=1)
    positive_mask = total_ss > 0

    shares = np.zeros_like(values)
    nonzero_rows = row_sums > 0
    both = positive_mask & nonzero_rows
    shares[both] = values[both] / row_sums[both, np.newaxis]
    # If row_sum == 0 but total_ss > 0, distribute equally.
    equal_rows = positive_mask & ~nonzero_rows
    shares[equal_rows] = 1.0 / values.shape[1]

    out = np.where(
        positive_mask[:, np.newaxis],
        shares * total_ss[:, np.newaxis],
        0.0,
    )
    return pd.DataFrame(out, columns=predictions.columns)


_RETIREMENT_VARS = {
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "self_employed_pension_contributions",
}

_SS_SUBCOMPONENT_VARS = {
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
}


def derive_clone_capped_childcare_expenses(
    donor_pre_subsidy: np.ndarray,
    donor_capped: np.ndarray,
    clone_pre_subsidy: np.ndarray,
    clone_person_data: pd.DataFrame,
    clone_spm_unit_ids: np.ndarray,
) -> np.ndarray:
    """Derive clone-half capped childcare from clone inputs.

    The CPS provides both pre-subsidy childcare and the SPM-specific
    capped childcare deduction. For the clone half, we impute only the
    pre-subsidy amount, then deterministically rebuild the capped amount
    instead of letting a second QRF predict it independently.

    We preserve the donor's observed capping share while also respecting
    the clone's own earnings cap. This keeps the clone-half value
    consistent with pre-subsidy childcare and avoids impossible outputs
    such as capped childcare exceeding pre-subsidy childcare.
    """

    donor_pre_subsidy = np.asarray(donor_pre_subsidy, dtype=float)
    donor_capped = np.asarray(donor_capped, dtype=float)
    clone_pre_subsidy = np.asarray(clone_pre_subsidy, dtype=float)
    clone_spm_unit_ids = np.asarray(clone_spm_unit_ids)

    donor_cap_share = np.divide(
        donor_capped,
        donor_pre_subsidy,
        out=np.zeros_like(donor_capped, dtype=float),
        where=donor_pre_subsidy > 0,
    )
    donor_cap_share = np.clip(donor_cap_share, 0.0, 1.0)
    capped_from_share = np.maximum(clone_pre_subsidy, 0.0) * donor_cap_share

    if clone_person_data.empty:
        earnings_cap = np.zeros(len(clone_spm_unit_ids), dtype=float)
    else:
        eligible = clone_person_data["is_parent_proxy"].astype(bool)
        parent_rows = clone_person_data.loc[
            eligible, ["spm_unit_id", "age", "earnings"]
        ].copy()
        if parent_rows.empty:
            earnings_cap = np.zeros(len(clone_spm_unit_ids), dtype=float)
        else:
            parent_rows["earnings"] = parent_rows["earnings"].clip(lower=0.0)
            parent_rows["age_rank"] = parent_rows.groupby("spm_unit_id")["age"].rank(
                method="first", ascending=False
            )
            top_two = parent_rows[parent_rows["age_rank"] <= 2].sort_values(
                ["spm_unit_id", "age_rank"]
            )
            earnings_cap_by_unit = top_two.groupby("spm_unit_id")["earnings"].agg(
                lambda values: (
                    float(values.iloc[0])
                    if len(values) == 1
                    else float(np.minimum(values.iloc[0], values.iloc[1]))
                )
            )
            earnings_cap = earnings_cap_by_unit.reindex(
                clone_spm_unit_ids, fill_value=0.0
            ).to_numpy(dtype=float)

    return np.minimum(capped_from_share, earnings_cap)


def _rebuild_clone_capped_childcare_expenses(
    data: dict,
    time_period: int,
    cps_sim,
) -> np.ndarray:
    """Rebuild clone-half capped childcare expenses after stage-2 imputation."""

    n_persons_half = len(data["person_id"][time_period]) // 2
    n_spm_units_half = len(data["spm_unit_id"][time_period]) // 2

    person_roles = cps_sim.calculate_dataframe(
        ["age", "is_tax_unit_head", "is_tax_unit_spouse"]
    )
    if len(person_roles) != n_persons_half:
        raise ValueError(
            "Unexpected person role frame length while rebuilding clone childcare "
            f"expenses: got {len(person_roles)}, expected {n_persons_half}"
        )

    clone_person_data = pd.DataFrame(
        {
            "spm_unit_id": data["person_spm_unit_id"][time_period][n_persons_half:],
            "age": person_roles["age"].values,
            "is_parent_proxy": (
                person_roles["is_tax_unit_head"].values
                | person_roles["is_tax_unit_spouse"].values
            ),
            "earnings": (
                data["employment_income"][time_period][n_persons_half:]
                + data["self_employment_income"][time_period][n_persons_half:]
            ),
        }
    )

    donor_pre_subsidy = data["spm_unit_pre_subsidy_childcare_expenses"][time_period][
        :n_spm_units_half
    ]
    donor_capped = data["spm_unit_capped_work_childcare_expenses"][time_period][
        :n_spm_units_half
    ]
    clone_pre_subsidy = data["spm_unit_pre_subsidy_childcare_expenses"][time_period][
        n_spm_units_half:
    ]
    clone_spm_unit_ids = data["spm_unit_id"][time_period][n_spm_units_half:]

    return derive_clone_capped_childcare_expenses(
        donor_pre_subsidy=donor_pre_subsidy,
        donor_capped=donor_capped,
        clone_pre_subsidy=clone_pre_subsidy,
        clone_person_data=clone_person_data,
        clone_spm_unit_ids=clone_spm_unit_ids,
    )


def _apply_post_processing(predictions, X_test, time_period, data):
    """Apply retirement constraints and SS reconciliation."""
    ret_cols = [c for c in predictions.columns if c in _RETIREMENT_VARS]
    if ret_cols:
        constrained = apply_retirement_constraints(
            predictions[ret_cols], X_test, time_period
        )
        for col in ret_cols:
            predictions[col] = constrained[col]

    ss_cols = [c for c in predictions.columns if c in _SS_SUBCOMPONENT_VARS]
    if ss_cols:
        n_half = len(data["person_id"][time_period]) // 2
        total_ss = data["social_security"][time_period][n_half:]
        reconciled = reconcile_ss_subcomponents(predictions[ss_cols], total_ss)
        for col in ss_cols:
            predictions[col] = reconciled[col]

    org_cols = [c for c in predictions.columns if c in ORG_IMPUTED_VARIABLES]
    if org_cols:
        n_half = len(data["person_id"][time_period]) // 2
        weekly_hours = (
            predictions["weekly_hours_worked"].values
            if "weekly_hours_worked" in predictions.columns
            else data["weekly_hours_worked"][time_period][n_half:]
        )
        receiver = pd.DataFrame(
            {
                "employment_income": X_test["employment_income"].values,
                "weekly_hours_worked": np.asarray(weekly_hours, dtype=np.float32),
            }
        )
        constrained = apply_org_domain_constraints(
            predictions[org_cols],
            receiver,
            self_employment_income=X_test["self_employment_income"].values,
        )
        for col in org_cols:
            predictions[col] = constrained[col]

    return predictions


def _splice_cps_only_predictions(
    data: dict,
    predictions: pd.DataFrame,
    time_period: int,
    dataset_path: str,
) -> dict:
    """Replace PUF clone half of CPS-only variables with QRF predictions.

    After ``puf_clone_dataset()`` the CPS-only variables in the second
    half are naive copies of the CPS donor values. This function
    replaces them with the second-stage QRF predictions that are
    consistent with the clone's PUF-imputed income.

    Args:
        data: Extended dataset dict (already doubled).
        predictions: DataFrame from ``_impute_cps_only_variables()``.
        time_period: Tax year.
        dataset_path: Path to CPS h5 file for entity mapping.

    Returns:
        Modified data dict with CPS-only variables spliced in.
    """
    from policyengine_us import Microsimulation

    cps_sim = Microsimulation(dataset=dataset_path)
    tbs = cps_sim.tax_benefit_system

    # Pre-compute half-lengths per entity so we split each
    # variable's array at the correct midpoint.
    entity_half_lengths = {}
    for entity_key in ["person", "tax_unit", "spm_unit", "family", "household"]:
        id_var = f"{entity_key}_id"
        if id_var in data:
            entity_half_lengths[entity_key] = len(data[id_var][time_period]) // 2

    for var in CPS_ONLY_IMPUTED_VARIABLES:
        if var not in data or var not in predictions.columns:
            continue

        pred_values = predictions[var].values
        var_meta = tbs.variables.get(var)
        entity_key = var_meta.entity.key if var_meta is not None else "person"

        if entity_key != "person":
            pred_values = cps_sim.populations[entity_key].value_from_first_person(
                pred_values
            )

        n_half = entity_half_lengths.get(entity_key, len(data[var][time_period]) // 2)
        if len(pred_values) != n_half:
            raise ValueError(
                f"Stage-2 prediction for '{var}' has {len(pred_values)} "
                f"entries but expected {n_half} (half of {entity_key})"
            )
        values = data[var][time_period]
        # First half: keep original CPS values.
        # Second half: replace with QRF predictions.
        cps_half = values[:n_half]
        new_values = np.concatenate([cps_half, pred_values])
        data[var] = {time_period: new_values}

    if (
        "spm_unit_capped_work_childcare_expenses" in data
        and "spm_unit_pre_subsidy_childcare_expenses" in data
    ):
        n_half = entity_half_lengths.get(
            "spm_unit",
            len(data["spm_unit_capped_work_childcare_expenses"][time_period]) // 2,
        )
        cps_half = data["spm_unit_capped_work_childcare_expenses"][time_period][:n_half]
        clone_half = _rebuild_clone_capped_childcare_expenses(
            data=data,
            time_period=time_period,
            cps_sim=cps_sim,
        )
        data["spm_unit_capped_work_childcare_expenses"] = {
            time_period: np.concatenate([cps_half, clone_half])
        }

    del cps_sim
    return data


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation

        from policyengine_us_data.calibration.clone_and_assign import (
            load_global_block_distribution,
        )
        from policyengine_us_data.calibration.puf_impute import (
            puf_clone_dataset,
        )

        logger.info("Loading CPS dataset: %s", self.cps)
        cps_sim = Microsimulation(dataset=self.cps)
        data = cps_sim.dataset.load_dataset()
        del cps_sim

        data_dict = {}
        for var in data:
            data_dict[var] = {self.time_period: data[var][...]}

        n_hh = len(data_dict["household_id"][self.time_period])
        _, _, block_states, block_probs = load_global_block_distribution()
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(block_states), size=n_hh, p=block_probs)
        state_fips = block_states[indices]

        logger.info("PUF clone with dataset: %s", self.puf)
        new_data = puf_clone_dataset(
            data=data_dict,
            state_fips=state_fips,
            time_period=self.time_period,
            puf_dataset=self.puf,
            dataset_path=str(self.cps.file_path),
        )

        # Stage 2a: donor-impute CPS feature variables for PUF clones.
        logger.info("Stage-2a: rematching CPS features for PUF clones")
        clone_feature_predictions = _impute_clone_cps_features(
            data=new_data,
            time_period=self.time_period,
            dataset_path=str(self.cps.file_path),
        )
        new_data = _splice_clone_feature_predictions(
            data=new_data,
            predictions=clone_feature_predictions,
            time_period=self.time_period,
        )

        # Stage 2b: QRF-impute CPS-only continuous variables for PUF clones.
        # Train on CPS data using demographics + PUF-imputed income
        # as predictors, so the PUF clone half gets values consistent
        # with its imputed income rather than naive donor duplication.
        logger.info("Stage-2b: imputing CPS-only variables for PUF clones")
        cps_only_predictions = _impute_cps_only_variables(
            data=new_data,
            time_period=self.time_period,
            dataset_path=str(self.cps.file_path),
        )
        new_data = _splice_cps_only_predictions(
            data=new_data,
            predictions=cps_only_predictions,
            time_period=self.time_period,
            dataset_path=str(self.cps.file_path),
        )

        new_data = self._rename_imputed_to_inputs(new_data)
        if _supports_structural_mortgage_inputs():
            new_data = impute_tax_unit_mortgage_balance_hints(
                new_data,
                self.time_period,
            )
            new_data = convert_mortgage_interest_to_structural_inputs(
                new_data,
                self.time_period,
            )
        new_data = self._drop_formula_variables(new_data)
        self.save_dataset(new_data)

    @classmethod
    def _rename_imputed_to_inputs(cls, data):
        """Rename QRF-imputed formula vars to their leaf inputs.

        The QRF imputes formula-level aggregates (e.g.
        taxable_pension_income) but the engine needs leaf inputs
        (e.g. taxable_private_pension_income) so formulas work.
        """
        for formula_var, input_var in cls._IMPUTED_TO_INPUT.items():
            if formula_var in data:
                logger.info(
                    "Renaming %s -> %s (leaf input)",
                    formula_var,
                    input_var,
                )
                data[input_var] = data.pop(formula_var)
        return data

    # Variables with formulas/adds that must still be stored.
    # Includes IDs needed before formulas run and tax-unit-level
    # QRF-imputed vars that can't be renamed to person-level leaves
    # due to entity shape mismatch.
    _KEEP_FORMULA_VARS = {
        "person_id",
        "self_employed_pension_contribution_ald",
        "self_employed_health_insurance_ald",
    }

    @classmethod
    def _keep_formula_vars(cls):
        keep = set(cls._KEEP_FORMULA_VARS)
        if not _supports_structural_mortgage_inputs():
            keep.add("interest_deduction")
        return keep

    # QRF imputes formula-level variables (e.g. taxable_pension_income)
    # but we must store them under leaf input names so
    # _drop_formula_variables doesn't discard them. The engine then
    # recomputes the formula var from its adds.
    # NOTE: only same-entity renames here; cross-entity vars
    # (tax_unit -> person) go in _KEEP_FORMULA_VARS instead.
    _IMPUTED_TO_INPUT = {
        "taxable_pension_income": "taxable_private_pension_income",
        "tax_exempt_pension_income": "tax_exempt_private_pension_income",
    }

    @classmethod
    def _drop_formula_variables(cls, data):
        """Remove variables that are computed by policyengine-us.

        Variables with formulas, ``adds``, or ``subtracts`` are
        recomputed by the simulation engine, so storing them wastes
        space and can mislead validation.

        Aggregate variables whose ``adds`` include a behavioral-
        response input (e.g. ``employment_income_before_lsr``) are
        renamed to that input before dropping so the raw data is
        preserved under the correct input-variable name.
        """
        from policyengine_us import CountryTaxBenefitSystem

        tbs = CountryTaxBenefitSystem()

        _RESPONSE_SUFFIXES = ("_before_lsr", "_before_response")
        for name, var in tbs.variables.items():
            if name not in data:
                continue
            for add_var in getattr(var, "adds", None) or []:
                if any(add_var.endswith(s) for s in _RESPONSE_SUFFIXES):
                    if add_var not in data:
                        logger.info(
                            "Renaming %s -> %s before drop",
                            name,
                            add_var,
                        )
                        data[add_var] = data.pop(name)
                    break

        formula_vars = {
            name
            for name, var in tbs.variables.items()
            if (hasattr(var, "formulas") and len(var.formulas) > 0)
            or getattr(var, "adds", None)
            or getattr(var, "subtracts", None)
        } - cls._keep_formula_vars()
        dropped = sorted(set(data.keys()) & formula_vars)
        if dropped:
            logger.info(
                "Dropping %d formula variables: %s",
                len(dropped),
                dropped,
            )
            for var in dropped:
                del data[var]
        return data


class ExtendedCPS_2024(ExtendedCPS):
    cps = CPS_2024_Full
    puf = PUF_2024
    name = "extended_cps_2024"
    label = "Extended CPS (2024)"
    file_path = STORAGE_FOLDER / "extended_cps_2024.h5"
    time_period = 2024


class ExtendedCPS_2024_Half(ExtendedCPS):
    cps = CPS_2024
    puf = PUF_2024
    name = "extended_cps_2024_half"
    label = "Extended CPS 2024 (half sample)"
    file_path = STORAGE_FOLDER / "extended_cps_2024_half.h5"
    time_period = 2024


if __name__ == "__main__":
    ExtendedCPS_2024().generate()
    ExtendedCPS_2024_Half().generate()
