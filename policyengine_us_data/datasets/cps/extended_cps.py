import logging
import time
from typing import Type

import numpy as np
import pandas as pd
from policyengine_core.data import Dataset

from policyengine_us_data.calibration.formulaic_inputs import (
    FORMULAIC_SPM_INPUTS_TO_DROP,
)
from policyengine_us_data.calibration.puf_impute import (
    CLONE_ORIGIN_FLAGS,
    IMPUTED_VARIABLES,
    OVERRIDDEN_IMPUTED_VARIABLES,
)
from policyengine_us_data.datasets.cps.cps import (
    CPS,
    CPS_2024,
    CPS_2024_Full,
    ESI_POLICYHOLDER_VARIABLE,
    FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES,
    FLSA_OVERTIME_OCCUPATION_CODES,
    _open_dataset_read_only,
    derive_flsa_overtime_premium,
    load_take_up_rate,
)
from policyengine_us_data.datasets.cps.medicaid_cost import (
    add_medicaid_cost_if_enrolled_to_time_period_data,
)
from policyengine_us_data.datasets.cps.takeup import prioritize_reported_recipients
from policyengine_us_data.datasets.org import (
    ORG_IMPUTED_VARIABLES,
    apply_org_domain_constraints,
)
from policyengine_us_data.datasets.sipp import (
    SSI_DISABILITY_CRITERIA_VARIABLE,
    SSI_DISABILITY_MODEL_PREDICTORS,
    get_ssi_disability_model,
    predict_ssi_disability_criteria,
)
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.datasets.puf import PUF, PUF_2024
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.aotc import (
    maximum_american_opportunity_credit_per_student,
    qualifying_expenses_from_american_opportunity_credit,
)
from policyengine_us_data.utils.mortgage_interest import (
    STRUCTURAL_MORTGAGE_VARIABLES,
    convert_mortgage_interest_to_structural_inputs,
    impute_tax_unit_mortgage_balance_hints,
)
from policyengine_us_data.utils.policyengine import has_policyengine_us_variables
from policyengine_us_data.utils.dataset_validation import (
    assert_no_computed_policyengine_us_variables_exported,
)
from policyengine_us_data.utils.retirement_limits import (
    get_retirement_limits,
    get_se_pension_limits,
)
from policyengine_us_data.utils.randomness import seeded_rng

logger = logging.getLogger(__name__)


AOTC_ELIGIBILITY_INPUTS = (
    "is_pursuing_credential_for_american_opportunity_credit",
    "attends_eligible_educational_institution_for_american_opportunity_credit",
    "is_enrolled_at_least_half_time_for_american_opportunity_credit",
    "has_american_opportunity_credit_1098_t_or_exception",
    "has_american_opportunity_credit_institution_ein",
    "has_completed_first_four_years_of_postsecondary_education",
    "has_felony_drug_conviction",
    "american_opportunity_credit_claimed_prior_years",
)


LLC_ELIGIBILITY_INPUTS = (
    "attends_eligible_educational_institution_for_lifetime_learning_credit",
    "has_lifetime_learning_credit_1098_t_or_exception",
)


def _supports_aotc_eligibility_inputs() -> bool:
    return has_policyengine_us_variables(*AOTC_ELIGIBILITY_INPUTS)


def _supports_llc_eligibility_inputs() -> bool:
    return has_policyengine_us_variables(*LLC_ELIGIBILITY_INPUTS)


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

PUF_IMPUTED_VARIABLES = set(IMPUTED_VARIABLES) | set(OVERRIDDEN_IMPUTED_VARIABLES)

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

_OVERTIME_OCCUPATION_CODES = dict(FLSA_OVERTIME_OCCUPATION_CODES)
FLSA_OVERTIME_PREMIUM_VARIABLE = "fsla_overtime_premium"
FLSA_OVERTIME_PREMIUM_INPUTS = (
    "employment_income",
    "hours_worked_last_week",
    "weeks_worked",
    "is_paid_hourly",
    "has_never_worked",
    "is_military",
    "is_executive_administrative_professional",
    "is_farmer_fisher",
    "is_computer_scientist",
)
CLONE_DERIVED_VARIABLES = frozenset({FLSA_OVERTIME_PREMIUM_VARIABLE})
# CPS-only variables that should be QRF-imputed for the PUF clone half
# instead of naively duplicated from the CPS donor. Most demographics,
# IDs, weights, and random seeds are fine to duplicate; the categorical
# clone features above are rematched separately, and clone-derived variables
# are recomputed from final clone inputs after QRF splicing.
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
    "child_support_received",
    "veterans_benefits",
    "workers_compensation",
    "educational_assistance",
    "financial_assistance",
    "survivor_benefits",
    "disability_benefits",
    SSI_DISABILITY_CRITERIA_VARIABLE,
    "strike_benefits",
    "receives_wic",
    # SPM variables
    "receives_housing_assistance",
    "spm_unit_energy_subsidy",
    "spm_unit_pre_subsidy_childcare_expenses",
    # Medical expenses
    "employer_sponsored_insurance_premiums",
    "health_insurance_premiums_without_medicare_part_b",
    "other_health_insurance_premiums",
    "over_the_counter_health_expenses",
    "other_medical_expenses",
    "child_support_expense",
    # Hours/employment
    "weekly_hours_worked",
    "hours_worked_last_week",
    "weeks_worked",
    # ORG labor-market variables
    "hourly_wage",
    "is_paid_hourly",
    "is_union_member_or_covered",
    # Previous year income
    "employment_income_last_year",
    "self_employment_income_last_year",
]

# Set for O(1) lookup in the splice loop.
_CPS_ONLY_SET = set(CPS_ONLY_IMPUTED_VARIABLES)

_CLONE_REFRESH_GEOGRAPHY_VARIABLES = {
    "block_geoid",
    "cbsa_code",
    "congressional_district_geoid",
    "county",
    "county_fips",
    "place_fips",
    "puma",
    "sldl",
    "sldu",
    "state_fips",
    "tract_geoid",
    "vtd",
    "zcta",
    "zip_code",
}

_CLONE_REFRESH_ANCHOR_VARIABLES = {
    "age",
}

_CLONE_REFRESH_STRUCTURAL_ROLE_VARIABLES = {
    "is_household_head",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
    "is_tax_unit_head_or_spouse",
    "is_family_head",
    "is_family_spouse",
    "is_family_dependent",
    "is_spm_unit_head",
    "is_spm_unit_spouse",
    "is_spm_unit_dependent",
}

# Predictors used for the second-stage CPS-only imputation: demographics
# plus key income variables that were already imputed from PUF data.
CPS_STAGE2_DEMOGRAPHIC_PREDICTORS = [
    "age",
    "is_male",
    "has_esi",
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


def _first_half_person_values(data: dict, variable: str, time_period: int):
    """Return original-CPS-half values for person-level variables."""
    if variable not in data:
        return None

    values = data[variable][time_period]
    n_persons = len(data["person_id"][time_period])
    if len(values) != n_persons:
        return None

    return np.asarray(values[: n_persons // 2])


def _is_structural_clone_variable(variable: str) -> bool:
    """Return whether a variable should remain copied, not rematched."""
    return (
        variable.endswith("_id")
        or variable.endswith("_weight")
        or variable in _CLONE_REFRESH_GEOGRAPHY_VARIABLES
        or variable in CLONE_ORIGIN_FLAGS.values()
        or variable in _CLONE_REFRESH_ANCHOR_VARIABLES
        or variable in _CLONE_REFRESH_STRUCTURAL_ROLE_VARIABLES
        or variable in _STAGE2_COMPUTED_PREDICTORS
    )


def _cps_clone_feature_variables_for_data(
    data: dict,
    time_period: int,
) -> list[str]:
    """Return person-level CPS-only fields to donor-rematch onto PUF clones.

    The PUF clone starts as a literal copy of each CPS donor, then selected
    tax/income fields are replaced with PUF-imputed values. Any remaining
    person-level CPS-only field should be refreshed from CPS donors unless it
    is structural, a PUF-imputed field, or a QRF-handled CPS-only output.
    """
    result = []
    seen = set()
    explicit_clone_features = set(CPS_CLONE_FEATURE_VARIABLES)
    for variable in [*CPS_CLONE_FEATURE_VARIABLES, *data.keys()]:
        if variable in seen:
            continue
        seen.add(variable)
        if (
            variable in PUF_IMPUTED_VARIABLES
            or variable in _CPS_ONLY_SET
            or variable in CLONE_DERIVED_VARIABLES
        ):
            continue
        is_explicit_clone_feature = variable in explicit_clone_features
        if not is_explicit_clone_feature and _is_structural_clone_variable(variable):
            continue
        if (
            not is_explicit_clone_feature
            and _first_half_person_values(data, variable, time_period) is None
        ):
            continue
        result.append(variable)
    return result


def _build_cps_train_frame(
    cps_sim,
    data: dict,
    time_period: int,
    variables: list[str],
) -> pd.DataFrame:
    """Build original-CPS-half training values from PE or stored data."""
    tbs = getattr(cps_sim, "tax_benefit_system", None)
    if tbs is None:
        calculable_variables = variables
    else:
        calculable_variables = [
            variable for variable in variables if variable in tbs.variables
        ]
    if calculable_variables:
        train = cps_sim.calculate_dataframe(calculable_variables).copy()
    else:
        n_half = len(data["person_id"][time_period]) // 2
        train = pd.DataFrame(index=np.arange(n_half))

    for variable in variables:
        if variable in train.columns:
            continue
        values = _first_half_person_values(data, variable, time_period)
        if values is not None:
            train[variable] = values

    return train


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


def _build_ssi_disability_clone_receiver(
    predictions: pd.DataFrame,
    X_test: pd.DataFrame,
    data: dict,
    time_period: int,
) -> pd.DataFrame:
    """Build SIPP SSI disability model inputs for PUF clone records."""
    n = len(X_test)
    receiver = pd.DataFrame(index=X_test.index)
    for predictor in SSI_DISABILITY_MODEL_PREDICTORS:
        values = None
        if (
            predictor == "has_disability_income"
            and "disability_benefits" in predictions
        ):
            values = predictions["disability_benefits"].to_numpy() > 0
        elif predictor in predictions:
            values = predictions[predictor].to_numpy()
        elif predictor in X_test:
            values = X_test[predictor].to_numpy()
        else:
            clone_values = _clone_half_person_values(data, predictor, time_period)
            if clone_values is not None and len(clone_values) == n:
                values = clone_values

        if values is None and predictor == "is_female" and "is_male" in X_test:
            values = ~X_test["is_male"].astype(bool).to_numpy()
        if values is None:
            values = np.zeros(n)

        receiver[predictor] = values

    return receiver


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
        FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES,
    )
    return pd.DataFrame(derived)


def _derive_clone_flsa_overtime_premium(data: dict, time_period: int) -> dict:
    """Recompute clone-half FLSA overtime premiums from final clone inputs."""
    if FLSA_OVERTIME_PREMIUM_VARIABLE not in data:
        return data

    missing = [
        variable
        for variable in FLSA_OVERTIME_PREMIUM_INPUTS
        if variable not in data or time_period not in data[variable]
    ]
    if missing:
        raise ValueError(
            "Cannot derive clone FLSA overtime premium; missing inputs: "
            + ", ".join(missing)
        )

    values = np.array(data[FLSA_OVERTIME_PREMIUM_VARIABLE][time_period], copy=True)
    n_persons = len(data["person_id"][time_period])
    if len(values) != n_persons:
        raise ValueError(
            "fsla_overtime_premium must be person-level to derive clone values"
        )

    n_half = n_persons // 2
    clone_inputs = {
        variable: np.asarray(data[variable][time_period][n_half:])
        for variable in FLSA_OVERTIME_PREMIUM_INPUTS
    }
    values[n_half:] = derive_flsa_overtime_premium(
        time_period=time_period,
        **clone_inputs,
    ).astype(values.dtype, copy=False)
    data[FLSA_OVERTIME_PREMIUM_VARIABLE] = {time_period: values}
    return data


def _impute_clone_cps_features(
    data: dict,
    time_period: int,
    dataset_path: str,
) -> pd.DataFrame:
    """Rematch CPS demographic/occupation features for the clone half."""
    from policyengine_us import Microsimulation
    from sklearn.neighbors import NearestNeighbors

    cps_sim = Microsimulation(dataset=dataset_path)
    feature_variables = _cps_clone_feature_variables_for_data(data, time_period)
    X_train = _build_cps_train_frame(
        cps_sim,
        data,
        time_period,
        CPS_CLONE_FEATURE_PREDICTORS + feature_variables,
    )
    available_outputs = [
        variable for variable in feature_variables if variable in X_train.columns
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


@pipeline_node(
    PipelineNode(
        id="clone_features",
        label="Splice Clone Features",
        node_type="process",
        description=(
            "Replaces clone-half CPS feature variables with donor-matched "
            "predictions so doubled records retain plausible demographics and "
            "occupation labels."
        ),
        status="transitional",
        stability="moving",
        pathways=["data_build"],
        artifacts_in=["qrf_pass2", "record_double"],
        artifacts_out=["clone_feature_splice"],
        pydoc=True,
    )
)
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


@pipeline_node(
    PipelineNode(
        id="cps_only",
        label="Impute CPS-Only Variables",
        node_type="process",
        description=(
            "Runs the second-stage CPS-only QRF imputation for PUF clone "
            "records inside the extended CPS build."
        ),
        status="transitional",
        stability="moving",
        pathways=["data_build"],
        artifacts_in=["record_double", "preprocess_cps"],
        artifacts_out=["cps_only_predictions"],
        pydoc=True,
    )
)
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

_PUF_COMPUTED_INTERMEDIATES_AFTER_CLONE = {
    "cdcc_relevant_expenses",
    "pre_tax_contributions",
    "self_employed_health_insurance_ald",
    "self_employed_pension_contribution_ald",
}

_STAGE2_COMPUTED_PREDICTORS = {
    "is_male",
    "is_tax_unit_dependent",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "tax_unit_count_dependents",
    "tax_unit_is_joint",
}

_STAGE2_COMPUTED_OUTPUTS_TO_DROP = {
    "employment_income_last_year",
}

_COMPUTED_AGGREGATE_INPUT_RENAMES = {
    "employment_income": "employment_income_before_lsr",
    "long_term_capital_gains": "long_term_capital_gains_before_response",
    "self_employment_income": "self_employment_income_before_lsr",
    "sstb_self_employment_income": "sstb_self_employment_income_before_lsr",
    "weekly_hours_worked": "weekly_hours_worked_before_lsr",
}

_HOUSING_ASSISTANCE_FORMULA_OUTPUTS = {
    "housing_assistance",
    "spm_unit_capped_housing_subsidy",
}
_FINAL_COMPUTED_OUTPUTS_TO_DROP = {
    *FORMULAIC_SPM_INPUTS_TO_DROP,
    "dividend_income",
    "interest_income",
    "rent",
    "spm_unit_capped_work_childcare_expenses",
}
# The PE formula reconstruction is a guard against missing housing assistance
# inputs, not a calibration target for the Census SPM raw housing-subsidy field.
# Production CPS builds have a roughly 49% formula/raw ratio, so leave margin
# for that observed gap while still catching clearly broken reconstructions.
_MIN_MODELED_HOUSING_SHARE_OF_BENCHMARK = 0.45


class _InMemoryTimePeriodDataset(Dataset):
    name = "extended_cps_validation"
    label = "Extended CPS validation"
    data_format = Dataset.TIME_PERIOD_ARRAYS
    file_path = STORAGE_FOLDER / "extended_cps_validation.h5"

    def __init__(self, data: dict, time_period: int):
        self._data = data
        self.time_period = time_period
        super().__init__()

    def load(self):
        return self._data

    def load_dataset(self):
        return self._data


def _load_raw_spm_capped_housing_subsidy(
    cps_dataset,
    time_period: int,
    target_spm_unit_ids=None,
):
    """Load Census SPM capped housing subsidy for validation only."""

    raw_cps = getattr(cps_dataset, "raw_cps", None)
    if raw_cps is None:
        return None

    with _open_dataset_read_only(raw_cps) as raw_data:
        spm_unit = raw_data["spm_unit"]
        if "SPM_CAPHOUSESUB" not in spm_unit.columns:
            return None
        values = np.asarray(spm_unit["SPM_CAPHOUSESUB"], dtype=float)
        if target_spm_unit_ids is not None:
            if "SPM_ID" in spm_unit.columns:
                raw_spm_unit_ids = np.asarray(spm_unit["SPM_ID"])
            else:
                raw_spm_unit_ids = np.asarray(spm_unit.index)
            raw_index = pd.Index(raw_spm_unit_ids.astype(str))
            target_index = pd.Index(np.asarray(target_spm_unit_ids).astype(str))
            aligned = pd.Series(values, index=raw_index).reindex(target_index)
            if aligned.isna().any():
                missing_count = int(aligned.isna().sum())
                logger.warning(
                    "Skipping raw SPM capped housing subsidy validation benchmark "
                    "because %d CPS SPM unit IDs are absent from raw ASEC.",
                    missing_count,
                )
                return None
            values = aligned.to_numpy(dtype=float)

    return {time_period: values}


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

    if "employer_sponsored_insurance_premiums" in predictions.columns:
        policyholder = _clone_half_person_values(
            data, ESI_POLICYHOLDER_VARIABLE, time_period
        )
        if policyholder is not None:
            predictions.loc[
                ~np.asarray(policyholder, dtype=bool),
                "employer_sponsored_insurance_premiums",
            ] = 0

    if SSI_DISABILITY_CRITERIA_VARIABLE in predictions.columns:
        receiver = _build_ssi_disability_clone_receiver(
            predictions,
            X_test,
            data,
            time_period,
        )
        disability_screen = predict_ssi_disability_criteria(
            get_ssi_disability_model(time_period=time_period),
            receiver,
        )
        predictions[SSI_DISABILITY_CRITERIA_VARIABLE] = disability_screen

    return predictions


@pipeline_node(
    PipelineNode(
        id="qrf_pass2",
        label="Splice CPS-Only Predictions",
        node_type="process",
        description=(
            "Writes second-stage CPS-only QRF predictions back into the PUF "
            "clone half of the extended CPS record set."
        ),
        status="transitional",
        stability="moving",
        pathways=["data_build"],
        artifacts_in=["cps_only_predictions"],
        artifacts_out=["extended_cps_stage2"],
        pydoc=True,
    )
)
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

    del cps_sim
    return data


class ExtendedCPS(Dataset):
    cps: Type[CPS]
    puf: Type[PUF]
    data_format = Dataset.TIME_PERIOD_ARRAYS

    def generate(self):
        from policyengine_us import Microsimulation

        from policyengine_us_data.calibration.clone_and_assign import (
            assign_geography_within_state_county,
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
        spm_unit_ids = data_dict.get("spm_unit_id", {}).get(self.time_period)
        raw_spm_capped_housing_subsidy = _load_raw_spm_capped_housing_subsidy(
            self.cps,
            self.time_period,
            target_spm_unit_ids=spm_unit_ids,
        )
        if raw_spm_capped_housing_subsidy is not None:
            data_dict["spm_unit_capped_housing_subsidy"] = (
                raw_spm_capped_housing_subsidy
            )

        state_fips = data_dict["state_fips"][self.time_period]
        county_fips = data_dict.get("county_fips", {}).get(self.time_period)
        geography = assign_geography_within_state_county(
            state_fips=state_fips,
            county_fips=county_fips,
            seed=42,
        )

        logger.info("PUF clone with dataset: %s", self.puf)
        new_data = puf_clone_dataset(
            data=data_dict,
            state_fips=geography.state_fips,
            block_geoid=geography.block_geoid,
            cd_geoid=geography.cd_geoid,
            county_fips=geography.county_fips,
            time_period=self.time_period,
            puf_dataset=self.puf,
            dataset_path=str(self.cps.file_path),
        )
        new_data = self._drop_puf_computed_intermediates(new_data)

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
        new_data = _derive_clone_flsa_overtime_premium(new_data, self.time_period)
        new_data = self._finalize_stage2_computed_variables(new_data)

        new_data = self._impute_aotc_eligibility_inputs(new_data, self.time_period)
        new_data = self._impute_llc_eligibility_inputs(new_data, self.time_period)
        new_data = self._rename_imputed_to_inputs(new_data)
        new_data = self._reassign_housing_assistance_takeup_with_geography(
            new_data,
            self.time_period,
        )
        new_data = add_medicaid_cost_if_enrolled_to_time_period_data(
            new_data,
            self.time_period,
        )
        new_data = self._validate_housing_assistance_microsimulation(
            new_data,
            self.time_period,
        )
        new_data = self._drop_housing_assistance_formula_outputs(new_data)
        if _supports_structural_mortgage_inputs():
            had_positive_mortgage_input = self._has_positive_mortgage_input(
                new_data,
                self.time_period,
            )
            new_data = impute_tax_unit_mortgage_balance_hints(
                new_data,
                self.time_period,
            )
            new_data = convert_mortgage_interest_to_structural_inputs(
                new_data,
                self.time_period,
            )
            self._validate_structural_mortgage_conversion(
                new_data,
                self.time_period,
                had_positive_mortgage_input,
            )
        new_data = self._drop_final_computed_outputs(new_data)
        new_data = self._assert_no_computed_variables_exported(
            new_data,
            self.time_period,
        )
        self.save_dataset(new_data)

    @classmethod
    def _impute_aotc_eligibility_inputs(cls, data, time_period):
        """Convert AOTC source signals to person eligibility inputs."""
        credit = data.get("american_opportunity_credit", {}).get(time_period)
        tax_unit_ids = data.get("tax_unit_id", {}).get(time_period)
        person_tax_unit_ids = data.get("person_tax_unit_id", {}).get(time_period)
        tuition = data.get("qualified_tuition_expenses", {}).get(time_period)
        if tax_unit_ids is None or person_tax_unit_ids is None or tuition is None:
            return data

        credit = np.asarray(credit) if credit is not None else None
        tax_unit_ids = np.asarray(tax_unit_ids)
        person_tax_unit_ids = np.asarray(person_tax_unit_ids)
        tuition = np.array(tuition, copy=True)
        if (credit is not None and len(credit) != len(tax_unit_ids)) or len(
            tuition
        ) != len(person_tax_unit_ids):
            logger.warning(
                "Skipping AOTC eligibility imputation due to entity length mismatch"
            )
            return data

        aotc_student = np.zeros(len(person_tax_unit_ids), dtype=bool)

        full_time = data.get("is_full_time_college_student", {}).get(time_period)
        full_time = (
            np.asarray(full_time, dtype=bool)
            if full_time is not None
            else np.zeros(len(person_tax_unit_ids), dtype=bool)
        )
        dependent = data.get("is_tax_unit_dependent", {}).get(time_period)
        dependent = (
            np.asarray(dependent, dtype=bool)
            if dependent is not None
            else np.zeros(len(person_tax_unit_ids), dtype=bool)
        )

        adjusted_tuition_count = 0
        signal_tax_unit_count = 0
        if credit is not None:
            positive_credit = credit > 0
            if not positive_credit.any():
                return data

            positive_credit_units = tax_unit_ids[positive_credit]
            signal_tax_unit_count = int(positive_credit.sum())
            credit_by_tax_unit_id = dict(zip(tax_unit_ids, credit))
            max_student_credit = maximum_american_opportunity_credit_per_student(
                time_period
            )
            for tax_unit_id in positive_credit_units:
                member_indices = np.flatnonzero(person_tax_unit_ids == tax_unit_id)
                if member_indices.size == 0 or max_student_credit <= 0:
                    continue

                tuition_indices = member_indices[tuition[member_indices] > 0]
                candidate_groups = []
                if tuition_indices.size > 0:
                    candidate_groups.append(tuition_indices)
                candidate_groups.extend(
                    (
                        member_indices[full_time[member_indices]],
                        member_indices[dependent[member_indices]],
                        member_indices,
                    )
                )
                ordered_candidates = []
                seen = set()
                for group in candidate_groups:
                    for index in group:
                        if index not in seen:
                            ordered_candidates.append(index)
                            seen.add(index)

                remaining_credit = float(credit_by_tax_unit_id[tax_unit_id])
                for selected in ordered_candidates:
                    if remaining_credit <= 0:
                        break
                    student_credit = min(remaining_credit, max_student_credit)
                    target_tuition = (
                        qualifying_expenses_from_american_opportunity_credit(
                            student_credit,
                            time_period,
                        )
                    )
                    if tuition[selected] != target_tuition:
                        adjusted_tuition_count += 1
                    aotc_student[selected] = True
                    tuition[selected] = target_tuition
                    remaining_credit -= student_credit
        else:
            aotc_student = tuition > 0
            if not aotc_student.any():
                return data
            signal_tax_unit_count = len(np.unique(person_tax_unit_ids[aotc_student]))

        if not _supports_aotc_eligibility_inputs():
            existing = data.get("is_eligible_for_american_opportunity_credit", {}).get(
                time_period
            )
            values = (
                np.asarray(existing, dtype=bool).copy()
                if existing is not None
                else np.zeros(len(person_tax_unit_ids), dtype=bool)
            )
            values[aotc_student] = True
            data["is_eligible_for_american_opportunity_credit"] = {time_period: values}
            data["qualified_tuition_expenses"] = {time_period: tuition}
            logger.info(
                "AOTC eligibility imputation populated the legacy "
                "eligibility input for %d people across %d tax units "
                "and adjusted tuition for %d people",
                int(aotc_student.sum()),
                signal_tax_unit_count,
                adjusted_tuition_count,
            )
            return data

        for variable in (
            "is_pursuing_credential_for_american_opportunity_credit",
            "attends_eligible_educational_institution_for_american_opportunity_credit",
            "is_enrolled_at_least_half_time_for_american_opportunity_credit",
            "has_american_opportunity_credit_1098_t_or_exception",
            "has_american_opportunity_credit_institution_ein",
        ):
            existing = data.get(variable, {}).get(time_period)
            values = (
                np.asarray(existing, dtype=bool).copy()
                if existing is not None
                else np.zeros(len(person_tax_unit_ids), dtype=bool)
            )
            values[aotc_student] = True
            data[variable] = {time_period: values}

        for variable in (
            "has_completed_first_four_years_of_postsecondary_education",
            "has_felony_drug_conviction",
        ):
            existing = data.get(variable, {}).get(time_period)
            values = (
                np.asarray(existing, dtype=bool).copy()
                if existing is not None
                else np.zeros(len(person_tax_unit_ids), dtype=bool)
            )
            values[aotc_student] = False
            data[variable] = {time_period: values}

        existing_prior_years = data.get(
            "american_opportunity_credit_claimed_prior_years", {}
        ).get(time_period)
        prior_years = (
            np.asarray(existing_prior_years).copy()
            if existing_prior_years is not None
            else np.zeros(len(person_tax_unit_ids), dtype=np.int8)
        )
        prior_years[aotc_student] = np.minimum(prior_years[aotc_student], 3)
        data["american_opportunity_credit_claimed_prior_years"] = {
            time_period: prior_years
        }
        data["qualified_tuition_expenses"] = {time_period: tuition}
        logger.info(
            "AOTC eligibility imputation populated inputs for %d people "
            "across %d tax units and adjusted tuition for %d people",
            int(aotc_student.sum()),
            signal_tax_unit_count,
            adjusted_tuition_count,
        )
        return data

    @classmethod
    def _impute_llc_eligibility_inputs(cls, data, time_period):
        """Populate LLC factual eligibility inputs for non-AOTC tuition records."""

        if not _supports_llc_eligibility_inputs():
            return data

        person_tax_unit_ids = data.get("person_tax_unit_id", {}).get(time_period)
        tuition = data.get("qualified_tuition_expenses", {}).get(time_period)
        if person_tax_unit_ids is None or tuition is None:
            return data

        person_tax_unit_ids = np.asarray(person_tax_unit_ids)
        tuition = np.asarray(tuition)
        if len(tuition) != len(person_tax_unit_ids):
            logger.warning(
                "Skipping LLC eligibility imputation due to entity length mismatch"
            )
            return data

        aotc_student = data.get(
            "is_pursuing_credential_for_american_opportunity_credit",
            {},
        ).get(time_period)
        if aotc_student is None:
            aotc_student = data.get(
                "is_eligible_for_american_opportunity_credit",
                {},
            ).get(time_period)
        aotc_student = (
            np.asarray(aotc_student, dtype=bool)
            if aotc_student is not None
            else np.zeros(len(person_tax_unit_ids), dtype=bool)
        )

        llc_student = (tuition > 0) & ~aotc_student
        if not llc_student.any():
            return data

        for variable in LLC_ELIGIBILITY_INPUTS:
            existing = data.get(variable, {}).get(time_period)
            values = (
                np.asarray(existing, dtype=bool).copy()
                if existing is not None
                else np.zeros(len(person_tax_unit_ids), dtype=bool)
            )
            values[llc_student] = True
            data[variable] = {time_period: values}

        logger.info(
            "LLC eligibility imputation populated inputs for %d people "
            "across %d tax units",
            int(llc_student.sum()),
            int(np.unique(person_tax_unit_ids[llc_student]).size),
        )
        return data

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

    @classmethod
    def _drop_puf_computed_intermediates(cls, data):
        """Drop PUF outputs that are construction-only in Extended CPS."""

        dropped = sorted(set(data) & _PUF_COMPUTED_INTERMEDIATES_AFTER_CLONE)
        if dropped:
            logger.info(
                "Dropping %d PUF computed intermediates after clone stage: %s",
                len(dropped),
                dropped,
            )
            for variable in dropped:
                del data[variable]
        return data

    @classmethod
    def _finalize_stage2_computed_variables(cls, data):
        """Remove or rename computed variables after their final stage-2 use."""

        for source, target in _COMPUTED_AGGREGATE_INPUT_RENAMES.items():
            if source not in data:
                continue
            if target not in data:
                logger.info(
                    "Renaming %s -> %s after stage-2 predictor use",
                    source,
                    target,
                )
                data[target] = data.pop(source)
            else:
                logger.info(
                    "Dropping %s after stage-2 predictor use; %s already exists",
                    source,
                    target,
                )
                del data[source]

        if "social_security" in data and _SS_SUBCOMPONENT_VARS <= set(data):
            logger.info("Dropping social_security after reconciling leaf subcomponents")
            del data["social_security"]

        dropped = sorted(
            set(data) & (_STAGE2_COMPUTED_PREDICTORS | _STAGE2_COMPUTED_OUTPUTS_TO_DROP)
        )
        if dropped:
            logger.info(
                "Dropping %d stage-2 computed variables after final use: %s",
                len(dropped),
                dropped,
            )
            for variable in dropped:
                del data[variable]
        return data

    @staticmethod
    def _has_positive_mortgage_input(data, time_period):
        values = data.get("deductible_mortgage_interest", {}).get(time_period)
        if values is not None and np.any(np.asarray(values) > 0):
            return True
        return False

    @staticmethod
    def _validate_structural_mortgage_conversion(
        data,
        time_period,
        had_positive_mortgage_input,
    ):
        if not had_positive_mortgage_input:
            return
        mortgage_values = data.get("first_home_mortgage_interest", {}).get(time_period)
        person_values = data.get("home_mortgage_interest", {}).get(time_period)
        if (
            mortgage_values is not None and np.any(np.asarray(mortgage_values) > 0)
        ) or (person_values is not None and np.any(np.asarray(person_values) > 0)):
            return
        raise RuntimeError(
            "Structural mortgage conversion lost positive mortgage inputs."
        )

    @classmethod
    @pipeline_node(
        PipelineNode(
            id="housing_assistance_microsim_validation",
            label="Validate Housing Assistance Microsimulation",
            node_type="process",
            description=(
                "Runs a temporary microsimulation before final export to ensure "
                "housing-assistance leaf inputs reconstruct nonzero modeled "
                "housing assistance and capped SPM housing subsidy."
            ),
            status="transitional",
            stability="moving",
            pathways=["data_build"],
            artifacts_in=["extended_cps_stage2"],
            artifacts_out=["housing_validated_extended_cps"],
            pydoc=True,
        )
    )
    def _validate_housing_assistance_microsimulation(
        cls,
        data,
        time_period,
        microsimulation_cls=None,
    ):
        """Check formula-reconstructed housing assistance before export.

        The final H5 must not export formula outputs such as ``housing_assistance``.
        This guard verifies that the remaining leaf inputs still make those
        formulas produce nonzero values before the export contract strips or
        rejects computed variables.
        """

        receives = data.get("receives_housing_assistance", {}).get(time_period)
        takes_up = data.get("takes_up_housing_assistance_if_eligible", {}).get(
            time_period
        )
        if receives is None and takes_up is None:
            return data

        receives = (
            np.asarray(receives, dtype=bool)
            if receives is not None
            else np.zeros_like(np.asarray(takes_up, dtype=bool))
        )
        takes_up = (
            np.asarray(takes_up, dtype=bool)
            if takes_up is not None
            else np.zeros_like(receives, dtype=bool)
        )
        if not (receives | takes_up).any():
            return data

        validation_data = {
            variable: values
            for variable, values in data.items()
            if variable not in _HOUSING_ASSISTANCE_FORMULA_OUTPUTS
        }
        if microsimulation_cls is None:
            from policyengine_us import Microsimulation

            microsimulation_cls = Microsimulation

        simulation = microsimulation_cls(
            dataset=_InMemoryTimePeriodDataset(validation_data, time_period)
        )
        housing_assistance = simulation.calculate("housing_assistance", time_period)
        capped_housing_subsidy = simulation.calculate(
            "spm_unit_capped_housing_subsidy",
            time_period,
        )
        housing_total = float(housing_assistance.sum())
        capped_total = float(capped_housing_subsidy.sum())
        if housing_total <= 0 or capped_total <= 0:
            raise RuntimeError(
                "Housing assistance inputs do not reconstruct modeled benefits: "
                f"housing_assistance={housing_total:,.0f}, "
                f"spm_unit_capped_housing_subsidy={capped_total:,.0f}. "
                "Check receives_housing_assistance, "
                "takes_up_housing_assistance_if_eligible, county_fips, rent, "
                "and HUD payment-standard inputs before dropping formula outputs."
            )

        benchmark = data.get("spm_unit_capped_housing_subsidy", {}).get(time_period)
        if benchmark is None:
            return data

        from microdf import MicroSeries

        spm_unit_weight = simulation.calculate(
            "spm_unit_weight",
            time_period,
            use_weights=False,
        )
        weights = np.asarray(getattr(spm_unit_weight, "values", spm_unit_weight))
        benchmark_total = float(
            MicroSeries(np.asarray(benchmark, dtype=float), weights=weights).sum()
        )
        if benchmark_total <= 0:
            return data

        minimum_total = benchmark_total * _MIN_MODELED_HOUSING_SHARE_OF_BENCHMARK
        if capped_total < minimum_total:
            raise RuntimeError(
                "Modeled capped housing subsidy is implausibly small relative "
                "to the raw ASEC SPM housing subsidy benchmark: "
                f"modeled={capped_total:,.0f}, benchmark={benchmark_total:,.0f}. "
                "This likely means a required formula input is missing before "
                "housing assistance formula outputs are dropped from the final export."
            )
        return data

    @classmethod
    def _reassign_housing_assistance_takeup_with_geography(
        cls,
        data,
        time_period,
        microsimulation_cls=None,
        take_up_rate=None,
        draws=None,
    ):
        """Recompute housing-assistance take-up after county assignment.

        CPS add_takeup runs before the ExtendedCPS geography assignment, so
        HUD income-limit eligibility can only anchor on reported recipients at
        that point. Reassign here, after county_fips is present and after PUF
        clone income variables have been spliced in, so reported recipients are
        preserved while non-reported take-up is drawn from the full HUD-eligible
        pool.
        """

        if "county_fips" not in data or time_period not in data["county_fips"]:
            return data

        receives = data.get("receives_housing_assistance", {}).get(time_period)
        existing_takeup = data.get("takes_up_housing_assistance_if_eligible", {}).get(
            time_period
        )
        if receives is None and existing_takeup is None:
            return data

        if microsimulation_cls is None:
            from policyengine_us import Microsimulation

            microsimulation_cls = Microsimulation

        validation_data = {
            variable: values
            for variable, values in data.items()
            if variable not in _HOUSING_ASSISTANCE_FORMULA_OUTPUTS
        }
        simulation = microsimulation_cls(
            dataset=_InMemoryTimePeriodDataset(validation_data, time_period)
        )
        eligible = simulation.calculate(
            "is_eligible_for_housing_assistance",
            time_period,
        )
        eligible = np.asarray(getattr(eligible, "values", eligible), dtype=bool)
        spm_unit_weight = simulation.calculate(
            "spm_unit_weight",
            time_period,
            use_weights=False,
        )
        weights = np.asarray(
            getattr(spm_unit_weight, "values", spm_unit_weight),
            dtype=float,
        )

        if receives is None:
            receives = np.zeros_like(eligible, dtype=bool)
        else:
            receives = np.asarray(receives, dtype=bool)

        if len(receives) != len(eligible):
            raise ValueError(
                "receives_housing_assistance length does not match HUD "
                "eligibility length when reassigning housing assistance "
                f"take-up: got {len(receives)}, expected {len(eligible)}."
            )

        if draws is None:
            rng = seeded_rng("takes_up_housing_assistance_if_eligible")
            draws = rng.random(len(receives))
        if take_up_rate is None:
            take_up_rate = load_take_up_rate("housing_assistance", time_period)

        draws = np.asarray(draws)
        reassigned_takeup = np.zeros_like(receives, dtype=bool)
        assignment_groups = (weights > 0, weights <= 0)
        for assignment_group in assignment_groups:
            if not assignment_group.any():
                continue
            reassigned_takeup[assignment_group] = prioritize_reported_recipients(
                receives[assignment_group],
                take_up_rate,
                draws[assignment_group],
                eligible_mask=eligible[assignment_group],
            )

        data["takes_up_housing_assistance_if_eligible"] = {
            time_period: reassigned_takeup
        }
        return data

    @classmethod
    def _drop_housing_assistance_formula_outputs(cls, data):
        """Remove housing assistance formula outputs after validation."""

        for variable in sorted(set(data) & _HOUSING_ASSISTANCE_FORMULA_OUTPUTS):
            del data[variable]
        return data

    @classmethod
    def _drop_final_computed_outputs(cls, data):
        """Remove final aggregates that policyengine-us recomputes from leaves."""

        for variable in sorted(set(data) & _FINAL_COMPUTED_OUTPUTS_TO_DROP):
            del data[variable]
        return data

    # QRF imputes formula-level variables (e.g. taxable_pension_income)
    # but we must store them under leaf input names. The engine then
    # recomputes the formula var from its adds.
    _IMPUTED_TO_INPUT = {
        "medicare_enrolled": "takes_up_medicare_if_eligible",
        "taxable_pension_income": "taxable_private_pension_income",
        "tax_exempt_pension_income": "tax_exempt_private_pension_income",
    }

    @classmethod
    @pipeline_node(
        PipelineNode(
            id="computed_export_contract",
            label="Validate Leaf-Input Export",
            node_type="process",
            description=(
                "Fails the build if the final export still contains "
                "variables computed by policyengine-us formulas, adds, or "
                "subtracts."
            ),
            status="transitional",
            stability="moving",
            pathways=["data_build"],
            artifacts_in=["extended_cps_stage2"],
            artifacts_out=["validated_extended_cps"],
            pydoc=True,
        )
    )
    def _assert_no_computed_variables_exported(cls, data, time_period):
        """Assert that final exported variables are leaf inputs."""

        from policyengine_us import CountryTaxBenefitSystem

        assert_no_computed_policyengine_us_variables_exported(
            variable_names=data.keys(),
            time_period=time_period,
            tax_benefit_system=CountryTaxBenefitSystem(),
            dataset_name=cls.name,
        )
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
