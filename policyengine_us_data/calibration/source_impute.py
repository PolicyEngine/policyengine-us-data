"""Non-PUF QRF imputations from donor surveys.

Re-imputes variables from ACS, SIPP, ORG, and SCF donor surveys.
Only ACS and ORG include state_fips as a QRF predictor. SIPP and SCF
lack state data, so their imputations use only demographic and
financial predictors.

Sources and variables:
    ACS  -> rent, real_estate_taxes  (with state predictor)
    SIPP -> tip_income, bank_account_assets, stock_assets,
            bond_assets, household_vehicles_owned,
            household_vehicles_value  (no state predictor)
    ORG  -> hourly_wage, is_paid_hourly,
            is_union_member_or_covered
    SCF  -> net_worth, auto_loan_balance, auto_loan_interest,
            SCF-only balance-sheet components, and
            50/50 source-model averaging for overlapping financial assets
            (no state predictor)

Usage in unified calibration pipeline:
    1. Load raw CPS
    2. Clone Nx, assign geography
    3. impute_source_variables()  <-- this module
    4. PUF clone + QRF impute (puf_impute.py)
    5. PE simulate, build matrix, calibrate
"""

import gc
import h5py
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from microimpute.models.qrf import QRF
from policyengine_us_data.datasets.cps.tipped_occupation import (
    derive_any_treasury_tipped_occupation_code,
    derive_is_tipped_occupation,
)
from policyengine_us_data.datasets.sipp.sipp import (
    ASSET_JOB_EARNINGS_COLUMNS,
    ASSET_PREDICTORS,
    SIPP_ASSET_ALLOCATION_COLUMNS,
    SIPP_ASSET_TARGET_ALLOCATION_COLUMNS,
    SIPP_ASSET_TARGET_SOURCE_COLUMNS,
    SIPP_TIP_AMOUNT_COLUMNS,
    SIPP_TIP_AMOUNT_TO_ALLOCATION_COLUMN,
    SIPP_VEHICLE_TARGET_ALLOCATION_COLUMNS,
    SSA_DISABILITY_SCREEN_VARIABLE,
    SSI_DISABILITY_COMPATIBILITY_VARIABLE,
    SSI_DISABILITY_DIFFICULTY_PREDICTORS,
    SSI_DISABILITY_EXPORT_VARIABLES,
    VEHICLE_MODEL_PREDICTORS,
    build_vehicle_training_frame,
    get_ssi_disability_model,
    predict_ssa_disability_screen,
    preserve_under_65_ssa_disability_screen,
)

from policyengine_us_data.datasets.org import (
    ORG_BOOL_VARIABLES,
    ORG_IMPUTED_VARIABLES,
    build_org_receiver_frame,
    predict_org_features,
)
from policyengine_us_data.utils.asset_imputation import (
    SCF_NET_WORTH_TARGET,
    SCF_FINANCIAL_ASSET_POLICY_VARIABLES,
    SCF_HOUSEHOLD_ASSET_POLICY_VARIABLES,
    SCF_NET_WORTH_COMPONENT_VARIABLES,
    add_scf_financial_asset_targets,
    add_scf_household_asset_targets,
    add_scf_net_worth_target,
    add_scf_net_worth_component_targets,
    aggregate_person_values_to_reference_households,
    align_household_values_to_reference_households,
    build_household_vehicle_receiver,
    combine_sipp_and_scf_financial_assets,
    combine_sipp_and_scf_household_assets,
    compute_net_worth_from_components,
    rebalance_scf_net_worth_components,
    require_scf_net_worth_formula_targets,
)
from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode
from policyengine_us_data.utils.source_quality import (
    cap_training_sample,
    filter_positive_finite_weight_rows,
    require_columns_present,
    target_observed_source_masks,
)

logger = logging.getLogger(__name__)

ACS_IMPUTED_VARIABLES = [
    "rent",
    "real_estate_taxes",
]
ACS_TARGET_ALLOCATION_COLUMNS = {
    "rent": ["rent_is_allocated"],
    "real_estate_taxes": ["real_estate_taxes_is_allocated"],
}

SIPP_IMPUTED_VARIABLES = [
    "tip_income",
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
    *SSI_DISABILITY_EXPORT_VARIABLES,
    "household_vehicles_owned",
    "household_vehicles_value",
]

SCF_CORE_IMPUTED_VARIABLES = [
    "auto_loan_balance",
    "auto_loan_interest",
]

SCF_IMPUTED_VARIABLES = [
    "net_worth",
    *SCF_CORE_IMPUTED_VARIABLES,
    *SCF_NET_WORTH_COMPONENT_VARIABLES,
]

ALL_SOURCE_VARIABLES = (
    ACS_IMPUTED_VARIABLES
    + SIPP_IMPUTED_VARIABLES
    + ORG_IMPUTED_VARIABLES
    + SCF_IMPUTED_VARIABLES
)

ACS_PREDICTORS = [
    "is_household_head",
    "age",
    "is_male",
    "tenure_type",
    "employment_income",
    "self_employment_income",
    "social_security",
    "pension_income",
    "household_size",
]

SIPP_TIPS_PREDICTORS = [
    "employment_income",
    "age",
    "count_under_18",
    "count_under_6",
    "is_tipped_occupation",
]

SIPP_ASSETS_PREDICTORS = ASSET_PREDICTORS

SCF_PREDICTORS = [
    "age",
    "is_female",
    "cps_race",
    "is_married",
    "own_children_in_household",
    "employment_income",
    "interest_dividend_income",
    "social_security_pension_income",
]


TENURE_TYPE_MAP = {
    "OWNED_WITH_MORTGAGE": 1,
    "OWNED_OUTRIGHT": 1,
    "RENTED": 2,
    "NONE": 0,
}

SIPP_JOB_OCCUPATION_COLUMNS = [f"TJB{i}_OCC" for i in range(1, 8)]


def _encode_tenure_type(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tenure_type enum strings to numeric codes."""
    if "tenure_type" in df.columns:
        df["tenure_type"] = (
            df["tenure_type"]
            .astype(str)
            .map(TENURE_TYPE_MAP)
            .fillna(0)
            .astype(np.float32)
        )
    return df


@pipeline_node(
    PipelineNode(
        id="source_impute",
        label="Source-Impute Stratified CPS",
        node_type="entrypoint",
        description="Apply ACS, SIPP, ORG, and SCF donor imputations to the stratified CPS calibration input.",
        source_file="policyengine_us_data/calibration/source_impute.py",
        status="current",
        stability="moving",
        pathways=["data_build"],
        artifacts_in=["stratified_extended_cps_2024.h5"],
        artifacts_out=["source_imputed_stratified_extended_cps_2024.h5"],
        validation_commands=[
            "uv run pytest tests/unit/calibration/test_source_impute.py"
        ],
    )
)
def impute_source_variables(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int = 2024,
    dataset_path: Optional[str] = None,
    skip_acs: bool = False,
    skip_sipp: bool = False,
    skip_org: bool = False,
    skip_scf: bool = False,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Re-impute ACS/SIPP/ORG/SCF variables from donor surveys.

    Overwrites existing imputed values in data. ACS uses
    state_fips as a QRF predictor; ORG uses state plus labor-market
    predictors; SIPP and SCF use only demographic and financial
    predictors (no state data).

    Args:
        data: CPS dataset dict {variable: {time_period: array}}.
        state_fips: State FIPS per household.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.
        skip_acs: Skip ACS imputation.
        skip_sipp: Skip SIPP imputation.
        skip_org: Skip ORG imputation.
        skip_scf: Skip SCF imputation.

    Returns:
        Updated data dict with re-imputed variables.
    """
    data["state_fips"] = {
        time_period: state_fips.astype(np.int32),
    }

    if not skip_acs:
        logger.info("Imputing ACS variables with state predictor")
        data = _impute_acs(data, state_fips, time_period, dataset_path)

    if not skip_sipp:
        logger.info("Imputing SIPP variables")
        data = _impute_sipp(data, state_fips, time_period, dataset_path)

    if not skip_org:
        logger.info("Imputing ORG variables")
        data = _impute_org(data, state_fips, time_period, dataset_path)

    if not skip_scf:
        logger.info("Imputing SCF variables")
        data = _impute_scf(data, state_fips, time_period, dataset_path)

    return data


def _build_cps_receiver(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    dataset_path: Optional[str],
    pe_variables: list,
) -> pd.DataFrame:
    """Build CPS receiver DataFrame from Microsimulation.

    Args:
        data: CPS data dict.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.
        pe_variables: List of PE variable names to compute.

    Returns:
        DataFrame with requested columns.
    """
    if dataset_path is not None:
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=dataset_path)
        tbs = sim.tax_benefit_system
        valid_vars = [v for v in pe_variables if v in tbs.variables]
        if valid_vars:
            df = sim.calculate_dataframe(valid_vars)
        else:
            df = pd.DataFrame(index=range(len(data["person_id"][time_period])))
        del sim
    else:
        df = pd.DataFrame()

    for var in pe_variables:
        if var not in df.columns and var in data:
            df[var] = data[var][time_period].astype(np.float32)

    return df


def _get_variable_entity(variable_name: str) -> str:
    """Return the entity key for a PE variable."""
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()
    var = tbs.variables.get(variable_name)
    if var is None:
        return "person"
    return var.entity.key


def _person_state_fips(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
) -> np.ndarray:
    """Map household-level state_fips to person level.

    Args:
        data: CPS data dict.
        state_fips: State FIPS per household.
        time_period: Tax year.

    Returns:
        Person-level state FIPS array.
    """
    hh_ids_person = data.get("person_household_id", {}).get(time_period)
    if hh_ids_person is not None:
        hh_ids = data["household_id"][time_period]
        hh_to_idx = {int(hh_id): i for i, hh_id in enumerate(hh_ids)}
        return np.array([state_fips[hh_to_idx[int(hh_id)]] for hh_id in hh_ids_person])
    # Fallback: distribute persons across households as evenly
    # as possible (first households get any remainder).
    n_hh = len(data["household_id"][time_period])
    n_persons = len(data["person_id"][time_period])
    base, remainder = divmod(n_persons, n_hh)
    counts = np.full(n_hh, base, dtype=int)
    counts[:remainder] += 1
    return np.repeat(state_fips, counts)


def _person_is_married(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    n_persons: int,
) -> np.ndarray:
    """Return a person-level married flag from CPS-compatible inputs."""
    if "is_married" in data and time_period in data["is_married"]:
        values = np.asarray(data["is_married"][time_period])
        if len(values) == n_persons:
            return values.astype(np.float32)

    marital_unit_id = data.get("person_marital_unit_id", {}).get(time_period)
    if marital_unit_id is not None and len(marital_unit_id) == n_persons:
        marital_unit_id = np.asarray(marital_unit_id)
        counts = pd.Series(marital_unit_id).map(
            pd.Series(marital_unit_id).value_counts()
        )
        return (counts.to_numpy() > 1).astype(np.float32)

    return np.zeros(n_persons, dtype=np.float32)


def _add_person_household_counts(
    df: pd.DataFrame,
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
) -> pd.DataFrame:
    """Add household composition predictors to a person-level CPS frame."""
    if "age" not in df.columns and "age" in data:
        df["age"] = data["age"][time_period].astype(np.float32)

    hh_ids_person = data.get("person_household_id", {}).get(time_period)
    if hh_ids_person is None or "age" not in df.columns:
        df["count_under_18"] = 0.0
        df["count_under_6"] = 0.0
        df["household_size"] = 1.0
        return df

    age_df = pd.DataFrame(
        {
            "hh": hh_ids_person,
            "age": np.asarray(df["age"]),
        }
    )
    grouped = age_df.groupby("hh")["age"]
    df["count_under_18"] = (
        grouped.transform(lambda values: (values < 18).sum())
        .to_numpy()
        .astype(np.float32)
    )
    df["count_under_6"] = (
        grouped.transform(lambda values: (values < 6).sum())
        .to_numpy()
        .astype(np.float32)
    )
    df["household_size"] = grouped.transform("size").to_numpy().astype(np.float32)
    return df


def _add_sipp_asset_predictors(asset_df: pd.DataFrame) -> pd.DataFrame:
    """Add SIPP-side liquid-asset model predictors without SSI receipt."""
    asset_df = asset_df.copy()
    asset_df["bank_account_assets"] = asset_df["TVAL_BANK"].fillna(0)
    asset_df["stock_assets"] = asset_df["TVAL_STMF"].fillna(0)
    asset_df["bond_assets"] = asset_df["TVAL_BOND"].fillna(0)
    asset_df["age"] = asset_df.TAGE
    asset_df["is_female"] = asset_df.ESEX == 2
    asset_df["is_married"] = asset_df.EMS == 1

    job_cols = [col for col in ASSET_JOB_EARNINGS_COLUMNS if col in asset_df]
    if job_cols:
        asset_df["employment_income"] = asset_df[job_cols].fillna(0).sum(axis=1) * 12
    elif "TPTOTINC" in asset_df:
        asset_df["employment_income"] = asset_df.TPTOTINC.fillna(0) * 12
    else:
        asset_df["employment_income"] = 0.0

    asset_df["interest_income"] = (
        asset_df["TINC_BANK"].fillna(0) + asset_df["TINC_BOND"].fillna(0)
    ) * 12
    asset_df["dividend_income"] = asset_df["TINC_STMF"].fillna(0) * 12
    asset_df["rental_income"] = asset_df["TINC_RENT"].fillna(0) * 12
    asset_df["social_security"] = asset_df["TSSSAMT"].fillna(0) * 12
    asset_df["retirement_income"] = asset_df["TRETINCAMT"].fillna(0) * 12
    asset_df["non_ssi_income"] = (
        asset_df["employment_income"]
        + asset_df["social_security"]
        + asset_df["retirement_income"]
    )
    asset_df["household_weight"] = asset_df.WPFINWGT

    asset_df["is_under_18"] = asset_df.TAGE < 18
    asset_df["is_under_6"] = asset_df.TAGE < 6
    grouped = asset_df.groupby("SSUID")
    asset_df["count_under_18"] = grouped["is_under_18"].transform("sum")
    asset_df["count_under_6"] = grouped["is_under_6"].transform("sum")
    asset_df["household_size"] = grouped["PNUM"].transform("count")
    return asset_df


def _add_cps_asset_predictors(
    cps_asset_df: pd.DataFrame,
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
) -> pd.DataFrame:
    """Add CPS-side predictors aligned to the SIPP liquid-asset model."""
    cps_asset_df = cps_asset_df.copy()
    n_persons = len(cps_asset_df)

    if "is_male" in cps_asset_df.columns:
        cps_asset_df["is_female"] = (~cps_asset_df["is_male"].astype(bool)).astype(
            np.float32
        )
    elif "is_female" in data:
        cps_asset_df["is_female"] = data["is_female"][time_period].astype(np.float32)
    else:
        cps_asset_df["is_female"] = 0.0

    cps_asset_df["is_married"] = _person_is_married(
        data,
        time_period,
        n_persons,
    )
    cps_asset_df = _add_person_household_counts(cps_asset_df, data, time_period)

    for var in [
        "employment_income",
        "interest_income",
        "dividend_income",
        "rental_income",
        "social_security",
        "pension_income",
        "retirement_distributions",
    ]:
        if var in cps_asset_df.columns:
            continue
        if var in data:
            cps_asset_df[var] = data[var][time_period].astype(np.float32)
        else:
            cps_asset_df[var] = 0.0

    cps_asset_df["retirement_income"] = cps_asset_df["pension_income"].fillna(
        0
    ) + cps_asset_df["retirement_distributions"].fillna(0)
    cps_asset_df["non_ssi_income"] = (
        cps_asset_df["employment_income"].fillna(0)
        + cps_asset_df["social_security"].fillna(0)
        + cps_asset_df["retirement_income"].fillna(0)
    )

    for predictor in SIPP_ASSETS_PREDICTORS:
        if predictor not in cps_asset_df.columns:
            cps_asset_df[predictor] = 0.0
        cps_asset_df[predictor] = cps_asset_df[predictor].fillna(0).astype(np.float32)

    return cps_asset_df


@pipeline_node(
    PipelineNode(
        id="acs_qrf",
        label="ACS QRF Imputation",
        node_type="library",
        description="Impute rent and real estate tax variables from ACS donor data.",
        source_file="policyengine_us_data/calibration/source_impute.py",
        status="current",
        stability="moving",
        pathways=["data_build"],
        validation_commands=[
            "uv run pytest tests/unit/calibration/test_source_impute.py"
        ],
    )
)
def _impute_acs(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Impute rent and real_estate_taxes from ACS with state.

    Args:
        data: CPS data dict.
        state_fips: State FIPS per household.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.

    Returns:
        Updated data dict.
    """
    from policyengine_us import Microsimulation

    from policyengine_us_data.datasets.acs.acs import ACS_2022

    acs = Microsimulation(dataset=ACS_2022)
    predictors = ACS_PREDICTORS + ["state_fips"]

    acs_df = acs.calculate_dataframe(
        ACS_PREDICTORS + ACS_IMPUTED_VARIABLES, map_to="person"
    )
    acs_df["state_fips"] = acs.calculate("state_fips", map_to="person").values.astype(
        np.float32
    )
    required_acs_flags = [
        column
        for columns in ACS_TARGET_ALLOCATION_COLUMNS.values()
        for column in columns
    ]
    with h5py.File(ACS_2022.file_path, "r") as acs_h5:
        require_columns_present(
            acs_h5,
            required_acs_flags,
            source_name="ACS_2022 artifact",
        )
        for flag_columns in ACS_TARGET_ALLOCATION_COLUMNS.values():
            for flag_column in flag_columns:
                acs_df[flag_column] = np.asarray(acs_h5[flag_column], dtype=bool)

    train_df = acs_df[acs_df.is_household_head].copy()
    train_df = _encode_tenure_type(train_df)
    del acs

    if dataset_path is not None:
        cps_sim = Microsimulation(dataset=dataset_path)
        cps_df = cps_sim.calculate_dataframe(ACS_PREDICTORS, map_to="person")
        del cps_sim
    else:
        cps_df = pd.DataFrame()
        for pred in ACS_PREDICTORS:
            if pred in data:
                cps_df[pred] = data[pred][time_period].astype(np.float32)

    cps_df = _encode_tenure_type(cps_df)

    person_states = _person_state_fips(data, state_fips, time_period)
    cps_df["state_fips"] = person_states.astype(np.float32)

    mask = (
        cps_df.is_household_head.values
        if "is_household_head" in cps_df.columns
        else np.ones(len(cps_df), dtype=bool)
    )
    cps_heads = cps_df[mask]

    logger.info(
        "ACS QRF: %d train, %d test, %d predictors",
        len(train_df),
        len(cps_heads),
        len(predictors),
    )
    acs_target_filters = target_observed_source_masks(
        train_df,
        targets=ACS_IMPUTED_VARIABLES,
        target_allocation_flag_columns=ACS_TARGET_ALLOCATION_COLUMNS,
    )
    train_df, acs_target_filters = cap_training_sample(
        train_df,
        max_train_samples=10_000,
        seed_name="calibration_acs_source_imputation_training_sample",
        target_filters=acs_target_filters,
    )
    fitted = QRF().fit(
        X_train=train_df,
        predictors=predictors,
        imputed_variables=ACS_IMPUTED_VARIABLES,
        target_filters=acs_target_filters,
    )
    predictions = fitted.predict(X_test=cps_heads)

    n_persons = len(data["person_id"][time_period])
    for var in ACS_IMPUTED_VARIABLES:
        values = np.zeros(n_persons, dtype=np.float32)
        values[mask] = predictions[var].values
        data[var] = {time_period: values}
    data["pre_subsidy_rent"] = {time_period: data["rent"][time_period].copy()}

    del fitted, predictions
    gc.collect()

    logger.info("ACS imputation complete: rent, real_estate_taxes")
    return data


@pipeline_node(
    PipelineNode(
        id="sipp_qrf",
        label="SIPP QRF Imputation",
        node_type="library",
        description="Impute tips, liquid assets, and vehicle assets from SIPP donor data.",
        source_file="policyengine_us_data/calibration/source_impute.py",
        status="current",
        stability="moving",
        pathways=["data_build"],
        validation_commands=[
            "uv run pytest tests/unit/calibration/test_source_impute.py"
        ],
    )
)
def _impute_sipp(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Impute tip_income, liquid assets, and vehicle signals from SIPP.

    Args:
        data: CPS data dict.
        state_fips: State FIPS per household.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.

    Returns:
        Updated data dict.
    """
    from huggingface_hub import hf_hub_download
    from policyengine_us_data.storage import STORAGE_FOLDER

    hf_hub_download(
        repo_id="PolicyEngine/policyengine-us-data",
        filename="pu2023_slim.csv",
        repo_type="model",
        local_dir=STORAGE_FOLDER,
    )
    sipp_df = pd.read_csv(STORAGE_FOLDER / "pu2023_slim.csv")

    tip_amount_columns = [
        column for column in SIPP_TIP_AMOUNT_COLUMNS if column in sipp_df
    ]
    tip_allocation_columns = [
        SIPP_TIP_AMOUNT_TO_ALLOCATION_COLUMN[column] for column in tip_amount_columns
    ]
    require_columns_present(
        sipp_df.columns,
        tip_allocation_columns,
        source_name="SIPP slim tip donor file",
    )
    sipp_df["tip_income"] = sipp_df[tip_amount_columns].fillna(0).sum(axis=1) * 12
    sipp_df["employment_income"] = sipp_df.TPTOTINC * 12
    sipp_df["age"] = sipp_df.TAGE
    sipp_df["household_weight"] = sipp_df.WPFINWGT
    sipp_df["household_id"] = sipp_df.SSUID
    sipp_df["treasury_tipped_occupation_code"] = (
        derive_any_treasury_tipped_occupation_code(sipp_df[SIPP_JOB_OCCUPATION_COLUMNS])
    )
    sipp_df["is_tipped_occupation"] = derive_is_tipped_occupation(
        sipp_df["treasury_tipped_occupation_code"]
    )

    if "MONTHCODE" in sipp_df:
        sipp_df = sipp_df[sipp_df["MONTHCODE"] == 12].copy()
    sipp_df["is_under_18"] = sipp_df.TAGE < 18
    sipp_df["is_under_6"] = sipp_df.TAGE < 6
    sipp_df["count_under_18"] = (
        sipp_df.groupby("SSUID")["is_under_18"].sum().loc[sipp_df.SSUID.values].values
    )
    sipp_df["count_under_6"] = (
        sipp_df.groupby("SSUID")["is_under_6"].sum().loc[sipp_df.SSUID.values].values
    )

    tip_target_filters = target_observed_source_masks(
        sipp_df,
        targets=["tip_income"],
        target_source_columns={"tip_income": tip_amount_columns},
        target_allocation_flag_columns={"tip_income": tip_allocation_columns},
        require_nonmissing_source=False,
    )

    tip_cols = [
        "household_id",
        "employment_income",
        "tip_income",
        "count_under_18",
        "count_under_6",
        "age",
        "is_tipped_occupation",
        "household_weight",
    ]
    tip_train = sipp_df[tip_cols].dropna()
    tip_train, tip_target_filters = filter_positive_finite_weight_rows(
        tip_train,
        weight_col="household_weight",
        target_filters=tip_target_filters,
        context_name="SIPP source tip donor",
    )
    tip_train, tip_target_filters = cap_training_sample(
        tip_train,
        max_train_samples=10_000,
        seed_name="calibration_sipp_tip_training_sample",
        target_filters=tip_target_filters,
    )

    cps_tip_df = _build_cps_receiver(
        data, time_period, dataset_path, ["employment_income", "age"]
    )
    person_ages = data["age"][time_period]
    hh_ids_person = data.get("person_household_id", {}).get(time_period)
    if hh_ids_person is not None:
        age_df = pd.DataFrame({"hh": hh_ids_person, "age": person_ages})
        under_18 = age_df.groupby("hh")["age"].apply(lambda x: (x < 18).sum())
        under_6 = age_df.groupby("hh")["age"].apply(lambda x: (x < 6).sum())
        cps_tip_df["count_under_18"] = under_18.loc[hh_ids_person].values.astype(
            np.float32
        )
        cps_tip_df["count_under_6"] = under_6.loc[hh_ids_person].values.astype(
            np.float32
        )
    else:
        cps_tip_df["count_under_18"] = 0.0
        cps_tip_df["count_under_6"] = 0.0
    if "treasury_tipped_occupation_code" in data:
        cps_tip_df["is_tipped_occupation"] = derive_is_tipped_occupation(
            data["treasury_tipped_occupation_code"][time_period]
        ).astype(np.float32)
    else:
        cps_tip_df["is_tipped_occupation"] = 0.0

    logger.info(
        "SIPP tips QRF: %d train, %d test",
        len(tip_train),
        len(cps_tip_df),
    )
    fitted = QRF().fit(
        X_train=tip_train,
        predictors=SIPP_TIPS_PREDICTORS,
        imputed_variables=["tip_income"],
        target_filters=tip_target_filters,
        weight_col="household_weight",
    )
    tip_preds = fitted.predict(X_test=cps_tip_df)
    data["tip_income"] = {
        time_period: tip_preds["tip_income"].values,
    }
    del fitted, tip_preds
    gc.collect()

    logger.info("SIPP tip imputation complete")

    # Asset imputation
    try:
        hf_hub_download(
            repo_id="PolicyEngine/policyengine-us-data",
            filename="pu2023.csv",
            repo_type="model",
            local_dir=STORAGE_FOLDER,
        )
        asset_cols = (
            [
                "SSUID",
                "PNUM",
                "MONTHCODE",
                "WPFINWGT",
                "TAGE",
                "ESEX",
                "EMS",
                "TSSSAMT",
                "TRETINCAMT",
                "TVAL_BANK",
                "TVAL_STMF",
                "TVAL_BOND",
                "TINC_BANK",
                "TINC_STMF",
                "TINC_BOND",
                "TINC_RENT",
            ]
            + ASSET_JOB_EARNINGS_COLUMNS
            + SIPP_ASSET_ALLOCATION_COLUMNS
        )
        asset_df = pd.read_csv(
            STORAGE_FOLDER / "pu2023.csv",
            delimiter="|",
            usecols=asset_cols,
        )
        asset_df = asset_df[asset_df.MONTHCODE == 12]
        asset_df = _add_sipp_asset_predictors(asset_df)

        asset_train_cols = [
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
            "household_weight",
            *SIPP_ASSETS_PREDICTORS,
            *[
                column
                for columns in SIPP_ASSET_TARGET_SOURCE_COLUMNS.values()
                for column in columns
            ],
            *SIPP_ASSET_ALLOCATION_COLUMNS,
        ]
        asset_train = asset_df[asset_train_cols].copy()

        cps_asset_df = _build_cps_receiver(
            data,
            time_period,
            dataset_path,
            [
                "employment_income",
                "interest_income",
                "dividend_income",
                "rental_income",
                "social_security",
                "pension_income",
                "retirement_distributions",
                "age",
                "is_male",
            ],
        )
        cps_asset_df = _add_cps_asset_predictors(
            cps_asset_df,
            data,
            time_period,
        )

        asset_vars = [
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
        ]
        asset_target_filters = target_observed_source_masks(
            asset_train,
            targets=asset_vars,
            target_source_columns=SIPP_ASSET_TARGET_SOURCE_COLUMNS,
            target_allocation_flag_columns=SIPP_ASSET_TARGET_ALLOCATION_COLUMNS,
        )
        asset_train, asset_target_filters = filter_positive_finite_weight_rows(
            asset_train,
            weight_col="household_weight",
            target_filters=asset_target_filters,
            context_name="SIPP source asset donor",
        )
        asset_train, asset_target_filters = cap_training_sample(
            asset_train,
            max_train_samples=20_000,
            seed_name="calibration_sipp_asset_training_sample",
            target_filters=asset_target_filters,
        )
        logger.info(
            "SIPP assets QRF: %d train, %d test",
            len(asset_train),
            len(cps_asset_df),
        )
        fitted = QRF().fit(
            X_train=asset_train,
            predictors=SIPP_ASSETS_PREDICTORS,
            imputed_variables=asset_vars,
            target_filters=asset_target_filters,
            weight_col="household_weight",
        )
        asset_preds = fitted.predict(X_test=cps_asset_df)

        for var in asset_vars:
            data[var] = {
                time_period: asset_preds[var].values,
            }
        del fitted, asset_preds
        gc.collect()

        logger.info("SIPP asset imputation complete")

        cps_ssi_df = _build_cps_receiver(
            data,
            time_period,
            dataset_path,
            [
                "employment_income",
                "interest_income",
                "dividend_income",
                "rental_income",
                "age",
                "is_male",
                *SSI_DISABILITY_DIFFICULTY_PREDICTORS,
                "social_security_disability",
                "disability_benefits",
            ],
        )
        if "is_male" in cps_ssi_df.columns:
            cps_ssi_df["is_female"] = (~cps_ssi_df["is_male"].astype(bool)).astype(
                np.float32
            )
        else:
            cps_ssi_df["is_female"] = 0.0
        if "is_married" in data:
            cps_ssi_df["is_married"] = data["is_married"][time_period].astype(
                np.float32
            )
        else:
            cps_ssi_df["is_married"] = 0.0
        cps_ssi_df["count_under_18"] = (
            cps_tip_df["count_under_18"]
            if "count_under_18" in cps_tip_df.columns
            else 0.0
        )
        for var in asset_vars:
            cps_ssi_df[var] = data[var][time_period].astype(np.float32)
        for var in [
            "interest_income",
            "dividend_income",
            "rental_income",
            *SSI_DISABILITY_DIFFICULTY_PREDICTORS,
            "social_security_disability",
        ]:
            if var not in cps_ssi_df.columns:
                cps_ssi_df[var] = data.get(var, {}).get(
                    time_period, np.zeros(len(cps_ssi_df))
                )
        if "disability_benefits" in cps_ssi_df.columns:
            disability_benefits = cps_ssi_df["disability_benefits"]
        else:
            disability_benefits = data.get("disability_benefits", {}).get(
                time_period, np.zeros(len(cps_ssi_df))
            )
        cps_ssi_df["has_disability_income"] = (
            np.asarray(disability_benefits).astype(float) > 0
        )

        ssi_disability_model = get_ssi_disability_model(time_period=time_period)
        would_pass_ssa_disability_screen = predict_ssa_disability_screen(
            ssi_disability_model,
            cps_ssi_df,
        )
        existing_ssa_disability_screen = data.get(
            SSA_DISABILITY_SCREEN_VARIABLE, {}
        ).get(time_period)
        existing_meets_ssi_disability_criteria = data.get(
            SSI_DISABILITY_COMPATIBILITY_VARIABLE, {}
        ).get(time_period)
        ssi_reported = data.get("ssi_reported", {}).get(time_period)
        would_pass_ssa_disability_screen = preserve_under_65_ssa_disability_screen(
            would_pass_ssa_disability_screen,
            age=data["age"][time_period],
            ssi_reported=ssi_reported,
            existing_ssa_disability_screen=existing_ssa_disability_screen,
            existing_meets_ssi_disability_criteria=existing_meets_ssi_disability_criteria,
        )
        data[SSA_DISABILITY_SCREEN_VARIABLE] = {
            time_period: would_pass_ssa_disability_screen
        }
        data[SSI_DISABILITY_COMPATIBILITY_VARIABLE] = {
            time_period: would_pass_ssa_disability_screen
        }

        logger.info("SIPP SSA disability-screen imputation complete")

        vehicle_train = build_vehicle_training_frame()

        cps_vehicle_df = _build_cps_receiver(
            data,
            time_period,
            dataset_path,
            [
                "employment_income",
                "interest_income",
                "dividend_income",
                "rental_income",
                "age",
                "is_male",
                "is_household_head",
            ],
        )
        cps_vehicle_df["person_household_id"] = data["person_household_id"][
            time_period
        ].astype(np.int64)
        if "is_male" in cps_vehicle_df.columns:
            cps_vehicle_df["is_female"] = (
                ~cps_vehicle_df["is_male"].astype(bool)
            ).astype(np.float32)
        else:
            cps_vehicle_df["is_female"] = 0.0
        cps_vehicle_df["is_married"] = _person_is_married(
            data,
            time_period,
            len(cps_vehicle_df),
        )
        for cap_var in [
            "interest_income",
            "dividend_income",
            "rental_income",
            "is_household_head",
        ]:
            if cap_var not in cps_vehicle_df.columns:
                cps_vehicle_df[cap_var] = 0.0

        vehicle_receiver = build_household_vehicle_receiver(
            cps_vehicle_df,
            tenure_type=data.get("tenure_type", {}).get(time_period),
        )

        logger.info(
            "SIPP vehicle QRF: %d train, %d test",
            len(vehicle_train),
            len(vehicle_receiver),
        )
        vehicle_vars = [
            "household_vehicles_owned",
            "household_vehicles_value",
        ]
        vehicle_target_filters = target_observed_source_masks(
            vehicle_train,
            targets=vehicle_vars,
            target_allocation_flag_columns=SIPP_VEHICLE_TARGET_ALLOCATION_COLUMNS,
        )
        vehicle_train, vehicle_target_filters = filter_positive_finite_weight_rows(
            vehicle_train,
            weight_col="household_weight",
            target_filters=vehicle_target_filters,
            context_name="SIPP source vehicle donor",
        )
        vehicle_train, vehicle_target_filters = cap_training_sample(
            vehicle_train,
            max_train_samples=20_000,
            seed_name="calibration_sipp_vehicle_training_sample",
            target_filters=vehicle_target_filters,
        )
        fitted = QRF().fit(
            X_train=vehicle_train,
            predictors=VEHICLE_MODEL_PREDICTORS,
            imputed_variables=vehicle_vars,
            target_filters=vehicle_target_filters,
            weight_col="household_weight",
        )
        vehicle_preds = fitted.predict(X_test=vehicle_receiver)
        data["household_vehicles_owned"] = {
            time_period: np.clip(
                np.rint(vehicle_preds["household_vehicles_owned"].values),
                0,
                None,
            ).astype(np.int32)
        }
        data["household_vehicles_value"] = {
            time_period: np.clip(
                vehicle_preds["household_vehicles_value"].values,
                0,
                None,
            ).astype(np.float32)
        }
        del fitted, vehicle_preds
        gc.collect()

        logger.info("SIPP vehicle imputation complete")

    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        logger.warning(
            "SIPP asset imputation failed: %s. Keeping existing values.",
            e,
        )

    return data


@pipeline_node(
    PipelineNode(
        id="scf_qrf",
        label="SCF QRF Imputation",
        node_type="library",
        description="Impute net worth and auto-loan variables from SCF donor data.",
        source_file="policyengine_us_data/calibration/source_impute.py",
        status="current",
        stability="moving",
        pathways=["data_build"],
        validation_commands=[
            "uv run pytest tests/unit/calibration/test_source_impute.py"
        ],
    )
)
def _impute_scf(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Impute net_worth and auto_loan from SCF.

    Args:
        data: CPS data dict.
        state_fips: State FIPS per household.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.

    Returns:
        Updated data dict.
    """
    from microimpute.models.qrf import QRF

    from policyengine_us_data.datasets.scf.scf import SCF_2022

    scf_dataset = SCF_2022()
    scf_data = scf_dataset.load_dataset()
    scf_df = pd.DataFrame(dict(scf_data))

    scf_predictors = SCF_PREDICTORS

    available_preds = [p for p in scf_predictors if p in scf_df.columns]
    missing_preds = [p for p in scf_predictors if p not in scf_df.columns]
    if missing_preds:
        logger.warning("SCF missing predictors: %s", missing_preds)
        scf_predictors = available_preds

    scf_net_worth_targets = add_scf_net_worth_target(scf_df)
    scf_financial_asset_targets = add_scf_financial_asset_targets(scf_df)
    scf_household_asset_targets = add_scf_household_asset_targets(scf_df)
    scf_component_targets = add_scf_net_worth_component_targets(scf_df)
    require_scf_net_worth_formula_targets(
        scf_financial_asset_targets=scf_financial_asset_targets,
        scf_household_asset_targets=scf_household_asset_targets,
        scf_component_targets=scf_component_targets,
        scf_net_worth_targets=scf_net_worth_targets,
    )

    available_vars = [v for v in SCF_CORE_IMPUTED_VARIABLES if v in scf_df.columns]
    qrf_vars = (
        [v for v in scf_net_worth_targets if v in scf_df.columns]
        + available_vars
        + [v for v in scf_financial_asset_targets if v in scf_df.columns]
        + [v for v in scf_household_asset_targets if v in scf_df.columns]
        + [v for v in scf_component_targets if v in scf_df.columns]
    )
    if not qrf_vars:
        logger.warning("No SCF imputed variables available. Skipping.")
        return data

    weights = scf_df.get("wgt")

    donor = scf_df[scf_predictors + qrf_vars].copy()
    if weights is not None:
        donor["wgt"] = weights
    donor = donor.dropna(subset=scf_predictors)
    donor = donor.sample(frac=0.5, random_state=42).reset_index(drop=True)

    pe_vars = [
        "age",
        "is_male",
        "employment_income",
    ]
    cps_df = _build_cps_receiver(data, time_period, dataset_path, pe_vars)

    if "is_male" in cps_df.columns:
        cps_df["is_female"] = (~cps_df["is_male"].astype(bool)).astype(np.float32)
    else:
        cps_df["is_female"] = 0.0

    for var in [
        "cps_race",
        "is_married",
        "own_children_in_household",
    ]:
        if var in data:
            cps_df[var] = data[var][time_period].astype(np.float32)
        else:
            cps_df[var] = 0.0

    for var in [
        "taxable_interest_income",
        "tax_exempt_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
    ]:
        if var in data:
            cps_df[var] = data[var][time_period].astype(np.float32)
    cps_df["interest_dividend_income"] = (
        cps_df.get("taxable_interest_income", 0)
        + cps_df.get("tax_exempt_interest_income", 0)
        + cps_df.get("qualified_dividend_income", 0)
        + cps_df.get("non_qualified_dividend_income", 0)
    ).astype(np.float32)

    for var in [
        "tax_exempt_private_pension_income",
        "taxable_private_pension_income",
        "social_security_retirement",
    ]:
        if var in data:
            cps_df[var] = data[var][time_period].astype(np.float32)
    cps_df["social_security_pension_income"] = (
        cps_df.get("tax_exempt_private_pension_income", 0)
        + cps_df.get("taxable_private_pension_income", 0)
        + cps_df.get("social_security_retirement", 0)
    ).astype(np.float32)

    qrf = QRF()
    logger.info(
        "SCF QRF: %d train, %d test, vars=%s",
        len(donor),
        len(cps_df),
        qrf_vars,
    )
    fitted = qrf.fit(
        X_train=donor,
        predictors=scf_predictors,
        imputed_variables=qrf_vars,
        weight_col="wgt" if weights is not None else None,
        tune_hyperparameters=False,
    )
    preds = fitted.predict(X_test=cps_df)

    hh_ids = data["household_id"][time_period]
    person_hh_ids = data.get("person_household_id", {}).get(time_period)

    for var in available_vars:
        person_vals = preds[var].values
        entity = _get_variable_entity(var)
        if entity == "household" and person_hh_ids is not None:
            hh_vals = np.zeros(len(hh_ids), dtype=np.float32)
            hh_to_idx = {int(hid): i for i, hid in enumerate(hh_ids)}
            seen = set()
            for p_idx, p_hh in enumerate(person_hh_ids):
                hh_key = int(p_hh)
                if hh_key not in seen:
                    seen.add(hh_key)
                    hh_vals[hh_to_idx[hh_key]] = person_vals[p_idx]
            data[var] = {time_period: hh_vals}
            logger.info(
                "  %s: person(%d) -> household(%d)",
                var,
                len(person_vals),
                len(hh_vals),
            )
        else:
            data[var] = {time_period: person_vals}

    person_hh_ids = data.get("person_household_id", {}).get(time_period)
    if person_hh_ids is not None:
        first_person_mask = ~pd.Series(person_hh_ids).duplicated().values
        reference_household_ids = person_hh_ids[first_person_mask]
        for var in SCF_NET_WORTH_COMPONENT_VARIABLES:
            if var in preds:
                data[var] = {
                    time_period: preds.loc[first_person_mask, var].values.astype(
                        np.float32
                    )
                }
        for scf_var, policy_var in SCF_FINANCIAL_ASSET_POLICY_VARIABLES.items():
            if scf_var not in preds or policy_var not in data:
                continue
            data[policy_var] = {
                time_period: combine_sipp_and_scf_financial_assets(
                    sipp_values=data[policy_var][time_period],
                    scf_household_values=preds.loc[first_person_mask, scf_var].values,
                    person_household_ids=person_hh_ids,
                    reference_person_mask=first_person_mask,
                    time_period=time_period,
                )
            }
        for scf_var, policy_var in SCF_HOUSEHOLD_ASSET_POLICY_VARIABLES.items():
            if scf_var not in preds or policy_var not in data:
                continue
            data[policy_var] = {
                time_period: combine_sipp_and_scf_household_assets(
                    sipp_household_values=data[policy_var][time_period],
                    scf_household_values=preds.loc[first_person_mask, scf_var].values,
                    household_ids=hh_ids,
                    reference_household_ids=reference_household_ids,
                    time_period=time_period,
                )
            }
        net_worth_components = {}
        for var in ("bank_account_assets", "stock_assets", "bond_assets"):
            if var in data:
                net_worth_components[var] = (
                    aggregate_person_values_to_reference_households(
                        data[var][time_period],
                        person_hh_ids,
                        first_person_mask,
                    )
                )
        if "household_vehicles_value" in data:
            net_worth_components["household_vehicles_value"] = (
                align_household_values_to_reference_households(
                    data["household_vehicles_value"][time_period],
                    hh_ids,
                    reference_household_ids,
                )
            )
        for var in SCF_NET_WORTH_COMPONENT_VARIABLES:
            if var in data:
                net_worth_components[var] = data[var][time_period]
        if SCF_NET_WORTH_TARGET in preds:
            net_worth_components = rebalance_scf_net_worth_components(
                components=net_worth_components,
                target_net_worth=preds.loc[
                    first_person_mask, SCF_NET_WORTH_TARGET
                ].values.astype(np.float32),
            )
            for var in SCF_NET_WORTH_COMPONENT_VARIABLES:
                if var in net_worth_components:
                    data[var] = {time_period: net_worth_components[var]}
        data["net_worth"] = {
            time_period: compute_net_worth_from_components(
                components=net_worth_components,
            )
        }

    del fitted, preds
    gc.collect()

    logger.info("SCF imputation complete: %s", SCF_IMPUTED_VARIABLES)
    return data


def _impute_org(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Impute ORG-only labor-market variables onto CPS persons."""
    pe_vars = [
        "age",
        "is_male",
        "is_hispanic",
        "cps_race",
        "employment_income",
        "weekly_hours_worked",
        "self_employment_income",
    ]
    cps_df = _build_cps_receiver(data, time_period, dataset_path, pe_vars)

    if "is_male" in cps_df.columns:
        is_female = (~cps_df["is_male"].astype(bool)).astype(np.float32).values
    elif "is_female" in data:
        is_female = data["is_female"][time_period].astype(np.float32)
    else:
        is_female = np.zeros(len(cps_df), dtype=np.float32)

    person_states = _person_state_fips(data, state_fips, time_period)
    receiver = build_org_receiver_frame(
        age=cps_df["age"].values,
        is_female=is_female,
        is_hispanic=cps_df["is_hispanic"].values,
        cps_race=cps_df["cps_race"].values,
        state_fips=person_states,
        employment_income=cps_df["employment_income"].values,
        weekly_hours_worked=cps_df["weekly_hours_worked"].values,
    )
    self_employment_income = (
        cps_df["self_employment_income"].values
        if "self_employment_income" in cps_df.columns
        else None
    )
    n_persons = len(data["person_id"][time_period])
    predictions = predict_org_features(
        receiver,
        self_employment_income=self_employment_income,
    )

    for var in ORG_IMPUTED_VARIABLES:
        values = predictions[var].values
        if len(values) != n_persons:
            raise ValueError(
                f"ORG prediction for '{var}' has {len(values)} entries "
                f"but dataset has {n_persons} persons"
            )
        if var in ORG_BOOL_VARIABLES:
            data[var] = {time_period: values.astype(bool)}
        else:
            data[var] = {time_period: values.astype(np.float32)}

    logger.info("ORG imputation complete: %s", ORG_IMPUTED_VARIABLES)
    return data
