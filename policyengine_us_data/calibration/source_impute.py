"""Non-PUF QRF imputations with state_fips as predictor.

Re-imputes variables from ACS, SIPP, and SCF donor surveys
with state_fips included as a QRF predictor. This runs after
geography assignment so imputations reflect assigned state.

Sources and variables:
    ACS  -> rent, real_estate_taxes
    SIPP -> tip_income, bank_account_assets, stock_assets,
            bond_assets
    SCF  -> net_worth, auto_loan_balance, auto_loan_interest

Usage in unified calibration pipeline:
    1. Load raw CPS
    2. Clone Nx, assign geography
    3. impute_source_variables()  <-- this module
    4. PUF clone + QRF impute (puf_impute.py)
    5. PE simulate, build matrix, calibrate
"""

import gc
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Variables imputed from each source
ACS_IMPUTED_VARIABLES = [
    "rent",
    "real_estate_taxes",
]

SIPP_IMPUTED_VARIABLES = [
    "tip_income",
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
]

SCF_IMPUTED_VARIABLES = [
    "net_worth",
    "auto_loan_balance",
    "auto_loan_interest",
]

ALL_SOURCE_VARIABLES = (
    ACS_IMPUTED_VARIABLES + SIPP_IMPUTED_VARIABLES + SCF_IMPUTED_VARIABLES
)

# Predictors for each source (state_fips always appended)
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
]

SIPP_ASSETS_PREDICTORS = [
    "employment_income",
    "age",
    "is_female",
    "is_married",
    "count_under_18",
]

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


def impute_source_variables(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int = 2024,
    dataset_path: Optional[str] = None,
    skip_acs: bool = False,
    skip_sipp: bool = False,
    skip_scf: bool = False,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Re-impute ACS/SIPP/SCF variables with state as predictor.

    Overwrites existing imputed values in data with new values
    that use assigned state_fips as a QRF predictor.

    Args:
        data: CPS dataset dict {variable: {time_period: array}}.
        state_fips: State FIPS per household, shape
            (n_households,).
        time_period: Tax year.
        dataset_path: Path to CPS h5 file (for computing
            demographic predictors via Microsimulation).
        skip_acs: Skip ACS imputation (rent, real_estate_taxes).
        skip_sipp: Skip SIPP imputation (tips, assets).
        skip_scf: Skip SCF imputation (net_worth, auto_loan).

    Returns:
        Updated data dict with re-imputed variables.
    """
    # Add state_fips to data (household level)
    data["state_fips"] = {
        time_period: state_fips.astype(np.int32),
    }

    if not skip_acs:
        logger.info("Imputing ACS variables with state predictor")
        data = _impute_acs(data, state_fips, time_period, dataset_path)

    if not skip_sipp:
        logger.info("Imputing SIPP variables with state predictor")
        data = _impute_sipp(data, state_fips, time_period, dataset_path)

    if not skip_scf:
        logger.info("Imputing SCF variables with state predictor")
        data = _impute_scf(data, state_fips, time_period, dataset_path)

    return data


def _build_cps_receiver(
    data: Dict[str, Dict[int, np.ndarray]],
    time_period: int,
    dataset_path: Optional[str],
    pe_variables: list,
) -> pd.DataFrame:
    """Build CPS receiver DataFrame from Microsimulation.

    Uses Microsimulation for standard PE variables, falls back
    to data dict for variables not in the PE tax-benefit system.

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
        # Only request variables that exist in PE
        tbs = sim.tax_benefit_system
        valid_vars = [v for v in pe_variables if v in tbs.variables]
        if valid_vars:
            df = sim.calculate_dataframe(valid_vars)
        else:
            df = pd.DataFrame(index=range(len(data["person_id"][time_period])))
        del sim
    else:
        df = pd.DataFrame()

    # Add any remaining variables from data dict
    for var in pe_variables:
        if var not in df.columns and var in data:
            df[var] = data[var][time_period].astype(np.float32)

    return df


def _get_variable_entity(variable_name: str) -> str:
    """Return the entity key ('person', 'household', etc.) for a PE variable."""
    from policyengine_us import CountryTaxBenefitSystem

    tbs = CountryTaxBenefitSystem()
    var = tbs.variables.get(variable_name)
    if var is None:
        return "person"  # Default to person if unknown
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
        return np.array(
            [state_fips[hh_to_idx[int(hh_id)]] for hh_id in hh_ids_person]
        )
    n_hh = len(data["household_id"][time_period])
    n_persons = len(data["person_id"][time_period])
    return np.repeat(state_fips, n_persons // n_hh)


def _impute_acs(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Impute rent and real_estate_taxes from ACS with state.

    Trains QRF on ACS_2022 with state_fips as predictor,
    predicts on CPS household heads.

    Args:
        data: CPS data dict.
        state_fips: State FIPS per household.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.

    Returns:
        Updated data dict.
    """
    from microimpute.models.qrf import QRF
    from policyengine_us import Microsimulation

    from policyengine_us_data.datasets.acs.acs import ACS_2022

    # Load ACS donor data
    acs = Microsimulation(dataset=ACS_2022)
    predictors = ACS_PREDICTORS + ["state_fips"]

    # ACS has state — use it directly
    acs_df = acs.calculate_dataframe(ACS_PREDICTORS + ACS_IMPUTED_VARIABLES)
    acs_df["state_fips"] = acs.calculate(
        "state_fips", map_to="person"
    ).values.astype(np.float32)

    # Filter to household heads and sample
    train_df = acs_df[acs_df.is_household_head].sample(10_000, random_state=42)
    # Convert tenure_type to numeric for QRF
    if "tenure_type" in train_df.columns:
        train_df["tenure_type"] = (
            train_df["tenure_type"]
            .astype(str)
            .map(
                {
                    "OWNED_WITH_MORTGAGE": 1,
                    "OWNED_OUTRIGHT": 1,
                    "RENTED": 2,
                    "NONE": 0,
                }
            )
            .fillna(0)
            .astype(np.float32)
        )
    del acs

    # Build CPS receiver data
    if dataset_path is not None:
        cps_sim = Microsimulation(dataset=dataset_path)
        cps_df = cps_sim.calculate_dataframe(ACS_PREDICTORS)
        del cps_sim
    else:
        cps_df = pd.DataFrame()
        for pred in ACS_PREDICTORS:
            if pred in data:
                cps_df[pred] = data[pred][time_period].astype(np.float32)

    # Convert tenure_type to numeric
    if "tenure_type" in cps_df.columns:
        cps_df["tenure_type"] = (
            cps_df["tenure_type"]
            .astype(str)
            .map(
                {
                    "OWNED_WITH_MORTGAGE": 1,
                    "OWNED_OUTRIGHT": 1,
                    "RENTED": 2,
                    "NONE": 0,
                }
            )
            .fillna(0)
            .astype(np.float32)
        )

    # Add person-level state_fips
    person_states = _person_state_fips(data, state_fips, time_period)
    cps_df["state_fips"] = person_states.astype(np.float32)

    # Filter to household heads
    mask = (
        cps_df.is_household_head.values
        if "is_household_head" in cps_df.columns
        else np.ones(len(cps_df), dtype=bool)
    )
    cps_heads = cps_df[mask]

    # Train and predict
    qrf = QRF()
    logger.info(
        "ACS QRF: %d train, %d test, %d predictors",
        len(train_df),
        len(cps_heads),
        len(predictors),
    )
    fitted = qrf.fit(
        X_train=train_df,
        predictors=predictors,
        imputed_variables=ACS_IMPUTED_VARIABLES,
    )
    predictions = fitted.predict(X_test=cps_heads)

    # Write back (household heads only)
    n_persons = len(data["person_id"][time_period])
    for var in ACS_IMPUTED_VARIABLES:
        values = np.zeros(n_persons, dtype=np.float32)
        values[mask] = predictions[var].values
        data[var] = {time_period: values}
    # Also set pre_subsidy_rent = rent, housing_assistance = 0
    data["pre_subsidy_rent"] = {time_period: data["rent"][time_period].copy()}

    del fitted, predictions
    gc.collect()

    logger.info("ACS imputation complete: rent, real_estate_taxes")
    return data


def _impute_sipp(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Impute tip_income and liquid assets from SIPP with state.

    Trains QRF on SIPP 2023 with state_fips as predictor.
    Since SIPP doesn't have state, random states are assigned
    to donor records.

    Args:
        data: CPS data dict.
        state_fips: State FIPS per household.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.

    Returns:
        Updated data dict.
    """
    from microimpute.models.qrf import QRF
    from policyengine_us import Microsimulation

    from policyengine_us_data.calibration.clone_and_assign import (
        load_global_block_distribution,
    )

    # Load state distribution for random assignment to donor
    _, _, donor_states, block_probs = load_global_block_distribution()
    rng = np.random.default_rng(seed=88)

    # --- Tips imputation ---
    from policyengine_us_data.datasets.sipp.sipp import (
        train_tip_model,
    )

    # We need to retrain with state — can't reuse pickled model.
    # Load SIPP tip training data directly.
    from policyengine_us_data.storage import STORAGE_FOLDER
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="PolicyEngine/policyengine-us-data",
        filename="pu2023_slim.csv",
        repo_type="model",
        local_dir=STORAGE_FOLDER,
    )
    sipp_df = pd.read_csv(STORAGE_FOLDER / "pu2023_slim.csv")

    # Prepare SIPP tip data (matching sipp.py logic)
    sipp_df["tip_income"] = (
        sipp_df[sipp_df.columns[sipp_df.columns.str.contains("TXAMT")]]
        .fillna(0)
        .sum(axis=1)
        * 12
    )
    sipp_df["employment_income"] = sipp_df.TPTOTINC * 12
    sipp_df["age"] = sipp_df.TAGE
    sipp_df["household_weight"] = sipp_df.WPFINWGT
    sipp_df["household_id"] = sipp_df.SSUID

    sipp_df["is_under_18"] = sipp_df.TAGE < 18
    sipp_df["is_under_6"] = sipp_df.TAGE < 6
    sipp_df["count_under_18"] = (
        sipp_df.groupby("SSUID")["is_under_18"]
        .sum()
        .loc[sipp_df.SSUID.values]
        .values
    )
    sipp_df["count_under_6"] = (
        sipp_df.groupby("SSUID")["is_under_6"]
        .sum()
        .loc[sipp_df.SSUID.values]
        .values
    )

    tip_cols = [
        "household_id",
        "employment_income",
        "tip_income",
        "count_under_18",
        "count_under_6",
        "age",
        "household_weight",
    ]
    tip_train = sipp_df[tip_cols].dropna()
    tip_train = tip_train.loc[
        rng.choice(
            tip_train.index,
            size=min(10_000, len(tip_train)),
            replace=True,
            p=(tip_train.household_weight / tip_train.household_weight.sum()),
        )
    ]

    # Assign random states to SIPP donor
    tip_state_idx = rng.choice(
        len(donor_states), size=len(tip_train), p=block_probs
    )
    tip_train["state_fips"] = donor_states[tip_state_idx].astype(np.float32)

    # Build CPS receiver for tips
    # count_under_18/6 aren't PE variables — compute from data
    cps_tip_df = _build_cps_receiver(
        data, time_period, dataset_path, ["employment_income", "age"]
    )
    # Compute household child counts from ages
    person_ages = data["age"][time_period]
    hh_ids_person = data.get("person_household_id", {}).get(time_period)
    if hh_ids_person is not None:
        age_df = pd.DataFrame({"hh": hh_ids_person, "age": person_ages})
        under_18 = age_df.groupby("hh")["age"].apply(lambda x: (x < 18).sum())
        under_6 = age_df.groupby("hh")["age"].apply(lambda x: (x < 6).sum())
        cps_tip_df["count_under_18"] = under_18.loc[
            hh_ids_person
        ].values.astype(np.float32)
        cps_tip_df["count_under_6"] = under_6.loc[hh_ids_person].values.astype(
            np.float32
        )
    else:
        cps_tip_df["count_under_18"] = 0.0
        cps_tip_df["count_under_6"] = 0.0

    person_states = _person_state_fips(data, state_fips, time_period)
    cps_tip_df["state_fips"] = person_states.astype(np.float32)

    # Train and predict tips
    tip_predictors = SIPP_TIPS_PREDICTORS + ["state_fips"]
    qrf = QRF()
    logger.info(
        "SIPP tips QRF: %d train, %d test",
        len(tip_train),
        len(cps_tip_df),
    )
    fitted = qrf.fit(
        X_train=tip_train,
        predictors=tip_predictors,
        imputed_variables=["tip_income"],
    )
    tip_preds = fitted.predict(X_test=cps_tip_df)
    data["tip_income"] = {
        time_period: tip_preds["tip_income"].values,
    }
    del fitted, tip_preds
    gc.collect()

    logger.info("SIPP tip imputation complete")

    # --- Asset imputation ---
    # Reload SIPP for assets (uses full file)
    try:
        hf_hub_download(
            repo_id="PolicyEngine/policyengine-us-data",
            filename="pu2023.csv",
            repo_type="model",
            local_dir=STORAGE_FOLDER,
        )
        asset_cols = [
            "SSUID",
            "PNUM",
            "MONTHCODE",
            "WPFINWGT",
            "TAGE",
            "ESEX",
            "EMS",
            "TPTOTINC",
            "TVAL_BANK",
            "TVAL_STMF",
            "TVAL_BOND",
        ]
        asset_df = pd.read_csv(
            STORAGE_FOLDER / "pu2023.csv",
            delimiter="|",
            usecols=asset_cols,
        )
        asset_df = asset_df[asset_df.MONTHCODE == 12]

        asset_df["bank_account_assets"] = asset_df["TVAL_BANK"].fillna(0)
        asset_df["stock_assets"] = asset_df["TVAL_STMF"].fillna(0)
        asset_df["bond_assets"] = asset_df["TVAL_BOND"].fillna(0)
        asset_df["age"] = asset_df.TAGE
        asset_df["is_female"] = asset_df.ESEX == 2
        asset_df["is_married"] = asset_df.EMS == 1
        asset_df["employment_income"] = asset_df.TPTOTINC * 12
        asset_df["household_weight"] = asset_df.WPFINWGT
        asset_df["is_under_18"] = asset_df.TAGE < 18
        asset_df["count_under_18"] = (
            asset_df.groupby("SSUID")["is_under_18"]
            .sum()
            .loc[asset_df.SSUID.values]
            .values
        )

        asset_train_cols = [
            "employment_income",
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
            "age",
            "is_female",
            "is_married",
            "count_under_18",
            "household_weight",
        ]
        asset_train = asset_df[asset_train_cols].dropna()
        asset_train = asset_train.loc[
            rng.choice(
                asset_train.index,
                size=min(20_000, len(asset_train)),
                replace=True,
                p=(
                    asset_train.household_weight
                    / asset_train.household_weight.sum()
                ),
            )
        ]

        # Assign random states to SIPP donor
        asset_state_idx = rng.choice(
            len(donor_states),
            size=len(asset_train),
            p=block_probs,
        )
        asset_train["state_fips"] = donor_states[asset_state_idx].astype(
            np.float32
        )

        # Build CPS receiver for assets
        # is_female, is_married, count_under_18 need special
        # handling — is_male is PE, is_married is Family-level
        cps_asset_df = _build_cps_receiver(
            data,
            time_period,
            dataset_path,
            ["employment_income", "age", "is_male"],
        )
        # is_female = NOT is_male
        if "is_male" in cps_asset_df.columns:
            cps_asset_df["is_female"] = (
                ~cps_asset_df["is_male"].astype(bool)
            ).astype(np.float32)
        else:
            cps_asset_df["is_female"] = 0.0
        # is_married from marital_unit membership
        if "is_married" in data:
            cps_asset_df["is_married"] = data["is_married"][
                time_period
            ].astype(np.float32)
        else:
            cps_asset_df["is_married"] = 0.0
        # count_under_18
        cps_asset_df["count_under_18"] = (
            cps_tip_df["count_under_18"]
            if "count_under_18" in cps_tip_df.columns
            else 0.0
        )

        cps_asset_df["state_fips"] = person_states.astype(np.float32)

        asset_predictors = SIPP_ASSETS_PREDICTORS + ["state_fips"]
        asset_vars = [
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
        ]
        qrf = QRF()
        logger.info(
            "SIPP assets QRF: %d train, %d test",
            len(asset_train),
            len(cps_asset_df),
        )
        fitted = qrf.fit(
            X_train=asset_train,
            predictors=asset_predictors,
            imputed_variables=asset_vars,
        )
        asset_preds = fitted.predict(X_test=cps_asset_df)

        for var in asset_vars:
            data[var] = {
                time_period: asset_preds[var].values,
            }
        del fitted, asset_preds
        gc.collect()

        logger.info("SIPP asset imputation complete")

    except Exception as e:
        logger.warning(
            "SIPP asset imputation failed: %s. " "Keeping existing values.",
            e,
        )

    return data


def _impute_scf(
    data: Dict[str, Dict[int, np.ndarray]],
    state_fips: np.ndarray,
    time_period: int,
    dataset_path: Optional[str] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Impute net_worth and auto_loan from SCF with state.

    Trains QRF on SCF_2022 with state_fips as predictor.
    Since SCF doesn't have state, random states are assigned
    to donor records.

    Args:
        data: CPS data dict.
        state_fips: State FIPS per household.
        time_period: Tax year.
        dataset_path: Path to CPS h5 for Microsimulation.

    Returns:
        Updated data dict.
    """
    from microimpute.models.qrf import QRF
    from policyengine_us import Microsimulation

    from policyengine_us_data.calibration.clone_and_assign import (
        load_global_block_distribution,
    )
    from policyengine_us_data.datasets.scf.scf import SCF_2022

    # Load state distribution for random assignment
    _, _, donor_states, block_probs = load_global_block_distribution()
    rng = np.random.default_rng(seed=77)

    # Load SCF donor data
    scf_dataset = SCF_2022()
    scf_data = scf_dataset.load_dataset()
    scf_df = pd.DataFrame({key: scf_data[key] for key in scf_data.keys()})

    # Assign random states to SCF
    scf_state_idx = rng.choice(
        len(donor_states), size=len(scf_df), p=block_probs
    )
    scf_df["state_fips"] = donor_states[scf_state_idx].astype(np.float32)

    scf_predictors = SCF_PREDICTORS + ["state_fips"]

    # Check which predictors are available
    available_preds = [p for p in scf_predictors if p in scf_df.columns]
    missing_preds = [p for p in scf_predictors if p not in scf_df.columns]
    if missing_preds:
        logger.warning("SCF missing predictors: %s", missing_preds)
        scf_predictors = available_preds

    scf_vars = SCF_IMPUTED_VARIABLES
    # SCF uses 'networth' not 'net_worth'
    scf_rename = {}
    if "networth" in scf_df.columns and "net_worth" not in scf_df.columns:
        scf_df["net_worth"] = scf_df["networth"]
        scf_rename["networth"] = "net_worth"

    available_vars = [v for v in scf_vars if v in scf_df.columns]
    if not available_vars:
        logger.warning("No SCF imputed variables available. Skipping.")
        return data

    weights = scf_df.get("wgt")

    # Sample SCF for training
    donor = scf_df[scf_predictors + available_vars].copy()
    if weights is not None:
        donor["wgt"] = weights
    donor = donor.dropna(subset=scf_predictors)
    donor = donor.sample(frac=0.5, random_state=42).reset_index(drop=True)

    # Build CPS receiver — many predictors are derived
    # Use PE Microsimulation for what it knows, derive the rest
    pe_vars = [
        "age",
        "is_male",
        "employment_income",
    ]
    cps_df = _build_cps_receiver(data, time_period, dataset_path, pe_vars)

    # Derive is_female from is_male
    if "is_male" in cps_df.columns:
        cps_df["is_female"] = (~cps_df["is_male"].astype(bool)).astype(
            np.float32
        )
    else:
        cps_df["is_female"] = 0.0

    # Derived predictors from data dict
    for var in [
        "cps_race",
        "is_married",
        "own_children_in_household",
    ]:
        if var in data:
            cps_df[var] = data[var][time_period].astype(np.float32)
        else:
            cps_df[var] = 0.0

    # Composite income predictors (matching cps.py SCF logic)
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

    person_states = _person_state_fips(data, state_fips, time_period)
    cps_df["state_fips"] = person_states.astype(np.float32)

    # Train and predict
    qrf = QRF()
    logger.info(
        "SCF QRF: %d train, %d test, vars=%s",
        len(donor),
        len(cps_df),
        available_vars,
    )
    fitted = qrf.fit(
        X_train=donor,
        predictors=scf_predictors,
        imputed_variables=available_vars,
        weight_col="wgt" if weights is not None else None,
        tune_hyperparameters=False,
    )
    preds = fitted.predict(X_test=cps_df)

    # SCF variables (net_worth, auto_loan_*) are household-level,
    # but QRF predicts at person level.  Aggregate back to household
    # by taking the first person's value in each household.
    hh_ids = data["household_id"][time_period]
    person_hh_ids = data.get("person_household_id", {}).get(time_period)

    for var in available_vars:
        person_vals = preds[var].values
        entity = _get_variable_entity(var)
        if entity == "household" and person_hh_ids is not None:
            # Map person-level predictions to household level
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
                var, len(person_vals), len(hh_vals),
            )
        else:
            data[var] = {time_period: person_vals}

    del fitted, preds
    gc.collect()

    logger.info("SCF imputation complete: %s", available_vars)
    return data
