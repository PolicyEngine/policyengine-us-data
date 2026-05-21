import pandas as pd
import numpy as np
from microimpute.models.qrf import QRF
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.randomness import seeded_rng
import pickle
from huggingface_hub import hf_hub_download
from policyengine_us_data.datasets.cps.tipped_occupation import (
    derive_any_treasury_tipped_occupation_code,
    derive_is_tipped_occupation,
)
from policyengine_us_data.utils.source_quality import (
    filter_observed_source_rows,
    sipp_allocation_flag_for,
    target_observed_source_masks,
)


SIPP_JOB_OCCUPATION_COLUMNS = [f"TJB{i}_OCC" for i in range(1, 8)]
SIPP_TIP_AMOUNT_COLUMNS = [f"TJB{i}_TXAMT" for i in range(1, 8)]
SIPP_TIP_ALLOCATION_COLUMNS = [
    sipp_allocation_flag_for(column) for column in SIPP_TIP_AMOUNT_COLUMNS
]
TIP_MODEL_PREDICTORS = [
    "employment_income",
    "age",
    "count_under_18",
    "count_under_6",
    "is_tipped_occupation",
]

VEHICLE_MODEL_PREDICTORS = [
    "household_employment_income",
    "household_interest_income",
    "household_dividend_income",
    "household_rental_income",
    "reference_age",
    "reference_is_female",
    "reference_is_married",
    "count_under_18",
    "household_size",
    "is_homeowner",
]

SSI_DISABILITY_MODEL_VARIABLE = "meets_ssi_disability_criteria"

SSI_DISABILITY_MODEL_PREDICTORS = [
    "age",
    "is_female",
    "is_married",
    "employment_income",
    "interest_income",
    "dividend_income",
    "rental_income",
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
    "count_under_18",
    "is_disabled",
    "social_security_disability",
    "has_disability_income",
]


def train_tip_model():
    DOWNLOAD_FULL_SIPP = False

    if DOWNLOAD_FULL_SIPP:
        hf_hub_download(
            repo_id="PolicyEngine/policyengine-us-data",
            filename="pu2023.csv",
            repo_type="model",
            local_dir=STORAGE_FOLDER,
        )
        cols = [
            "SSUID",
            "PNUM",
            "MONTHCODE",
            "ERESIDENCEID",
            "ERELRPE",
            "SPANEL",
            "SWAVE",
            "WPFINWGT",
            "ESEX",
            "TAGE",
            "TAGE_EHC",
            "ERACE",
            "EORIGIN",
            "EEDUC",
            "EDEPCLM",
            "EMS",
            "EFSTATUS",
            "TJB1_TXAMT",
            "TJB1_MSUM",
            "TJB1_OCC",
            "TJB1_IND",
            "AJB1_TXAMT",
            "TPTOTINC",
        ]

        for col in cols:
            if "JB1" in col:
                for i in range(2, 8):
                    cols.append(col.replace("JB1", f"JB{i}"))

        df = pd.read_csv(
            STORAGE_FOLDER / "pu2023.csv",
            delimiter="|",
            usecols=cols,
        )

    else:
        hf_hub_download(
            repo_id="PolicyEngine/policyengine-us-data",
            filename="pu2023_slim.csv",
            repo_type="model",
            local_dir=STORAGE_FOLDER,
        )
        df = pd.read_csv(
            STORAGE_FOLDER / "pu2023_slim.csv",
        )
    # Sum tip dollar-amount columns (TJB*_TXAMT) across all jobs.
    # Previously used `str.contains("TXAMT")`, which also picked up
    # AJB*_TXAMT Census allocation flags (small ints 0/1/2 indicating
    # imputation status) and added them to the dollar totals.
    tip_amount_columns = [column for column in SIPP_TIP_AMOUNT_COLUMNS if column in df]
    df["tip_income"] = df[tip_amount_columns].fillna(0).sum(axis=1) * 12
    df["employment_income"] = df.TPTOTINC * 12
    df["is_under_18"] = (df.TAGE < 18) & (df.MONTHCODE == 12)
    df["is_under_6"] = (df.TAGE < 6) & (df.MONTHCODE == 12)
    df["count_under_18"] = (
        df.groupby("SSUID")["is_under_18"].sum().loc[df.SSUID.values].values
    )
    df["count_under_6"] = (
        df.groupby("SSUID")["is_under_6"].sum().loc[df.SSUID.values].values
    )
    df["household_weight"] = df.WPFINWGT
    df["household_id"] = df.SSUID
    df["age"] = df.TAGE
    df["treasury_tipped_occupation_code"] = derive_any_treasury_tipped_occupation_code(
        df[SIPP_JOB_OCCUPATION_COLUMNS]
    )
    df["is_tipped_occupation"] = derive_is_tipped_occupation(
        df["treasury_tipped_occupation_code"]
    )

    # SIPP data are monthly (one row per person × MONTHCODE 1..12).
    # tip_income and employment_income above were annualized as
    # ``month_value * 12``, which is only right for a single reference
    # month. Without this filter the training frame had 12 rows per
    # person — each annualized from a different month — so the QRF
    # treated Jan-income-annualized and Dec-income-annualized as
    # separate observations and mixed seasonal tip amounts
    # (restaurant, holiday) with the annual figures. Filter to
    # December (end of year) so every training row represents one
    # person-year.
    df = df[df["MONTHCODE"] == 12]
    tip_target_filters = target_observed_source_masks(
        df,
        targets=["tip_income"],
        target_source_columns={"tip_income": tip_amount_columns},
        target_allocation_flag_columns={"tip_income": SIPP_TIP_ALLOCATION_COLUMNS},
        require_nonmissing_source=False,
    )

    sipp = df[
        [
            "household_id",
            "employment_income",
            "tip_income",
            "count_under_18",
            "count_under_6",
            "age",
            "is_tipped_occupation",
            "household_weight",
        ]
    ]

    sipp = sipp[~sipp.isna().any(axis=1)]

    model = QRF(max_train_samples=10_000)

    model = model.fit(
        X_train=sipp,
        predictors=TIP_MODEL_PREDICTORS,
        imputed_variables=["tip_income"],
        target_filters=tip_target_filters,
        weight_col="household_weight",
    )

    return model


def get_tip_model() -> QRF:
    model_path = STORAGE_FOLDER / "tips_tipped_occ_v3.pkl"

    if not model_path.exists():
        model = train_tip_model()

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model


# Asset imputation from SIPP 2023
# Imputes asset categories separately for policy flexibility

ASSET_JOB_EARNINGS_COLUMNS = [f"TJB{i}_MSUM" for i in range(1, 8)]
SIPP_ASSET_TARGET_SOURCE_COLUMNS = {
    "bank_account_assets": ["TVAL_BANK"],
    "stock_assets": ["TVAL_STMF"],
    "bond_assets": ["TVAL_BOND"],
}
SIPP_ASSET_TARGET_ALLOCATION_COLUMNS = {
    target: [sipp_allocation_flag_for(column) for column in columns]
    for target, columns in SIPP_ASSET_TARGET_SOURCE_COLUMNS.items()
}
SIPP_ASSET_ALLOCATION_COLUMNS = sorted(
    {
        column
        for columns in SIPP_ASSET_TARGET_ALLOCATION_COLUMNS.values()
        for column in columns
    }
)

ASSET_COLUMNS = (
    [
        "SSUID",
        "PNUM",
        "MONTHCODE",
        "SPANEL",
        "SWAVE",
        "WPFINWGT",
        "TAGE",
        "ESEX",
        "EMS",
        "TSSSAMT",
        "TRETINCAMT",
        # Asset values (person-level sums from SIPP)
        "TVAL_BANK",  # Checking, savings, money market
        "TVAL_STMF",  # Stocks and mutual funds
        "TVAL_BOND",  # Bonds and government securities
        # Income from assets (monthly, person-level)
        "TINC_BANK",  # Interest from bank accounts
        "TINC_STMF",  # Dividends from stocks/mutual funds
        "TINC_BOND",  # Interest from bonds
        "TINC_RENT",  # Rental income
    ]
    + ASSET_JOB_EARNINGS_COLUMNS
    + SIPP_ASSET_ALLOCATION_COLUMNS
)

ASSET_PREDICTORS = [
    "employment_income",
    "interest_income",
    "dividend_income",
    "rental_income",
    "social_security",
    "retirement_income",
    "non_ssi_income",
    "age",
    "is_female",
    "is_married",
    "count_under_18",
    "count_under_6",
    "household_size",
]

SSI_DISABILITY_INCOME_AMOUNT_COLUMNS = [
    "TDIS1AMT",
    "TDIS2AMT",
    "TDIS3AMT",
    "TDIS4AMT",
    "TDIS5AMT",
    "TDIS6AMT",
    "TDIS7AMT",
    "TDIS8AMT",
    "TDIS9AMT",
    "TDIS10AMT",
]
SSI_DISABILITY_LABEL_SOURCE_COLUMNS = ["RSSI_YRYN", "ESSI_BRSN"]
SSI_DISABILITY_LABEL_ALLOCATION_COLUMNS = [
    sipp_allocation_flag_for(column) for column in SSI_DISABILITY_LABEL_SOURCE_COLUMNS
]

SSI_DISABILITY_COLUMNS = sorted(
    set(
        ASSET_COLUMNS
        + [
            "TPTOTINC",
            "RSSI_YRYN",
            "EDISABL",
            "EHLTHCOND",
            "RDIS",
            "RDIS_ALT",
            "EDISANY",
            "ENJ_NOWRK3",
            "ESSRSN2YN",
            "ESSI_BRSN",
            *SSI_DISABILITY_INCOME_AMOUNT_COLUMNS,
            *SSI_DISABILITY_LABEL_ALLOCATION_COLUMNS,
        ]
    )
)

SIPP_VEHICLE_TARGET_ALLOCATION_COLUMNS = {
    "household_vehicles_owned": [sipp_allocation_flag_for("TVEH_NUM")],
    "household_vehicles_value": [sipp_allocation_flag_for("THVAL_VEH")],
}

VEHICLE_COLUMNS = [
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
    "EMS",
    "TPTOTINC",
    "TINC_BANK",
    "TINC_STMF",
    "TINC_BOND",
    "TINC_RENT",
    "TVEH_NUM",
    "THVAL_VEH",
    "THVAL_HOME",
    "AVEH_NUM",
    "AHVAL_VEH",
]


def _add_asset_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Add SIPP predictors shared by legacy and source-impute asset models."""
    df = df.copy()
    df["age"] = df.TAGE
    df["is_female"] = df.ESEX == 2
    df["is_married"] = df.EMS == 1
    df["household_weight"] = df.WPFINWGT
    df["household_id"] = df.SSUID

    job_cols = [col for col in ASSET_JOB_EARNINGS_COLUMNS if col in df]
    if job_cols:
        df["employment_income"] = df[job_cols].fillna(0).sum(axis=1) * 12
    elif "TPTOTINC" in df:
        df["employment_income"] = df.TPTOTINC.fillna(0) * 12
    else:
        df["employment_income"] = 0.0

    df["interest_income"] = (df["TINC_BANK"].fillna(0) + df["TINC_BOND"].fillna(0)) * 12
    df["dividend_income"] = df["TINC_STMF"].fillna(0) * 12
    df["rental_income"] = df["TINC_RENT"].fillna(0) * 12
    df["social_security"] = df["TSSSAMT"].fillna(0) * 12
    df["retirement_income"] = df["TRETINCAMT"].fillna(0) * 12
    df["non_ssi_income"] = (
        df["employment_income"] + df["social_security"] + df["retirement_income"]
    )

    df["is_under_18"] = df.TAGE < 18
    df["is_under_6"] = df.TAGE < 6
    grouped = df.groupby("SSUID")
    df["count_under_18"] = grouped["is_under_18"].transform("sum")
    df["count_under_6"] = grouped["is_under_6"].transform("sum")
    df["household_size"] = grouped["PNUM"].transform("count")

    return df


def _yes(df: pd.DataFrame, column: str) -> pd.Series:
    values = df[column] if column in df else pd.Series(0, index=df.index)
    return values.fillna(0).astype(float).eq(1)


def _ssi_financial_candidate_mask(
    df: pd.DataFrame, time_period: int = 2024
) -> pd.Series:
    """Approximate non-disability SSI financial eligibility in SIPP.

    This is only a training-frame screen. It avoids treating people whose
    resources or income make SSI receipt structurally unlikely as clean
    non-disabled labels.
    """
    try:
        from policyengine_us import CountryTaxBenefitSystem

        p = CountryTaxBenefitSystem().parameters(f"{time_period}-01-01").gov.ssa.ssi
        individual_resource_limit = float(p.eligibility.resources.limit.individual)
        couple_resource_limit = float(p.eligibility.resources.limit.couple)
        individual_fbr = float(p.amount.individual)
        couple_fbr = float(p.amount.couple)
    except Exception:
        individual_resource_limit = 2_000.0
        couple_resource_limit = 3_000.0
        individual_fbr = 943.0
        couple_fbr = 1_415.0

    resource_limit = np.where(
        df["is_married"].astype(bool),
        couple_resource_limit,
        individual_resource_limit,
    )
    monthly_income_limit = np.where(
        df["is_married"].astype(bool),
        couple_fbr,
        individual_fbr,
    )
    liquid_resources = (
        df["bank_account_assets"].fillna(0)
        + df["stock_assets"].fillna(0)
        + df["bond_assets"].fillna(0)
    )
    monthly_income = df["TPTOTINC"].fillna(0)
    return (liquid_resources <= resource_limit) & (
        monthly_income <= monthly_income_limit * 2
    )


def build_ssi_disability_training_frame(
    df: pd.DataFrame, time_period: int = 2024
) -> pd.DataFrame:
    """Build SIPP training rows for latent SSI disability criteria."""
    df = df[df.MONTHCODE == 12].copy()

    df["bank_account_assets"] = df["TVAL_BANK"].fillna(0)
    df["stock_assets"] = df["TVAL_STMF"].fillna(0)
    df["bond_assets"] = df["TVAL_BOND"].fillna(0)
    df["age"] = df.TAGE
    df["is_female"] = df.ESEX == 2
    df["is_married"] = df.EMS == 1
    df["employment_income"] = df.TPTOTINC.fillna(0) * 12
    df["interest_income"] = (df["TINC_BANK"].fillna(0) + df["TINC_BOND"].fillna(0)) * 12
    df["dividend_income"] = df["TINC_STMF"].fillna(0) * 12
    df["rental_income"] = df["TINC_RENT"].fillna(0) * 12
    df["household_weight"] = df.WPFINWGT.fillna(0)
    df["is_under_18"] = df.TAGE < 18
    df["count_under_18"] = (
        df.groupby("SSUID")["is_under_18"].sum().loc[df.SSUID.values].values
    )

    disability_income_amount = pd.Series(0.0, index=df.index)
    for column in SSI_DISABILITY_INCOME_AMOUNT_COLUMNS:
        if column in df:
            disability_income_amount += df[column].fillna(0)

    df["is_disabled"] = (
        _yes(df, "RDIS_ALT")
        | _yes(df, "RDIS")
        | _yes(df, "EDISABL")
        | _yes(df, "EHLTHCOND")
        | _yes(df, "ENJ_NOWRK3")
    )
    df["social_security_disability"] = _yes(df, "ESSRSN2YN")
    df["has_disability_income"] = _yes(df, "EDISANY") | disability_income_amount.gt(0)

    received_ssi = _yes(df, "RSSI_YRYN")
    under_65 = df["age"] < 65
    disabled_or_blind_reason = (
        df.get("ESSI_BRSN", pd.Series(-9, index=df.index))
        .fillna(-9)
        .astype(float)
        .eq(1)
    )
    aged_reason = (
        df.get("ESSI_BRSN", pd.Series(-9, index=df.index))
        .fillna(-9)
        .astype(float)
        .eq(2)
    )
    df[SSI_DISABILITY_MODEL_VARIABLE] = (
        received_ssi & under_65 & (disabled_or_blind_reason | ~aged_reason)
    )

    financial_candidate = _ssi_financial_candidate_mask(df, time_period=time_period)
    df["ssi_disability_training_candidate"] = (financial_candidate & under_65) | df[
        SSI_DISABILITY_MODEL_VARIABLE
    ]
    df = filter_observed_source_rows(
        df,
        target_name=SSI_DISABILITY_MODEL_VARIABLE,
        source_columns=SSI_DISABILITY_LABEL_SOURCE_COLUMNS,
        allocation_flag_columns=SSI_DISABILITY_LABEL_ALLOCATION_COLUMNS,
    )

    columns = SSI_DISABILITY_MODEL_PREDICTORS + [
        SSI_DISABILITY_MODEL_VARIABLE,
        "ssi_disability_training_candidate",
        "household_weight",
    ]
    return df[columns].dropna()


def prepare_ssi_disability_receiver(df: pd.DataFrame) -> pd.DataFrame:
    """Return receiver predictors expected by the SSI disability model."""
    df = df.copy()
    for predictor in SSI_DISABILITY_MODEL_PREDICTORS:
        if predictor not in df:
            df[predictor] = 0
    return df[SSI_DISABILITY_MODEL_PREDICTORS].fillna(0)


def _coerce_ssi_disability_signal(values) -> np.ndarray:
    series = pd.Series(values)
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float).gt(0).to_numpy(dtype=bool)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def apply_ssi_disability_signal_screen(
    meets_ssi_disability_criteria: np.ndarray,
    is_disabled: np.ndarray,
    social_security_disability: np.ndarray,
    has_disability_income: np.ndarray,
) -> np.ndarray:
    """Require at least one observed disability signal before accepting imputation."""
    disability_signal = (
        _coerce_ssi_disability_signal(is_disabled)
        | _coerce_ssi_disability_signal(social_security_disability)
        | _coerce_ssi_disability_signal(has_disability_income)
    )
    return np.asarray(meets_ssi_disability_criteria, dtype=bool) & disability_signal


def preserve_under_65_ssi_disability_criteria(
    meets_ssi_disability_criteria: np.ndarray,
    age: np.ndarray,
    ssi_reported: np.ndarray | None = None,
    existing_meets_ssi_disability_criteria: np.ndarray | None = None,
) -> np.ndarray:
    """Preserve observed under-65 SSI disability criteria anchors."""
    result = np.asarray(meets_ssi_disability_criteria, dtype=bool).copy()
    under_65 = pd.Series(age).fillna(np.inf).astype(float).lt(65).to_numpy()

    if ssi_reported is not None:
        reported_ssi = pd.Series(ssi_reported).fillna(0).astype(float).gt(0).to_numpy()
        result |= reported_ssi & under_65

    if existing_meets_ssi_disability_criteria is not None:
        result |= (
            _coerce_ssi_disability_signal(existing_meets_ssi_disability_criteria)
            & under_65
        )

    return result


def coerce_ssi_disability_predictions(values) -> np.ndarray:
    """Convert classifier labels to booleans without treating 'False' as true."""
    series = pd.Series(values)
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float).ne(0).to_numpy(dtype=bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def predict_ssi_disability_criteria(model, receiver_df: pd.DataFrame) -> np.ndarray:
    """Predict SSI disability criteria before applying dynamic policy screens."""
    receiver = prepare_ssi_disability_receiver(receiver_df)
    predictions = model.predict(X_test=receiver[SSI_DISABILITY_MODEL_PREDICTORS])
    meets_ssi_disability_criteria = coerce_ssi_disability_predictions(
        predictions[SSI_DISABILITY_MODEL_VARIABLE]
    )
    return apply_ssi_disability_signal_screen(
        meets_ssi_disability_criteria,
        receiver["is_disabled"],
        receiver["social_security_disability"],
        receiver["has_disability_income"],
    )


def train_asset_model():
    """Train QRF model for liquid asset categories using SIPP 2023 data.

    Imputes three asset categories separately:
    - bank_account_assets: checking, savings, money market (TVAL_BANK)
    - stock_assets: stocks and mutual funds (TVAL_STMF)
    - bond_assets: bonds and government securities (TVAL_BOND)

    Policy models can then define countable resources based on rules.
    """
    hf_hub_download(
        repo_id="PolicyEngine/policyengine-us-data",
        filename="pu2023.csv",
        repo_type="model",
        local_dir=STORAGE_FOLDER,
    )

    df = pd.read_csv(
        STORAGE_FOLDER / "pu2023.csv",
        delimiter="|",
        usecols=ASSET_COLUMNS,
    )

    # Filter to December (end of year values) to get annual snapshot
    df = df[df.MONTHCODE == 12]

    # Rename SIPP variables to policy-neutral names
    df["bank_account_assets"] = df["TVAL_BANK"].fillna(0)
    df["stock_assets"] = df["TVAL_STMF"].fillna(0)
    df["bond_assets"] = df["TVAL_BOND"].fillna(0)

    df = _add_asset_predictors(df)

    sipp = df[
        [
            "household_id",
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
            "household_weight",
            *ASSET_PREDICTORS,
            *[
                column
                for columns in SIPP_ASSET_TARGET_SOURCE_COLUMNS.values()
                for column in columns
            ],
            *SIPP_ASSET_ALLOCATION_COLUMNS,
        ]
    ]

    asset_vars = [
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
    ]
    model = QRF(max_train_samples=20_000)
    model = model.fit(
        X_train=sipp,
        predictors=ASSET_PREDICTORS,
        imputed_variables=asset_vars,
        target_filters=target_observed_source_masks(
            sipp,
            targets=asset_vars,
            target_source_columns=SIPP_ASSET_TARGET_SOURCE_COLUMNS,
            target_allocation_flag_columns=SIPP_ASSET_TARGET_ALLOCATION_COLUMNS,
        ),
        weight_col="household_weight",
    )

    return model


def get_asset_model() -> QRF:
    """Get or train the liquid asset imputation model."""
    model_path = STORAGE_FOLDER / "liquid_assets_v3.pkl"

    if not model_path.exists():
        model = train_asset_model()

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model


def train_ssi_disability_model(time_period: int = 2024):
    """Train a boolean model for likely SSI disability criteria."""
    hf_hub_download(
        repo_id="PolicyEngine/policyengine-us-data",
        filename="pu2023.csv",
        repo_type="model",
        local_dir=STORAGE_FOLDER,
    )

    df = pd.read_csv(
        STORAGE_FOLDER / "pu2023.csv",
        delimiter="|",
        usecols=SSI_DISABILITY_COLUMNS,
    )
    sipp = build_ssi_disability_training_frame(df, time_period=time_period)
    sipp = sipp[sipp["ssi_disability_training_candidate"]].drop(
        columns=["ssi_disability_training_candidate"]
    )

    if sipp[SSI_DISABILITY_MODEL_VARIABLE].nunique() < 2:
        raise ValueError(
            "SIPP SSI disability training frame must contain both positive "
            "and negative labels."
        )

    ssi_rng = seeded_rng("sipp_ssi_disability_model_training_sample")
    weights = sipp.household_weight / sipp.household_weight.sum()
    sipp = sipp.loc[
        ssi_rng.choice(
            sipp.index,
            size=min(20_000, len(sipp)),
            replace=True,
            p=weights,
        )
    ]

    model = QRF()
    model = model.fit(
        X_train=sipp,
        predictors=SSI_DISABILITY_MODEL_PREDICTORS,
        imputed_variables=[SSI_DISABILITY_MODEL_VARIABLE],
    )

    return model


def get_ssi_disability_model(time_period: int = 2024) -> QRF:
    """Get or train the SSI disability criteria imputation model."""
    model_path = STORAGE_FOLDER / f"ssi_disability_criteria_v2_{time_period}.pkl"

    if not model_path.exists():
        model = train_ssi_disability_model(time_period=time_period)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model


def build_vehicle_training_frame() -> pd.DataFrame:
    """Build a household-level SIPP frame for vehicle asset imputation."""
    hf_hub_download(
        repo_id="PolicyEngine/policyengine-us-data",
        filename="pu2023.csv",
        repo_type="model",
        local_dir=STORAGE_FOLDER,
    )

    df = pd.read_csv(
        STORAGE_FOLDER / "pu2023.csv",
        delimiter="|",
        usecols=VEHICLE_COLUMNS,
    )
    df = df[df.MONTHCODE == 12].copy()

    df["employment_income"] = df.TPTOTINC.fillna(0) * 12
    df["interest_income"] = (df["TINC_BANK"].fillna(0) + df["TINC_BOND"].fillna(0)) * 12
    df["dividend_income"] = df["TINC_STMF"].fillna(0) * 12
    df["rental_income"] = df["TINC_RENT"].fillna(0) * 12
    df["is_under_18"] = df["TAGE"].fillna(0) < 18

    grouped = df.groupby("SSUID")

    reference_idx = grouped["TAGE"].idxmax()
    reference_people = (
        df.loc[reference_idx, ["SSUID", "TAGE", "ESEX", "EMS"]]
        .rename(
            columns={
                "TAGE": "reference_age",
                "ESEX": "reference_sex",
                "EMS": "reference_marital_status",
            }
        )
        .set_index("SSUID")
    )

    household = pd.DataFrame(
        {
            "household_id": grouped["SSUID"].first(),
            "household_weight": grouped["WPFINWGT"].first().fillna(0),
            "household_employment_income": grouped["employment_income"].sum(),
            "household_interest_income": grouped["interest_income"].sum(),
            "household_dividend_income": grouped["dividend_income"].sum(),
            "household_rental_income": grouped["rental_income"].sum(),
            "count_under_18": grouped["is_under_18"].sum(),
            "household_size": grouped.size(),
            "household_vehicles_owned": grouped["TVEH_NUM"].max().fillna(0),
            "household_vehicles_value": grouped["THVAL_VEH"].first().fillna(0),
            "AVEH_NUM": grouped["AVEH_NUM"].max().fillna(0),
            "AHVAL_VEH": grouped["AHVAL_VEH"].first().fillna(0),
            "is_homeowner": (grouped["THVAL_HOME"].first().fillna(0) > 0).astype(
                np.float32
            ),
        }
    ).reset_index(drop=True)

    household = household.merge(
        reference_people,
        left_on="household_id",
        right_index=True,
        how="left",
    )
    household["reference_is_female"] = (
        household["reference_sex"].fillna(1) == 2
    ).astype(np.float32)
    household["reference_is_married"] = (
        household["reference_marital_status"].fillna(0) == 1
    ).astype(np.float32)

    household = household.drop(
        columns=["reference_sex", "reference_marital_status"],
        errors="ignore",
    )
    household = household.fillna(0)
    return household


def train_vehicle_model():
    """Train a household-level vehicle asset model from SIPP 2023."""
    sipp = build_vehicle_training_frame()
    sipp = sipp[~sipp.isna().any(axis=1)]
    vehicle_vars = [
        "household_vehicles_owned",
        "household_vehicles_value",
    ]
    model = QRF(max_train_samples=20_000)
    model = model.fit(
        X_train=sipp,
        predictors=VEHICLE_MODEL_PREDICTORS,
        imputed_variables=vehicle_vars,
        target_filters=target_observed_source_masks(
            sipp,
            targets=vehicle_vars,
            target_allocation_flag_columns=SIPP_VEHICLE_TARGET_ALLOCATION_COLUMNS,
        ),
        weight_col="household_weight",
    )
    return model


def get_vehicle_model() -> QRF:
    """Get or train the household vehicle imputation model."""
    model_path = STORAGE_FOLDER / "household_vehicle_assets_v2.pkl"

    if not model_path.exists():
        model = train_vehicle_model()

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model
