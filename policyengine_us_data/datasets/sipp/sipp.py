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


SIPP_JOB_OCCUPATION_COLUMNS = [f"TJB{i}_OCC" for i in range(1, 8)]
TIP_MODEL_PREDICTORS = [
    "employment_income",
    "age",
    "count_under_18",
    "count_under_6",
    "is_tipped_occupation",
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
    df["tip_income"] = (
        df[df.columns[df.columns.str.match(r"TJB\d_TXAMT")]].fillna(0).sum(axis=1) * 12
    )
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

    # Seed the weighted resample so the tip-model training frame (and
    # the pickled QRF) is deterministic across rebuilds of the cache.
    tip_rng = seeded_rng("sipp_tip_model_training_sample")
    sipp = sipp.loc[
        tip_rng.choice(
            sipp.index,
            size=10_000,
            replace=True,
            p=sipp.household_weight / sipp.household_weight.sum(),
        )
    ]

    model = QRF()

    model = model.fit(
        X_train=sipp,
        predictors=TIP_MODEL_PREDICTORS,
        imputed_variables=["tip_income"],
    )

    return model


def get_tip_model() -> QRF:
    model_path = STORAGE_FOLDER / "tips_tipped_occ_v2.pkl"

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

ASSET_COLUMNS = [
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "SPANEL",
    "SWAVE",
    "WPFINWGT",
    "TAGE",
    "ESEX",
    "EMS",
    "TPTOTINC",
    # Asset values (person-level sums from SIPP)
    "TVAL_BANK",  # Checking, savings, money market
    "TVAL_STMF",  # Stocks and mutual funds
    "TVAL_BOND",  # Bonds and government securities
    # Income from assets (monthly, person-level)
    "TINC_BANK",  # Interest from bank accounts
    "TINC_STMF",  # Dividends from stocks/mutual funds
    "TINC_BOND",  # Interest from bonds
    "TINC_RENT",  # Rental income
    # SSI receipt (for validation)
    "RSSI_YRYN",  # Received SSI in at least one month
]


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

    # Prepare predictors
    df["age"] = df.TAGE
    df["is_female"] = df.ESEX == 2
    df["is_married"] = df.EMS == 1
    df["employment_income"] = df.TPTOTINC * 12
    df["household_weight"] = df.WPFINWGT
    df["household_id"] = df.SSUID

    # Capital income predictors (annualized from monthly SIPP)
    # Maps to CPS: interest_income, dividend_income, rental_income
    df["interest_income"] = (df["TINC_BANK"].fillna(0) + df["TINC_BOND"].fillna(0)) * 12
    df["dividend_income"] = df["TINC_STMF"].fillna(0) * 12
    df["rental_income"] = df["TINC_RENT"].fillna(0) * 12

    # Calculate household-level counts
    df["is_under_18"] = df.TAGE < 18
    df["count_under_18"] = (
        df.groupby("SSUID")["is_under_18"].sum().loc[df.SSUID.values].values
    )

    sipp = df[
        [
            "household_id",
            "employment_income",
            "interest_income",
            "dividend_income",
            "rental_income",
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
            "age",
            "is_female",
            "is_married",
            "count_under_18",
            "household_weight",
        ]
    ]

    sipp = sipp[~sipp.isna().any(axis=1)]

    # Seed the weighted resample so the asset-model training frame
    # (and the pickled QRF) is deterministic across rebuilds.
    asset_rng = seeded_rng("sipp_asset_model_training_sample")
    sipp = sipp.loc[
        asset_rng.choice(
            sipp.index,
            size=min(20_000, len(sipp)),
            replace=True,
            p=sipp.household_weight / sipp.household_weight.sum(),
        )
    ]

    model = QRF()

    model = model.fit(
        X_train=sipp,
        predictors=[
            "employment_income",
            "interest_income",
            "dividend_income",
            "rental_income",
            "age",
            "is_female",
            "is_married",
            "count_under_18",
        ],
        imputed_variables=[
            "bank_account_assets",
            "stock_assets",
            "bond_assets",
        ],
    )

    return model


def get_asset_model() -> QRF:
    """Get or train the liquid asset imputation model."""
    model_path = STORAGE_FOLDER / "liquid_assets.pkl"

    if not model_path.exists():
        model = train_asset_model()

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model
