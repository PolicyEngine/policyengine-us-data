import pandas as pd
from microdf import MicroDataFrame
import numpy as np
from policyengine_us import Microsimulation
from microimpute.models import QRF
from policyengine_us_data.storage import STORAGE_FOLDER
import pickle
from huggingface_hub import hf_hub_download
import os

test_lite = os.environ.get("TEST_LITE")


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
    # Sum tip columns (AJB*_TXAMT + TJB*_TXAMT) across all jobs.
    df["tip_income"] = (
        df[df.columns[df.columns.str.contains("TXAMT")]].fillna(0).sum(axis=1)
        * 12
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

    sipp = df[
        [
            "household_id",
            "employment_income",
            "tip_income",
            "count_under_18",
            "count_under_6",
            "age",
            "household_weight",
        ]
    ]

    sipp = sipp[~sipp.isna().any(axis=1)]

    sipp = sipp.loc[
        np.random.choice(
            sipp.index,
            size=10_000,
            replace=True,
            p=sipp.household_weight / sipp.household_weight.sum(),
        )
    ]

    model = QRF()

    model = model.fit(
        X_train=sipp,
        predictors=[
            "employment_income",
            "age",
            "count_under_18",
            "count_under_6",
        ],
        imputed_variables=["tip_income"],
    )

    return model


def get_tip_model() -> QRF:
    model_path = STORAGE_FOLDER / "tips.pkl"

    if not model_path.exists():
        model = train_tip_model()

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model
