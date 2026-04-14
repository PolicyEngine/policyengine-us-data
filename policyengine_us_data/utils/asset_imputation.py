import numpy as np
import pandas as pd


def build_household_asset_receiver(
    person_df: pd.DataFrame,
    tenure_type: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build household-level predictors for SIPP household asset imputation."""
    if (
        "household_id" not in person_df.columns
        and "person_household_id" in person_df.columns
    ):
        person_df = person_df.rename(columns={"person_household_id": "household_id"})

    work = person_df.copy()

    for col in [
        "employment_income",
        "interest_income",
        "dividend_income",
        "rental_income",
        "age",
        "is_female",
        "is_married",
    ]:
        if col not in work.columns:
            work[col] = 0.0

    work["is_under_18"] = work["age"] < 18

    household_agg = (
        work.groupby("household_id")
        .agg(
            household_employment_income=("employment_income", "sum"),
            household_interest_income=("interest_income", "sum"),
            household_dividend_income=("dividend_income", "sum"),
            household_rental_income=("rental_income", "sum"),
            count_under_18=("is_under_18", "sum"),
            household_size=("household_id", "size"),
        )
        .reset_index()
    )

    if "is_household_head" in work.columns:
        heads = work[work["is_household_head"].astype(bool)].copy()
    else:
        heads = work.groupby("household_id", as_index=False).first()

    heads = (
        heads.sort_values("household_id")
        .drop_duplicates("household_id")
        .loc[:, ["household_id", "age", "is_female", "is_married"]]
        .rename(
            columns={
                "age": "reference_age",
                "is_female": "reference_is_female",
                "is_married": "reference_is_married",
            }
        )
    )

    receiver = household_agg.merge(heads, on="household_id", how="left")

    if tenure_type is not None:
        tenure = pd.Series(tenure_type)
        receiver["is_homeowner"] = (
            tenure.astype(str)
            .isin(
                [
                    "OWNED_OUTRIGHT",
                    "OWNED_WITH_MORTGAGE",
                    "b'OWNED_OUTRIGHT'",
                    "b'OWNED_WITH_MORTGAGE'",
                ]
            )
            .astype(np.float32)
        )
    else:
        receiver["is_homeowner"] = 0.0

    for col in [
        "reference_age",
        "reference_is_female",
        "reference_is_married",
        "count_under_18",
        "household_size",
    ]:
        receiver[col] = receiver[col].fillna(0).astype(np.float32)

    for col in [
        "household_employment_income",
        "household_interest_income",
        "household_dividend_income",
        "household_rental_income",
        "is_homeowner",
    ]:
        receiver[col] = receiver[col].fillna(0).astype(np.float32)

    return receiver


def build_household_vehicle_receiver(
    person_df: pd.DataFrame,
    tenure_type: np.ndarray | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for the household asset receiver."""
    return build_household_asset_receiver(
        person_df=person_df,
        tenure_type=tenure_type,
    )
