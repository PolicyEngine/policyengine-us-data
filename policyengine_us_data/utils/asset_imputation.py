from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SIPP_LIQUID_ASSET_VARIABLES = (
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
)
SIPP_VEHICLE_ASSET_VARIABLES = ("household_vehicles_value",)
SCF_NET_WORTH_VARIABLE = "net_worth"
SCF_BALANCE_SHEET_DEBT_VARIABLES = ("auto_loan_balance",)
SCF_FINANCIAL_ASSET_TARGETS = {
    "scf_bank_account_assets": ("liq",),
    "scf_stock_assets": ("stocks", "nmmf"),
    "scf_bond_assets": ("bond",),
}
SCF_FINANCIAL_ASSET_POLICY_VARIABLES = {
    "scf_bank_account_assets": "bank_account_assets",
    "scf_stock_assets": "stock_assets",
    "scf_bond_assets": "bond_assets",
}

EXPOSED_NET_WORTH_COMPONENT_VARIABLES = (
    SIPP_LIQUID_ASSET_VARIABLES
    + SIPP_VEHICLE_ASSET_VARIABLES
    + SCF_BALANCE_SHEET_DEBT_VARIABLES
)
NET_WORTH_COMPONENT_SIGNS = {
    "auto_loan_balance": -1.0,
}
UNOBSERVED_NET_WORTH_COMPONENT_GROUPS = (
    "primary_residence_value",
    "mortgage_debt",
    "retirement_assets",
    "business_equity",
    "other_real_estate",
    "other_financial_assets",
    "other_debts",
)
NET_WORTH_COMPONENTS_ARE_COMPLETE = False
FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY = 0.5


@dataclass(frozen=True)
class NetWorthReconciliationReport:
    """Summary of a household-level net worth reconciliation check."""

    components_are_complete: bool
    available_component_variables: tuple[str, ...]
    missing_component_variables: tuple[str, ...]
    unobserved_component_groups: tuple[str, ...]
    max_abs_difference: float | None
    is_reconciled: bool | None
    message: str


def check_household_net_worth_reconciliation(
    data: Mapping[str, Sequence[float]],
    *,
    component_variables: Sequence[str] = EXPOSED_NET_WORTH_COMPONENT_VARIABLES,
    net_worth_variable: str = SCF_NET_WORTH_VARIABLE,
    component_signs: Mapping[str, float] = NET_WORTH_COMPONENT_SIGNS,
    components_are_complete: bool = NET_WORTH_COMPONENTS_ARE_COMPLETE,
    rtol: float = 1e-6,
    atol: float = 1.0,
) -> NetWorthReconciliationReport:
    """Check whether household net worth equals signed balance-sheet components.

    The current CPS asset fields are intentionally not a complete balance sheet:
    liquid assets and vehicles are imputed from SIPP, while net worth and auto
    loan balances are imputed from SCF. Leave ``components_are_complete`` false
    for current public datasets. Set it to true only for a household-aligned data
    frame whose component variables are intended to exhaust net worth.
    """
    component_variables = tuple(component_variables)
    available_components = tuple(
        variable for variable in component_variables if variable in data
    )
    missing_components = tuple(
        variable for variable in component_variables if variable not in data
    )

    if not components_are_complete:
        return NetWorthReconciliationReport(
            components_are_complete=False,
            available_component_variables=available_components,
            missing_component_variables=missing_components,
            unobserved_component_groups=UNOBSERVED_NET_WORTH_COMPONENT_GROUPS,
            max_abs_difference=None,
            is_reconciled=None,
            message=(
                "Net worth is an independently imputed SCF aggregate. The "
                "available SIPP/SCF asset fields are partial and should not be "
                "expected to reconstruct it."
            ),
        )

    if net_worth_variable not in data:
        raise KeyError(f"Missing net worth variable: {net_worth_variable}")
    if missing_components:
        raise KeyError(
            "Cannot reconcile net worth with a complete component set because "
            f"these component variables are missing: {', '.join(missing_components)}"
        )

    net_worth = np.asarray(data[net_worth_variable], dtype=float)
    component_total = np.zeros_like(net_worth, dtype=float)

    for variable in component_variables:
        values = np.asarray(data[variable], dtype=float)
        if values.shape != net_worth.shape:
            raise ValueError(
                f"{variable} has shape {values.shape}, but {net_worth_variable} "
                f"has shape {net_worth.shape}. Reconciliation data must already "
                "be aligned to household rows."
            )
        component_total += component_signs.get(variable, 1.0) * values

    difference = net_worth - component_total
    max_abs_difference = (
        float(np.nanmax(np.abs(difference))) if difference.size else 0.0
    )
    is_reconciled = bool(
        np.allclose(net_worth, component_total, rtol=rtol, atol=atol, equal_nan=True)
    )

    return NetWorthReconciliationReport(
        components_are_complete=True,
        available_component_variables=available_components,
        missing_component_variables=(),
        unobserved_component_groups=(),
        max_abs_difference=max_abs_difference,
        is_reconciled=is_reconciled,
        message=(
            "Net worth reconciles to the signed component variables."
            if is_reconciled
            else "Net worth does not reconcile to the signed component variables."
        ),
    )


def add_scf_financial_asset_targets(scf: pd.DataFrame) -> tuple[str, ...]:
    """Add SCF financial asset targets comparable to SIPP policy leaves."""
    added_targets = []
    for target, source_columns in SCF_FINANCIAL_ASSET_TARGETS.items():
        if all(column in scf.columns for column in source_columns):
            scf[target] = sum(scf[column].fillna(0) for column in source_columns)
            added_targets.append(target)
    return tuple(added_targets)


def _stable_unit_interval(key: str) -> float:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def financial_asset_source_is_scf(
    household_ids: Sequence,
    *,
    time_period: int,
    probability: float = FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY,
) -> np.ndarray:
    """Return a stable 50/50 source-model draw for financial assets.

    The draw is at the household asset-block level, so bank accounts, stocks,
    and bonds all come from the same source for a household.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")

    household_ids = np.asarray(household_ids)
    draws_by_household = {
        household_id: (
            _stable_unit_interval(
                f"financial_asset_source:{time_period}:{household_id}"
            )
            < probability
        )
        for household_id in pd.unique(household_ids)
    }
    return np.array(
        [draws_by_household[household_id] for household_id in household_ids],
        dtype=bool,
    )


def combine_sipp_and_scf_financial_assets(
    *,
    sipp_values: Sequence[float],
    scf_household_values: Sequence[float],
    person_household_ids: Sequence,
    reference_person_mask: Sequence[bool],
    time_period: int,
) -> np.ndarray:
    """Apply a stable 50/50 SIPP/SCF source draw to a person-level asset leaf."""
    sipp_values = np.asarray(sipp_values, dtype=np.float32)
    scf_household_values = np.asarray(scf_household_values, dtype=np.float32)
    person_household_ids = np.asarray(person_household_ids)
    reference_person_mask = np.asarray(reference_person_mask, dtype=bool)

    if sipp_values.shape != person_household_ids.shape:
        raise ValueError(
            "sipp_values and person_household_ids must have the same shape"
        )
    if reference_person_mask.shape != person_household_ids.shape:
        raise ValueError(
            "reference_person_mask and person_household_ids must have the same shape"
        )
    if scf_household_values.shape[0] != reference_person_mask.sum():
        raise ValueError(
            "scf_household_values must contain one value per reference person"
        )

    scf_person_values = np.zeros_like(sipp_values, dtype=np.float32)
    scf_person_values[reference_person_mask] = scf_household_values
    use_scf = financial_asset_source_is_scf(
        person_household_ids,
        time_period=time_period,
    )
    return np.where(use_scf, scf_person_values, sipp_values).astype(np.float32)


def build_household_vehicle_receiver(
    person_df: pd.DataFrame,
    tenure_type: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build household-level predictors for vehicle asset imputation.

    The donor model is household-level, so we aggregate CPS person-level
    predictors into one row per household and anchor demographic predictors
    on the household head when available.
    """
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
