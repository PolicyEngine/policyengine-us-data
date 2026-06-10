"""Smoke test the PUF QRF real-half imputation change.

This deliberately avoids `Microsimulation(dataset=PUF_2015)`, because local CPS
artifacts can be stale relative to the installed PE-US schema. It uses the raw
2015 PUF CSV plus the 2024 CPS H5 to test the core failure mode:

* old path: unweighted, demographic-only QRF draws
* fixed path: PUF-weighted, income-conditioned QRF draws

The numbers are diagnostic, not release validation totals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from microimpute.models.qrf import QRF

from policyengine_us_data.calibration.puf_impute import (
    DEMOGRAPHIC_PREDICTORS,
    PUF_SUBSAMPLE_TARGET,
    PUF_WEIGHT_COLUMN,
    REAL_HALF_INCOME_PREDICTORS,
    _stratified_subsample_index,
)
from policyengine_us_data.storage import STORAGE_FOLDER


TARGETS = [
    "charitable_cash_donations",
    "charitable_non_cash_donations",
]

PUF_RENAMES = {
    "adjusted_gross_income": "E00100",
    "charitable_cash_donations": "E19800",
    "charitable_non_cash_donations": "E20100",
    "employment_income": "E00200",
    "self_employment_income": "E00900",
    "taxable_interest_income": "E00300",
    "tax_exempt_interest_income": "E00400",
    "qualified_dividend_income": "E00650",
    "short_term_capital_gains": "P22250",
    "long_term_capital_gains": "P23250",
    "farm_income": "T27800",
    "taxable_pension_income": "E01700",
    "taxable_private_pension_income": "E01700",
    "taxable_ira_distributions": "E01400",
    "taxable_unemployment_compensation": "E02300",
    "social_security": "E02400",
    "social_security_retirement": "E02400",
}


def _decode_age(age_range: pd.Series) -> np.ndarray:
    midpoints = {
        0: 40,
        1: 22,
        2: 30,
        3: 40,
        4: 50,
        5: 60,
        6: 72,
        7: 82,
    }
    return age_range.fillna(0).round().astype(int).map(midpoints).fillna(40).values


def _load_puf_tax_units(storage_folder: Path) -> pd.DataFrame:
    puf_cols = [
        "RECID",
        "S006",
        "E00100",
        "E00200",
        "E00300",
        "E00400",
        "E00600",
        "E00650",
        "E00900",
        "E01400",
        "E01500",
        "E01700",
        "E02300",
        "E02400",
        "E19800",
        "E20100",
        "E25850",
        "E25860",
        "P22250",
        "P23250",
        "T27800",
        "MARS",
        "XTOT",
    ]
    demo_cols = ["RECID", "AGERANGE", "GENDER"]
    puf = pd.read_csv(storage_folder / "puf_2015.csv", usecols=puf_cols)
    demographics = pd.read_csv(
        storage_folder / "demographics_2015.csv", usecols=demo_cols
    )
    puf = puf.merge(demographics, on="RECID", how="left")
    frame = pd.DataFrame(index=puf.index)
    frame[PUF_WEIGHT_COLUMN] = puf["S006"].fillna(0).astype(float) / 100
    frame["age"] = _decode_age(puf["AGERANGE"])
    frame["is_male"] = (puf["GENDER"].fillna(0) == 1).astype(float)
    frame["tax_unit_is_joint"] = (puf["MARS"].fillna(0) == 2).astype(float)
    frame["tax_unit_count_dependents"] = np.maximum(
        puf["XTOT"].fillna(1).astype(float) - 1 - frame["tax_unit_is_joint"],
        0,
    )
    frame["is_tax_unit_head"] = 1.0
    frame["is_tax_unit_spouse"] = 0.0
    frame["is_tax_unit_dependent"] = 0.0
    for target, source in PUF_RENAMES.items():
        frame[target] = puf[source].fillna(0).astype(float)
    frame["non_qualified_dividend_income"] = puf["E00600"].fillna(0).astype(
        float
    ) - puf["E00650"].fillna(0).astype(float)
    frame["rental_income"] = puf["E25850"].fillna(0).astype(float) - puf[
        "E25860"
    ].fillna(0).astype(float)
    frame["tax_exempt_pension_income"] = puf["E01500"].fillna(0).astype(float) - puf[
        "E01700"
    ].fillna(0).astype(float)
    frame["tax_exempt_private_pension_income"] = frame["tax_exempt_pension_income"]
    frame["tax_exempt_ira_distributions"] = 0.0
    frame["social_security_disability"] = 0.0
    frame["social_security_survivors"] = 0.0
    frame["social_security_dependents"] = 0.0
    return frame


def _series_by_id(values: np.ndarray, ids: np.ndarray) -> pd.Series:
    return pd.Series(values).groupby(ids).sum()


def _load_cps_tax_units(cps_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    with h5py.File(cps_path, "r") as f:
        tax_unit_ids = f["tax_unit_id"][:]
        person_tax_unit_ids = f["person_tax_unit_id"][:]
        person_household_ids = f["person_household_id"][:]
        household_weights = pd.Series(
            f["household_weight"][:],
            index=f["household_id"][:],
        )

        first_person_by_tax_unit = (
            pd.Series(np.arange(len(person_tax_unit_ids)))
            .groupby(person_tax_unit_ids)
            .first()
            .reindex(tax_unit_ids)
        )
        first_person_idx = first_person_by_tax_unit.values.astype(int)
        tax_unit_household_ids = person_household_ids[first_person_idx]
        weights = household_weights.reindex(tax_unit_household_ids).fillna(0).values

        frame = pd.DataFrame(index=np.arange(len(tax_unit_ids)))
        person_age = f["age"][:].astype(float)
        adult_count = _series_by_id(
            (person_age >= 18).astype(float), person_tax_unit_ids
        )
        child_count = _series_by_id(
            (person_age < 19).astype(float), person_tax_unit_ids
        )
        frame["age"] = person_age[first_person_idx]
        frame["is_male"] = 1.0 - f["is_female"][:][first_person_idx].astype(float)
        frame["tax_unit_is_joint"] = (
            adult_count.reindex(tax_unit_ids).fillna(0).values >= 2
        ).astype(float)
        frame["tax_unit_count_dependents"] = (
            child_count.reindex(tax_unit_ids).fillna(0).values
        )
        frame["is_tax_unit_head"] = 1.0
        frame["is_tax_unit_spouse"] = 0.0
        frame["is_tax_unit_dependent"] = 0.0

        for variable in REAL_HALF_INCOME_PREDICTORS:
            if variable not in f:
                continue
            by_tax_unit = _series_by_id(f[variable][:], person_tax_unit_ids)
            frame[variable] = by_tax_unit.reindex(tax_unit_ids).fillna(0).values

    return frame, weights


def _weighted_total(values: pd.Series | np.ndarray, weights: np.ndarray) -> float:
    return float(np.asarray(values, dtype=float) @ np.asarray(weights, dtype=float))


def _run_qrf(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: list[str],
    weight_col: str | None,
    max_train_samples: int | None,
) -> pd.DataFrame:
    qrf = QRF(
        log_level="WARNING",
        memory_efficient=True,
        max_train_samples=max_train_samples,
    )
    return qrf.fit_predict(
        X_train=train,
        X_test=test,
        predictors=predictors,
        imputed_variables=TARGETS,
        weight_col=weight_col,
        n_jobs=1,
    )


def _format_billions(value: float) -> str:
    return f"${value / 1e9:,.1f}B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cps-path",
        type=Path,
        default=STORAGE_FOLDER / "cps_2024.h5",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=PUF_SUBSAMPLE_TARGET,
    )
    args = parser.parse_args()

    puf = _load_puf_tax_units(STORAGE_FOLDER)
    cps, cps_weights = _load_cps_tax_units(args.cps_path)
    if "adjusted_gross_income" not in cps:
        cps["adjusted_gross_income"] = cps[
            [column for column in REAL_HALF_INCOME_PREDICTORS if column in cps]
        ].sum(axis=1)
    income_predictors = [
        predictor
        for predictor in REAL_HALF_INCOME_PREDICTORS
        if predictor in cps and predictor in puf
    ]
    weighted_predictors = DEMOGRAPHIC_PREDICTORS + income_predictors

    old_idx = _stratified_subsample_index(
        puf["adjusted_gross_income"].values,
        target_n=args.max_train_samples,
    )
    old_train = puf.iloc[old_idx].reset_index(drop=True)
    weighted_train = puf.loc[puf[PUF_WEIGHT_COLUMN] > 0].reset_index(drop=True)

    print(f"PUF rows: {len(puf):,}; CPS tax units: {len(cps):,}")
    print(f"Old unweighted train rows: {len(old_train):,}")
    print(f"Weighted train rows before QRF sampling: {len(weighted_train):,}")
    print(f"Income predictors used: {', '.join(income_predictors)}")

    old_pred = _run_qrf(
        train=old_train,
        test=cps,
        predictors=DEMOGRAPHIC_PREDICTORS,
        weight_col=None,
        max_train_samples=None,
    )
    fixed_pred = _run_qrf(
        train=weighted_train,
        test=cps,
        predictors=weighted_predictors,
        weight_col=PUF_WEIGHT_COLUMN,
        max_train_samples=args.max_train_samples,
    )

    rows = []
    for target in TARGETS:
        puf_total = _weighted_total(puf[target], puf[PUF_WEIGHT_COLUMN].values)
        old_total = _weighted_total(old_pred[target], cps_weights)
        fixed_total = _weighted_total(fixed_pred[target], cps_weights)
        rows.append(
            {
                "target": target,
                "puf_weighted": _format_billions(puf_total),
                "old_qrf": _format_billions(old_total),
                "old_vs_puf": f"{old_total / puf_total:,.1f}x",
                "fixed_qrf": _format_billions(fixed_total),
                "fixed_vs_puf": f"{fixed_total / puf_total:,.1f}x",
            }
        )

    puf_total = sum(
        _weighted_total(puf[target], puf[PUF_WEIGHT_COLUMN].values)
        for target in TARGETS
    )
    old_total = sum(
        _weighted_total(old_pred[target], cps_weights) for target in TARGETS
    )
    fixed_total = sum(
        _weighted_total(fixed_pred[target], cps_weights) for target in TARGETS
    )
    rows.append(
        {
            "target": "combined_charitable",
            "puf_weighted": _format_billions(puf_total),
            "old_qrf": _format_billions(old_total),
            "old_vs_puf": f"{old_total / puf_total:,.1f}x",
            "fixed_qrf": _format_billions(fixed_total),
            "fixed_vs_puf": f"{fixed_total / puf_total:,.1f}x",
        }
    )

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
