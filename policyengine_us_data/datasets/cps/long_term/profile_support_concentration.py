from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SS_COMPONENTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
)
PAYROLL_COMPONENTS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
)


def _read_year_array(store: h5py.File, name: str, year: int) -> np.ndarray:
    return store[name][str(year)][()]


def _build_household_lookup(household_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(household_ids)
    sorted_ids = household_ids[order]
    return sorted_ids, order


def _household_index(sorted_household_ids: np.ndarray, order: np.ndarray, person_household_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_household_ids, person_household_ids)
    if np.any(positions >= len(sorted_household_ids)):
        raise ValueError("Person household ids exceed household id support")
    matched = sorted_household_ids[positions]
    if not np.array_equal(matched, person_household_ids):
        raise ValueError("Person household ids do not match household-level ids")
    return order[positions]


def _load_year(path: Path, year: int) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as store:
        household_ids = _read_year_array(store, "household_id", year).astype(np.int64)
        household_weights = _read_year_array(store, "household_weight", year).astype(float)
        person_household_ids = _read_year_array(store, "person_household_id", year).astype(np.int64)
        ages = _read_year_array(store, "age", year).astype(float)
        payroll = np.zeros_like(ages, dtype=float)
        for component in PAYROLL_COMPONENTS:
            payroll += _read_year_array(store, component, year).astype(float)
        social_security = np.zeros_like(ages, dtype=float)
        for component in SS_COMPONENTS:
            social_security += _read_year_array(store, component, year).astype(float)

    sorted_ids, order = _build_household_lookup(household_ids)
    person_household_index = _household_index(sorted_ids, order, person_household_ids)

    return {
        "household_ids": household_ids,
        "household_weights": household_weights,
        "person_household_ids": person_household_ids,
        "person_household_index": person_household_index,
        "ages": ages,
        "payroll": payroll,
        "social_security": social_security,
    }


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum())
    denom = float(np.dot(weights, weights))
    if total <= 0 or denom <= 0:
        return 0.0
    return total**2 / denom


def _top_households(data: dict[str, np.ndarray], top_n: int) -> list[dict[str, object]]:
    weights = data["household_weights"]
    top_idx = np.argsort(weights)[-top_n:][::-1]
    records: list[dict[str, object]] = []
    for idx in top_idx:
        if weights[idx] <= 0:
            continue
        mask = data["person_household_index"] == idx
        ages = np.sort(data["ages"][mask]).astype(int).tolist()
        payroll_total = float(data["payroll"][mask].sum())
        social_security_total = float(data["social_security"][mask].sum())
        records.append(
            {
                "household_id": int(data["household_ids"][idx]),
                "weight": float(weights[idx]),
                "weight_share_pct": float(weights[idx] / weights.sum() * 100),
                "ages": ages,
                "payroll_proxy": payroll_total,
                "social_security_total": social_security_total,
            }
        )
    return records


def profile_support(path: Path, year: int, *, top_n: int) -> dict[str, object]:
    data = _load_year(path, year)
    household_weights = data["household_weights"]
    positive_mask = household_weights > 0
    sorted_weights = np.sort(household_weights)
    person_weights = household_weights[data["person_household_index"]]
    ages = data["ages"]
    payroll = data["payroll"]

    overall_nonworking = payroll <= 0
    age_85_plus = ages >= 85

    return {
        "path": str(path),
        "year": year,
        "positive_household_count": int(positive_mask.sum()),
        "positive_household_pct": float(positive_mask.mean() * 100),
        "effective_sample_size": _effective_sample_size(household_weights),
        "top_10_weight_share_pct": float(sorted_weights[-10:].sum() / household_weights.sum() * 100),
        "top_100_weight_share_pct": float(sorted_weights[-100:].sum() / household_weights.sum() * 100),
        "weighted_nonworking_share_pct": float(
            person_weights[overall_nonworking].sum() / person_weights.sum() * 100
        ),
        "weighted_nonworking_share_85_plus_pct": float(
            person_weights[age_85_plus & overall_nonworking].sum()
            / person_weights[age_85_plus].sum()
            * 100
        )
        if person_weights[age_85_plus].sum() > 0
        else 0.0,
        "top_households": _top_households(data, top_n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile late-year support concentration in projected household datasets."
    )
    parser.add_argument("dataset", type=Path, help="Projected year-specific H5 dataset.")
    parser.add_argument("year", type=int, help="Projection year stored in the dataset.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top households to emit.")
    args = parser.parse_args()

    report = profile_support(args.dataset, args.year, top_n=args.top_n)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
