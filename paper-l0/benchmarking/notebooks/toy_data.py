"""Synthetic toy household survey used by the GREG and IPF walkthrough notebooks.

The dataset is ten households distributed across two districts. Each household
has a design weight of 100, a household size, an adult/child breakdown, and a
household income. The targets defined below are internally consistent so any
reasonable calibration run should hit them to within a small tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


HOUSEHOLDS = pd.DataFrame(
    {
        "hh_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "district": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "n_adults": [2, 2, 1, 2, 2, 2, 1, 2, 3, 1],
        "n_children": [0, 2, 0, 1, 0, 3, 1, 1, 1, 0],
        "income": [
            30_000,
            60_000,
            25_000,
            45_000,
            70_000,
            40_000,
            20_000,
            80_000,
            90_000,
            15_000,
        ],
        "design_weight": [100.0] * 10,
    }
)
HOUSEHOLDS["hh_size"] = HOUSEHOLDS["n_adults"] + HOUSEHOLDS["n_children"]


# Targets, defined as (target_name, target_value, coefficient_column).
# Each coefficient column already exists on the household table.
TARGETS = pd.DataFrame(
    [
        ("district_A_households", 520.0, "is_A"),
        ("district_A_adults", 940.0, "adults_in_A"),
        ("district_A_children", 310.0, "children_in_A"),
        ("district_A_income", 23_500_000.0, "income_in_A"),
        ("district_B_households", 480.0, "is_B"),
        ("district_B_adults", 1_000.0, "adults_in_B"),
        ("district_B_children", 600.0, "children_in_B"),
        ("district_B_income", 25_000_000.0, "income_in_B"),
    ],
    columns=["target_name", "value", "coef_col"],
)


# IPF-eligible subset: count-style targets only.
IPF_TARGETS = TARGETS[~TARGETS["target_name"].str.endswith("_income")].reset_index(
    drop=True
)


def household_table_with_coefficients() -> pd.DataFrame:
    """Return HOUSEHOLDS augmented with the per-target coefficient columns."""
    hh = HOUSEHOLDS.copy()
    hh["is_A"] = (hh["district"] == "A").astype(float)
    hh["is_B"] = (hh["district"] == "B").astype(float)
    hh["adults_in_A"] = hh["n_adults"] * hh["is_A"]
    hh["children_in_A"] = hh["n_children"] * hh["is_A"]
    hh["income_in_A"] = hh["income"] * hh["is_A"]
    hh["adults_in_B"] = hh["n_adults"] * hh["is_B"]
    hh["children_in_B"] = hh["n_children"] * hh["is_B"]
    hh["income_in_B"] = hh["income"] * hh["is_B"]
    return hh


def build_target_matrix(hh: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    """Return a (n_targets, n_units) dense matrix of per-unit coefficients."""
    return np.stack([hh[col].to_numpy(dtype=float) for col in targets["coef_col"]])


def baseline_totals(hh: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Realized weighted totals using the design weights."""
    X = build_target_matrix(hh, targets)
    w = hh["design_weight"].to_numpy(dtype=float)
    realized = X @ w
    return pd.DataFrame(
        {
            "target_name": targets["target_name"].to_numpy(),
            "target_value": targets["value"].to_numpy(dtype=float),
            "baseline_weighted_total": realized,
        }
    )


def diagnostic_table(
    hh: pd.DataFrame,
    targets: pd.DataFrame,
    fitted_weights: np.ndarray,
) -> pd.DataFrame:
    """Compare fitted weighted totals to target values."""
    X = build_target_matrix(hh, targets)
    base_w = hh["design_weight"].to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "target_name": targets["target_name"].to_numpy(),
            "target_value": targets["value"].to_numpy(dtype=float),
            "baseline_weighted_total": X @ base_w,
            "fitted_weighted_total": X @ fitted_weights,
        }
    )
    out["abs_rel_error"] = (
        out["fitted_weighted_total"] - out["target_value"]
    ).abs() / out["target_value"].abs()
    return out
