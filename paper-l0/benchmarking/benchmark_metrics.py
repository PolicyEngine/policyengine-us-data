from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy.io import mmread


def load_targets_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_common_metrics(
    weights: np.ndarray,
    targets_df: pd.DataFrame,
    matrix_path: str | Path,
) -> Dict:
    X_sparse = mmread(str(matrix_path)).tocsr()
    estimates = X_sparse.dot(weights)
    true_values = targets_df["value"].to_numpy(dtype=np.float64)
    rel_errors = np.where(
        np.abs(true_values) > 0,
        (estimates - true_values) / np.abs(true_values),
        0.0,
    )
    abs_rel = np.abs(rel_errors)
    achievable = np.asarray(X_sparse.sum(axis=1)).ravel() > 0

    active_weights = weights[weights > 0]
    weight_sum = float(weights.sum())
    ess = (
        float(weight_sum**2 / np.square(weights).sum())
        if np.square(weights).sum() > 0
        else 0.0
    )

    metrics = {
        "n_targets": int(len(true_values)),
        "n_units": int(len(weights)),
        "n_achievable_targets": int(achievable.sum()),
        "mean_abs_rel_error": float(abs_rel.mean()),
        "median_abs_rel_error": float(np.median(abs_rel)),
        "p95_abs_rel_error": float(np.quantile(abs_rel, 0.95)),
        "max_abs_rel_error": float(abs_rel.max()),
        "ess": ess,
        "active_record_count": int((weights > 0).sum()),
        "negative_weight_share": float((weights < 0).mean()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "weight_median": float(np.median(weights)),
        "nonzero_weight_min": float(active_weights.min())
        if len(active_weights)
        else 0.0,
        "nonzero_weight_max": float(active_weights.max())
        if len(active_weights)
        else 0.0,
    }

    if "geo_level" in targets_df.columns:
        by_geo = {}
        for geo_level, group in targets_df.assign(abs_rel_error=abs_rel).groupby(
            "geo_level"
        ):
            vals = group["abs_rel_error"].to_numpy(dtype=np.float64)
            by_geo[str(geo_level)] = {
                "n_targets": int(len(vals)),
                "mean_abs_rel_error": float(vals.mean()),
                "median_abs_rel_error": float(np.median(vals)),
                "p95_abs_rel_error": float(np.quantile(vals, 0.95)),
                "max_abs_rel_error": float(vals.max()),
            }
        metrics["by_geo_level"] = by_geo

    if "variable" in targets_df.columns:
        by_variable = {}
        enriched = targets_df.assign(abs_rel_error=abs_rel)
        for variable, group in enriched.groupby("variable"):
            vals = group["abs_rel_error"].to_numpy(dtype=np.float64)
            by_variable[str(variable)] = {
                "n_targets": int(len(vals)),
                "mean_abs_rel_error": float(vals.mean()),
                "median_abs_rel_error": float(np.median(vals)),
            }
        metrics["by_variable"] = by_variable

    return metrics


def write_method_summary(summary: Dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
