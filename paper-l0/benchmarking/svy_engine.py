"""In-process `svy` weight-calibration engines for the benchmark.

Replaces the R subprocess runners (`runners/greg_runner.R`,
`runners/ipf_runner.R`) with calls into the `svy` Python package:

- GREG (classical unbounded linear calibration, ``cal.linear`` /
  Deville-Sarndal 1992) reuses the repo's CI-tested
  ``policyengine_us_data.datasets.cps.long_term.calibration.GregCalibrator``.
- IPF (raking) uses ``svy``'s raking with a household-mean wrapper that
  reproduces ``surveysd::ipf(meanHH = TRUE)`` behaviour.

Both consume the exact same exported bundle files the R runners read
(MatrixMarket design matrix, target-metadata CSV, ``initial_weights.npy``,
and for IPF the unit/target metadata CSVs), so the surrounding CLI, scoring,
and matched-target logic are unchanged.

The engines raise ``RuntimeError`` on failure so the suite records a failed
row (mirroring the R runners' non-zero-exit-as-result behaviour) rather than
crashing the whole run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.io import mmread


def fit_greg_svy(
    *,
    matrix_path: str | Path,
    targets_path: str | Path,
    initial_weights_path: str | Path,
    options: dict | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Fit classical linear GREG weights with ``svy``.

    Reuses :class:`GregCalibrator` (the repo's existing svy GREG adapter,
    ``bounded=False`` == classical unbounded linear calibration). Consumes the
    same inputs ``greg_runner.R`` read: the ``X`` (targets x units)
    MatrixMarket matrix, ``target_metadata.csv`` (``value`` column), and the
    ``initial_weights.npy`` base weights.

    Args:
        matrix_path: MatrixMarket ``X_targets_by_units.mtx`` (targets x units).
        targets_path: Target-metadata CSV with a ``value`` column.
        initial_weights_path: ``.npy`` base/design weights (length n_units).
        options: ``method_options.greg`` dict. ``maxit``/``epsilon`` are inert
            under svy's closed-form linear solve and are reported back in
            ``greg_ignored_options``. ``max_aux_bytes`` (optional) caps the
            densified aux-matrix size, raising before a ``MemoryError``.

    Returns:
        ``(weights, diagnostics)`` where ``weights`` is the length-n_units
        vector of full calibrated weights and ``diagnostics`` records the
        engine and any ignored options.
    """
    options = options or {}

    # GregCalibrator lazily imports svy/polars in __init__, so importing the
    # class itself is cheap and does not require the calibration extra.
    from policyengine_us_data.datasets.cps.long_term.calibration import (
        GregCalibrator,
    )

    X = mmread(str(matrix_path)).tocsr()  # (n_targets, n_units)
    targets_df = pd.read_csv(targets_path)
    values = targets_df["value"].to_numpy(dtype=np.float64)
    initial_weights = np.load(initial_weights_path).astype(np.float64)

    n_targets, n_units = X.shape
    if len(values) != n_targets:
        raise RuntimeError(
            "GREG target count mismatch: "
            f"{len(values)} target rows vs {n_targets} matrix rows"
        )
    if len(initial_weights) != n_units:
        raise RuntimeError(
            "GREG unit count mismatch: "
            f"{len(initial_weights)} weights vs {n_units} matrix columns"
        )

    # svy densifies aux_vars; guard the largest tiers so an oversized problem
    # is a tidy failed row, not an opaque MemoryError (same ceiling R had).
    max_aux_bytes = options.get("max_aux_bytes")
    if max_aux_bytes is not None and n_units * n_targets * 8 > int(max_aux_bytes):
        raise RuntimeError(
            "GREG aux matrix too large for svy: "
            f"{n_units} units x {n_targets} targets exceeds max_aux_bytes="
            f"{int(max_aux_bytes)}"
        )

    # aux_vars is (units x targets); control values are positionally aligned to
    # the matrix rows, so unique stringified indices key both consistently.
    aux_vars = X.T.tocsr()
    control = {str(i): float(values[i]) for i in range(n_targets)}

    try:
        weights = GregCalibrator().calibrate(
            samp_weight=initial_weights,
            aux_vars=aux_vars,
            control=control,
        )
    except Exception as exc:  # noqa: BLE001 - convert to reportable failed row
        raise RuntimeError(f"svy GREG calibration failed: {exc}") from exc

    diagnostics = {
        "greg_engine": "svy",
        "greg_ignored_options": [key for key in ("maxit", "epsilon") if key in options],
        "greg_aux_shape": [int(n_units), int(n_targets)],
    }
    return np.asarray(weights, dtype=np.float64), diagnostics


def _collapse_rows_to_unit_weights(
    unit_index: np.ndarray, row_weights: np.ndarray, n_units: int
) -> np.ndarray:
    """Collapse per-row weights to a length-``n_units`` vector.

    Rows sharing a ``unit_index`` must carry the same weight (guaranteed by the
    household-mean pass); the spread check keeps that assumption honest. Mirrors
    ``benchmark_cli._collapse_ipf_rows_to_unit_weights``.
    """
    df = pd.DataFrame({"unit_index": unit_index.astype(np.int64), "w": row_weights})
    if (df["unit_index"] < 0).any() or (df["unit_index"] >= n_units).any():
        raise RuntimeError("IPF unit_index values fall outside the weight vector")
    spread = df.groupby("unit_index")["w"].agg(lambda s: float(s.max() - s.min()))
    if (spread > 1e-9).any():
        raise RuntimeError(
            "IPF produced inconsistent fitted weights within the same unit_index"
        )
    by_unit = (
        df.groupby("unit_index")["w"]
        .first()
        .reindex(np.arange(n_units, dtype=np.int64))
    )
    if by_unit.isna().any():
        raise RuntimeError("Aggregated IPF weights do not cover the full unit range")
    return by_unit.to_numpy(dtype=np.float64)


def _parse_ipf_controls(targets: pd.DataFrame, frame: pd.DataFrame):
    """Translate the categorical-margin metadata into svy ``rake`` controls.

    Returns ``(controls, scopes)``. Each ``margin_id`` becomes one raking
    column; crossed (multi-variable) margins materialize a composite column on
    ``frame`` whose category label joins the per-variable cell labels with
    ``|`` (matching the ``cell`` string order).
    """
    controls: Dict[str, Dict[str, float]] = {}
    scopes: Dict[str, str] = {}
    for margin_id, group in targets.groupby("margin_id", sort=False):
        variables = str(group["variables"].iloc[0]).split("|")
        scope_values = set(group["scope"].astype(str))
        if len(scope_values) != 1:
            raise RuntimeError(f"Margin {margin_id!r} has inconsistent scope values")
        scopes[str(margin_id)] = scope_values.pop()

        if len(variables) == 1:
            column = variables[0]
            if column not in frame.columns:
                raise RuntimeError(f"Margin variable {column!r} missing from unit data")
            # Categories are compared against string cell labels; normalize the
            # column to str so e.g. an integer state_fips matches "state_fips=6".
            frame[column] = frame[column].astype(str)
        else:
            column = "__".join(variables)
            missing = [v for v in variables if v not in frame.columns]
            if missing:
                raise RuntimeError(f"Margin variables {missing} missing from unit data")
            frame[column] = frame[variables].astype(str).agg("|".join, axis=1)

        cell_targets: Dict[str, float] = {}
        for _, row in group.iterrows():
            parts = str(row["cell"]).split("|")
            labels = [part.split("=", 1)[1] for part in parts]
            cell_targets["|".join(labels)] = float(row["target_value"])
        controls[column] = cell_targets
    return controls, scopes


def fit_ipf_svy(
    *,
    unit_metadata_path: str | Path,
    ipf_target_metadata_path: str | Path,
    initial_weights_path: str | Path,
    options: dict | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Fit IPF (raking) weights with ``svy``, reproducing ``surveysd`` meanHH.

    Consumes the same inputs ``ipf_runner.R`` read: ``unit_metadata.csv`` (one
    row per cloned unit-or-person, with ``unit_index``, a household id, and the
    categorical columns), ``ipf_target_metadata.csv`` (categorical margins),
    and ``initial_weights.npy``. After raking, weights are averaged within each
    household and collapsed to a length-n_units vector — the analogue of
    ``surveysd::ipf(meanHH = TRUE)``.
    """
    options = options or {}
    import polars as pl
    import svy

    unit = pd.read_csv(unit_metadata_path)
    targets = pd.read_csv(ipf_target_metadata_path)
    initial_weights = np.load(initial_weights_path).astype(np.float64)
    n_units = int(len(initial_weights))

    if targets.empty:
        raise RuntimeError("IPF target metadata is empty; nothing to run.")
    if "target_type" in targets.columns:
        unsupported = set(targets["target_type"].astype(str)) - {"categorical_margin"}
        if unsupported:
            raise RuntimeError(
                f"svy IPF supports target_type='categorical_margin' only; got {sorted(unsupported)}"
            )
    if "scope" in targets.columns:
        scopes = set(targets["scope"].astype(str))
        unsupported_scopes = scopes - {"person", "household"}
        if unsupported_scopes:
            raise RuntimeError(
                f"svy IPF supports person/household scope only; got {sorted(unsupported_scopes)}"
            )
        if len(scopes) > 1:
            # Mixed person+household in one run needs surveysd's two-level
            # conP/conH/meanHH loop, which svy raking on a single flat frame
            # cannot reproduce. Surface as a reportable failure (the suite
            # records a failed row) rather than silently mis-counting.
            raise RuntimeError(
                "svy IPF engine supports single-scope runs (all-person or "
                "all-household) only; this bundle mixes person and household "
                "margins, which requires surveysd's two-level conP/conH/meanHH "
                "loop. Run this IPF bundle with the R engine or split by scope."
            )
    if "unit_index" not in unit.columns:
        raise RuntimeError("Unit metadata must include a unit_index column for IPF")

    weight_col = str(options.get("weight_col", "base_weight"))
    hid_col = str(options.get("household_id_col", "household_id"))

    unit_index = unit["unit_index"].astype(np.int64).to_numpy()
    if unit_index.min() < 0 or unit_index.max() >= n_units:
        raise RuntimeError("Unit metadata unit_index values fall outside the weights")

    frame = unit.copy()
    frame["__w0"] = (
        unit[weight_col].astype(np.float64).to_numpy()
        if weight_col in unit.columns
        else initial_weights[unit_index]
    )

    controls, _ = _parse_ipf_controls(targets, frame)

    # surveysd::ipf leaves units outside a margin's targeted cells at their
    # current weight (e.g. units in untargeted geographies). svy raking instead
    # requires every observed category to carry a control total. Reproduce the
    # surveysd "leave untouched" semantics by padding each margin with its
    # uncovered observed categories at their current base-weight totals, which
    # raking then holds fixed.
    base = frame["__w0"].to_numpy(dtype=np.float64)
    n_padded = 0
    for column, cell_targets in controls.items():
        observed = frame[column].astype(str)
        for category in observed.unique():
            if category not in cell_targets:
                cell_targets[category] = float(
                    base[(observed == category).to_numpy()].sum()
                )
                n_padded += 1

    bound = float(options.get("bound", 10.0))
    tol = min(float(options.get("epsP", 1e-6)), float(options.get("epsH", 1e-2)))
    max_iter = int(options.get("max_iter", 200))

    sample = svy.Sample(pl.from_pandas(frame), design=svy.Design(wgt="__w0"))
    try:
        raked = sample.weighting.rake(
            controls=controls,
            wgt_name="__rk",
            up_bound=bound,
            ll_bound=1.0 / bound,
            tol=tol,
            max_iter=max_iter,
        )
    except Exception as exc:  # noqa: BLE001 - convert to reportable failed row
        raise RuntimeError(f"svy IPF (rake) failed: {exc}") from exc

    raked_w = raked.data.get_column("__rk").to_numpy().astype(np.float64)

    # meanHH: average raked weight within each household, then collapse to units.
    group_col = hid_col if hid_col in frame.columns else "unit_index"
    mean_df = pd.DataFrame({"_g": frame[group_col].to_numpy(), "_w": raked_w})
    mean_df["_mean"] = mean_df.groupby("_g")["_w"].transform("mean")
    weights = _collapse_rows_to_unit_weights(
        unit_index, mean_df["_mean"].to_numpy(), n_units
    )

    return weights, {
        "ipf_engine": "svy",
        "ipf_meanhh_group": group_col,
        "ipf_padded_uncovered_cells": int(n_padded),
    }
