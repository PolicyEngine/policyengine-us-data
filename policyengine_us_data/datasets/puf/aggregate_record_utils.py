"""Helpers for disaggregating IRS aggregate PUF rows."""

from __future__ import annotations

from importlib.resources import files
import re

import numpy as np
import pandas as pd
import yaml

AGGREGATE_RECIDS = [999996, 999997, 999998, 999999]
SYNTHETIC_RECID_START = 1_000_000

SCREENED_FIELDS = [
    "E00200",  # Wages
    "P23250",  # Long-term capital gains
    "P22250",  # Short-term capital gains
    "E00650",  # Qualified dividends
    "E00300",  # Taxable interest
    "E26270",  # Partnership / S-corp
    "E00900",  # Business income
    "E02100",  # Farm income
    "E00400",  # Tax-exempt interest
    "E00600",  # Ordinary dividends
]

_STRUCTURAL_COLUMNS = {"MARS", "XTOT", "DSI", "EIC"}
_AMOUNT_COLUMN_PATTERN = re.compile(r"^(?:[EPT]\d+|S\d{5})$")
_AGI_CAP_100M_PLUS = 1_250_000_000
_MAX_AGI_DOMINANCE = 0.20
_SELECTION_POWER = 24
_NUMERIC_TOL = 1e-9

_YAML_PATH = (
    files("policyengine_us_data") / "datasets" / "puf" / "aggregate_record_totals.yaml"
)
with open(_YAML_PATH, "r", encoding="utf-8") as _f:
    _META = yaml.safe_load(_f)

BUCKET_META = _META["buckets"]


def _choose_n_synthetic(pop_weight: float) -> int:
    """Choose the number of synthetic records for an aggregate bucket."""
    return int(min(40, max(20, round(pop_weight / 10))))


def _assign_weights(pop_weight: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Assign integer synthetic weights summing to the published return count."""
    total_weight = int(round(pop_weight))
    base = max(total_weight // n, 3)
    weights = np.full(n, base, dtype=int)
    remainder = total_weight - weights.sum()

    if remainder > 0:
        bump_idx = rng.choice(n, size=remainder, replace=False)
        weights[bump_idx] += 1
    elif remainder < 0:
        reducible = np.where(weights > 1)[0]
        reduce_count = min(-remainder, len(reducible))
        reduce_idx = rng.choice(reducible, size=reduce_count, replace=False)
        weights[reduce_idx] -= 1

    gap = total_weight - weights.sum()
    if gap != 0:
        weights[0] += gap

    return weights


def _get_bucket_mask(df: pd.DataFrame, recid: int) -> pd.Series:
    """Return the AGI-bucket mask for a given aggregate RECID."""
    if recid == 999996:
        return df.E00100 < 0
    if recid == 999997:
        return (df.E00100 >= 0) & (df.E00100 < 10_000_000)
    if recid == 999998:
        return (df.E00100 >= 10_000_000) & (df.E00100 < 100_000_000)
    if recid == 999999:
        return df.E00100 >= 100_000_000
    raise ValueError(f"Unknown aggregate RECID {recid}")


def _get_amount_columns(columns: pd.Index | list[str]) -> list[str]:
    """Return raw PUF columns that behave like amount fields."""
    return [c for c in columns if _AMOUNT_COLUMN_PATTERN.match(c)]


def _get_bucket_targets(row: pd.Series) -> tuple[float, float, float]:
    """Return aggregate bucket weights and AGI targets."""
    pop_weight = float(row.S006) / 100
    target_mean_agi = float(row.E00100)
    target_total_agi = pop_weight * target_mean_agi
    return pop_weight, target_mean_agi, target_total_agi


def _get_donor_bucket(regular: pd.DataFrame, recid: int) -> pd.DataFrame:
    """Return donor records for one aggregate bucket, with a safe fallback."""
    donor_bucket = regular[_get_bucket_mask(regular, recid)].copy()
    if donor_bucket.empty:
        return regular.copy()
    return donor_bucket


def _coerce_amount_columns(
    selected: pd.DataFrame, amount_columns: list[str]
) -> pd.DataFrame:
    """Ensure donor amount columns are numeric before calibration."""
    return selected.astype(
        {column: float for column in amount_columns if column in selected.columns}
    )


def compute_aggregate_eligibility_scores(
    df: pd.DataFrame,
    screened_fields: list[str] | None = None,
    reference_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Score records on how 'aggregate-like' they are."""

    fields = screened_fields or SCREENED_FIELDS
    reference = df if reference_df is None else reference_df
    present_fields = [field for field in fields if field in df.columns]
    if not present_fields:
        return pd.Series(0.0, index=df.index, dtype=float)

    max_scores = np.zeros(len(df), dtype=float)

    for field in present_fields:
        values = pd.to_numeric(df[field], errors="coerce").fillna(0.0)
        reference_values = pd.to_numeric(reference[field], errors="coerce").fillna(0.0)
        field_scores = np.zeros(len(df), dtype=float)

        positive_mask = values > 0
        reference_positive = np.sort(reference_values[reference_values > 0].to_numpy())
        if positive_mask.any() and len(reference_positive) > 0:
            positive_scores = np.searchsorted(
                reference_positive,
                values[positive_mask].to_numpy(),
                side="right",
            ) / len(reference_positive)
            field_scores[positive_mask.to_numpy()] = positive_scores

        negative_mask = values < 0
        reference_negative = np.sort(
            (-reference_values[reference_values < 0]).to_numpy()
        )
        if negative_mask.any() and len(reference_negative) > 0:
            negative_scores = np.searchsorted(
                reference_negative,
                (-values[negative_mask]).to_numpy(),
                side="right",
            ) / len(reference_negative)
            field_scores[negative_mask.to_numpy()] = np.maximum(
                field_scores[negative_mask.to_numpy()],
                negative_scores,
            )

        max_scores = np.maximum(max_scores, field_scores)

    return pd.Series(max_scores, index=df.index, dtype=float)


def _project_weighted_sum_to_bounds(
    values: np.ndarray,
    weights: np.ndarray,
    target_total: float,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    """Adjust values so the weighted sum matches the target within bounds."""

    projected = np.clip(values.astype(float), lower, upper)

    for _ in range(max_iter):
        residual = float(target_total - np.dot(projected, weights))
        if abs(residual) <= 1e-6:
            return projected

        slack = upper - projected if residual > 0 else projected - lower
        free = slack > _NUMERIC_TOL
        if not free.any():
            break

        basis = np.abs(projected[free])
        if basis.sum() <= _NUMERIC_TOL:
            basis = np.ones(free.sum(), dtype=float)

        denom = float(np.dot(weights[free], basis))
        if denom <= _NUMERIC_TOL:
            basis = np.ones(free.sum(), dtype=float)
            denom = float(weights[free].sum())

        delta = residual * basis / denom
        if residual > 0:
            delta = np.minimum(delta, slack[free])
        else:
            delta = -np.minimum(-delta, slack[free])

        projected[free] += delta
        projected = np.clip(projected, lower, upper)

    residual = float(target_total - np.dot(projected, weights))
    if abs(residual) > 1e-6:
        slack = upper - projected if residual > 0 else projected - lower
        free = np.where(slack > _NUMERIC_TOL)[0]
        if len(free) > 0:
            best = free[np.argmax(slack[free] * weights[free])]
            projected[best] = np.clip(
                projected[best] + residual / weights[best],
                lower[best],
                upper[best],
            )

    return projected


def _allocate_weighted_values(
    base_values: np.ndarray,
    weights: np.ndarray,
    target_total: float,
    lower: np.ndarray | float | None = None,
    upper: np.ndarray | float | None = None,
) -> np.ndarray:
    """Allocate a weighted total across records using donor magnitudes."""

    base_values = np.asarray(base_values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(base_values)

    if abs(target_total) <= 1e-6:
        return np.zeros(n, dtype=float)

    if target_total > 0 and np.any(base_values > 0):
        active = base_values > 0
    elif target_total < 0 and np.any(base_values < 0):
        active = base_values < 0
    elif np.any(np.abs(base_values) > _NUMERIC_TOL):
        active = np.abs(base_values) > _NUMERIC_TOL
    else:
        active = np.ones(n, dtype=bool)

    allocated = np.zeros(n, dtype=float)
    magnitudes = np.abs(base_values[active])
    if magnitudes.sum() <= _NUMERIC_TOL:
        magnitudes = np.ones(active.sum(), dtype=float)

    denom = float(np.dot(weights[active], magnitudes))
    if denom <= _NUMERIC_TOL:
        magnitudes = np.ones(active.sum(), dtype=float)
        denom = float(weights[active].sum())

    allocated[active] = np.sign(target_total) * magnitudes * abs(target_total) / denom

    if lower is None and upper is None:
        return allocated

    if lower is None:
        lower_array = np.full(n, -np.inf, dtype=float)
    elif np.isscalar(lower):
        lower_array = np.full(n, float(lower), dtype=float)
    else:
        lower_array = np.asarray(lower, dtype=float)

    if upper is None:
        upper_array = np.full(n, np.inf, dtype=float)
    elif np.isscalar(upper):
        upper_array = np.full(n, float(upper), dtype=float)
    else:
        upper_array = np.asarray(upper, dtype=float)

    return _project_weighted_sum_to_bounds(
        allocated,
        weights,
        target_total,
        lower_array,
        upper_array,
    )


def _allocate_agi_values(
    donor_agi: np.ndarray,
    weights: np.ndarray,
    recid: int,
    target_total: float,
) -> np.ndarray:
    """Allocate AGI values with bucket bounds and dominance constraints."""

    donor_agi = np.asarray(donor_agi, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(donor_agi)
    meta = BUCKET_META[recid]

    dominance_cap = _MAX_AGI_DOMINANCE * abs(target_total) / weights

    if recid == 999996:
        lower = -dominance_cap
        upper = np.zeros(n, dtype=float)
    else:
        bucket_lower = float(meta["agi_lower"])
        bucket_upper = (
            _AGI_CAP_100M_PLUS
            if np.isinf(meta["agi_upper"])
            else float(meta["agi_upper"])
        )
        lower = np.full(n, max(bucket_lower, 0.0), dtype=float)
        upper = np.minimum(np.full(n, bucket_upper, dtype=float), dominance_cap)

    base = np.abs(donor_agi)
    return _allocate_weighted_values(
        base_values=base,
        weights=weights,
        target_total=target_total,
        lower=lower,
        upper=upper,
    )


def _selection_probabilities(
    donor_bucket: pd.DataFrame,
    donor_scores: pd.Series,
    target_mean_agi: float,
) -> np.ndarray:
    """Create donor-selection probabilities from extremeness and AGI proximity."""

    scores = donor_scores.loc[donor_bucket.index].to_numpy(dtype=float)
    score_mass = np.clip(scores, 1e-6, None) ** _SELECTION_POWER

    donor_abs_agi = np.abs(donor_bucket.E00100.to_numpy(dtype=float))
    target_abs_agi = max(abs(float(target_mean_agi)), 1.0)
    agi_distance = np.abs(np.log1p(donor_abs_agi) - np.log1p(target_abs_agi))
    agi_mass = 1.0 / (1.0 + agi_distance)

    probabilities = score_mass * np.sqrt(agi_mass)
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        probabilities = np.ones(len(donor_bucket), dtype=float)

    return probabilities / probabilities.sum()


def _sample_bucket_donors(
    donor_bucket: pd.DataFrame,
    donor_scores: pd.Series,
    target_mean_agi: float,
    n_syn: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample donor templates for one aggregate bucket."""

    probabilities = _selection_probabilities(
        donor_bucket=donor_bucket,
        donor_scores=donor_scores,
        target_mean_agi=target_mean_agi,
    )
    replace = len(donor_bucket) < n_syn
    selected_index = rng.choice(
        donor_bucket.index.to_numpy(),
        size=n_syn,
        replace=replace,
        p=probabilities,
    )
    return donor_bucket.loc[selected_index].reset_index(drop=True).copy()


def _apply_structural_templates(
    synthetic: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    """Copy donor structural variables without scaling them."""
    for column in _STRUCTURAL_COLUMNS:
        if column in synthetic.columns:
            synthetic[column] = selected[column].round().astype(int)

    if "MARS" not in synthetic.columns or "XTOT" not in synthetic.columns:
        return

    joint_mask = synthetic["MARS"] == 2
    synthetic.loc[joint_mask, "XTOT"] = np.maximum(
        synthetic.loc[joint_mask, "XTOT"],
        2,
    )
    synthetic["XTOT"] = synthetic["XTOT"].clip(lower=0, upper=5).astype(int)


def _calibrate_amount_columns(
    synthetic: pd.DataFrame,
    selected: pd.DataFrame,
    row: pd.Series,
    recid: int,
    pop_weight: float,
    target_total_agi: float,
    amount_columns: list[str],
    synthetic_weights: np.ndarray,
) -> None:
    """Match bucket-level amount totals while preserving donor structure."""
    synthetic["E00100"] = _allocate_agi_values(
        donor_agi=selected["E00100"].to_numpy(dtype=float),
        weights=synthetic_weights,
        recid=recid,
        target_total=target_total_agi,
    )

    for column in amount_columns:
        if column == "E00100":
            continue

        target_total = pop_weight * float(row.get(column, 0))
        synthetic[column] = _allocate_weighted_values(
            base_values=selected[column].to_numpy(dtype=float),
            weights=synthetic_weights,
            target_total=target_total,
        )


def _disaggregate_bucket(
    recid: int,
    row: pd.Series,
    regular: pd.DataFrame,
    amount_columns: list[str],
    donor_scores: pd.Series,
    next_recid: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create synthetic donor records for one aggregate bucket."""
    pop_weight, target_mean_agi, target_total_agi = _get_bucket_targets(row)
    donor_bucket = _get_donor_bucket(regular, recid)

    n_syn = _choose_n_synthetic(pop_weight)
    synthetic_weights = _assign_weights(pop_weight, n_syn, rng).astype(float)

    selected = _sample_bucket_donors(
        donor_bucket=donor_bucket,
        donor_scores=donor_scores,
        target_mean_agi=target_mean_agi,
        n_syn=n_syn,
        rng=rng,
    )
    selected = _coerce_amount_columns(selected, amount_columns)

    synthetic = selected.copy()
    synthetic["RECID"] = np.arange(next_recid, next_recid + n_syn, dtype=int)
    synthetic["S006"] = (synthetic_weights.astype(int) * 100).astype(int)

    _apply_structural_templates(synthetic, selected)
    _calibrate_amount_columns(
        synthetic=synthetic,
        selected=selected,
        row=row,
        recid=recid,
        pop_weight=pop_weight,
        target_total_agi=target_total_agi,
        amount_columns=amount_columns,
        synthetic_weights=synthetic_weights,
    )
    return synthetic
