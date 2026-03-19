"""Disaggregate PUF aggregate records into synthetic individual records.

The IRS PUF has 4 aggregate records (MARS=0, RECID 999996-999999) that
bundle ~1,214 ultra-high-income filers for disclosure protection.  The
income values are per-return averages (weight x amount = population total).

This module synthesizes ~120 weighted records (not 1,214 unit-weight)
using a conservative approach:
  1. Truncated lognormal AGI with hard bucket bounds
  2. Donor-based variable templates
  3. Dirichlet composition shares with high concentration
  4. Exact weighted-total calibration
"""

from importlib.resources import files

import numpy as np
import pandas as pd
import yaml
from scipy.stats import truncnorm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGGREGATE_RECIDS = [999996, 999997, 999998, 999999]
SYNTHETIC_RECID_START = 1_000_000

MAJOR_COMPONENTS = [
    "E00200",  # Wages
    "P23250",  # Long-term capital gains
    "P22250",  # Short-term capital gains
    "E00650",  # Qualified dividends
    "E00300",  # Taxable interest
    "E26270",  # Partnership / S-corp
    "E00900",  # Business income (Sch C)
    "E02100",  # Farm income (Sch F)
]

# MARS distribution for ultra-high-income filers (from SOI data).
_MARS_VALUES = np.array([1, 2, 3, 4])
_MARS_PROBS = np.array([0.20, 0.75, 0.03, 0.02])

# Donor pool: regular records with |AGI| >= this threshold.
DONOR_AGI_THRESHOLD = 2_000_000

# Per-bucket lognormal sigma and AGI caps.
_BUCKET_SIGMA = {
    999996: 1.0,  # negative AGI: high dispersion
    999997: 0.20,  # <$10M
    999998: 0.30,  # $10M-$100M
    999999: 0.35,  # $100M+
}

# Cap for the open-ended $100M+ bucket (3x typical mean).
_AGI_CAP_100M_PLUS = 1_250_000_000

# Per-bucket Dirichlet concentration (higher = less noise).
_BUCKET_CONCENTRATION = {
    999996: 50,
    999997: 80,
    999998: 50,
    999999: 35,
}

# Component-to-AGI caps (absolute value).
_COMPONENT_CAPS = {
    "E00200": 1.5,  # wages
    "P23250": 2.0,  # LTCG
    "P22250": 1.0,  # STCG
    "E00650": 1.0,  # qual div
    "E00300": 0.5,  # interest
    "E26270": 1.0,  # partnership
    "E00900": 0.75,  # business
    "E02100": 0.5,  # farm
}

# Max share of bucket AGI any single record can carry.
_MAX_AGI_DOMINANCE = 0.20

# ---------------------------------------------------------------------------
# Load YAML metadata
# ---------------------------------------------------------------------------

_YAML_PATH = (
    files("policyengine_us_data") / "datasets" / "puf" / "aggregate_record_totals.yaml"
)
with open(_YAML_PATH, "r", encoding="utf-8") as _f:
    _META = yaml.safe_load(_f)

BUCKET_META = _META["buckets"]
COMBINED_NONZERO = _META["combined_nonzero_counts"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _choose_n_synthetic(pop_weight: float) -> int:
    """Choose number of synthetic records for a bucket.

    Target ~10 filers per synthetic record, clamped to [20, 40].
    """
    return int(min(40, max(20, round(pop_weight / 10))))


def _assign_weights(pop_weight: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Assign integer weights summing to pop_weight.

    Returns array of length n with values >= 3.
    """
    w_int = int(round(pop_weight))
    base = max(w_int // n, 3)
    weights = np.full(n, base)
    remainder = w_int - base * n
    if remainder > 0:
        bump_idx = rng.choice(n, size=remainder, replace=False)
        weights[bump_idx] += 1
    elif remainder < 0:
        # Reduce some weights (but keep >= 3)
        reduce_idx = rng.choice(n, size=-remainder, replace=False)
        weights[reduce_idx] -= 1
        weights = np.maximum(weights, 3)
    return weights


def _draw_truncated_lognormal(
    n: int,
    lower: float,
    upper: float,
    sigma: float,
    target_total: float,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw AGI values from truncated lognormal, rescaled to total.

    Uses scipy's truncnorm in log-space to draw from a truncated
    lognormal, then rescales draws to match exact weighted total.
    """
    if n == 0:
        return np.array([])

    target_mean = target_total / weights.sum()

    # mu estimate (untruncated lognormal mean formula)
    mu = np.log(max(target_mean, lower + 1)) - sigma**2 / 2

    # Draw from truncated lognormal via scipy truncnorm in log-space
    a = (np.log(lower) - mu) / sigma
    b = (np.log(upper) - mu) / sigma
    log_vals = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng)
    vals = np.exp(log_vals)

    # Rescale so weighted sum matches total
    current_weighted = (vals * weights).sum()
    if abs(current_weighted) > 0:
        vals *= target_total / current_weighted

    # Reclip to bucket bounds
    vals = np.clip(vals, lower, upper)

    # Final exact adjustment: distribute residual proportionally
    residual = target_total - (vals * weights).sum()
    if abs(residual) > 1:
        vals += residual / weights.sum()
        vals = np.clip(vals, lower, upper)

    return vals


def _draw_lognormal_negative(
    n: int,
    target_total: float,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw negative AGI values from lognormal on |AGI|."""
    if n == 0:
        return np.array([])
    mean_abs = abs(target_total / weights.sum())
    sigma = 1.0
    mu = np.log(max(mean_abs, 1)) - sigma**2 / 2
    vals = -rng.lognormal(mean=mu, sigma=sigma, size=n)
    # Rescale
    current = (vals * weights).sum()
    if abs(current) > 0:
        vals *= target_total / current
    return vals


def _compute_shares(row: pd.Series, components: list) -> dict:
    """Compute income component shares of |AGI| for a donor row."""
    agi = abs(row.get("E00100", 0))
    if agi < 1:
        agi = 1
    return {c: row.get(c, 0) / agi for c in components}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def disaggregate_aggregate_records(
    puf: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Replace 4 aggregate PUF records with ~120 weighted synthetic records.

    Called BEFORE preprocess_puf, so S006 is still in raw hundredths.
    Population total for each variable = (S006 / 100) * value.

    Synthetic records get variable weights (S006 = weight * 100)
    and MARS drawn from a high-income distribution.

    Parameters
    ----------
    puf : pd.DataFrame
        Raw PUF before preprocess (S006 in hundredths).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        PUF with aggregate rows removed and synthetic rows appended.
    """
    rng = np.random.default_rng(seed)

    agg_mask = puf.RECID.isin(AGGREGATE_RECIDS)
    if agg_mask.sum() == 0:
        return puf

    agg_rows = puf[agg_mask].copy()
    regular = puf[~agg_mask].copy()

    # Build donor pool
    donor_mask = regular.E00100.abs() >= DONOR_AGI_THRESHOLD
    donors = regular[donor_mask].copy()

    # Columns to carry through (all numeric PUF columns except meta)
    meta_cols = {"RECID", "S006", "MARS"}
    income_cols = [
        c for c in puf.columns if c not in meta_cols and puf[c].dtype.kind in ("f", "i")
    ]

    all_synthetic = []
    next_recid = SYNTHETIC_RECID_START

    for recid in AGGREGATE_RECIDS:
        row = agg_rows[agg_rows.RECID == recid].iloc[0]
        meta = BUCKET_META[recid]
        agi_lower = meta["agi_lower"]
        agi_upper = meta["agi_upper"]

        # True weight and population total
        pop_weight = row.S006 / 100
        total_agi = pop_weight * row.E00100

        # Step 1: Choose n synthetic records and assign weights
        n_syn = _choose_n_synthetic(pop_weight)
        syn_weights = _assign_weights(pop_weight, n_syn, rng)

        # Step 2: Generate AGI
        sigma = _BUCKET_SIGMA[recid]
        if agi_upper <= 0:
            synthetic_agi = _draw_lognormal_negative(n_syn, total_agi, syn_weights, rng)
        else:
            # Apply cap for open-ended bucket
            effective_upper = agi_upper
            if np.isinf(agi_upper):
                effective_upper = _AGI_CAP_100M_PLUS
            effective_lower = max(agi_lower, 1)

            synthetic_agi = _draw_truncated_lognormal(
                n_syn,
                effective_lower,
                effective_upper,
                sigma,
                total_agi,
                syn_weights,
                rng,
            )

        # Dominance check: no record > 20% of total
        if abs(total_agi) > 0:
            max_agi = _MAX_AGI_DOMINANCE * abs(total_agi)
            over = np.abs(synthetic_agi * syn_weights) > max_agi
            if over.any():
                synthetic_agi[over] = np.sign(synthetic_agi[over]) * (
                    max_agi / syn_weights[over]
                )
                # Re-calibrate
                current = (synthetic_agi * syn_weights).sum()
                if abs(current) > 0:
                    synthetic_agi *= total_agi / current
                    if agi_upper > 0:
                        eff_u = _AGI_CAP_100M_PLUS if np.isinf(agi_upper) else agi_upper
                        synthetic_agi = np.clip(
                            synthetic_agi,
                            max(agi_lower, 1),
                            eff_u,
                        )

        # Step 3: Compute aggregate shares for Dirichlet
        agg_shares = _compute_shares(row, MAJOR_COMPONENTS)
        concentration = _BUCKET_CONCENTRATION[recid]

        # Step 4: Sample donors by AGI proximity
        donor_agis = donors.E00100.values
        abs_diff = np.abs(donor_agis[:, None] - synthetic_agi[None, :])
        proximity = 1.0 / (abs_diff + 1e6)
        proximity /= proximity.sum(axis=0, keepdims=True)
        donor_indices = np.array(
            [rng.choice(len(donors), p=proximity[:, i]) for i in range(n_syn)]
        )
        sampled_donors = donors.iloc[donor_indices]

        # Step 5: Build synthetic records
        synthetic_rows = []
        for i in range(n_syn):
            syn = {}
            syn["RECID"] = next_recid
            next_recid += 1
            syn["S006"] = int(syn_weights[i]) * 100
            syn["MARS"] = rng.choice(_MARS_VALUES, p=_MARS_PROBS)

            syn_agi = synthetic_agi[i]
            abs_agi = max(abs(syn_agi), 1)
            donor_row = sampled_donors.iloc[i]

            # Dirichlet shares centered on aggregate shares
            raw_shares = np.array(
                [max(abs(agg_shares[c]), 0.001) for c in MAJOR_COMPONENTS]
            )
            noisy = rng.dirichlet(raw_shares * concentration)

            # Apply to AGI with sign from aggregate record
            for j, c in enumerate(MAJOR_COMPONENTS):
                sign = np.sign(agg_shares[c]) if agg_shares[c] != 0 else 1
                raw_val = noisy[j] * syn_agi * sign
                # Cap component
                cap = _COMPONENT_CAPS.get(c, 2.0)
                raw_val = np.clip(raw_val, -cap * abs_agi, cap * abs_agi)
                syn[c] = raw_val

            # Wages must be non-negative for positive AGI
            if syn_agi >= 0:
                syn["E00200"] = max(syn.get("E00200", 0), 0)

            # AGI
            syn["E00100"] = syn_agi

            # Secondary variables via donor scaling
            donor_agi = donor_row.E00100
            if abs(donor_agi) < 1:
                donor_agi = 1
            scale_ratio = np.clip(syn_agi / donor_agi, -5, 5)

            for c in income_cols:
                if c in MAJOR_COMPONENTS or c == "E00100":
                    continue
                donor_val = donor_row.get(c, 0)
                if pd.isna(donor_val):
                    donor_val = 0
                val = donor_val * scale_ratio
                # Kill floating point noise
                if abs(val) < 1e-6:
                    val = 0
                syn[c] = val

            synthetic_rows.append(syn)

        all_synthetic.append(pd.DataFrame(synthetic_rows))

    synthetic_df = pd.concat(all_synthetic, ignore_index=True)

    # ---- Weighted calibration ----
    # 3 iterations of bounded multiplicative adjustment
    offset = 0
    for recid in AGGREGATE_RECIDS:
        row = agg_rows[agg_rows.RECID == recid].iloc[0]
        meta = BUCKET_META[recid]
        pop_weight = row.S006 / 100
        n_syn = _choose_n_synthetic(pop_weight)

        bucket_start = SYNTHETIC_RECID_START + offset
        bucket_end = bucket_start + n_syn
        offset += n_syn

        bucket_mask = (synthetic_df.RECID >= bucket_start) & (
            synthetic_df.RECID < bucket_end
        )
        if bucket_mask.sum() == 0:
            continue

        # Weights for this bucket
        bw = synthetic_df.loc[bucket_mask, "S006"].values / 100

        for _ in range(3):
            for c in income_cols:
                target = pop_weight * row.get(c, 0)
                if pd.isna(target):
                    target = 0

                vals = synthetic_df.loc[bucket_mask, c].values
                current_total = (vals * bw).sum()

                if abs(target) < 1:
                    synthetic_df.loc[bucket_mask, c] = 0
                    continue

                if abs(current_total) < 1:
                    # Distribute evenly
                    synthetic_df.loc[bucket_mask, c] = target / bw.sum()
                    continue

                adj = np.clip(target / current_total, 0.5, 2.0)
                synthetic_df.loc[bucket_mask, c] *= adj

    # Final exact calibration pass
    offset = 0
    for recid in AGGREGATE_RECIDS:
        row = agg_rows[agg_rows.RECID == recid].iloc[0]
        meta = BUCKET_META[recid]
        pop_weight = row.S006 / 100
        n_syn = _choose_n_synthetic(pop_weight)

        bucket_start = SYNTHETIC_RECID_START + offset
        bucket_end = bucket_start + n_syn
        offset += n_syn

        bucket_mask = (synthetic_df.RECID >= bucket_start) & (
            synthetic_df.RECID < bucket_end
        )
        if bucket_mask.sum() == 0:
            continue

        bw = synthetic_df.loc[bucket_mask, "S006"].values / 100

        for c in income_cols:
            target = pop_weight * row.get(c, 0)
            if pd.isna(target):
                target = 0
            vals = synthetic_df.loc[bucket_mask, c].values
            current_total = (vals * bw).sum()
            if abs(target) < 1:
                synthetic_df.loc[bucket_mask, c] = 0
            elif abs(current_total) > 0:
                synthetic_df.loc[bucket_mask, c] *= target / current_total

    # Assemble output
    for c in puf.columns:
        if c not in synthetic_df.columns:
            synthetic_df[c] = 0

    synthetic_df = synthetic_df[puf.columns]

    result = pd.concat([regular, synthetic_df], ignore_index=True)

    n_syn_total = len(synthetic_df)
    n_removed = agg_mask.sum()
    print(
        f"Disaggregated {n_removed} aggregate records into "
        f"{n_syn_total} synthetic records"
    )

    return result
