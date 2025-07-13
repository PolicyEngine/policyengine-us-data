# ------------------------------------------------------------------------------
# Helper – classify rows as 'nation' or 'state' and extract a
# geography‑free metric key.
# ------------------------------------------------------------------------------

import re

_ST_RE  = re.compile(r"^[A-Za-z]{2}$")            # two‑letter code, any case
_US_RE  = re.compile(r"^US(\d{2})$")              # FIPS root, e.g. US37
_FIPS2STATE = {                                   #                     
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
    "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
    "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
    "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
    "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
    "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
    "55":"WI","56":"WY","72":"PR"
}
_STATE_CODES = set(_FIPS2STATE.values()) | {"DC"}   # quick membership test

# ---------------------------------------------------------------------------
#  Central place to normalise metric names:
#     canonical = ALIASES.get(raw_metric, raw_metric)
#  – Use the canonical key for *all* grouping and comparisons.
# ---------------------------------------------------------------------------

ALIASES: dict[str, str] = {
    # ---------------------------------------------------
    # Affordable Care Act
    # ---------------------------------------------------
    # IRS publishes ACA data by state; the nation‑level total sits under
    # a generic “gov/” prefix in your file.
    "irs/aca_enrollment" : "gov/aca_enrollment",
    "irs/aca_spending"   : "gov/aca_spending",

    # ---------------------------------------------------
    # Medicaid
    # ---------------------------------------------------
    # State totals live under IRS, national total under HHS.
    "irs/medicaid_enrollment" : "hhs/medicaid_enrollment",

    # ---------------------------------------------------
    # Supplemental Nutrition Assistance Program (SNAP)
    # ---------------------------------------------------
    # • `snap-cost` – state FIPS‑rooted rows published by USDA and used by
    #   CBO for the baseline ➜ compare with CBO’s national outlay series.
    # • `snap-hhs`  – state beneficiary counts published by HHS; there is
    #   **no** national head‑count row in the file, so we leave that *un‑*
    #   aliased on purpose.
    "snap-cost" : "cbo/snap",

    # ---------------------------------------------------
    # Property‑tax offsets used in SPM
    # ---------------------------------------------------
    # State property‑tax totals under “real_estate_taxes” align with the
    # single national figure published by Census.
    "real_estate_taxes" : "census/real_estate_taxes",
}


def _parse_name(col: str):
    """
    Returns
    -------
    level   : 'nation' | 'state'
    geo_id  : 'USA' or two‑letter state code (upper‑case)
    metric  : geography‑free identifier (same string for nation & state rows)
    """
    parts = col.split("/")

    # ---------- Case 1: root == 'nation' -------------------------------------
    if parts[0] == "nation":
        #  If the *last* segment is a valid US‑state code → treat as state‑level
        last = parts[-1].upper()
        if last in _STATE_CODES:
            metric = "/".join(parts[1:-1])          # drop 'nation' + state code
            return "state", last, metric
        # otherwise it really is a national row
        return "nation", "USA", "/".join(parts[1:])

    # ---------- Case 2: FIPS‑coded root, e.g. 'US37/...' ---------------------
    m = _US_RE.match(parts[0])
    if m:
        geo = _FIPS2STATE.get(m.group(1), m.group(1))
        return "state", geo, "/".join(parts[1:])

    # ---------- Case 3: explicit 'state/' prefix -----------------------------
    if parts[0] == "state":
        for i, seg in enumerate(parts[1:], 1):
            if seg.upper() in _STATE_CODES:
                metric = "/".join(p for j, p in enumerate(parts[1:], 1) if j != i)
                return "state", seg.upper(), metric

    # ---------- Case 4: generic “…/<STATE>” tail -----------------------------
    if parts[-1].upper() in _STATE_CODES:
        return "state", parts[-1].upper(), "/".join(parts[:-1])

    # ---------- Fallback – treat as state row with unknown geo ---------------
    return "state", "??", "/".join(parts)



# ------------------------------------------------------------------------------
# 2.  Build look‑up tables once – this is cheap (~3000 rows)
# ------------------------------------------------------------------------------

def _build_index(names: List[str]) -> pd.DataFrame:
    records = [(*_parse_name(n), idx, n) for idx, n in enumerate(names)]
    df = pd.DataFrame(
        records,
        columns=["level", "geo_id", "metric", "i", "fullname"]
    )
    return df

# ------------------------------------------------------------------------------
# 3.  Flag inconsistencies
# ------------------------------------------------------------------------------

def flag_inconsistencies(targets: np.ndarray,
                         names:   List[str],
                         tol: float = 0.01
                         ) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per *metric* that has at least two
    geography levels.  For each row we show:

        metric, national_total, sum_state, rel_gap_state,
                 bad_states (boolean list),
                 example_child_metric (for debugging)

    *rel_gap_state* is (sum_state – national_total) / |national_total|.
    A value whose absolute size exceeds *tol* is flagged.
    """
    idx = _build_index(names)
    df = idx.copy()
    df["value"] = targets[df["i"]]
    df["metric_norm"] = df["metric"].map(ALIASES).fillna(df["metric"])

    # ---- Compute national level totals
    nat = (df[df.level == "nation"]
             .groupby("metric_norm")["value"]
             .sum()
             .rename("nat_total"))

    # ---- Compute state totals (whatever rows we have labelled 'state')
    st  = (df[df.level == "state"]
             .groupby("metric_norm")["value"]
             .sum()
             .rename("sum_state"))

    report = pd.concat([nat, st], axis=1, join="inner").reset_index()

    # ---- Gap and flag
    report["rel_gap_state"] = (report["sum_state"] - report["nat_total"]).abs() \
                              / report["nat_total"].abs().replace(0, np.nan)
    report["state_inconsistent"] = report["rel_gap_state"] > tol

    # ---- Which individual states break their own internal hierarchy?
    # For every state + metric where we have *both* a state total row and
    # more granular sub‑state rows, check they match.
    bad_states = []
    for metric, grp in df[df.level != "nation"].groupby("metric_norm"):
        state_vals = grp[grp.level == "state"]
        if state_vals.empty:         # nothing to compare inside this metric
            continue
        # For each state that has both state‑level & district rows:
        for st_code, st_grp in grp.groupby("geo_id"):
            if len(st_grp) <= 1:     # no child rows
                continue
            state_total = st_grp[st_grp.level == "state"]["value"].sum()
            child_total = st_grp[st_grp.level != "state"]["value"].sum()
            if state_total == 0:
                continue
            rel_gap = abs(child_total - state_total) / abs(state_total)
            if rel_gap > tol:
                bad_states.append((metric, st_code, rel_gap))

    report["bad_states"] = report["metric_norm"].apply(
        lambda m: [s for (met, s, _) in bad_states if met == m]
    )

    return report.sort_values("rel_gap_state", ascending=False)

# ------------------------------------------------------------------------------
# 4.  Repair inconsistencies (down‑weight fine geography to match coarse)
# ------------------------------------------------------------------------------

def repair_inconsistencies(targets: np.ndarray,
                           names:   List[str],
                           inplace: bool = False,
                           tol: float = 0.01
                           ) -> np.ndarray:
    """
    Returns a *new* array (unless `inplace=True`) in which the child
    geographies have been scaled **proportionally** so that

        Σ states   == national total   for every metric, and
        Σ children == state total      for every (state, metric)

    Coarser levels are never changed; only the finer rows are rescaled.
    """
    if not inplace:
        targets = targets.copy()

    idx = _build_index(names)
    values = targets

    # ---- 1. nation -> states -------------------------------------------------
    df = idx.copy()
    df["value"] = values[df["i"]]
    df["metric_norm"] = df["metric"].map(ALIASES).fillna(df["metric"])

    # Work metric by metric
    for metric, grp in df.groupby("metric_norm"):
        nat_val = grp.loc[grp.level == "nation", "value"]
        if nat_val.empty:
            continue                    # no national comparator
        nat_val = nat_val.iloc[0]

        states = grp[grp.level == "state"]
        total_state = states["value"].sum()
        if total_state == 0 or abs(total_state - nat_val) / abs(nat_val) <= tol:
            continue

        scale = nat_val / total_state
        values[states["i"]] *= scale    # proportional rescale

        # ---- 2. Within each state: state -> district/child ------------------
        df.loc[states.index, "value"] = values[states["i"]]
        # Build fresh groups after the state adjustment
        for geo_id, st_grp in grp.groupby("geo_id"):
            st_rows   = st_grp[st_grp.level == "state"]
            child_rows= st_grp[st_grp.level != "state"]
            if st_rows.empty or child_rows.empty:
                continue
            st_total = st_rows["value"].sum()
            child_total = child_rows["value"].sum()
            if st_total == 0 or abs(child_total - st_total) / abs(st_total) <= tol:
                continue
            child_scale = st_total / child_total
            values[child_rows["i"]] *= child_scale

    return values



from typing import Callable, Iterable, Tuple, List, Sequence
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1.  Decide what “an entire set of targets” means
# ---------------------------------------------------------------------------
#
# We keep using the canonical‑metric key that powers flag_inconsistencies:
#
#       canonical = ALIASES.get(raw_metric, raw_metric)
#
# For example
#   • every row whose canonical key is  'adjusted_gross_income/amount'
#     (all states + any national rows)
#   • every row that *contains* 'census/age/'  (all 5‑year age buckets)
#   • etc.
#
# So the *selector* you pass to drop_targets() is a function that consumes the
# canonical key and returns True if that column should be deleted.
# ---------------------------------------------------------------------------

def _canonical_names(colnames: Sequence[str]) -> List[str]:
    """Return the canonical metric string for every raw column name."""
    return [ALIASES.get(_parse_name(c)[2], _parse_name(c)[2]) for c in colnames]

# ---------------------------------------------------------------------------
# 2.  Main public helper
# ---------------------------------------------------------------------------

def drop_targets(loss_matrix : pd.DataFrame,
                 targets      : np.ndarray,
                 selector     : Callable[[str], bool]
                 ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Parameters
    ----------
    loss_matrix : original matrix whose *columns* are the raw metric names.
    targets     : 1‑D vector whose rows are aligned with those columns.
    selector    : function(canonical_metric:str) -> bool
                  Return True to delete that entire column / row.

    Returns
    -------
    (new_matrix, new_targets)
    """
    if targets.ndim != 1 or len(targets) != loss_matrix.shape[1]:
        raise ValueError("`targets` must be 1‑D and aligned with loss_matrix columns")

    canon = _canonical_names(loss_matrix.columns)
    mask  = np.fromiter((not selector(c) for c in canon), dtype=bool, count=len(canon))

    new_matrix  = loss_matrix.loc[:, mask]     # keep the columns we want
    new_targets = targets[mask]                # keep the aligned rows
    return new_matrix, new_targets

# ---------------------------------------------------------------------------
# 3.  Convenience wrappers for common use‑cases
# ---------------------------------------------------------------------------

def drop_metrics_by_prefix(loss_matrix: pd.DataFrame,
                           targets     : np.ndarray,
                           *prefixes   : str
                           ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Remove everything whose canonical metric *starts with* any of `prefixes`.
    Example:
        lm, tgt = drop_metrics_by_prefix(lm, tgt,
                                         'adjusted_gross_income/amount',
                                         'adjusted_gross_income/count')
    """
    pref = tuple(p.rstrip("/") for p in prefixes)           # normalise
    return drop_targets(loss_matrix, targets,
                        lambda m: m.startswith(pref))

def drop_metrics_by_regex(loss_matrix: pd.DataFrame,
                          targets     : np.ndarray,
                          pattern     : str
                          ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Remove every metric whose canonical name matches the regex `pattern`."""
    import re
    rx = re.compile(pattern)
    return drop_targets(loss_matrix, targets, lambda m: bool(rx.search(m)))


