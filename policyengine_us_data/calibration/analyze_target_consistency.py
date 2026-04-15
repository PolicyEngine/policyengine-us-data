"""Cross-level consistency analysis for calibration target inputs.

Reads `unified_diagnostics.csv` from a calibration run, parses target strings
into (geo_id, variable, filter), and runs two integrity checks:

1. Cross-level: for each (variable, filter) that appears at 2+ geo levels
   (national, state, district) with an identical filter string, compare
   `national` vs `sum(states)` vs `sum(districts)` and flag internal
   disagreements.

2. Bucket coverage: for each (variable, geo_id) where the filter set contains
   both an unbucketed aggregate and AGI-bucketed variants sharing the same
   base filter, verify that sum(bucket values) ~= aggregate value.

Writes a CSV report and prints a console summary of flagged entries. No data
mutation. Not a test — a read-only diagnostic.

Usage:
    python -m policyengine_us_data.calibration.analyze_target_consistency
    python -m policyengine_us_data.calibration.analyze_target_consistency \\
        --diagnostics path/to/unified_diagnostics.csv \\
        --output path/to/report.csv \\
        --tolerance-pct 1.0
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIAGNOSTICS_PATH = (
    "policyengine_us_data/storage/calibration/national/unified_diagnostics.csv"
)
DEFAULT_OUTPUT_PATH = (
    "policyengine_us_data/storage/calibration/target_consistency_report.csv"
)

AGI_CLAUSE_RE = re.compile(r"^adjusted_gross_income(<|>=)[^,\]]+$")
AGI_LOWER_RE = re.compile(r"adjusted_gross_income>=(-?[\w.]+)")
AGI_UPPER_RE = re.compile(r"adjusted_gross_income<(-?[\w.]+)")


def parse_target(target: str) -> tuple[str, str, str]:
    """Split 'geo/variable/filter' into (geo, variable, filter).

    Filter defaults to empty string when not present.
    """
    parts = target.split("/", 2)
    geo = parts[0]
    variable = parts[1] if len(parts) > 1 else ""
    filt = parts[2] if len(parts) > 2 else ""
    return geo, variable, filt


def classify_geo_level(geo: str) -> str:
    if geo == "national":
        return "national"
    if geo.startswith("state_"):
        return "state"
    if geo.startswith("cd_"):
        return "district"
    return "other"


def strip_agi_bucket_clauses(filt: str) -> str:
    """Return filter with any adjusted_gross_income<X / >=Y clauses removed.

    Used as the base-filter key for bucket-family detection. If the input is
    bracketed like '[a,b,c]', the output preserves the brackets.
    """
    if not filt:
        return filt
    bracketed = filt.startswith("[") and filt.endswith("]")
    inner = filt[1:-1] if bracketed else filt
    if not inner:
        return filt
    clauses = inner.split(",")
    kept = [c for c in clauses if not AGI_CLAUSE_RE.match(c)]
    if not kept:
        return "[]" if bracketed else ""
    result = ",".join(kept)
    return f"[{result}]" if bracketed else result


def has_agi_bucket_clause(filt: str) -> bool:
    return strip_agi_bucket_clauses(filt) != filt


def _parse_edge(raw: str) -> float:
    """Parse an AGI boundary token into a float, handling inf and -inf."""
    if raw == "inf":
        return float("inf")
    if raw == "-inf":
        return float("-inf")
    return float(raw)


def extract_agi_bounds(filt: str) -> tuple[float, float] | None:
    """Return (lower, upper) AGI bounds from a filter string, or None if
    the filter does not contain an AGI bucket clause."""
    lo_match = AGI_LOWER_RE.search(filt)
    hi_match = AGI_UPPER_RE.search(filt)
    if not lo_match or not hi_match:
        return None
    try:
        return _parse_edge(lo_match.group(1)), _parse_edge(hi_match.group(1))
    except ValueError:
        return None


def buckets_form_complete_partition(
    bounds: list[tuple[float, float]],
) -> bool:
    """True if the given (lower, upper) pairs partition (-inf, inf) with no
    gaps or overlaps."""
    if not bounds:
        return False
    sorted_bounds = sorted(bounds, key=lambda b: b[0])
    if sorted_bounds[0][0] != float("-inf"):
        return False
    if sorted_bounds[-1][1] != float("inf"):
        return False
    for i in range(len(sorted_bounds) - 1):
        if sorted_bounds[i][1] != sorted_bounds[i + 1][0]:
            return False
    return True


def classify_severity(max_abs_pct: float) -> str:
    if pd.isna(max_abs_pct):
        return "ok"
    if max_abs_pct < 1.0:
        return "ok"
    if max_abs_pct < 5.0:
        return "flag"
    if max_abs_pct < 15.0:
        return "significant"
    return "critical"


def _safe_pct_diff(numerator_value: float, denominator_value: float) -> float:
    if pd.isna(numerator_value) or pd.isna(denominator_value) or denominator_value == 0:
        return np.nan
    return (numerator_value - denominator_value) / denominator_value * 100.0


def load_diagnostics(path: str | Path) -> pd.DataFrame:
    """Load the diagnostics CSV and add parsed columns."""
    df = pd.read_csv(path)
    if "target" not in df.columns or "true_value" not in df.columns:
        raise ValueError(
            f"{path} is missing required columns 'target' and 'true_value'"
        )
    parsed = df["target"].apply(parse_target)
    df["geo_id"] = [p[0] for p in parsed]
    df["variable"] = [p[1] for p in parsed]
    df["filter"] = [p[2] for p in parsed]
    df["geo_level"] = df["geo_id"].apply(classify_geo_level)
    return df


def cross_level_check(df: pd.DataFrame) -> pd.DataFrame:
    """Same-filter comparison across national / state / district levels.

    For each (variable, filter), compute national true_value, sum of state
    true_values, and sum of district true_values, then report pairwise
    relative differences. Only groups with 2+ geo levels populated are
    returned.
    """
    agg = (
        df.groupby(["variable", "filter", "geo_level"])
        .agg(sum_value=("true_value", "sum"), n=("true_value", "size"))
        .reset_index()
    )
    values = agg.pivot_table(
        index=["variable", "filter"],
        columns="geo_level",
        values="sum_value",
        aggfunc="first",
    )
    counts = agg.pivot_table(
        index=["variable", "filter"],
        columns="geo_level",
        values="n",
        aggfunc="first",
        fill_value=0,
    )

    rows = []
    for (variable, filt), vrow in values.iterrows():
        national = vrow.get("national", np.nan)
        state_sum = vrow.get("state", np.nan)
        district_sum = vrow.get("district", np.nan)
        levels_present = sum(
            1 for x in (national, state_sum, district_sum) if pd.notna(x)
        )
        if levels_present < 2:
            continue

        n_states = int(counts.loc[(variable, filt)].get("state", 0) or 0)
        n_districts = int(counts.loc[(variable, filt)].get("district", 0) or 0)

        rel_ns = _safe_pct_diff(national, state_sum)
        rel_nd = _safe_pct_diff(national, district_sum)
        rel_sd = _safe_pct_diff(state_sum, district_sum)
        diffs = [d for d in (rel_ns, rel_nd, rel_sd) if pd.notna(d)]
        max_abs = max(abs(d) for d in diffs) if diffs else np.nan

        rows.append(
            {
                "check_type": "cross_level",
                "variable": variable,
                "filter": filt,
                "geo_level": "",
                "geo_id": "",
                "n_states": n_states,
                "n_districts": n_districts,
                "national": national,
                "sum_states": state_sum,
                "sum_districts": district_sum,
                "aggregate": np.nan,
                "sum_buckets": np.nan,
                "n_buckets": 0,
                "rel_nat_vs_state_pct": rel_ns,
                "rel_nat_vs_dist_pct": rel_nd,
                "rel_state_vs_dist_pct": rel_sd,
                "sum_vs_agg_pct": np.nan,
                "max_abs_rel_pct": max_abs,
                "severity": classify_severity(max_abs),
            }
        )
    return pd.DataFrame(rows)


def bucket_coverage_check(df: pd.DataFrame) -> pd.DataFrame:
    """AGI-bucket-vs-aggregate consistency within each geo_id.

    For each (variable, geo_id, base_filter), check whether the sum of
    AGI-bucketed true_values matches the unbucketed aggregate true_value.
    Only families that contain both at least one aggregate row and at least
    two bucket rows are considered.
    """
    work = df.copy()
    work["base_filter"] = work["filter"].apply(strip_agi_bucket_clauses)
    work["is_bucket"] = work["filter"] != work["base_filter"]

    rows = []
    grouped = work.groupby(["variable", "geo_id", "base_filter"])
    for (variable, geo_id, base_filter), family in grouped:
        buckets = family[family["is_bucket"]]
        aggregates = family[~family["is_bucket"]]
        if aggregates.empty or len(buckets) < 2:
            continue
        if len(aggregates) > 1:
            continue

        bounds = [extract_agi_bounds(f) for f in buckets["filter"].tolist()]
        if any(b is None for b in bounds):
            continue
        if not buckets_form_complete_partition(bounds):
            continue

        agg_value = float(aggregates["true_value"].iloc[0])
        bucket_sum = float(buckets["true_value"].sum())
        rel_diff = _safe_pct_diff(bucket_sum, agg_value)
        max_abs = abs(rel_diff) if pd.notna(rel_diff) else np.nan

        rows.append(
            {
                "check_type": "bucket_coverage",
                "variable": variable,
                "filter": base_filter,
                "geo_level": classify_geo_level(geo_id),
                "geo_id": geo_id,
                "n_states": 0,
                "n_districts": 0,
                "national": np.nan,
                "sum_states": np.nan,
                "sum_districts": np.nan,
                "aggregate": agg_value,
                "sum_buckets": bucket_sum,
                "n_buckets": len(buckets),
                "rel_nat_vs_state_pct": np.nan,
                "rel_nat_vs_dist_pct": np.nan,
                "rel_state_vs_dist_pct": np.nan,
                "sum_vs_agg_pct": rel_diff,
                "max_abs_rel_pct": max_abs,
                "severity": classify_severity(max_abs),
            }
        )
    return pd.DataFrame(rows)


def _format_currency_brief(value: float) -> str:
    if pd.isna(value):
        return "—"
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value / 1e12:,.2f}T"
    if abs_value >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs_value >= 1e6:
        return f"${value / 1e6:,.2f}M"
    if abs_value >= 1e3:
        return f"${value / 1e3:,.1f}K"
    return f"${value:,.0f}"


def _format_rel_pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.2f}%"


def print_summary(report: pd.DataFrame, tolerance_pct: float) -> None:
    total = len(report)
    severities = (
        report["severity"]
        .value_counts()
        .reindex(["critical", "significant", "flag", "ok"], fill_value=0)
    )
    flagged = report[report["severity"] != "ok"].copy()
    flagged = flagged.sort_values(
        ["severity", "max_abs_rel_pct"], ascending=[True, False]
    )

    print()
    print("=" * 100)
    print("Target consistency report")
    print("=" * 100)
    print(
        f"Tolerance: {tolerance_pct:.2f}%   "
        f"Total comparable entries: {total}   "
        f"Flagged: {len(flagged)}"
    )
    print(f"  critical (>15%):     {int(severities['critical'])}")
    print(f"  significant (5-15%): {int(severities['significant'])}")
    print(f"  flag (1-5%):         {int(severities['flag'])}")
    print(f"  ok (<1%):            {int(severities['ok'])}")
    print()

    if flagged.empty:
        print("No cross-level or bucket-coverage inconsistencies above tolerance.")
        return

    print("Flagged entries (sorted by severity, then max_abs_rel_pct):")
    print()
    header = (
        f"{'severity':<12} {'check':<17} {'variable':<32} "
        f"{'national':>14} {'sum_state':>14} {'sum_dist':>14} "
        f"{'aggregate':>14} {'sum_buckets':>14} {'max_abs':>10}"
    )
    print(header)
    print("-" * len(header))

    for _, r in flagged.head(40).iterrows():
        row = (
            f"{r['severity']:<12} "
            f"{r['check_type']:<17} "
            f"{r['variable'][:30]:<32} "
            f"{_format_currency_brief(r['national']):>14} "
            f"{_format_currency_brief(r['sum_states']):>14} "
            f"{_format_currency_brief(r['sum_districts']):>14} "
            f"{_format_currency_brief(r['aggregate']):>14} "
            f"{_format_currency_brief(r['sum_buckets']):>14} "
            f"{_format_rel_pct(r['max_abs_rel_pct']):>10}"
        )
        print(row)
        print(f"  filter: {r['filter']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze calibration target inputs for cross-level and "
            "bucket-coverage consistency."
        )
    )
    parser.add_argument(
        "--diagnostics",
        default=DEFAULT_DIAGNOSTICS_PATH,
        help=(f"Path to unified_diagnostics.csv (default: {DEFAULT_DIAGNOSTICS_PATH})"),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=1.0,
        help="Minimum |rel_diff| to flag (default: 1.0)",
    )
    args = parser.parse_args(argv)

    diagnostics_path = Path(args.diagnostics)
    if not diagnostics_path.exists():
        parser.error(f"Diagnostics file not found: {diagnostics_path}")

    df = load_diagnostics(diagnostics_path)
    cross = cross_level_check(df)
    bucket = bucket_coverage_check(df)

    report = pd.concat([cross, bucket], ignore_index=True)
    if report.empty:
        print("No comparable entries found in diagnostics.")
        return 0

    report = report.sort_values("max_abs_rel_pct", ascending=False, na_position="last")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Wrote {len(report)} rows to {output_path}")

    print_summary(report, args.tolerance_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
