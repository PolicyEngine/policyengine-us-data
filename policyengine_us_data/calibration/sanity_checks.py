"""Structural integrity checks for calibrated H5 files.

Run standalone:
    python -m policyengine_us_data.calibration.sanity_checks path/to/file.h5

Or integrated via validate_staging.py --sanity-only.
"""

import logging
from typing import List

import h5py
import numpy as np

from policyengine_us_data.pipeline_metadata import pipeline_node
from policyengine_us_data.pipeline_schema import PipelineNode

logger = logging.getLogger(__name__)

WEEKS_IN_YEAR = 52
STANDARD_HOURS_PER_WEEK = 40
OVERTIME_RATE_MULTIPLIER = 1.5
HOURLY_WAGE_INCOME_RELATIVE_TOLERANCE = 0.10
HOURLY_WAGE_INCOME_MISMATCH_SHARE_WARN_THRESHOLD = 0.25
HOURLY_WAGE_INCOME_MEAN_ABS_REL_ERROR_WARN_THRESHOLD = 0.20

KEY_MONETARY_VARS = [
    "employment_income",
    "adjusted_gross_income",
    "snap",
    "ssi",
    "eitc",
    "social_security",
    "income_tax_before_credits",
]

TAKEUP_VARS = [
    "takes_up_snap_if_eligible",
    "takes_up_ssi_if_eligible",
    "takes_up_aca_if_eligible",
    "takes_up_medicaid_if_eligible",
    "takes_up_tanf_if_eligible",
    "takes_up_head_start_if_eligible",
    "takes_up_early_head_start_if_eligible",
    "takes_up_dc_ptc",
]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    return float(np.average(values, weights=weights))


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if values.size == 0 or weights.sum() <= 0:
        return 0.0

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative_weights = np.cumsum(weights)
    cutoff = quantile * cumulative_weights[-1]
    return float(values[np.searchsorted(cumulative_weights, cutoff, side="left")])


def _format_hourly_wage_income_detail(
    *,
    comparable_count: int,
    comparable_weight: float,
    mismatch_count: int,
    mismatch_share: float,
    mean_abs_rel_error: float,
    p90_abs_rel_error: float,
    over_income_share: float,
    tolerance: float,
) -> str:
    return (
        f"{mismatch_count:,}/{comparable_count:,} unweighted mismatches; "
        f"{mismatch_share:.1%} weighted mismatch share at "
        f">{tolerance:.0%} tolerance among {comparable_weight:,.0f} weighted workers; "
        f"mean absolute relative gap {mean_abs_rel_error:.1%}; "
        f"p90 absolute relative gap {p90_abs_rel_error:.1%}; "
        f"{over_income_share:.1%} imply annual wages above employment_income "
        f"by >{tolerance:.0%}"
    )


def build_hourly_wage_income_consistency_diagnostics(
    employment_income: np.ndarray,
    hourly_wage: np.ndarray,
    hours_worked_last_week: np.ndarray,
    is_paid_hourly: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    relative_tolerance: float = HOURLY_WAGE_INCOME_RELATIVE_TOLERANCE,
    mismatch_share_warn_threshold: float = (
        HOURLY_WAGE_INCOME_MISMATCH_SHARE_WARN_THRESHOLD
    ),
    mean_abs_rel_error_warn_threshold: float = (
        HOURLY_WAGE_INCOME_MEAN_ABS_REL_ERROR_WARN_THRESHOLD
    ),
) -> List[dict]:
    """Compare hourly facts with annual employment income.

    Warns when more than 25 percent of comparable hourly workers differ by
    more than 10 percent, or when the weighted mean absolute relative gap
    exceeds 20 percent. These thresholds flag broad inconsistencies while
    allowing last-week hours and annual wages to differ for normal reasons.
    """
    employment_income = np.asarray(employment_income, dtype=float)
    hourly_wage = np.asarray(hourly_wage, dtype=float)
    hours_worked_last_week = np.asarray(hours_worked_last_week, dtype=float)
    is_paid_hourly = np.asarray(is_paid_hourly, dtype=bool)

    if weights is None:
        weights = np.ones_like(employment_income, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    straight_time_hours = np.minimum(hours_worked_last_week, STANDARD_HOURS_PER_WEEK)
    overtime_hours = np.maximum(hours_worked_last_week - STANDARD_HOURS_PER_WEEK, 0)
    straight_time_equivalent_hours = WEEKS_IN_YEAR * (
        straight_time_hours + overtime_hours * OVERTIME_RATE_MULTIPLIER
    )
    implied_annual_wages = hourly_wage * straight_time_equivalent_hours

    base_mask = (
        is_paid_hourly
        & (hourly_wage > 0)
        & (hours_worked_last_week > 0)
        & (employment_income > 0)
        & np.isfinite(implied_annual_wages)
        & np.isfinite(employment_income)
        & np.isfinite(weights)
        & (weights > 0)
    )

    results = []
    subsets = [
        ("hourly_wage_income_consistency", base_mask),
        (
            "hourly_wage_income_consistency_overtime",
            base_mask & (overtime_hours > 0),
        ),
    ]

    for check_name, mask in subsets:
        if not mask.any():
            results.append(
                {
                    "check": check_name,
                    "status": "SKIP",
                    "detail": "no comparable hourly workers",
                }
            )
            continue

        rel_gap = (
            implied_annual_wages[mask] - employment_income[mask]
        ) / employment_income[mask]
        subset_weights = weights[mask]
        mismatch = np.abs(rel_gap) >= relative_tolerance
        over_income = rel_gap >= relative_tolerance
        mismatch_share = _weighted_mean(mismatch.astype(float), subset_weights)
        mean_abs_rel_error = _weighted_mean(np.abs(rel_gap), subset_weights)
        p90_abs_rel_error = _weighted_quantile(
            np.abs(rel_gap),
            subset_weights,
            0.9,
        )
        over_income_share = _weighted_mean(
            over_income.astype(float),
            subset_weights,
        )

        warn = (
            mismatch_share > mismatch_share_warn_threshold
            or mean_abs_rel_error > mean_abs_rel_error_warn_threshold
        )
        results.append(
            {
                "check": check_name,
                "status": "WARN" if warn else "PASS",
                "detail": _format_hourly_wage_income_detail(
                    comparable_count=int(mask.sum()),
                    comparable_weight=float(subset_weights.sum()),
                    mismatch_count=int(mismatch.sum()),
                    mismatch_share=mismatch_share,
                    mean_abs_rel_error=mean_abs_rel_error,
                    p90_abs_rel_error=p90_abs_rel_error,
                    over_income_share=over_income_share,
                    tolerance=relative_tolerance,
                ),
            }
        )

    return results


@pipeline_node(
    PipelineNode(
        id="sanity_checks",
        label="Run H5 Sanity Checks",
        node_type="validation",
        description="Check calibrated H5 structure, weights, IDs, mappings, takeup, and aggregate sanity.",
        source_file="policyengine_us_data/calibration/sanity_checks.py",
        status="current",
        stability="moving",
        pathways=["local_h5"],
        validation_commands=[
            "uv run pytest tests/unit/calibration/test_validate_staging.py"
        ],
    )
)
def run_sanity_checks(
    h5_path: str,
    period: int = 2024,
) -> List[dict]:
    """Run structural integrity checks on an H5 file.

    Args:
        h5_path: Path to the H5 dataset file.
        period: Tax year (used for variable keys).

    Returns:
        List of {check, status, detail} dicts.
    """
    results = []

    def _get(f, path):
        """Resolve a slash path like 'var/2024' in the H5."""
        try:
            obj = f[path]
            if isinstance(obj, h5py.Dataset):
                return obj[:]
            return None
        except KeyError:
            return None

    def _get_person_weights(f, period, person_count, household_weights):
        if household_weights is None:
            return None
        if len(household_weights) == person_count:
            return household_weights

        person_hh_arr = _get(f, f"person_household_id/{period}")
        if person_hh_arr is None:
            person_hh_arr = _get(f, "person_household_id")
        hh_id_arr = _get(f, f"household_id/{period}")
        if hh_id_arr is None:
            hh_id_arr = _get(f, "household_id")
        if person_hh_arr is None or hh_id_arr is None:
            return None
        if len(hh_id_arr) != len(household_weights):
            return None

        household_weight_by_id = dict(zip(hh_id_arr.tolist(), household_weights))
        try:
            return np.array(
                [household_weight_by_id[hh_id] for hh_id in person_hh_arr.tolist()],
                dtype=float,
            )
        except KeyError:
            return None

    with h5py.File(h5_path, "r") as f:
        # 1. Weight non-negativity
        w_key = f"household_weight/{period}"
        weights = _get(f, w_key)
        if weights is not None:
            n_neg = int((weights < 0).sum())
            if n_neg > 0:
                results.append(
                    {
                        "check": "weight_non_negativity",
                        "status": "FAIL",
                        "detail": f"{n_neg} negative weights",
                    }
                )
            else:
                results.append(
                    {
                        "check": "weight_non_negativity",
                        "status": "PASS",
                        "detail": "",
                    }
                )
        else:
            results.append(
                {
                    "check": "weight_non_negativity",
                    "status": "SKIP",
                    "detail": f"key {w_key} not found",
                }
            )

        # 2. Entity ID uniqueness
        for entity in [
            "person",
            "household",
            "tax_unit",
            "spm_unit",
        ]:
            ids = _get(f, f"{entity}_id/{period}")
            if ids is None:
                ids = _get(f, f"{entity}_id")
            if ids is not None:
                n_dup = len(ids) - len(np.unique(ids))
                if n_dup > 0:
                    results.append(
                        {
                            "check": f"{entity}_id_uniqueness",
                            "status": "FAIL",
                            "detail": f"{n_dup} duplicate IDs",
                        }
                    )
                else:
                    results.append(
                        {
                            "check": f"{entity}_id_uniqueness",
                            "status": "PASS",
                            "detail": "",
                        }
                    )

        # 3. No NaN/Inf in key monetary variables
        for var in KEY_MONETARY_VARS:
            vals = _get(f, f"{var}/{period}")
            if vals is None:
                continue
            n_nan = int(np.isnan(vals).sum())
            n_inf = int(np.isinf(vals).sum())
            if n_nan > 0 or n_inf > 0:
                results.append(
                    {
                        "check": f"no_nan_inf_{var}",
                        "status": "FAIL",
                        "detail": f"{n_nan} NaN, {n_inf} Inf",
                    }
                )
            else:
                results.append(
                    {
                        "check": f"no_nan_inf_{var}",
                        "status": "PASS",
                        "detail": "",
                    }
                )

        # 4. Person-to-household mapping
        person_hh_arr = _get(f, f"person_household_id/{period}")
        if person_hh_arr is None:
            person_hh_arr = _get(f, "person_household_id")
        hh_id_arr = _get(f, f"household_id/{period}")
        if hh_id_arr is None:
            hh_id_arr = _get(f, "household_id")

        if person_hh_arr is not None and hh_id_arr is not None:
            person_hh = set(person_hh_arr.tolist())
            hh_ids = set(hh_id_arr.tolist())
            orphans = person_hh - hh_ids
            if orphans:
                results.append(
                    {
                        "check": "person_household_mapping",
                        "status": "FAIL",
                        "detail": (
                            f"{len(orphans)} persons map to non-existent households"
                        ),
                    }
                )
            else:
                results.append(
                    {
                        "check": "person_household_mapping",
                        "status": "PASS",
                        "detail": "",
                    }
                )

        # 5. Boolean takeup variables
        for var in TAKEUP_VARS:
            vals = _get(f, f"{var}/{period}")
            if vals is None:
                continue
            unique = set(np.unique(vals).tolist())
            valid = {True, False, 0, 1, 0.0, 1.0}
            bad = unique - valid
            if bad:
                results.append(
                    {
                        "check": f"boolean_takeup_{var}",
                        "status": "FAIL",
                        "detail": (f"unexpected values: {bad}"),
                    }
                )
            else:
                results.append(
                    {
                        "check": f"boolean_takeup_{var}",
                        "status": "PASS",
                        "detail": "",
                    }
                )

        # 6. Reasonable per-capita ranges
        if weights is not None:
            total_hh = weights.sum()
            if total_hh > 0:
                emp = _get(f, f"employment_income/{period}")
                if emp is not None:
                    total_emp = (emp * weights).sum()
                    per_hh = total_emp / total_hh
                    if per_hh < 10_000 or per_hh > 200_000:
                        results.append(
                            {
                                "check": "per_hh_employment_income",
                                "status": "WARN",
                                "detail": (f"${per_hh:,.0f}/hh (expected $10K-$200K)"),
                            }
                        )
                    else:
                        results.append(
                            {
                                "check": "per_hh_employment_income",
                                "status": "PASS",
                                "detail": f"${per_hh:,.0f}/hh",
                            }
                        )

                snap_arr = _get(f, f"snap/{period}")
                if snap_arr is not None:
                    total_snap = (snap_arr * weights).sum()
                    per_hh_snap = total_snap / total_hh
                    if per_hh_snap < 0 or per_hh_snap > 10_000:
                        results.append(
                            {
                                "check": "per_hh_snap",
                                "status": "WARN",
                                "detail": (
                                    f"${per_hh_snap:,.0f}/hh (expected $0-$10K)"
                                ),
                            }
                        )
                    else:
                        results.append(
                            {
                                "check": "per_hh_snap",
                                "status": "PASS",
                                "detail": f"${per_hh_snap:,.0f}/hh",
                            }
                        )

        employment_income = _get(f, f"employment_income/{period}")
        hourly_wage = _get(f, f"hourly_wage/{period}")
        hours_worked_last_week = _get(f, f"hours_worked_last_week/{period}")
        is_paid_hourly = _get(f, f"is_paid_hourly/{period}")
        hourly_inputs = [
            employment_income,
            hourly_wage,
            hours_worked_last_week,
            is_paid_hourly,
        ]
        if any(value is None for value in hourly_inputs):
            results.append(
                {
                    "check": "hourly_wage_income_consistency",
                    "status": "SKIP",
                    "detail": "missing one or more hourly wage consistency inputs",
                }
            )
            results.append(
                {
                    "check": "hourly_wage_income_consistency_overtime",
                    "status": "SKIP",
                    "detail": "missing one or more hourly wage consistency inputs",
                }
            )
        else:
            person_weights = _get_person_weights(
                f,
                period,
                len(employment_income),
                weights,
            )
            results.extend(
                build_hourly_wage_income_consistency_diagnostics(
                    employment_income=employment_income,
                    hourly_wage=hourly_wage,
                    hours_worked_last_week=hours_worked_last_week,
                    is_paid_hourly=is_paid_hourly,
                    weights=person_weights,
                )
            )

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run structural sanity checks on an H5 file"
    )
    parser.add_argument("h5_path", help="Path to the H5 file")
    parser.add_argument(
        "--period",
        type=int,
        default=2024,
        help="Tax year (default: 2024)",
    )
    args = parser.parse_args()

    results = run_sanity_checks(args.h5_path, args.period)

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_warn = sum(1 for r in results if r["status"] == "WARN")

    for r in results:
        icon = "PASS" if r["status"] == "PASS" else r["status"]
        detail = f" — {r['detail']}" if r["detail"] else ""
        print(f"  [{icon}] {r['check']}{detail}")

    print(f"\n{len(results)} checks: {n_fail} failures, {n_warn} warnings")
    if n_fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
