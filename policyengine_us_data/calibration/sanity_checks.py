"""Structural integrity checks for calibrated H5 files.

Run standalone:
    python -m policyengine_us_data.calibration.sanity_checks path/to/file.h5

Or integrated via validate_staging.py --sanity-only.
"""

import logging
from typing import List

import h5py
import numpy as np

logger = logging.getLogger(__name__)

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
    "takes_up_aca_ptc_if_eligible",
    "takes_up_medicaid_if_eligible",
    "takes_up_tanf_if_eligible",
    "takes_up_head_start_if_eligible",
    "takes_up_early_head_start_if_eligible",
    "takes_up_dc_property_tax_credit_if_eligible",
]


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

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        # 1. Weight non-negativity
        w_key = f"household_weight/{period}"
        if w_key in keys:
            weights = f[w_key][:]
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
            id_key = f"{entity}_id/{period}"
            if id_key not in keys:
                id_key = f"{entity}_id"
            if id_key in keys:
                ids = f[id_key][:]
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
            var_key = f"{var}/{period}"
            if var_key not in keys:
                continue
            vals = f[var_key][:]
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
        phh_key = f"person_household_id/{period}"
        hh_key = f"household_id/{period}"
        if phh_key not in keys:
            phh_key = "person_household_id"
        if hh_key not in keys:
            hh_key = "household_id"

        if phh_key in keys and hh_key in keys:
            person_hh = set(f[phh_key][:].tolist())
            hh_ids = set(f[hh_key][:].tolist())
            orphans = person_hh - hh_ids
            if orphans:
                results.append(
                    {
                        "check": "person_household_mapping",
                        "status": "FAIL",
                        "detail": (
                            f"{len(orphans)} persons map to "
                            "non-existent households"
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
            var_key = f"{var}/{period}"
            if var_key not in keys:
                continue
            vals = f[var_key][:]
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
        w_key = f"household_weight/{period}"
        emp_key = f"employment_income/{period}"
        snap_key = f"snap/{period}"
        if w_key in keys:
            weights = f[w_key][:]
            total_hh = weights.sum()
            if total_hh > 0:
                if emp_key in keys:
                    emp = f[emp_key][:]
                    total_emp = (emp * weights).sum()
                    per_hh = total_emp / total_hh
                    if per_hh < 10_000 or per_hh > 200_000:
                        results.append(
                            {
                                "check": ("per_hh_employment_income"),
                                "status": "WARN",
                                "detail": (
                                    f"${per_hh:,.0f}/hh "
                                    "(expected $10K-$200K)"
                                ),
                            }
                        )
                    else:
                        results.append(
                            {
                                "check": ("per_hh_employment_income"),
                                "status": "PASS",
                                "detail": f"${per_hh:,.0f}/hh",
                            }
                        )

                if snap_key in keys:
                    snap = f[snap_key][:]
                    total_snap = (snap * weights).sum()
                    per_hh_snap = total_snap / total_hh
                    if per_hh_snap < 0 or per_hh_snap > 10_000:
                        results.append(
                            {
                                "check": "per_hh_snap",
                                "status": "WARN",
                                "detail": (
                                    f"${per_hh_snap:,.0f}/hh "
                                    "(expected $0-$10K)"
                                ),
                            }
                        )
                    else:
                        results.append(
                            {
                                "check": "per_hh_snap",
                                "status": "PASS",
                                "detail": (f"${per_hh_snap:,.0f}/hh"),
                            }
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

    print(f"\n{len(results)} checks: " f"{n_fail} failures, {n_warn} warnings")
    if n_fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
