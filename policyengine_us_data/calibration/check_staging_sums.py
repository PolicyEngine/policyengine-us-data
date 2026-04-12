"""Sum key variables across all staging state H5 files.

Quick smoke test: loads all 51 state H5s, sums key variables,
compares to national references. No database needed. ~10 min runtime.

Usage:
    python -m policyengine_us_data.calibration.check_staging_sums
    python -m policyengine_us_data.calibration.check_staging_sums \
        --hf-prefix hf://policyengine/policyengine-us-data/staging
"""

import argparse

import pandas as pd

from policyengine_us_data.calibration.calibration_utils import (
    STATE_CODES,
)
from policyengine_us_data.db.etl_irs_soi import get_national_geography_soi_target

STATE_ABBRS = sorted(STATE_CODES.values())

VARIABLES = [
    "adjusted_gross_income",
    "employment_income",
    "total_self_employment_income",
    "tax_unit_partnership_s_corp_income",
    "taxable_pension_income",
    "dividend_income",
    "net_capital_gains",
    "rental_income",
    "taxable_interest_income",
    "social_security",
    "snap",
    "ssi",
    "income_tax_before_credits",
    "eitc",
    "non_refundable_ctc",
    "refundable_ctc",
    "real_estate_taxes",
    "rent",
    "is_pregnant",
    "person_count",
    "household_count",
]

DEFAULT_HF_PREFIX = "hf://policyengine/policyengine-us-data/staging/states"


def get_reference_summary(reference_year: int = 2024) -> str:
    refundable_ctc_target = get_national_geography_soi_target(
        "refundable_ctc",
        reference_year,
    )
    non_refundable_ctc_target = get_national_geography_soi_target(
        "non_refundable_ctc",
        reference_year,
    )
    return (
        "  SNAP ~$110B, SSI ~$60B, Social Security ~$1.2T\n"
        f"  EITC ~$60B, refundable CTC ~${refundable_ctc_target['amount'] / 1e9:.1f}B "
        f"(IRS SOI {refundable_ctc_target['source_year']}), "
        f"non-refundable CTC ~${non_refundable_ctc_target['amount'] / 1e9:.1f}B "
        f"(IRS SOI {non_refundable_ctc_target['source_year']})"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sum key variables across staging state H5 files"
    )
    parser.add_argument(
        "--hf-prefix",
        default=DEFAULT_HF_PREFIX,
        help=f"HF path prefix for state H5 files (default: {DEFAULT_HF_PREFIX})",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run ID to scope HF staging prefix (e.g. staging/{run_id}/states/...)",
    )
    args = parser.parse_args(argv)
    if args.run_id and args.hf_prefix == DEFAULT_HF_PREFIX:
        args.hf_prefix = (
            f"hf://policyengine/policyengine-us-data/staging/{args.run_id}/states"
        )

    from policyengine_us import Microsimulation

    results = {}
    errors = []

    for i, st in enumerate(STATE_ABBRS):
        print(
            f"[{i + 1}/{len(STATE_ABBRS)}] {st}...",
            end=" ",
            flush=True,
        )
        try:
            sim = Microsimulation(dataset=f"{args.hf_prefix}/{st}.h5")
            row = {}
            for var in VARIABLES:
                try:
                    row[var] = float(sim.calculate(var).sum())
                except Exception:
                    row[var] = None
            results[st] = row
            print("OK")
        except Exception as e:
            errors.append((st, str(e)))
            print(f"FAILED: {e}")

    df = pd.DataFrame(results).T
    df.index.name = "state"

    print("\n" + "=" * 70)
    print("NATIONAL TOTALS (sum across all states)")
    print("=" * 70)

    totals = df.sum()
    for var in VARIABLES:
        val = totals[var]
        if var in ("person_count", "household_count", "is_pregnant"):
            print(f"  {var:45s} {val:>15,.0f}")
        else:
            print(f"  {var:45s} ${val:>15,.0f}")

    print("\n" + "=" * 70)
    print("REFERENCE VALUES (approximate, for sanity checking)")
    print("=" * 70)
    print("  US GDP ~$29T, US population ~335M, ~130M households")
    print("  Total AGI ~$15T, Employment income ~$10T")
    print(get_reference_summary())

    if errors:
        print(f"\n{len(errors)} states failed:")
        for st, err in errors:
            print(f"  {st}: {err}")

    print("\nPer-state details saved to staging_sums.csv")
    df.to_csv("staging_sums.csv")


if __name__ == "__main__":
    main()
