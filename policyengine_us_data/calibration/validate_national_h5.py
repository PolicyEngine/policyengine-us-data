"""Validate a national US.h5 file against reference values.

Loads the national H5, computes key variables, and compares to
known national totals. Also runs structural sanity checks.

Usage:
    python -m policyengine_us_data.calibration.validate_national_h5
    python -m policyengine_us_data.calibration.validate_national_h5 \
        --h5-path path/to/US.h5
    python -m policyengine_us_data.calibration.validate_national_h5 \
        --hf-path hf://policyengine/policyengine-us-data/national/US.h5
"""

import argparse

VARIABLES = [
    "adjusted_gross_income",
    "employment_income",
    "self_employment_income",
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
    "refundable_ctc",
    "real_estate_taxes",
    "rent",
    "is_pregnant",
    "person_count",
    "household_count",
]

REFERENCES = {
    "person_count": (335_000_000, "~335M"),
    "household_count": (130_000_000, "~130M"),
    "adjusted_gross_income": (15_000_000_000_000, "~$15T"),
    "employment_income": (10_000_000_000_000, "~$10T"),
    "social_security": (1_200_000_000_000, "~$1.2T"),
    "snap": (110_000_000_000, "~$110B"),
    "ssi": (60_000_000_000, "~$60B"),
    "eitc": (60_000_000_000, "~$60B"),
    "refundable_ctc": (120_000_000_000, "~$120B"),
    "income_tax_before_credits": (4_000_000_000_000, "~$4T"),
}

DEFAULT_HF_PATH = "hf://policyengine/policyengine-us-data/national/US.h5"

COUNT_VARS = {"person_count", "household_count", "is_pregnant"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate national US.h5")
    parser.add_argument(
        "--h5-path",
        default=None,
        help="Local path to US.h5",
    )
    parser.add_argument(
        "--hf-path",
        default=DEFAULT_HF_PATH,
        help=f"HF path to US.h5 (default: {DEFAULT_HF_PATH})",
    )
    args = parser.parse_args(argv)

    dataset_path = args.h5_path or args.hf_path

    from policyengine_us import Microsimulation

    print(f"Loading {dataset_path}...")
    sim = Microsimulation(dataset=dataset_path)

    n_hh = sim.calculate("household_id", map_to="household").shape[0]
    print(f"Households in file: {n_hh:,}")

    print("\n" + "=" * 70)
    print("NATIONAL H5 VALUES")
    print("=" * 70)

    values = {}
    failures = []
    for var in VARIABLES:
        try:
            val = float(sim.calculate(var).sum())
            values[var] = val
            if var in COUNT_VARS:
                print(f"  {var:45s} {val:>15,.0f}")
            else:
                print(f"  {var:45s} ${val:>15,.0f}")
        except Exception as e:
            failures.append((var, str(e)))
            print(f"  {var:45s} FAILED: {e}")

    print("\n" + "=" * 70)
    print("COMPARISON TO REFERENCE VALUES")
    print("=" * 70)

    any_flag = False
    for var, (ref_val, ref_label) in REFERENCES.items():
        if var not in values:
            continue
        val = values[var]
        pct_diff = (val - ref_val) / ref_val * 100
        flag = " ***" if abs(pct_diff) > 30 else ""
        if flag:
            any_flag = True
        if var in COUNT_VARS:
            print(
                f"  {var:35s} {val:>15,.0f}  "
                f"ref {ref_label:>8s}  "
                f"({pct_diff:+.1f}%){flag}"
            )
        else:
            print(
                f"  {var:35s} ${val:>15,.0f}  "
                f"ref {ref_label:>8s}  "
                f"({pct_diff:+.1f}%){flag}"
            )

    if any_flag:
        print("\n*** = >30% deviation from reference. " "Investigate further.")

    if failures:
        print(f"\n{len(failures)} variables failed:")
        for var, err in failures:
            print(f"  {var}: {err}")

    print("\n" + "=" * 70)
    print("STRUCTURAL CHECKS")
    print("=" * 70)

    from policyengine_us_data.calibration.sanity_checks import (
        run_sanity_checks,
    )

    results = run_sanity_checks(dataset_path)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    for r in results:
        icon = (
            "PASS"
            if r["status"] == "PASS"
            else "FAIL" if r["status"] == "FAIL" else "WARN"
        )
        print(f"  [{icon}] {r['check']}: {r['detail']}")

    print(f"\n{n_pass}/{len(results)} passed, {n_fail} failed")

    return 0 if n_fail == 0 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
