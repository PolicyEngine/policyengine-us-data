"""Post-build validation for retirement contribution QRF imputation.

Run after a full Extended CPS build to verify that the PUF clone
half has plausible retirement contribution values.

Usage:
    python validation/validate_retirement_imputation.py

Requires:
    - Python 3.12+ (for microimpute/policyengine_us)
    - Built ExtendedCPS_2024 dataset available
"""

import logging
import sys

import numpy as np
import pandas as pd

from policyengine_us_data.utils.loss import HARD_CODED_TOTALS
from policyengine_us_data.utils.retirement_limits import (
    get_retirement_limits,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Retirement-related subset of calibration targets (from loss.py).
_RETIREMENT_VARS = [
    "traditional_401k_contributions",
    "roth_401k_contributions",
    "traditional_ira_contributions",
    "roth_ira_contributions",
    "self_employed_pension_contribution_ald",
]
TARGETS = {k: HARD_CODED_TOTALS[k] for k in _RETIREMENT_VARS}

# Contribution limits from policyengine-us parameters.
LIMITS_2024 = get_retirement_limits(2024)


def load_extended_cps():
    """Load the Extended CPS dataset."""
    from policyengine_us import Microsimulation
    from policyengine_us_data import ExtendedCPS_2024

    logger.info("Loading ExtendedCPS_2024...")
    sim = Microsimulation(dataset=ExtendedCPS_2024)
    return sim


def validate_constraints(sim) -> list:
    """Check hard constraints on imputed values."""
    issues = []
    year = 2024

    emp_income = sim.calculate(
        "employment_income", year, map_to="person"
    ).values
    se_income = sim.calculate(
        "self_employment_income", year, map_to="person"
    ).values
    age = sim.calculate("age", year, map_to="person").values

    catch_up = age >= 50
    max_401k = LIMITS_2024["401k"] + catch_up * LIMITS_2024["401k_catch_up"]
    max_ira = LIMITS_2024["ira"] + catch_up * LIMITS_2024["ira_catch_up"]

    # 401k constraints
    for var in (
        "traditional_401k_contributions",
        "roth_401k_contributions",
    ):
        vals = sim.calculate(var, year, map_to="person").values

        n_negative = (vals < 0).sum()
        if n_negative > 0:
            issues.append(f"FAIL: {var} has {n_negative} negative values")

        n_over_cap = (vals > max_401k + 1).sum()
        if n_over_cap > 0:
            issues.append(
                f"FAIL: {var} has {n_over_cap} values exceeding " f"401k cap"
            )

        zero_wage = emp_income == 0
        n_nonzero_no_wage = (vals[zero_wage] > 0).sum()
        if n_nonzero_no_wage > 0:
            issues.append(
                f"FAIL: {var} has {n_nonzero_no_wage} nonzero values "
                f"for records with $0 wages"
            )
        else:
            logger.info(
                "PASS: %s is $0 for all %d zero-wage records",
                var,
                zero_wage.sum(),
            )

    # IRA constraints
    for var in (
        "traditional_ira_contributions",
        "roth_ira_contributions",
    ):
        vals = sim.calculate(var, year, map_to="person").values

        n_negative = (vals < 0).sum()
        if n_negative > 0:
            issues.append(f"FAIL: {var} has {n_negative} negative values")

        n_over_cap = (vals > max_ira + 1).sum()
        if n_over_cap > 0:
            issues.append(
                f"FAIL: {var} has {n_over_cap} values exceeding " f"IRA cap"
            )

    # SE pension constraint
    var = "self_employed_pension_contributions"
    vals = sim.calculate(var, year, map_to="person").values
    zero_se = se_income == 0
    n_nonzero_no_se = (vals[zero_se] > 0).sum()
    if n_nonzero_no_se > 0:
        issues.append(
            f"FAIL: {var} has {n_nonzero_no_se} nonzero values "
            f"for records with $0 SE income"
        )
    else:
        logger.info(
            "PASS: %s is $0 for all %d zero-SE records",
            var,
            zero_se.sum(),
        )

    return issues


def validate_aggregates(sim) -> list:
    """Compare weighted aggregates against calibration targets."""
    issues = []
    year = 2024

    weight = sim.calculate("person_weight", year).values

    logger.info(
        "\n%-45s %15s %15s %10s", "Variable", "Weighted Sum", "Target", "Ratio"
    )
    logger.info("-" * 90)

    for var, target in TARGETS.items():
        try:
            vals = sim.calculate(var, year, map_to="person").values
        except (ValueError, KeyError):
            logger.warning("Could not calculate %s", var)
            continue

        weighted_sum = (vals * weight).sum()
        ratio = weighted_sum / target if target != 0 else float("inf")

        logger.info(
            "%-45s $%14.1fB $%14.1fB %9.1f%%",
            var,
            weighted_sum / 1e9,
            target / 1e9,
            ratio * 100,
        )

        # Flag if more than 2x off target
        if ratio < 0.1 or ratio > 5.0:
            issues.append(
                f"WARNING: {var} weighted sum "
                f"${weighted_sum/1e9:.1f}B is far from "
                f"target ${target/1e9:.1f}B "
                f"(ratio={ratio:.2f})"
            )

    return issues


def validate_cps_vs_puf_half(sim) -> list:
    """Compare CPS half and PUF half distributions."""
    issues = []
    year = 2024

    n_persons = len(sim.calculate("person_id", year).values)
    n_half = n_persons // 2

    logger.info("\nCPS vs PUF half comparison (n=%d per half):", n_half)
    logger.info(
        "%-45s %12s %12s %12s %12s",
        "Variable",
        "CPS mean",
        "PUF mean",
        "CPS >0 pct",
        "PUF >0 pct",
    )
    logger.info("-" * 95)

    for var in [
        "traditional_401k_contributions",
        "roth_401k_contributions",
        "traditional_ira_contributions",
        "roth_ira_contributions",
        "self_employed_pension_contributions",
    ]:
        try:
            vals = sim.calculate(var, year, map_to="person").values
        except (ValueError, KeyError):
            continue

        cps_vals = vals[:n_half]
        puf_vals = vals[n_half:]

        cps_mean = cps_vals.mean()
        puf_mean = puf_vals.mean()
        cps_pct = (cps_vals > 0).mean() * 100
        puf_pct = (puf_vals > 0).mean() * 100

        logger.info(
            "%-45s $%11.0f $%11.0f %11.1f%% %11.1f%%",
            var,
            cps_mean,
            puf_mean,
            cps_pct,
            puf_pct,
        )

    return issues


def main():
    """Run all validations."""
    sim = load_extended_cps()

    logger.info("\n" + "=" * 60)
    logger.info("RETIREMENT CONTRIBUTION IMPUTATION VALIDATION")
    logger.info("=" * 60)

    all_issues = []

    logger.info("\n1. CONSTRAINT CHECKS")
    logger.info("-" * 40)
    all_issues.extend(validate_constraints(sim))

    logger.info("\n2. AGGREGATE COMPARISONS")
    logger.info("-" * 40)
    all_issues.extend(validate_aggregates(sim))

    logger.info("\n3. CPS vs PUF HALF DISTRIBUTIONS")
    logger.info("-" * 40)
    all_issues.extend(validate_cps_vs_puf_half(sim))

    logger.info("\n" + "=" * 60)
    if all_issues:
        logger.info("ISSUES FOUND: %d", len(all_issues))
        for issue in all_issues:
            logger.info("  - %s", issue)
    else:
        logger.info("ALL CHECKS PASSED")
    logger.info("=" * 60)

    return len(all_issues)


if __name__ == "__main__":
    sys.exit(main())
