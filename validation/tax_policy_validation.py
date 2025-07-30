"""
Validate Enhanced CPS against tax policy benchmarks.

This script compares key tax policy metrics from the Enhanced CPS
against published estimates from CBO, JCT, and TPC.
"""

import pandas as pd
import numpy as np
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS


def calculate_effective_tax_rates():
    """Calculate effective tax rates by income decile."""
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)

    # Get income and tax variables
    income = sim.calculate("adjusted_gross_income", 2022)
    fed_tax = sim.calculate("income_tax", 2022)
    weights = sim.calculate("household_weight", 2022)

    # Calculate deciles
    deciles = pd.qcut(income, 10, labels=False, weights=weights)

    # Calculate effective rates by decile
    results = []
    for d in range(10):
        mask = deciles == d
        total_income = (income[mask] * weights[mask]).sum()
        total_tax = (fed_tax[mask] * weights[mask]).sum()
        eff_rate = total_tax / total_income if total_income > 0 else 0

        results.append(
            {
                "decile": d + 1,
                "mean_income": income[mask].mean(),
                "effective_rate": eff_rate * 100,
                "total_tax": total_tax / 1e9,  # billions
            }
        )

    return pd.DataFrame(results)


def validate_tax_expenditures():
    """Validate major tax expenditure totals."""
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)

    expenditures = {
        "mortgage_interest": {
            "variable": "mortgage_interest_deduction",
            "jct_estimate": 24.8,  # billions
        },
        "charitable": {
            "variable": "charitable_deduction",
            "jct_estimate": 65.3,
        },
        "salt": {
            "variable": "salt_deduction",
            "jct_estimate": 21.2,
        },
        "medical": {
            "variable": "medical_expense_deduction",
            "jct_estimate": 11.4,
        },
    }

    results = []
    for name, info in expenditures.items():
        amount = sim.calculate(info["variable"], 2022)
        weights = sim.calculate("tax_unit_weight", 2022)
        total = (amount * weights).sum() / 1e9  # billions

        results.append(
            {
                "expenditure": name,
                "enhanced_cps": total,
                "jct_estimate": info["jct_estimate"],
                "difference": total - info["jct_estimate"],
                "pct_difference": (total - info["jct_estimate"])
                / info["jct_estimate"]
                * 100,
            }
        )

    return pd.DataFrame(results)


def analyze_high_income_taxpayers():
    """Analyze representation of high-income taxpayers."""
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)

    agi = sim.calculate("adjusted_gross_income", 2022)
    weights = sim.calculate("tax_unit_weight", 2022)

    # Income thresholds
    thresholds = [100_000, 200_000, 500_000, 1_000_000, 10_000_000]

    results = []
    for threshold in thresholds:
        count = (weights[agi >= threshold]).sum()
        pct_returns = count / weights.sum() * 100
        total_agi = (
            agi[agi >= threshold] * weights[agi >= threshold]
        ).sum() / 1e9

        results.append(
            {
                "threshold": f"${threshold:,}+",
                "returns": int(count),
                "pct_returns": pct_returns,
                "total_agi_billions": total_agi,
            }
        )

    return pd.DataFrame(results)


def validate_state_revenues():
    """Validate state tax revenues by state."""
    sim = Microsimulation(dataset=EnhancedCPS, dataset_year=2022)

    # Get state tax and identifiers
    state_tax = sim.calculate("state_income_tax", 2022)
    state_code = sim.calculate("state_code", 2022)
    weights = sim.calculate("tax_unit_weight", 2022)

    # Aggregate by state
    results = []
    for state in state_code.unique():
        if state > 0:  # Valid state codes
            mask = state_code == state
            total = (state_tax[mask] * weights[mask]).sum() / 1e9

            results.append({"state_code": state, "revenue_billions": total})

    return pd.DataFrame(results).sort_values(
        "revenue_billions", ascending=False
    )


def generate_validation_report():
    """Generate comprehensive validation report."""
    print("Enhanced CPS Tax Policy Validation Report")
    print("=" * 50)

    print("\n1. Effective Tax Rates by Income Decile")
    etr_df = calculate_effective_tax_rates()
    print(etr_df.to_string(index=False))

    print("\n\n2. Tax Expenditure Validation (billions)")
    exp_df = validate_tax_expenditures()
    print(exp_df.to_string(index=False))

    print("\n\n3. High-Income Taxpayer Representation")
    high_df = analyze_high_income_taxpayers()
    print(high_df.to_string(index=False))

    print("\n\n4. Top 10 States by Income Tax Revenue")
    state_df = validate_state_revenues()
    print(state_df.head(10).to_string(index=False))

    # Save to files
    etr_df.to_csv("validation/effective_tax_rates.csv", index=False)
    exp_df.to_csv("validation/tax_expenditures.csv", index=False)
    high_df.to_csv("validation/high_income_analysis.csv", index=False)
    state_df.to_csv("validation/state_revenues.csv", index=False)

    print("\n\nValidation results saved to validation/ directory")


if __name__ == "__main__":
    generate_validation_report()
