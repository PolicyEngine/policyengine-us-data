# Long Term Projections
## Integrating Economic Uprating with Demographic Reweighting

## Executive Summary

This document outlines an innovative approach for projecting federal income tax revenue through 2100 that uniquely combines sophisticated economic microsimulation with demographic reweighting. By harmonizing PolicyEngine's state-of-the-art tax modeling with Social Security Administration demographic projections, we can isolate and quantify the fiscal impact of population aging while preserving the full complexity of the tax code.

## The Challenge

Projecting tax revenue over a 75-year horizon requires simultaneously modeling two distinct but interrelated dynamics:

**Economic Evolution**: How incomes, prices, and tax parameters change over time
- Wage growth and income distribution shifts
- Inflation affecting brackets and deductions
- Legislative changes and indexing rules
- Behavioral responses to tax policy

**Demographic Transformation**: How the population structure evolves
- Baby boom generation aging through retirement
- Declining birth rates reducing working-age population
- Increasing longevity extending retirement duration
- Shifting household composition patterns

Traditional approaches typically sacrifice either economic sophistication (using simplified tax calculations) or demographic realism (holding age distributions constant). Our methodology preserves both.

## Running Projections

Run projections using `run_household_projection.py`:

```bash
# Save calibrated datasets as .h5 files for each year
python ../policyengine_us_data/datasets/cps/long_term/run_household_projection.py 2027 --greg --use-ss --use-payroll --save-h5
```

**Arguments:**
- `END_YEAR`: Target year for projection (default: 2035)
- `--greg`: Use GREG calibration instead of IPF (optional)
- `--use-ss`: Include Social Security benefit totals as calibration target (requires --greg)
- `--use-payroll`: Include taxable payroll as calibration target (requires --greg)
- `--save-h5`: Save year-specific .h5 files to `./projected_datasets/` directory

## Data Sources

The long-term projections use two key SSA datasets:

1. **SSA Population Projections** (`SSPopJul_TR2024.csv`)
   - Source: [SSA 2024 Trustees Report - Single Year Age Demographic Projections](https://www.ssa.gov/oact/HistEst/Population/2024/Population2024.html)
   - Contains age-specific population projections through 2100
   - Used for demographic reweighting to match future population structure

2. **Social Security Cost Projections** (`social_security_aux.csv`)
   - Source: [SSA 2025 Trustees Report, Table VI.G9](https://www.ssa.gov/oact/TR/2025/index.html)
   - Contains OASDI benefit cost projections in CPI-indexed 2025 dollars
   - Used as calibration target in GREG method to ensure fiscal consistency

### Loading SSA Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from policyengine_us_data.storage import STORAGE_FOLDER

# Load SSA population data
ssa_pop = pd.read_csv(STORAGE_FOLDER / 'SSPopJul_TR2024.csv')
ssa_pop.head()

# Load Social Security auxiliary data
ss_aux = pd.read_csv(STORAGE_FOLDER / 'social_security_aux.csv')
ss_aux.head()
```

## Core Innovation

Our approach operates in two complementary stages:

### Stage 1: Economic Uprating

PolicyEngine's microsimulation engine projects each household's economic circumstances forward using:

**Sophisticated Income Modeling**

The system models 17 distinct income categories, each uprated according to its economic fundamentals:

*Primary Categories with Specific Projections:*
- Employment income (wages) - follows CBO wage growth projections
- Self-employment income - follows CBO business income projections
- Capital gains - follows CBO asset appreciation projections
- Interest income - follows CBO interest rate projections
- Dividend income - follows CBO corporate profit projections
- Pension income - follows CBO retirement income projections
- Social Security - follows SSA COLA projections (available through 2100)

### Stage 2: Demographic Reweighting

We offer two calibration methods for adjusting household weights to match SSA projections:

**Method 1: Iterative Proportional Fitting (IPF)**
- Traditional raking approach using Kullback-Leibler divergence
- Iteratively adjusts weights to match marginal distributions
- Robust to specification and always produces non-negative weights
- Default method for backward compatibility

**Method 2: Generalized Regression (GREG) Calibration**
- Modern calibration using chi-squared distance minimization
- Enables simultaneous calibration to categorical AND continuous variables
- Direct solution via matrix operations (no iteration needed)
- Required for incorporating Social Security benefit constraints

## Demonstrating the Calibration Methods

```python
from policyengine_us_data.datasets.cps.long_term.ssa_data import (
    load_ssa_age_projections,
    load_ssa_benefit_projections
)

# Get SSA population targets for a specific year
year = 2025
age_targets = load_ssa_age_projections(end_year=year)
print(f"\nAge distribution targets for {year}:")
print(f"Shape: {age_targets.shape}")
print(f"Total population: {age_targets[:, 0].sum() / 1000:.1f}M")

# Get Social Security benefit target
ss_target = load_ssa_benefit_projections(year)
print(f"\nSocial Security benefit target for {year}: ${ss_target / 1e9:.1f}B")
```

## PWBM Analysis: Eliminating Income Taxes on Social Security Benefits

**Source:** [Eliminating Income Taxes on Social Security Benefits](https://budgetmodel.wharton.upenn.edu/issues/2025/2/10/eliminating-income-taxes-on-social-security-benefits) (Penn Wharton Budget Model, February 10, 2025)

---

### Policy Analyzed
The Penn Wharton Budget Model (PWBM) analyzed a policy proposal to permanently eliminate all income taxes on Social Security benefits, effective January 1, 2025.

### Key Findings

* **Budgetary Impact:** The policy is projected to reduce federal revenues by **$1.45 trillion** over the 10-year budget window (2025-2034). Over the long term, it is projected to increase federal debt by 7 percent by 2054, relative to the current baseline.

* **Macroeconomic Impact:** The analysis finds the policy would have negative long-term effects on the economy.
    * It reduces incentives for households to save for retirement and to work.
    * This leads to a smaller capital stock (projected to be 4.2% lower by 2054).
    * The smaller capital stock results in lower average wages (1.8% lower by 2054) and lower GDP (2.1% lower by 2054).

* **Conventional Distributional Impact (Your Table):** The table you shared shows the annual "conventional" effects on household after-tax income.
    * The largest average *dollar* tax cuts go to households in the top 20 percent of the income distribution (quintiles 80-100%).
    * The largest *relative* gains (as a percentage of income) go to households in the fourth quintile (60-80%), who see a 1.6% increase in after-tax income by 2054.
    * The dollar amounts shown are in **nominal dollars** for each specified year, not adjusted to a single base year.

* **Dynamic (Lifetime) Impact:** When analyzing the policy's effects over a household's entire lifetime, PWBM finds:
    * The policy primarily benefits high-income households who are nearing or in retirement.
    * It negatively impacts all households under the age of 30 and all future generations, who would experience a net welfare loss due to the long-term effects of lower wages and higher federal debt.

## PolicyEngine's Analysis of Eliminating Income Taxes on Social Security Benefits

```python
import sys
import os
import pandas as pd
import numpy as np
import gc

from policyengine_us import Microsimulation
from policyengine_core.reforms import Reform

WHARTON_BENCHMARKS = {
    2026: {
        'First quintile': {'tax_change': 0, 'pct_change': 0.0},
        'Second quintile': {'tax_change': -15, 'pct_change': 0.0},
        'Middle quintile': {'tax_change': -340, 'pct_change': 0.5},
        'Fourth quintile': {'tax_change': -1135, 'pct_change': 1.1},
        '80-90%': {'tax_change': -1625, 'pct_change': 1.0},
        '90-95%': {'tax_change': -1590, 'pct_change': 0.7},
        '95-99%': {'tax_change': -2020, 'pct_change': 0.5},
        '99-99.9%': {'tax_change': -2205, 'pct_change': 0.2},
        'Top 0.1%': {'tax_change': -2450, 'pct_change': 0.0},
    },
    2034: {
        'First quintile': {'tax_change': 0, 'pct_change': 0.0},
        'Second quintile': {'tax_change': -45, 'pct_change': 0.1},
        'Middle quintile': {'tax_change': -615, 'pct_change': 0.8},
        'Fourth quintile': {'tax_change': -1630, 'pct_change': 1.2},
        '80-90%': {'tax_change': -2160, 'pct_change': 1.1},
        '90-95%': {'tax_change': -2160, 'pct_change': 0.7},
        '95-99%': {'tax_change': -2605, 'pct_change': 0.6},
        '99-99.9%': {'tax_change': -2715, 'pct_change': 0.2},
        'Top 0.1%': {'tax_change': -2970, 'pct_change': 0.0},
    },
    2054: {
        'First quintile': {'tax_change': -5, 'pct_change': 0.0},
        'Second quintile': {'tax_change': -275, 'pct_change': 0.3},
        'Middle quintile': {'tax_change': -1730, 'pct_change': 1.3},
        'Fourth quintile': {'tax_change': -3560, 'pct_change': 1.6},
        '80-90%': {'tax_change': -4075, 'pct_change': 1.2},
        '90-95%': {'tax_change': -4385, 'pct_change': 0.9},
        '95-99%': {'tax_change': -4565, 'pct_change': 0.6},
        '99-99.9%': {'tax_change': -4820, 'pct_change': 0.2},
        'Top 0.1%': {'tax_change': -5080, 'pct_change': 0.0},
    },
}

def run_analysis(dataset_path, year, income_rank_var = "household_net_income"):
    """Run Option 1 analysis for given dataset and year"""

    option1_reform = Reform.from_dict(
        {
            # Base rate parameters (0-50% bracket)
            "gov.irs.social_security.taxability.rate.base.benefit_cap": {
                "2026-01-01.2100-12-31": 0
            },
            "gov.irs.social_security.taxability.rate.base.excess": {
                "2026-01-01.2100-12-31": 0
            },
            # Additional rate parameters (50-85% bracket)
            "gov.irs.social_security.taxability.rate.additional.benefit_cap": {
                "2026-01-01.2100-12-31": 0
            },
            "gov.irs.social_security.taxability.rate.additional.bracket": {
                "2026-01-01.2100-12-31": 0
            },
            "gov.irs.social_security.taxability.rate.additional.excess": {
                "2026-01-01.2100-12-31": 0
            }
        }, country_id="us"
    )
    reform = Microsimulation(dataset=dataset_path, reform=option1_reform)

    # Get household data
    household_net_income_reform = reform.calculate("household_net_income", period=year, map_to="household")
    household_agi_reform = reform.calculate("adjusted_gross_income", period=year, map_to="household")
    income_tax_reform = reform.calculate("income_tax", period=year, map_to="household")

    del reform
    gc.collect()

    print(f"Loading dataset: {dataset_path}")
    baseline = Microsimulation(dataset=dataset_path)
    household_weight = baseline.calculate("household_weight", period=year)
    household_net_income_baseline = baseline.calculate("household_net_income", period=year, map_to="household")
    household_agi_baseline = baseline.calculate("adjusted_gross_income", period=year, map_to="household")
    income_tax_baseline = baseline.calculate("income_tax", period=year, map_to="household")

    # Calculate changes
    tax_change = income_tax_reform - income_tax_baseline
    income_change_pct = (
        (household_net_income_reform - household_net_income_baseline) / household_net_income_baseline
    ) * 100

    # Create DataFrame
    df = pd.DataFrame({
        'household_net_income': household_net_income_baseline,
        'weight': household_weight,
        'tax_change': tax_change,
        'income_change_pct': income_change_pct,
        'income_rank_var': baseline.calculate(income_rank_var, year, map_to="household")
    })

    # Calculate percentiles

    print(f"Ranking according to quantiles with: {income_rank_var}")
    df['income_percentile'] = df['income_rank_var'].rank(pct=True) * 100

    # Assign income groups
    def assign_income_group(percentile):
        if percentile <= 20:
            return 'First quintile'
        elif percentile <= 40:
            return 'Second quintile'
        elif percentile <= 60:
            return 'Middle quintile'
        elif percentile <= 80:
            return 'Fourth quintile'
        elif percentile <= 90:
            return '80-90%'
        elif percentile <= 95:
            return '90-95%'
        elif percentile <= 99:
            return '95-99%'
        elif percentile <= 99.9:
            return '99-99.9%'
        else:
            return 'Top 0.1%'

    df['income_group'] = df['income_percentile'].apply(assign_income_group)

    # Calculate aggregate revenue
    revenue_impact = (income_tax_reform.sum() - income_tax_baseline.sum()) / 1e9

    # Calculate by group
    results = []
    for group in ['First quintile', 'Second quintile', 'Middle quintile', 'Fourth quintile',
                  '80-90%', '90-95%', '95-99%', '99-99.9%', 'Top 0.1%']:
        group_data = df[df['income_group'] == group]
        if len(group_data) == 0:
            continue

        total_weight = group_data['weight'].sum()
        avg_tax_change = (group_data['tax_change'] * group_data['weight']).sum() / total_weight
        avg_income_change_pct = (group_data['income_change_pct'] * group_data['weight']).sum() / total_weight

        results.append({
            'group': group,
            'pe_tax_change': round(avg_tax_change),
            'pe_pct_change': round(avg_income_change_pct, 1),
        })

    return pd.DataFrame(results), revenue_impact

def generate_comparison_table(pe_results, year):
    """Generate comparison table with Wharton benchmark"""

    if year not in WHARTON_BENCHMARKS:
        print(f"Warning: No Wharton benchmark available for year {year}")
        return pe_results

    wharton_data = WHARTON_BENCHMARKS[year]

    comparison = []
    for _, row in pe_results.iterrows():
        group = row['group']
        wharton = wharton_data.get(group, {'tax_change': None, 'pct_change': None})

        pe_tax = row['pe_tax_change']
        wh_tax = wharton['tax_change']

        comparison.append({
            'Income Group': group,
            'PolicyEngine': f"${pe_tax:,}",
            'Wharton': f"${wh_tax:,}" if wh_tax is not None else 'N/A',
            'Difference': f"${(pe_tax - wh_tax):,}" if wh_tax is not None else 'N/A',
            'PE %': f"{row['pe_pct_change']}%",
            'Wharton %': f"{wharton['pct_change']}%" if wharton['pct_change'] is not None else 'N/A',
        })

    return pd.DataFrame(comparison)

# Example usage:
# dataset_path = 'hf://policyengine/test/2054.h5'
# year = 2054
# income_rank_variable = "household_net_income"
# pe_results, revenue_impact = run_analysis(dataset_path, year, income_rank_variable)
# comparison_table = generate_comparison_table(pe_results, year)
```
