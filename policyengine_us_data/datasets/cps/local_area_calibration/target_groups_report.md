# Calibration Target Groups Report

Generated: 2026-02-07

This document describes all target groups used in `create_calibration_package.py` for the X_sparse calibration matrix.

## Overview

- **Total groups created**: 61 (groups 0-60)
- **Groups excluded**: 22 group IDs
- **After filtering**: 39 included groups

### Full Run (436 CDs):
- Total targets before filtering: ~33,600
- After filtering: ~21,500 targets
- Matrix shape: (~21,500 targets x ~5,889,052 households)

## Exclusion List (from create_calibration_package.py)

```python
groups_to_exclude = [
    0,   # National Alimony Expense
    1,   # National Alimony Income
    2,   # National Charitable Deduction
    3,   # National Child Support Expense
    4,   # National Child Support Received
    8,   # National Interest Deduction
    10,  # National Medical Expense Deduction
    12,  # National Net Worth
    15,  # National Person Count (also excludes aca_ptc recipients)
    17,  # National Real Estate Taxes
    18,  # National Rent
    23,  # National Social Security Dependents (not modeled)
    26,  # National Social Security Survivors (not modeled)
    36,  # District ACA PTC (not ready yet)
    39,  # District EITC (use National CBO instead)
    42,  # District Income Tax Before Credits
    43,  # District Medical Expense Deduction
    44,  # District Net Capital Gains
    50,  # District Rental Income
    54,  # District Tax Unit Count
    55,  # District Tax Unit Partnership S Corp Income
    59,  # District Taxable Social Security
]
```

## All Target Groups

### National Level (Groups 0-33)

| Group | Variable | Targets | Status |
|-------|----------|---------|--------|
| 0 | Alimony Expense | 1 | EXCLUDED |
| 1 | Alimony Income | 1 | EXCLUDED |
| 2 | Charitable Deduction | 1 | EXCLUDED |
| 3 | Child Support Expense | 1 | EXCLUDED |
| 4 | Child Support Received | 1 | EXCLUDED |
| 5 | EITC | 1 | INCLUDED (CBO target) |
| 6 | Health Insurance Premiums (w/o Medicare Part B) | 1 | INCLUDED |
| 7 | Income Tax Positive | 1 | INCLUDED (CBO target) |
| 8 | Interest Deduction | 1 | EXCLUDED |
| 9 | Medicaid | 1 | INCLUDED |
| 10 | Medical Expense Deduction | 1 | EXCLUDED |
| 11 | Medicare Part B Premiums | 1 | INCLUDED |
| 12 | Net Worth | 1 | EXCLUDED |
| 13 | Other Medical Expenses | 1 | INCLUDED |
| 14 | Over The Counter Health Expenses | 1 | INCLUDED |
| 15 | Person Count | 3 | EXCLUDED |
| 16 | Qualified Business Income Deduction | 1 | INCLUDED |
| 17 | Real Estate Taxes | 1 | EXCLUDED |
| 18 | Rent | 1 | EXCLUDED |
| 19 | Roth IRA Contributions | 1 | INCLUDED |
| 20 | SALT Deduction | 1 | INCLUDED |
| 21 | SNAP | 1 | INCLUDED |
| 22 | Social Security | 1 | INCLUDED (CBO target) |
| 23 | Social Security Dependents | 1 | EXCLUDED (not modeled) |
| 24 | Social Security Disability | 1 | INCLUDED |
| 25 | Social Security Retirement | 1 | INCLUDED |
| 26 | Social Security Survivors | 1 | EXCLUDED (not modeled) |
| 27 | SPM Unit Capped Housing Subsidy | 1 | INCLUDED |
| 28 | SPM Unit Capped Work Childcare Expenses | 1 | INCLUDED |
| 29 | SSI | 1 | INCLUDED |
| 30 | TANF | 1 | INCLUDED |
| 31 | Tip Income | 1 | INCLUDED |
| 32 | Traditional IRA Contributions | 1 | INCLUDED |
| 33 | Unemployment Compensation | 1 | INCLUDED |

### State Level (Groups 34-35)

| Group | Variable | Targets (436 CDs) | Status |
|-------|----------|-------------------|--------|
| 34 | Person Count (= Medicaid enrollment) | 51 (one per state) | INCLUDED |
| 35 | SNAP | 51 (one per state) | INCLUDED |

### District Level (Groups 36-60)

*Target counts shown for full 436-CD mode*

| Group | Variable | Targets (436 CDs) | Status |
|-------|----------|-------------------|--------|
| 36 | ACA PTC | 436 | EXCLUDED (not ready yet) |
| 37 | Adjusted Gross Income | 436 | INCLUDED |
| 38 | Dividend Income | 436 | INCLUDED |
| 39 | EITC | 1,744 (4 per CD by # qualifying children) | EXCLUDED (use national CBO) |
| 40 | SNAP Household Count | 436 | INCLUDED |
| 41 | Income Tax | 436 | INCLUDED |
| 42 | Income Tax Before Credits | 436 | EXCLUDED |
| 43 | Medical Expense Deduction | 436 | EXCLUDED |
| 44 | Net Capital Gains | 436 | EXCLUDED |
| 45 | Person Count | ~12,200 (28 per CD: age + AGI distribution) | INCLUDED |
| 46 | Qualified Business Income Deduction | 436 | INCLUDED |
| 47 | Qualified Dividend Income | 436 | INCLUDED |
| 48 | Real Estate Taxes | 436 | INCLUDED |
| 49 | Refundable CTC | 436 | INCLUDED |
| 50 | Rental Income | 436 | EXCLUDED |
| 51 | SALT | 436 | INCLUDED |
| 52 | Self Employment Income | 436 | INCLUDED |
| 53 | Tax Exempt Interest Income | 436 | INCLUDED |
| 54 | Tax Unit Count | ~10,000 (23 per CD) | EXCLUDED |
| 55 | Tax Unit Partnership S Corp Income | 436 | EXCLUDED |
| 56 | Taxable Interest Income | 436 | INCLUDED |
| 57 | Taxable IRA Distributions | 436 | INCLUDED |
| 58 | Taxable Pension Income | 436 | INCLUDED |
| 59 | Taxable Social Security | 436 | EXCLUDED |
| 60 | Unemployment Compensation | 436 | INCLUDED |

## Included Targets Summary

After filtering, the calibration uses these target categories:

**National (21 targets):**
- EITC (CBO), Health insurance premiums, Income tax positive (CBO), Medicaid
- Medicare Part B premiums, Other medical expenses, OTC health expenses
- QBI deduction, Roth IRA contributions, SALT deduction, SNAP
- Social Security (CBO), Social Security Disability, Social Security Retirement
- SPM housing subsidy, SPM childcare expenses, SSI, TANF
- Tip income, Traditional IRA contributions, Unemployment compensation

**State (102 targets):**
- Medicaid enrollment by state (person_count, 51 states)
- SNAP benefits by state (51 states)

**District (~21,400 targets across 436 CDs):**
- AGI, Dividend income, SNAP household count, Income tax
- Person count (age + AGI distribution), QBI deduction
- Qualified dividends, Real estate taxes, Refundable CTC, SALT
- Self-employment income, Tax-exempt interest, Taxable interest
- Taxable IRA distributions, Taxable pensions, Unemployment compensation

## Notes

1. The target grouping is dynamic - created by `create_target_groups()` in `calibration_utils.py` based on geographic level (National -> State -> District) and alphabetically sorted variable name within each level.

2. Group 34 (State Person Count) represents **Medicaid enrollment** targets - these are `person_count` targets constrained to Medicaid-enrolled persons, added explicitly by the matrix builder for each state.

3. Group 45 (District Person Count) contains both age bracket targets AND AGI distribution targets:
   - 18 age brackets per CD
   - Plus AGI distribution targets
   - Total: ~12,200 targets for 436 CDs (~28 per CD)

4. Group 36 (District ACA PTC) is new and excluded pending validation.

5. Groups 39 (District EITC) is excluded because we prefer the national CBO EITC target (group 5) to avoid double-counting.
