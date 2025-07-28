# Introduction

PolicyEngine-US-Data is a package that creates representative microdata for the US, 
designed for input in the PolicyEngine tax-benefit microsimulation model. This tool 
allows users to explore the data sources, validation processes, and enhancements 
made to ensure accurate and reliable microsimulation results.

PolicyEngine is a tool with a clear purpose: for given assumptions about US government policy and US households, predicting what US households will look like in the next few years. To do that, we need both of two things:

* An accurate model of the effects of policy rules on households.
* An accurate representation of the current US household sector *now*.

This repository is dedicated to the second of those. In this documentation, we'll explain how we do that, but we'll also use our model (the first bullet) to see what we end up with when we combine the two, and measure up against other organisations doing the same thing.

## Data Enhancement Process

PolicyEngine US Data enhances the Census Bureau's Current Population Survey (CPS) through a two-stage process: imputation and reweighting. This creates the Enhanced CPS (ECPS) dataset that better represents the US population for microsimulation purposes.

### Stage 1: Imputation (Extended CPS)

The first stage extends the CPS by imputing detailed tax-related variables that are missing from the original survey but are available in the IRS Public Use File (PUF). We use a Quantile Random Forest (QRF) model to predict these variables based on characteristics available in both datasets.

#### Imputation Methodology

The imputation uses the following predictors from both CPS and PUF:
- `age` - Age of the person
- `is_male` - Binary indicator for male gender
- `tax_unit_is_joint` - Whether the tax unit files jointly
- `tax_unit_count_dependents` - Number of dependents in the tax unit
- `is_tax_unit_head` - Whether person is head of tax unit
- `is_tax_unit_spouse` - Whether person is spouse in tax unit
- `is_tax_unit_dependent` - Whether person is a dependent

#### Imputed Variables

The QRF model imputes 72 tax-related variables from the PUF to the CPS. The most significant variables by magnitude include:

**Employment and Business Income:**
- `employment_income` - Wages, salaries, and tips
- `partnership_s_corp_income` - Income from partnerships and S-corporations
- `self_employment_income` - Net profit from self-employment
- `w2_wages_from_qualified_business` - W-2 wages from qualified businesses
- `rental_income` - Net rental income
- `farm_income` - Net farm income

**Retirement and Social Security:**
- `social_security` - Total Social Security benefits
- `taxable_pension_income` - Taxable portion of pensions
- `tax_exempt_pension_income` - Tax-exempt pension income
- `taxable_ira_distributions` - Taxable IRA distributions

**Investment Income:**
- `long_term_capital_gains` - Long-term capital gains
- `short_term_capital_gains` - Short-term capital gains
- `qualified_dividend_income` - Qualified dividend income
- `non_qualified_dividend_income` - Non-qualified dividend income
- `taxable_interest_income` - Taxable interest income
- `tax_exempt_interest_income` - Tax-exempt interest income

**Deductions and Credits:**
- `interest_deduction` - Mortgage interest deduction
- `charitable_cash_donations` - Cash charitable contributions
- `charitable_non_cash_donations` - Non-cash charitable contributions
- `deductible_mortgage_interest` - Deductible mortgage interest
- `state_and_local_tax_deduction` - SALT deduction
- `medical_expense_deduction` - Medical expense deduction

**Other Income and Adjustments:**
- `unreimbursed_business_employee_expenses` - Unreimbursed employee expenses
- `pre_tax_contributions` - Pre-tax retirement contributions
- `self_employed_pension_contribution_ald` - Self-employed pension contributions
- `domestic_production_ald` - Domestic production activities deduction
- `self_employed_health_insurance_ald` - Self-employed health insurance deduction
- `alimony_income` and `alimony_expense` - Alimony received and paid
- `student_loan_interest` - Student loan interest deduction
- `educator_expense` - Educator expense deduction

**Tax Credits and Payments:**
- `foreign_tax_credit` - Foreign tax credit
- `american_opportunity_credit` - American Opportunity Tax Credit
- `savers_credit` - Retirement savings contributions credit
- `energy_efficient_home_improvement_credit` - Energy efficiency credits
- `excess_withheld_payroll_tax` - Excess payroll tax withheld

### Stage 2: Reweighting (Enhanced CPS)

The second stage adjusts household weights to match official statistics from multiple sources. This ensures the microdata produces accurate aggregates when used for policy simulation.

#### Reweighting Targets

The reweighting process calibrates to over 570 targets from the following sources:

**1. IRS Statistics of Income (SOI)**

The largest set of targets comes from IRS SOI data, including:
- **AGI-stratified totals and counts** for variables by filing status and taxable return status:
  - Adjusted gross income
  - Employment income
  - Business net profits and losses
  - Capital gains (gross amounts and losses)
  - Dividends (ordinary and qualified)
  - Partnership and S-corp income/losses
  - Interest income (taxable)
  - Pension income (total and taxable)
  - Social Security (total and taxable)
  - Unemployment compensation
  - Estate income/losses
  - IRA distributions
  - Rental income/losses
  - Tax-exempt interest

- **EITC statistics by number of children**:
  - Number of returns claiming EITC
  - Total EITC amounts

- **Aggregate statistics**:
  - Negative household market income (total and count)

**2. Census Bureau Data**

- **Population by single year of age** (ages 0-85) from the National Population Projections (np2023_d5_mid.csv)
- **Population by state** including total and under-5 populations
- **SPM threshold-based AGI distributions** by decile
- **Total infants** (age 0-1)
- **Hard-coded CPS-derived totals**:
  - Medical expenses: $385B (health insurance premiums), $278B (other medical), $112B (Medicare Part B), $72B (over-the-counter)
  - SPM thresholds total: $3,945B
  - Child support: $33B (both expense and received)
  - Work and childcare expenses: $348B
  - Housing subsidy: $35B
  - TANF: $9B
  - Alimony: $13B (both income and expense)
  - Real estate taxes: $500B
  - Rent: $735B

**3. Congressional Budget Office (CBO) Projections**

Program totals for:
- Income tax
- SNAP (food stamps)
- Social Security
- SSI (Supplemental Security Income)
- Unemployment compensation

**4. Treasury Department**

- EITC total expenditure

**5. Joint Committee on Taxation (JCT)**

Tax expenditure estimates for:
- State and local tax (SALT) deduction: $21.2B
- Medical expense deduction: $11.4B
- Charitable deduction: $65.3B
- Mortgage interest deduction: $24.8B

**6. Healthcare Spending by Age**

Age-stratified (10-year groups) spending for:
- Health insurance premiums
- Over-the-counter expenses
- Other medical expenses
- Medicare Part B premiums

#### Reweighting Algorithm

The reweighting uses PyTorch to optimize household weights through gradient descent:
1. Initial weights are log-transformed and slightly perturbed with noise
2. A loss function computes mean squared relative error between weighted sums and targets
3. Adam optimizer adjusts weights over 5,000 iterations
4. Dropout (5% rate) is applied during training for regularization
5. Final weights are exponentiated back to original scale

## Summary

The Enhanced CPS combines the demographic representativeness of the CPS with the tax detail of the PUF through imputation, then ensures accuracy through reweighting to official statistics. This creates a comprehensive microdata file suitable for detailed tax and benefit microsimulation while maintaining consistency with aggregate benchmarks from multiple authoritative sources.

