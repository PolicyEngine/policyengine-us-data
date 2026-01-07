# Appendix

## Appendix A: Implementation Code

### A.1 Quantile Regression Forest Implementation

The following code demonstrates the implementation of Quantile Regression Forests for variable imputation:

```python
from quantile_forest import RandomForestQuantileRegressor

qrf = RandomForestQuantileRegressor(
    n_estimators=100,
    min_samples_leaf=1,
    random_state=0
)
```

### A.2 PyTorch Optimization for Reweighting

The reweighting optimization uses PyTorch for gradient-based optimization:

```python
import torch

# Initialize with log of original weights
log_weights = torch.log(original_weights)
log_weights.requires_grad = True

# Adam optimizer
optimizer = torch.optim.Adam([log_weights], lr=0.1)

# Optimization loop
for iteration in range(5000):
    weights = torch.exp(log_weights)
    achieved = weights @ loss_matrix
    relative_errors = (achieved - targets) / targets
    loss = torch.mean(relative_errors ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Appendix B: Tables

### Table A1: Complete List of Imputed Variables

#### Variables Imputed from IRS Public Use File (67 variables)

**Income Variables:**
- employment_income
- partnership_s_corp_income
- social_security
- taxable_pension_income
- tax_exempt_pension_income
- long_term_capital_gains
- short_term_capital_gains
- taxable_ira_distributions
- self_employment_income
- qualified_dividend_income
- non_qualified_dividend_income
- rental_income
- taxable_unemployment_compensation
- taxable_interest_income
- tax_exempt_interest_income
- estate_income
- miscellaneous_income
- farm_income
- alimony_income
- farm_rent_income
- non_sch_d_capital_gains
- long_term_capital_gains_on_collectibles
- unrecaptured_section_1250_gain
- salt_refund_income

**Deductions and Adjustments:**
- interest_deduction
- unreimbursed_business_employee_expenses
- pre_tax_contributions
- charitable_cash_donations
- self_employed_pension_contribution_ald
- domestic_production_ald
- self_employed_health_insurance_ald
- charitable_non_cash_donations
- alimony_expense
- health_savings_account_ald
- student_loan_interest
- investment_income_elected_form_4952
- early_withdrawal_penalty
- educator_expense
- deductible_mortgage_interest

**Tax Credits:**
- cdcc_relevant_expenses
- foreign_tax_credit
- american_opportunity_credit
- general_business_credit
- energy_efficient_home_improvement_credit
- amt_foreign_tax_credit
- excess_withheld_payroll_tax
- savers_credit
- prior_year_minimum_tax_credit
- other_credits

**Qualified Business Income Variables:**
- w2_wages_from_qualified_business
- unadjusted_basis_qualified_property
- business_is_sstb
- qualified_reit_and_ptp_income
- qualified_bdc_income
- farm_operations_income
- estate_income_would_be_qualified
- farm_operations_income_would_be_qualified
- farm_rent_income_would_be_qualified
- partnership_s_corp_income_would_be_qualified
- rental_income_would_be_qualified
- self_employment_income_would_be_qualified

**Other Tax Variables:**
- traditional_ira_contributions
- qualified_tuition_expenses
- casualty_loss
- unreported_payroll_tax
- recapture_of_investment_credit

#### Variables Imputed from Survey of Income and Program Participation (1 variable)

- tip_income

#### Variables Imputed from Survey of Consumer Finances (3 variables)

- networth
- auto_loan_balance
- auto_loan_interest

#### Variables Imputed from American Community Survey (2 variables)

- rent
- real_estate_taxes