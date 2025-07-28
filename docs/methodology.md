# Methodology

The Enhanced CPS dataset combines survey and administrative data through a two-stage enhancement process: imputation and reweighting.

## Data Sources

The enhanced dataset builds upon three primary sources:

1. **Current Population Survey (CPS)**: Provides demographic detail and household structure
2. **IRS Public Use File (PUF)**: Contains detailed tax return information
3. **Administrative Targets**: Over 7,000 calibration targets from IRS SOI, Census, CBO, Treasury, JCT, and healthcare data

## Enhancement Process

### Stage 1: Tax Variable Imputation

We impute 72 tax variables from the PUF onto CPS records using Quantile Regression Forests (QRF). The imputation uses seven predictors available in both datasets:

- Age
- Gender (is_male)
- Tax unit filing status (is_joint)
- Number of dependents
- Tax unit role indicators (head, spouse, dependent)

The 72 imputed variables include:
- Employment and business income (6 variables)
- Retirement and Social Security (4 variables)
- Investment income (6 variables)
- Deductions (12 variables)
- Tax credits and adjustments (20 variables)
- Other income and special items (24 variables)

### Stage 2: Reweighting

After imputation, we optimize household weights to match administrative targets using gradient descent with PyTorch. The optimization minimizes mean squared relative error across all targets.

Key features:
- Log-transformed weights ensure positivity
- Dropout regularization (5% rate) prevents overfitting
- 5,000 iterations with Adam optimizer
- Learning rate of 0.1

## Calibration Targets

The loss matrix includes over 7,000 targets from six sources:

### IRS Statistics of Income
- Income components by AGI bracket and filing status
- Over 5,300 distinct targets

### Census Population
- Single-year age populations
- State-level demographics

### CBO Projections
- Program participation totals
- Tax revenue estimates

### JCT Tax Expenditures
- SALT deduction: $21.2B
- Charitable deduction: $65.3B
- Mortgage interest: $24.8B
- Medical expense: $11.4B

### Healthcare and Other Sources
- Spending by age group
- State-level program participation

## Validation

Results are validated through:
- Cross-validation on held-out targets
- Stability analysis across random seeds
- Comparison with official statistics

Detailed validation results are available in our [interactive dashboard](https://policyengine.github.io/policyengine-us-data/validation.html).