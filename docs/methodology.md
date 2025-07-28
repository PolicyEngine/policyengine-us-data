# Methodology

The Enhanced CPS dataset is created through a two-stage process: imputation followed by reweighting. This approach leverages the strengths of both data sources while mitigating their individual limitations.

## Overview

Our enhancement process consists of:

1. **Imputation Stage**: Use Quantile Regression Forests to impute 72 tax variables from the PUF onto CPS records
2. **Reweighting Stage**: Optimize household weights to match over 7,000 administrative targets

The imputation stage creates what we call the Extended CPS, which maintains the CPS structure while adding tax detail. The reweighting stage produces the final Enhanced CPS with weights calibrated to official statistics.

## Stage 1: Variable Imputation

We impute missing variables from multiple data sources using Quantile Regression Forests (QRF). This includes both tax variables from the PUF and additional variables from SIPP, SCF, and ACS.

### Quantile Regression Forests

We use Quantile Regression Forests (QRF), an extension of random forests that estimates conditional quantiles rather than conditional means. This approach better preserves distributional characteristics compared to standard imputation methods.

QRF works by:
- Building an ensemble of decision trees on the training data
- Storing all observations in leaf nodes (not just means)
- Estimating any quantile of the conditional distribution at prediction time
- Allowing us to sample from the full conditional distribution

### Implementation

We use the `quantile-forest` package, which provides efficient scikit-learn compatible QRF implementation:

```python
from quantile_forest import RandomForestQuantileRegressor

qrf = RandomForestQuantileRegressor(
    n_estimators=100,
    min_samples_leaf=1,
    random_state=0
)
```

### Predictor Variables

The imputation uses seven variables available in both datasets:
- Age of the person
- Gender indicator (is_male)
- Tax unit filing status (is_joint)
- Number of dependents in tax unit
- Tax unit role indicators (head, spouse, dependent)

These predictors capture key determinants of tax variables while being reliably measured in both datasets.

### Imputed Variables

We impute 72 tax-related variables spanning six categories:

**Employment and Business Income** (6 variables): employment income, partnership/S-corp income, self-employment income, W-2 wages from qualified businesses, rental income, farm income

**Retirement and Social Security** (4 variables): Social Security benefits, taxable pension income, tax-exempt pension income, taxable IRA distributions

**Investment Income** (6 variables): long and short-term capital gains, qualified and non-qualified dividend income, taxable and tax-exempt interest income

**Deductions** (12 variables): mortgage interest, charitable cash and non-cash donations, state and local taxes, medical expenses, casualty losses, various business deductions

**Tax Credits and Adjustments** (20 variables): foreign tax credit, education credits, retirement savings credit, energy credits, educator expenses, student loan interest, various above-the-line deductions

**Other Income and Special Items** (24 variables): alimony, unemployment compensation, estate income, miscellaneous income, various specialized gains and losses

### Additional Imputations

Beyond the 72 PUF tax variables, we impute:

**From SIPP (Survey of Income and Program Participation)**:
- Tip income using employment income, age, and household composition as predictors

**From SCF (Survey of Consumer Finances)**:
- Auto loan balances and interest
- Net worth components
- Uses SCF reference person definition for proper household matching

**From ACS (American Community Survey)**:
- Property taxes for homeowners
- Rent values for specific tenure types
- Additional housing characteristics

### Sampling Process

Rather than using point estimates, we sample from the conditional distribution:

1. Train QRF models on each source dataset
2. For each CPS record, estimate the conditional distribution
3. Sample from this distribution using a random quantile
4. Ensure consistency across related variables

This approach preserves realistic correlations between imputed variables.

## Stage 2: Reweighting

### Problem Formulation

Given:
- Loss matrix **M** ∈ ℝⁿˣᵐ (n households, m targets)  
- Target vector **t** ∈ ℝᵐ (official statistics)

We optimize log-transformed weights **w** to minimize mean squared relative error:

L(w) = (1/m) Σᵢ ((exp(w)ᵀMᵢ - tᵢ) / tᵢ)²

The log transformation ensures positive weights while allowing unconstrained optimization.

### Optimization

We use PyTorch for gradient-based optimization:

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

### Dropout Regularization

To prevent overfitting to calibration targets, we apply dropout during optimization:
- Randomly mask 5% of weights each iteration
- Replace masked weights with mean of unmasked weights
- Selected through sensitivity analysis on validation performance

### Calibration Targets

The loss matrix includes over 7,000 targets from six sources:

**IRS Statistics of Income** (5,300+ targets):
- Income by AGI bracket and filing status
- Counts of returns by category
- Aggregate income totals

**Census Data** (200+ targets):
- Population by single year of age
- State populations
- Demographic distributions

**Program Totals** (10+ targets):
- CBO projections for major programs
- Treasury EITC statistics

**Tax Expenditures** (4 targets):
- JCT estimates for major deductions

**Healthcare Spending** (40+ targets):
- Age-stratified expenditure patterns

**Other Sources** (1,500+ targets):
- State-level program participation
- Income distributions by geography

### Convergence

The optimization typically converges within 3,000 iterations. We run for 5,000 iterations to ensure stability. Convergence is monitored through:
- Loss value trajectory
- Weight stability
- Target achievement rates

## Validation

### Cross-Validation

We validate the methodology through:
- 5-fold cross-validation on calibration targets
- Stability testing across random seeds
- Out-of-sample prediction for imputation

### Quality Checks

Throughout the process, we verify:
- No negative weights
- Reasonable weight magnitudes
- Preservation of demographic relationships
- Consistency of household structures

## Implementation

The complete implementation is available at:
[https://github.com/PolicyEngine/policyengine-us-data](https://github.com/PolicyEngine/policyengine-us-data)

Key files:
- `policyengine_us_data/datasets/cps/extended_cps.py` - Imputation stage
- `policyengine_us_data/datasets/cps/enhanced_cps.py` - Reweighting stage
- `policyengine_us_data/utils/loss.py` - Loss matrix construction

The modular design allows researchers to modify or extend individual components while maintaining the overall framework.