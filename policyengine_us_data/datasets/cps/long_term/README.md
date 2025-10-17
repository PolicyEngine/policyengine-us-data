# Long-Term Income Tax Projections Using Demographic Reweighting

This directory contains a demographic-based income tax projection system that combines population dynamics with survey reweighting to forecast U.S. federal income tax revenue from 2025 to 2100.

## Overview

As the U.S. population ages and demographics shift, tax revenue will be significantly impacted. This model projects these changes by:

1. **Projecting population** by age bracket using demographic transition matrices
2. **Reweighting survey data** to match future demographics using Iterative Proportional Fitting (IPF)
3. **Calculating income tax revenue** under these demographic scenarios

## Key Findings (with current assumptions)

- Population: 332M (2025) → 85M (2100)
- Income Tax Revenue: $2.1T (2025) → $500B (2100)
- Working-age share (20-64): 60% → 45%
- Elderly share (65+): 20% → 38%

*Note: These projections assume constant birth rates and no immigration, representing a baseline scenario.*

## Files

### `run_full_projection.py`
**Main Script** - Orchestrates the complete pipeline from demographics to tax projections.
```bash
python run_full_projection.py
```
Outputs: `income_tax_projections.html` (interactive visualization)

### `create_reweighting_matrix.py`
**IPF Engine** - Contains the Iterative Proportional Fitting algorithm and functions to create the design matrix from CPS microdata.
- `iterative_proportional_fitting()` - Raking algorithm to reweight survey samples
- `create_age_design_matrix()` - Builds household×age_bracket matrix from CPS

### `age_projection.py`
**Demographics Module** - Handles population projections using transition matrices.
- `create_annual_transition_matrix()` - Builds year-over-year demographic transitions
- Contains baseline 2024 population vector and visualization functions

## Methodology

### 1. Demographic Transitions
The model uses an 18×18 transition matrix representing 5-year age brackets (0-4, 5-9, ..., 85+). Each year:
- 20% of each bracket ages into the next bracket
- Age-specific survival rates are applied
- Births are added to the 0-4 bracket

### 2. Iterative Proportional Fitting (IPF)
Also known as "raking," IPF adjusts survey weights to match population targets:
- Start with CPS household weights
- Iteratively adjust to match age bracket totals
- Preserve relative relationships within the data

### 3. Income Tax Calculation
- Uses PolicyEngine's microsimulation on 2024 CPS data
- Applies year-specific weights from IPF
- Assumes tax law and income distributions remain constant (in real terms)

## Key Assumptions

**Demographics:**
- Annual births: 3.8 million (constant)
- No immigration
- Mortality rates from simplified life tables

**Economics:**
- Tax law remains unchanged
- Real income distribution by age remains constant
- No behavioral responses to demographic changes

## Technical Details

### State-Space Representation
The system can be viewed as a discrete-time state-space model:
```
x(t+1) = T × x(t) + b
```
Where:
- `x(t)` = 18×1 population vector at time t
- `T` = 18×18 transition matrix
- `b` = births vector (only first element non-zero)

### IPF Convergence
The IPF algorithm minimizes the Kullback-Leibler divergence:
```
KL(w||w₀) = Σ w log(w/w₀)
```
Subject to: `X'w = y` (age bracket constraints)

Typical convergence: <100 iterations to relative error <1e-6

## Extensions

Potential improvements for more realistic projections:
- Immigration flows
- Dynamic fertility rates
- Economic feedback effects
- Policy responses
- Stochastic mortality shocks
- Cohort-specific income profiles

## Requirements

- Python 3.8+
- PolicyEngine-US
- NumPy, Pandas
- Plotly (for visualization)

## Citation

This methodology combines demographic projection techniques with survey reweighting methods commonly used in microsimulation modeling. The IPF algorithm implementation follows standard survey statistics practices.