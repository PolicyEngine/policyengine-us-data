# Methodology

The Enhanced CPS dataset is created through a two-stage process: imputation followed by reweighting. This approach leverages the strengths of both data sources while mitigating their individual limitations. The imputation stage uses Quantile Regression Forests to transfer 72 tax variables from the PUF onto CPS records, creating what we call the Extended CPS. The reweighting stage then optimizes household weights to match over 7,000 administrative targets, producing the final Enhanced CPS with weights calibrated to official statistics. A visual overview of this process is provided in Appendix Figure A1.

:::{card} Enhancement Process Overview
:shadow: md

```{mermaid}
flowchart TD
    A[CPS Data] --> B[Stage 1: Imputation]
    C[PUF Data] --> B
    D[SIPP/SCF/ACS] --> B
    B --> E[Extended CPS]
    E --> F[Stage 2: Reweighting]
    G[7,000+ Targets] --> F
    F --> H[Enhanced CPS]
```
:::

## Stage 1: Variable Imputation

We impute missing variables from multiple data sources using Quantile Regression Forests (QRF). This includes both tax variables from the PUF and additional variables from SIPP, SCF, and ACS.

### Quantile Regression Forests

We use Quantile Regression Forests (QRF), an extension of random forests that estimates conditional quantiles rather than conditional means. This approach better preserves distributional characteristics compared to standard imputation methods. QRF works by building an ensemble of decision trees on the training data, but unlike standard random forests, it stores all observations in leaf nodes rather than just their means. This enables estimation of any quantile of the conditional distribution at prediction time, allowing us to sample from the full conditional distribution rather than relying on point estimates.

### Implementation

We use the `quantile-forest` package, which provides efficient scikit-learn compatible QRF implementation. The specific implementation details are provided in Appendix A.1.

### Predictor Variables

The imputation uses seven variables available in both datasets. These include age of the person, a gender indicator, tax unit filing status (whether joint or separate), and the number of dependents in the tax unit. We also use tax unit role indicators specifying whether each person is the head, spouse, or dependent within their tax unit. These predictors capture key determinants of tax variables while being reliably measured in both datasets. The limited set of predictors ensures common support between the datasets while capturing the primary sources of variation in tax outcomes.

### Imputed Variables

We impute 72 tax-related variables spanning six categories: employment and business income (6 variables), retirement and Social Security (4 variables), investment income (6 variables), deductions (12 variables), tax credits and adjustments (20 variables), and other income and special items (24 variables). The complete list of imputed variables is provided in Appendix Table A1. These variables cover the major components needed for tax simulation while maintaining reasonable imputation quality given the available predictors.

### Additional Imputations

Beyond the 72 PUF tax variables, we impute additional variables from three other data sources. From the Survey of Income and Program Participation (SIPP), we impute tip income using employment income, age, and household composition as predictors. The Survey of Consumer Finances (SCF) provides data for imputing auto loan balances, interest payments, and net worth components. For SCF matching, we use their reference person definition to ensure proper household alignment. From the American Community Survey (ACS), we impute property taxes for homeowners, rent values for specific tenure types, and additional housing characteristics. These supplementary imputations fill gaps in the CPS that are important for comprehensive policy analysis but not available in tax data.

### Sampling Process

Rather than using point estimates, we sample from the conditional distribution to preserve realistic variation in the imputed variables. We first train QRF models on each source dataset, then for each CPS record, we estimate the conditional distribution of each variable given the predictors. We sample from this distribution using a random quantile drawn from a uniform distribution. To ensure consistency across related variables, we use the same random quantile for variables that should be correlated, such as different types of capital gains. This approach preserves realistic correlations between imputed variables while maintaining the marginal distributions observed in the source data.

## Stage 2: Reweighting

### Problem Formulation

The reweighting stage adjusts household weights to ensure the enhanced dataset matches known administrative totals. Given a loss matrix M ∈ R^{n×m} containing n households' contributions to m targets, and a target vector t ∈ R^m of official statistics, we optimize log-transformed weights w to minimize mean squared relative error. The objective function is L(w) = (1/m) Σ_i ((exp(w)^T M_i - t_i) / t_i)^2, where exp(w) represents the exponentiated weights applied to households. The log transformation ensures positive final weights while allowing unconstrained optimization.

### Optimization

We use PyTorch for gradient-based optimization with the Adam optimizer. The implementation uses log-transformed weights to ensure positivity constraints are satisfied throughout the optimization process. The detailed optimization code is provided in Appendix A.2.

### Dropout Regularization

To prevent overfitting to calibration targets, we apply dropout during optimization. We randomly mask 5% of weights each iteration and replace masked weights with the mean of unmasked weights. This percentage was selected through sensitivity analysis on validation performance, testing rates from 0% to 10%. The dropout helps ensure that no single household receives excessive weight in matching targets, improving the stability of policy simulations.

### Calibration Targets

The loss matrix includes over 7,000 targets from six sources:

::::{tab-set}

:::{tab-item} IRS SOI
**5,300+ targets** covering:
- Income by AGI bracket and filing status
- Counts of returns by category
- Aggregate income totals by source
- Deduction and credit utilization rates
:::

:::{tab-item} Census
**200+ targets** including:
- Population by single year of age (0-85)
- State total populations
- State populations under age 5
- Demographic distributions
:::

:::{tab-item} CBO/Treasury
**10+ targets** covering:
- SNAP participation and benefits
- SSI recipient counts
- EITC claims by family size
- Total federal revenues
:::

:::{tab-item} JCT
**4 major deductions**:
- State and local taxes: $21.2B
- Charitable contributions: $65.3B
- Mortgage interest: $24.8B
- Medical expenses: $11.4B
:::

:::{tab-item} Healthcare
**40+ targets** by age:
- Health insurance premiums
- Medicare Part B premiums
- Other medical expenses
- Over-the-counter health costs
:::

:::{tab-item} Other
**1,500+ targets** including:
- State program participation
- Income distributions by geography
- Local area statistics
- Additional administrative benchmarks
:::

::::

### Tax and Benefit Calculations

Our calibration process incorporates comprehensive tax and benefit calculations through PolicyEngine's microsimulation capabilities. This ensures that the reweighted dataset accurately reflects not just income distributions but also the complex interactions between tax liabilities and benefit eligibility.

For tax calculations, we model federal income tax with all major credits and deductions, as well as state and local taxes. The state and local tax (SALT) calculation involves three components. First, we calculate state and local income tax liabilities for each household based on their state of residence and income characteristics. Second, we incorporate property tax amounts, using the imputed values from ACS data for homeowners. Third, we calculate sales tax deductions using the IRS sales tax tables, which most taxpayers use instead of tracking actual sales tax payments. This comprehensive SALT modeling is crucial for accurately capturing itemized deductions and the impact of the SALT deduction cap.

The benefit calculations span major federal and state transfer programs. We model Supplemental Nutrition Assistance Program (SNAP) eligibility and benefit amounts based on household composition, income, and expenses. Supplemental Security Income (SSI) calculations consider both income and asset tests. For healthcare programs, we model Medicaid and Children's Health Insurance Program (CHIP) eligibility using state-specific income thresholds and household characteristics. The Affordable Care Act (ACA) premium tax credits are calculated based on household income relative to the federal poverty level and available benchmark plans. We also include Special Supplemental Nutrition Program for Women, Infants, and Children (WIC) eligibility and benefit calculations.

These tax and benefit calculations enter the calibration process through the loss matrix construction. Each household's calculated tax liabilities and benefit amounts contribute to aggregate targets for program participation and expenditures. This ensures that the final weights not only match income distributions but also produce realistic estimates of government revenues and transfer spending. The interaction between taxes and benefits is particularly important for analyzing reforms that affect both sides of the fiscal system, such as changes to refundable tax credits or means-tested benefits.

### Convergence

The optimization typically converges within 3,000 iterations. We run for 5,000 iterations to ensure stability. Convergence is monitored through the loss value trajectory, weight stability across iterations, and target achievement rates. The optimization is considered converged when the relative change in loss falls below 0.001% for 100 consecutive iterations.

## Validation

### Cross-Validation

We validate the methodology through three approaches. First, we employ 5-fold cross-validation on calibration targets, holding out subsets of targets to assess out-of-sample performance. Second, we test stability across multiple random seeds to ensure results are not sensitive to initialization. Third, we validate the imputation quality through out-of-sample prediction on held-out records from the source datasets.

### Quality Checks

Throughout the enhancement process, we implement several quality checks to ensure data integrity. We verify that all weights remain positive after optimization, as negative weights would violate the interpretation of survey weights as population representations. Weight magnitudes are checked to ensure no single household receives excessive influence on aggregate statistics. We preserve demographic relationships by verifying that household members maintain consistent relationships after reweighting. Finally, we ensure household structures remain intact, with all members of a household receiving the same weight adjustment factor.

## Implementation

The complete implementation is available at:
[https://github.com/PolicyEngine/policyengine-us-data](https://github.com/PolicyEngine/policyengine-us-data)

Key files:
- `policyengine_us_data/datasets/cps/extended_cps.py` - Imputation stage
- `policyengine_us_data/datasets/cps/enhanced_cps.py` - Reweighting stage
- `policyengine_us_data/utils/loss.py` - Loss matrix construction

The modular design allows researchers to modify or extend individual components while maintaining the overall framework.