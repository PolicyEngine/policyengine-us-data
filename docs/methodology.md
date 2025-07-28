# Methodology

The Enhanced CPS dataset is created through a two-stage process: imputation followed by reweighting. This approach leverages the strengths of both data sources while mitigating their individual limitations. The imputation stage uses Quantile Regression Forests to transfer 72 tax variables from the PUF onto CPS records, creating what we call the Extended CPS. The reweighting stage then optimizes household weights to match over 7,000 administrative targets, producing the final Enhanced CPS with weights calibrated to official statistics. A visual overview of this process is provided in Appendix Figure A1.

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

The loss matrix includes over 7,000 targets from six sources. IRS Statistics of Income provides the largest share with over 5,300 targets covering income by AGI bracket and filing status, counts of returns by category, and aggregate income totals. Census data contributes over 200 targets including population by single year of age, state populations, and demographic distributions. Program totals from CBO projections and Treasury EITC statistics add approximately 10 targets. Tax expenditure estimates from JCT cover four major deductions. Healthcare spending patterns stratified by age contribute over 40 targets. The remaining 1,500+ targets come from various sources including state-level program participation and income distributions by geography. The complete list of calibration targets is provided in our online documentation.

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