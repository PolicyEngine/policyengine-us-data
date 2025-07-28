# Data Sources

Our methodology combines two primary data sources with calibration targets from six administrative sources.

## Primary Data Sources

### Current Population Survey (CPS)

The Current Population Survey Annual Social and Economic Supplement (ASEC) serves as our base dataset. Conducted jointly by the Census Bureau and Bureau of Labor Statistics, the CPS ASEC surveys approximately 75,000 households annually.

Key features:
- Representative sample of US households
- Detailed demographic information
- Family and household relationships
- Geographic identifiers including state
- Program participation questions
- Self-reported income by source

Limitations:
- Income underreporting, especially at high incomes
- Limited tax detail
- No information on itemized deductions
- Topcoding of high values

### IRS Public Use File (PUF)

The IRS Statistics of Income Public Use File contains detailed tax return information from a stratified sample of individual income tax returns. The most recent PUF available is from tax year 2015, containing approximately 230,000 returns.

Key features:
- Accurate income reporting from tax returns
- Detailed breakdown of income sources
- Complete deduction information
- Tax credits and payments
- Sampling weights to represent all filers

Limitations:
- No demographic information beyond filing status
- No state identifiers
- Excludes non-filers
- Significant time lag (2015 data)
- No household structure

## Additional Data Sources for Imputation

Beyond the PUF, we incorporate data from three additional surveys to impute specific variables missing from the CPS:

### Survey of Income and Program Participation (SIPP)

The SIPP provides detailed income and program participation data. We use SIPP to impute:
- **Tip income**: Using a Quantile Regression Forest model trained on SIPP data, we impute tip income based on employment income, age, and household composition

### Survey of Consumer Finances (SCF)

The SCF provides comprehensive wealth and debt information. We use SCF to impute:
- **Auto loan balances**: Matched based on household demographics and income
- **Interest on auto loans**: Calculated from imputed balances
- **Net worth components**: Various wealth measures not available in CPS

The SCF imputation uses their reference person definition (male in mixed-sex couples or older person in same-sex couples) to ensure proper matching.

### American Community Survey (ACS)

The ACS provides detailed housing and geographic data. We use ACS to impute:
- **Property taxes**: For homeowners, imputed based on state, household income, and demographics
- **Rent values**: For specific tenure types where CPS data is incomplete
- **Housing characteristics**: Additional housing-related variables

These imputations use Quantile Regression Forests to preserve distributional characteristics while accounting for household heterogeneity.

## Calibration Data Sources

We calibrate the enhanced dataset to over 7,000 targets from six authoritative sources:

### IRS Statistics of Income (SOI)

Annual tabulations from tax returns provide income distributions by:
- Adjusted Gross Income (AGI) bracket
- Filing status
- Income type

We use SOI Table 1.4 which cross-tabulates income components by AGI ranges, creating over 5,300 distinct targets.

### Census Population Projections

National and state-level demographic targets from:
- Single-year-of-age populations (ages 0-85)
- State total populations
- State populations under age 5

### Congressional Budget Office (CBO)

Program participation and revenue projections:
- SNAP (Supplemental Nutrition Assistance Program)
- Social Security benefits
- Supplemental Security Income (SSI)
- Unemployment compensation
- Individual income tax revenue

### Joint Committee on Taxation (JCT)

Tax expenditure estimates for major deductions:
- State and local tax deduction: $21.2 billion
- Charitable contribution deduction: $65.3 billion
- Mortgage interest deduction: $24.8 billion
- Medical expense deduction: $11.4 billion

### Treasury Department

Additional program totals:
- Earned Income Tax Credit by number of children
- Total EITC expenditure

### Healthcare Spending Data

Age-stratified medical expenditures:
- Health insurance premiums (excluding Medicare Part B)
- Medicare Part B premiums
- Other medical expenses
- Over-the-counter health expenses

## Data Preparation

### CPS Processing

We use the CPS ASEC from survey year 2024 (covering calendar year 2023 income). The Census Bureau provides:
- Person-level records with demographics
- Hierarchical identifiers linking persons to families and households
- Initial survey weights

### PUF Processing

The 2015 PUF requires several adjustments:
- Dollar amounts uprated using SOI growth factors by income type
- Records filtered to remove those with insufficient data
- Weights normalized to represent the filing population

### Target Preparation

Administrative targets are collected for the appropriate year:
- Most targets use 2024 projections
- Historical data uprated using official growth rates
- State-level targets adjusted for population changes

The combination of these data sources enables us to create a dataset that maintains the CPS's demographic richness while achieving tax reporting accuracy comparable to administrative data.