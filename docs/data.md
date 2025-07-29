# Data sources

Our methodology combines two primary data sources with calibration targets from six administrative sources.

## Primary data sources

### Current Population Survey (CPS)

The Current Population Survey Annual Social and Economic Supplement (ASEC) serves as our base dataset. Conducted jointly by the Census Bureau and Bureau of Labor Statistics, the CPS ASEC expands the regular monthly CPS sample to survey more than 75,000 households annually, providing detailed income and poverty estimates.

The CPS provides several essential features for microsimulation modeling. It offers a representative sample of US households with detailed demographic information including age, education, race, and employment status. The survey captures family and household relationships through a comprehensive set of relationship codes that allow reconstruction of tax units and benefit units. Geographic identifiers down to the state level enable subnational policy analysis. The survey includes detailed questions about program participation in major transfer programs like SNAP, Medicaid, and housing assistance. Income data is collected by source, distinguishing between wages, self-employment, interest, dividends, and transfers.

However, the CPS faces well-documented limitations that necessitate enhancement. Income underreporting is particularly severe at high income levels, with {cite:t}`rothbaum2021` finding that the CPS captures only 50% of top incomes compared to tax records. The survey provides limited tax detail, lacking information on itemized deductions, tax credits, and capital gains realizations that are crucial for revenue estimation. High income values are topcoded to protect confidentiality, further limiting the ability to analyze tax policies affecting high earners.

### IRS Public Use File (PUF)

The IRS Statistics of Income Public Use File contains detailed tax return information from a stratified sample of individual income tax returns. The most recent PUF available is from tax year 2015, containing approximately 230,000 returns.

The PUF provides tax-related variables drawn directly from filed tax returns. It provides detailed breakdowns of income by source including wages, interest, dividends, capital gains, business income, and retirement distributions. The file contains complete information on itemized deductions such as mortgage interest, state and local taxes, and charitable contributions. All tax credits claimed by filers are included, from the earned income tax credit to education credits. The stratified sampling design oversamples high-income returns, providing better coverage of the income distribution's upper tail than survey data. Sampling weights allow researchers to produce population-representative estimates.

Despite these strengths, the PUF has significant limitations for comprehensive policy analysis. The file contains minimal demographic information, limited to filing status and exemptions claimed. Geographic identifiers are removed to protect taxpayer privacy, preventing state-level analysis. The population excludes non-filers, who represent approximately 20% of adults and are disproportionately low-income. The substantial time lag means the most recent data is nine years old as of 2024, missing recent economic and demographic changes. Perhaps most critically, the PUF lacks household structure, preventing analysis of how tax policies interact with transfer programs that operate at the household level.

## Additional data sources for imputation

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

## Calibration data sources

We calibrate the enhanced dataset to over 7,000 targets from six authoritative sources:

### IRS Statistics of Income (SOI)

Annual tabulations from tax returns provide income distributions by:
- Adjusted Gross Income (AGI) bracket
- Filing status
- Income type

We use SOI Table 1.4 which cross-tabulates income components by AGI ranges, creating over 5,300 distinct targets.

### Census population projections

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

### Healthcare spending data

Age-stratified medical expenditures:
- Health insurance premiums (excluding Medicare Part B)
- Medicare Part B premiums
- Other medical expenses
- Over-the-counter health expenses

## Addressing the temporal gap

The nine-year gap between the 2015 PUF and 2024 CPS presents a methodological challenge. Economic conditions, tax law, and demographic patterns have changed significantly since 2015. We address this temporal inconsistency through several approaches. Dollar amounts in the PUF are uprated using income-specific growth factors from IRS Statistics of Income publications, ensuring that income levels reflect current economic conditions. The calibration process forces the combined dataset to match contemporary administrative totals, partially compensating for demographic shifts. However, structural changes in the economy, such as the growth of gig work or shifts in retirement patterns, may not be fully captured. Users should consider this limitation when analyzing policies sensitive to recent economic trends.

## Data preparation

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

### Target preparation

Administrative targets are collected for the appropriate year:
- Most targets use 2024 projections
- Historical data uprated using official growth rates
- State-level targets adjusted for population changes

### Variable harmonization

Combining datasets requires careful harmonization of variable definitions and concepts. Tax units in the PUF must be mapped to CPS households, accounting for multi-generational households and unmarried partners. Income concepts differ between sources, with the PUF using tax definitions while the CPS follows survey conventions. For example, the PUF reports taxable Social Security benefits while the CPS reports total benefits received. We harmonize these differences by using PolicyEngine's tax calculator to compute tax concepts from CPS variables before imputation.

Time periods also require harmonization. The CPS collects income for the previous calendar year while asking about current-year program participation. The PUF reports tax year data with some income received in different calendar years. We align all amounts to a common tax year basis using payment timing assumptions documented in our code repository.

The combination of these data sources enables us to create a dataset that maintains the CPS's demographic richness while achieving tax reporting accuracy comparable to administrative data.