# Data Sources

Our methodology combines two primary data sources with calibration targets from administrative sources.

## Primary Data Sources

### Current Population Survey (CPS)

The Current Population Survey Annual Social and Economic Supplement (ASEC) serves as our base dataset. The Census Bureau and Bureau of Labor Statistics jointly conduct the CPS ASEC, surveying households annually.

The CPS provides features for microsimulation modeling. It offers a representative sample of US households with demographic information including age, education, race, and employment status. The survey captures family and household relationships through relationship codes that allow reconstruction of tax units and benefit units. Geographic identifiers down to the state level enable subnational policy analysis. The survey includes questions about program participation in transfer programs like SNAP, Medicaid, and housing assistance. The survey collects income data by source, distinguishing between wages, self-employment, interest, dividends, and transfers.

The CPS faces limitations that necessitate enhancement. Income underreporting is severe at high income levels {cite:p}`rothbaum2021`. The survey provides limited tax detail, lacking information on itemized deductions, tax credits, and capital gains realizations. The Census Bureau topcodes high income values to protect confidentiality. The survey's focus on cash income means it misses non-cash compensation like employer-provided health insurance premiums.

### IRS Public Use File (PUF)

The IRS Statistics of Income Public Use File contains tax return information that the IRS draws from a stratified sample of individual income tax returns.

The PUF provides tax-related variables drawn from filed tax returns. It provides breakdowns of income by source including wages, interest, dividends, capital gains, business income, and retirement distributions. The file includes itemized deductions such as mortgage interest, state and local taxes, and charitable contributions. The file includes tax credits that filers claim, from the earned income tax credit to education credits. The stratified sampling design oversamples high-income returns. Sampling weights allow researchers to produce population-representative estimates.

The PUF has limitations for policy analysis. The file contains minimal demographic information, limited to filing status and exemptions claimed. The IRS removes geographic identifiers to protect taxpayer privacy, which prevents state-level analysis. The population excludes non-filers. The PUF lacks household structure, preventing analysis of how tax policies interact with transfer programs that operate at the household level.

## Additional Data Sources for Imputation

Beyond the PUF, we incorporate data from three additional surveys to impute specific variables missing from the CPS:

### Survey of Income and Program Participation (SIPP)

The SIPP provides income and program participation data. We use SIPP primarily to impute tip income through a Quantile Regression Forest model trained on SIPP data, using employment income, age, and household composition as predictors.

### Survey of Consumer Finances (SCF)

The SCF provides wealth and debt information that we use to impute several financial variables missing from the CPS. We match auto loan balances based on household demographics and income, then calculate interest on auto loans from these imputed balances. Additionally, we impute various net worth components and other wealth measures not available in CPS. The SCF imputation uses their reference person definition to ensure proper matching.

### American Community Survey (ACS)

The ACS provides housing and geographic data that supplements the CPS housing information. For homeowners, we impute property taxes based on state of residence, household income, and demographic characteristics. We also impute rent values for specific tenure types where CPS data is incomplete, along with additional housing characteristics not captured in the CPS. These imputations use Quantile Regression Forests to preserve distributional characteristics while accounting for household heterogeneity.

## Calibration Data Sources

The calibration process uses targets from six administrative sources:

### IRS Statistics of Income (SOI)

The IRS SOI provides tax return aggregates by income level, filing status, and geography. These include counts of returns, aggregate income by source, deduction amounts, and credit utilization.

### Census Population Estimates

Census provides population counts by age, state, and other demographic characteristics.

### Congressional Budget Office

CBO provides projections for program participation and spending, including SNAP benefits, unemployment compensation, and tax revenues.

### Joint Committee on Taxation

JCT provides estimates of tax expenditures for major deductions and credits.

### Healthcare Spending Data

Various sources provide data on health insurance premiums, Medicare costs, and medical spending by age group.

### State Administrative Data

State-level program participation and spending data from various state agencies.

## Data Access and Documentation

The enhanced dataset is publicly available through Hugging Face at [https://huggingface.co/datasets/PolicyEngine/policyengine-us-data](https://huggingface.co/datasets/PolicyEngine/policyengine-us-data). We distribute the data as HDF5 files compatible with PolicyEngine and other microsimulation frameworks, with new releases accompanying each CPS vintage.

We maintain complete documentation of variable definitions, imputation procedures, and calibration targets in the project repository.