# Results

We validate our enhanced dataset against official statistics and compare its performance to both the original CPS and PUF datasets. Our validation framework covers over 7,000 distinct targets spanning demographic totals, program participation rates, and income components across the distribution.

## Validation against administrative totals

The Enhanced CPS is validated against all 7,000+ administrative targets used in the calibration process. While the dataset is explicitly calibrated to these targets, the large number and diversity of targets makes achieving good fit across all dimensions a challenge.

Detailed validation results are available in our interactive dashboard at [https://policyengine.github.io/policyengine-us-data/validation.html](https://policyengine.github.io/policyengine-us-data/validation.html).

## Target category performance

The enhanced dataset is calibrated to various categories of targets:

**IRS Income Components**: AGI-stratified income targets including employment income, capital gains, partnership and S-corp income, and dividend income across filing statuses and income ranges.

**Program Participation**: CBO projections for SNAP benefits, Social Security, unemployment compensation, and income tax revenue.

**Demographic Targets**: Population by single year of age, state populations, and healthcare spending by age group.

**Tax Expenditures**: JCT estimates for SALT deduction ($21.2B), charitable deduction ($65.3B), mortgage interest ($24.8B), and medical expense deduction ($11.4B).

### Tax expenditure validation

We validate tax expenditures against JCT estimates from the [2024 Tax Expenditures Report (JCX-48-24)](https://www.jct.gov/publications/2024/jcx-48-24/):

| Deduction | JCT Target |
|-----------|------------|
| SALT | \$21.2B |
| Charitable | \$65.3B |
| Mortgage Interest | \$24.8B |
| Medical Expense | \$11.4B |

These targets are used in the calibration process (`policyengine_us_data/utils/loss.py`, lines 622-628).

## Income distribution

Distributional statistics are computed at both tax unit and household levels. Tax unit metrics allow comparison with the PUF, while household metrics are relevant for many policy applications.

### Tax unit level metrics

The Enhanced CPS achieves distributional statistics between those of the CPS and PUF. The imputation of tax variables from the PUF increases measured inequality compared to the baseline CPS.

The results notebook (`docs/results.ipynb`) calculates distributional metrics at the tax unit level using the PolicyEngine microsimulation framework. See the notebook for current values.

### Household level metrics

For applications requiring household-level analysis, we also compute metrics over households rather than tax units. The PUF cannot provide household-level statistics as it lacks household structure.

## Poverty measurement

Poverty metrics require interpretation. The interaction between imputed tax variables and poverty measurement is complex, and results may differ from official statistics.

### Poverty rate comparison

The results notebook (`docs/results.ipynb`) calculates SPM poverty rates for both datasets. See the notebook for current values. Users analyzing poverty should:

- Compare results across different weight specifications
- Consider the impact of tax variable imputation
- Reference official poverty statistics for validation

## Weight distribution

The weight distribution reflects the enhancement methodology:

- Original CPS weights are relatively uniform
- Enhanced CPS weights show greater variation due to the calibration process
- Some records receive zero weight as the optimization selects representative combinations

## Policy Application: top tax rate reform

To demonstrate practical applications, we analyze a reform raising the top marginal tax rate from 37% to 39.6%. This reform affects high-income taxpayers and tests the dataset's ability to model policies targeting the top of the income distribution.

The Enhanced CPS incorporates income data from the PUF, enabling analysis of high-income tax reforms that would be difficult with the CPS alone.

## Validation dashboard

Our validation dashboard provides:
- Performance metrics for all 7,000+ targets
- Comparison across datasets
- Filtering by target category and source
- Regular updates with each data release

Visit [https://policyengine.github.io/policyengine-us-data/validation.html](https://policyengine.github.io/policyengine-us-data/validation.html) to explore results.

## Summary

The Enhanced CPS successfully combines the demographic detail of the CPS with tax precision approaching that of administrative data. While specific validation metrics depend on the target category, the dataset provides a suitable foundation for analyzing both tax and transfer policies.

Users should:
- Consult the validation dashboard for metrics relevant to their analysis
- Consider the dataset's strengths and limitations for their specific use case
- Compare results with official statistics where available