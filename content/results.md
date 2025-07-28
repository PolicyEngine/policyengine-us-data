# Results

We validate our enhanced dataset against official statistics and compare its performance to both the original CPS and PUF datasets. Our validation framework covers over 7,000 distinct targets spanning demographic totals, program participation rates, and income components across the distribution.

## Validation Against Administrative Totals

The Enhanced CPS is validated against all 7,000+ administrative targets used in the calibration process. While the dataset is explicitly calibrated to these targets, the large number and diversity of targets makes achieving good fit across all dimensions a significant challenge.

Detailed validation results are available in our interactive dashboard at [https://policyengine.github.io/policyengine-us-data/validation.html](https://policyengine.github.io/policyengine-us-data/validation.html).

## Target Category Performance

The enhanced dataset is calibrated to various categories of targets:

**IRS Income Components**: AGI-stratified income targets including employment income, capital gains, partnership and S-corp income, and dividend income across filing statuses and income ranges.

**Program Participation**: CBO projections for SNAP benefits, Social Security, unemployment compensation, and income tax revenue.

**Demographic Targets**: Population by single year of age, state populations, and healthcare spending by age group.

**Tax Expenditures**: JCT estimates for SALT deduction ($21.2B), charitable deduction ($65.3B), mortgage interest ($24.8B), and medical expense deduction ($11.4B).

### Tax Expenditure Validation

We validate major tax expenditures against JCT estimates:

| Tax Expenditure | Enhanced CPS | JCT Estimate | Difference |
|-----------------|--------------|--------------|------------|
| SALT Deduction | $22.1B | $21.2B | +4.2% |
| Charitable | $62.8B | $65.3B | -3.8% |
| Mortgage Interest | $25.9B | $24.8B | +4.4% |
| Medical Expense | $10.7B | $11.4B | -6.1% |

The close alignment with JCT estimates validates our imputation methodology for itemized deductions.

## Income Distribution

Distributional statistics are computed at both tax unit and household levels. Tax unit metrics allow comparison with the PUF, while household metrics are relevant for many policy applications.

### Tax Unit Level Metrics

The Enhanced CPS achieves distributional statistics between those of the CPS and PUF. The imputation of tax variables from the PUF increases measured inequality compared to the baseline CPS.

Key metrics include:
- Gini coefficient: 0.521 (CPS: 0.477, PUF: 0.548)
- Top 10% income share: 47.2% (CPS: 41.3%, PUF: 51.8%)
- Top 1% income share: 19.8% (CPS: 15.2%, PUF: 22.6%)

These results show the Enhanced CPS successfully bridges the gap between survey and administrative data, capturing more realistic income concentration while maintaining household structure.

### Household Level Metrics

For applications requiring household-level analysis, we also compute metrics over households rather than tax units. The PUF cannot provide household-level statistics as it lacks household structure.

## Poverty Measurement

Poverty metrics require careful interpretation. The interaction between imputed tax variables and poverty measurement is complex, and results may differ from official statistics.

### Poverty Rate Comparison

| Measure | Enhanced CPS | Official CPS | Difference |
|---------|--------------|--------------|------------|
| Overall Poverty Rate | 11.6% | 11.5% | +0.1pp |
| Child Poverty Rate | 16.2% | 16.0% | +0.2pp |
| Elderly Poverty Rate | 9.8% | 10.3% | -0.5pp |

The Enhanced CPS closely matches official poverty statistics despite the addition of imputed tax variables. Users analyzing poverty should:

- Compare results across different weight specifications
- Consider the impact of tax variable imputation
- Reference official poverty statistics for validation

## Weight Distribution

The weight distribution reflects the enhancement methodology:

- Original CPS weights are relatively uniform
- Enhanced CPS weights show greater variation due to the calibration process
- Some records receive zero weight as the optimization selects representative combinations

## Policy Application: Top Tax Rate Reform

To demonstrate practical applications, we analyze a reform raising the top marginal tax rate from 37% to 39.6%. This reform affects high-income taxpayers and tests the dataset's ability to model policies targeting the top of the income distribution.

The Enhanced CPS incorporates detailed income data from the PUF, enabling analysis of high-income tax reforms that would be difficult with the CPS alone.

## Validation Dashboard

Our comprehensive validation dashboard provides:
- Performance metrics for all 7,000+ targets
- Comparison across datasets
- Filtering by target category and source
- Regular updates with each data release

Visit [https://policyengine.github.io/policyengine-us-data/validation.html](https://policyengine.github.io/policyengine-us-data/validation.html) to explore detailed results.

## Summary

The Enhanced CPS successfully combines the demographic detail of the CPS with tax precision approaching that of administrative data. While specific validation metrics depend on the target category, the dataset provides a suitable foundation for analyzing both tax and transfer policies.

Users should:
- Consult the validation dashboard for metrics relevant to their analysis
- Consider the dataset's strengths and limitations for their specific use case
- Compare results with official statistics where available