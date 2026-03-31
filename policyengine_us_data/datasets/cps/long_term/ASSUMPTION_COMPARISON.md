# Long-Run Calibration Assumption Comparison

This note distinguishes between:

- hard microsimulation calibration targets, which directly shape household weights
- post-calibration benchmarks, which are compared against published source packages without being forced into the weights

The split is intentional. Long-run Taxation of Benefits (TOB) depends on tax-side assumptions that are not fully published as a household-level rule schedule in the Trustees/OACT materials.

| Component | Current `policyengine-us-data` approach | Trustees / OACT published approach | Calibration use |
| --- | --- | --- | --- |
| Population by age | SSA single-year age projections | SSA single-year age projections | Hard target |
| OASDI benefits | Named long-term target source package | Trustees or OACT-patched annual OASDI path | Hard target |
| Taxable payroll | Named long-term target source package | Trustees annual taxable payroll path | Hard target |
| Social Security benefit-tax thresholds | Literal current-law statutory thresholds remain fixed in nominal dollars | Trustees also describe statutory thresholds as remaining fixed in nominal dollars | Not separately targeted |
| Federal income-tax brackets | PolicyEngine tax simulation using its own income-tax parameter uprating path | Trustees assume periodic future bracket adjustments; after the tenth year, tax brackets are assumed to rise with average wages to avoid indefinite bracket creep | Not hard-targeted |
| OASDI TOB | Computed from the tax microsimulation and compared to the selected target source | Trustees/OACT publish annual revenue paths or ratios, but not a full public household-level micro rule schedule | Post-calibration benchmark |
| HI TOB | Computed from the tax microsimulation and compared to the selected target source | Trustees publish current-law HI TOB path; OACT OBBBA updates do not currently provide a full public annual HI replacement series | Post-calibration benchmark |
| OBBBA OASDI update | Available through named target source `oact_2025_08_05_provisional` | August 5, 2025 OACT letter provides annual OASDI changes through 2100 | Benchmark / target-source input |
| OBBBA HI update | Provisional bridge only in named target source | No equivalent full public annual HI replacement path located yet | Benchmark only |

## Practical interpretation

- `ss-payroll` is the core hard-target profile.
- `ss-payroll-tob` now means: calibrate on age + OASDI benefits + taxable payroll, then benchmark TOB against the selected source package.
- TOB remains important for model comparison, but it is no longer treated as a weight-identifying target in the long-run microsimulation contract.
