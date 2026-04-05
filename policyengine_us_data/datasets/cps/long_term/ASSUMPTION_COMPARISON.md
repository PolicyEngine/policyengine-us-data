# Long-Run Calibration Assumption Comparison

This note distinguishes between:

- hard microsimulation calibration targets, which directly shape household weights
- tax-side assumptions used to make those targets more comparable to the public Trustees/OACT methodology

The current long-run baseline now adopts a named tax-side assumption,
`trustees-core-thresholds-v1`, before hard-targeting TOB.

| Component | Current `policyengine-us-data` approach | Trustees / OACT published approach | Calibration use |
| --- | --- | --- | --- |
| Population by age | SSA single-year age projections | SSA single-year age projections | Hard target |
| OASDI benefits | Named long-term target source package | Trustees or OACT-patched annual OASDI path | Hard target |
| Taxable payroll | Named long-term target source package | Trustees annual taxable payroll path | Hard target |
| Social Security benefit-tax thresholds | Literal current-law statutory thresholds remain fixed in nominal dollars | Trustees also describe the statutory `$25k/$32k/$0` and `$34k/$44k` thresholds as remaining fixed in nominal dollars | Not separately targeted |
| Federal income-tax brackets | Core ordinary thresholds are wage-indexed after `2034` via `trustees-core-thresholds-v1` | Trustees assume periodic future bracket adjustments; after the tenth projection year, ordinary federal income-tax brackets are assumed to rise with average wages to avoid indefinite bracket creep | Tax-side assumption |
| Standard deduction / aged-blind addition / capital gains thresholds / AMT thresholds | Included in the same `trustees-core-thresholds-v1` bundle | Not parameterized publicly line-by-line, but these are the main additional federal thresholds most likely to affect long-run TOB | Tax-side assumption |
| OASDI TOB | Computed under the core-threshold tax assumption and targeted in `ss-payroll-tob` profiles | Trustees/OACT publish annual revenue paths or ratios, but not a full public household-level micro rule schedule | Hard target |
| HI TOB | Computed under the core-threshold tax assumption and targeted in `ss-payroll-tob` profiles | Trustees publish current-law HI TOB path; OACT OBBBA updates do not currently provide a full public annual HI replacement series | Hard target |
| OBBBA OASDI update | Available through named target source `oact_2025_08_05_provisional` | August 5, 2025 OACT letter provides annual OASDI changes through 2100 | Benchmark / target-source input |
| OBBBA HI update | Provisional bridge only in named target source | No equivalent full public annual HI replacement path located yet | Benchmark only |

## Practical interpretation

- `ss-payroll` remains the core non-TOB hard-target profile.
- `ss-payroll-tob` now means: calibrate on age + OASDI benefits + taxable payroll + TOB under `trustees-core-thresholds-v1`.
- The core-threshold bundle is a best-public approximation, not a literal public Trustees rules schedule.
- Trustees-consistent long-run TOB requires keeping two different tax-side ideas separate:
  - the Social Security benefit-tax thresholds remain fixed in nominal dollars
  - ordinary federal income-tax brackets are assumed to rise with average wages after the tenth projection year

## Primary-source references

- [SSA 2025 Trustees Report, V.C.7](https://www.ssa.gov/oact/tr/2025/V_C_prog.html)
  - States that the law specifies fixed threshold amounts for taxation of Social Security benefits and that those thresholds remain constant in future years.
  - Also states that, after the tenth year of the projection period, income-tax brackets are assumed to rise with average wages rather than with `C-CPI-U`.
- [26 U.S.C. § 86](https://www.law.cornell.edu/uscode/text/26/86)
  - Statutory basis for the Social Security benefit-tax threshold structure.
- [SSA 2025 Trustees Report, Table VI.G6](https://www.ssa.gov/OACT/TR/2025/VI_G3_OASDHI_dollars.html)
  - Published annual average wage index path through `2100`.
- [42 U.S.C. § 430](https://www.law.cornell.edu/uscode/text/42/430) and [20 CFR § 404.1048](https://www.law.cornell.edu/cfr/text/20/404.1048)
  - Statutory and regulatory basis for deriving the Social Security contribution and benefit base from the wage index.
