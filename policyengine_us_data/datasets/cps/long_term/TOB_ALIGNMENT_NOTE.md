# Long-Run TOB Alignment Note

This note records the current apples-to-apples comparison point for long-run
taxation of benefits (TOB).

## Why this note exists

Our earlier comparison mixed two different objects:

- `OASDI-only` income from taxation of benefits as a share of total OASDI
  benefits
- combined `OASDI + HI` credited taxes as a share of total OASDI benefits

The public Urban / DYNASIM and SSA Trustees discussion is about the former,
not the latter.

## Primary-source baseline

- SSA 2025 Trustees Report, [V.C.7](https://www.ssa.gov/oact/tr/2025/V_C_prog.html)
  says the benefit-tax thresholds are "constant in the future" and, after the
  tenth projection year, ordinary income-tax brackets "rise with average
  wages."
- Urban's 2024 DYNASIM appendix says it continues current indexing of income
  tax parameters indefinitely and keeps the Social Security benefit-tax
  thresholds at current nominal levels throughout the projection period. Source:
  [Urban 2024 appendix](https://www.urban.org/sites/default/files/2024-10/Does-the-2023-Social-Security-Expansion-Act-Improve-Equity-in-Key-Outcomes.pdf).
- SSA's published OASDI long-run target series in
  [Table IV.B2](https://www.ssa.gov/oact/tr/2025/lr4b2.html) and our local
  `trustees_2025_current_law.csv` target package imply OASDI-only TOB shares
  of about `6.0%` to `6.1%` of OASDI benefits in the late horizon.

## Current branch-local comparison

The table below uses one-year probe outputs produced on `2026-04-02` with:

- `policyengine-us-data` branch `codex/us-data-calibration-contract`
- `policyengine-us` branch `codex/extend-ss-cap-2100` or equivalent fix from
  [PR #7912](https://github.com/PolicyEngine/policyengine-us/pull/7912)
- profile `ss-payroll-tob`
- target source `trustees_2025_current_law`
- donor-composite support augmentation enabled

| Year | OASDI actual | OASDI target | OASDI gap | Combined actual | Combined target | Combined gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2075 | 9.43% | 6.01% | +3.42 pp | 12.58% | 11.09% | +1.50 pp |
| 2090 | 10.52% | 6.08% | +4.44 pp | 16.66% | 11.20% | +5.46 pp |
| 2100 | 11.16% | 6.10% | +5.06 pp | 19.35% | 11.22% | +8.14 pp |

Interpretation:

- The `OASDI-only` comparison is the right one for evaluating alignment to the
  public Urban / SSA discussion.
- On that comparable metric, the current corrected baseline is above both the
  Trustees-style target path (`~6.1%`) and Urban's public DYNASIM endpoint
  (`8.5%` in `2095`).
- The larger combined `OASDI + HI` shares are still useful internal diagnostics,
  but they should not be compared directly to the public `5.6%` / `8.5%`
  figures.

## Trustees-style bracket-growth sensitivity

We also ran a narrow tax-side sensitivity that keeps the calibrated household
weights fixed and changes only one assumption in the tax model:

- after `2034`, ordinary federal income-tax bracket thresholds are uprated with
  `NAWI` instead of `C-CPI-U`
- the Social Security benefit-tax thresholds remain fixed

This is intended as a best-public approximation to the Trustees statement that
ordinary income-tax brackets "rise with average wages" after the tenth
projection year.

| Year | Baseline OASDI | Wage-indexed-brackets OASDI | Trustees target | Remaining gap |
| --- | ---: | ---: | ---: | ---: |
| 2075 | 9.43% | 8.41% | 6.01% | +2.40 pp |
| 2090 | 10.52% | 7.85% | 6.08% | +1.76 pp |
| 2100 | 11.16% | 9.46% | 6.10% | +3.36 pp |

Interpretation:

- This tax-side assumption moves the modeled OASDI-only TOB share materially in
  the right direction.
- It explains a substantial share of the excess over the Trustees target,
  especially around `2090`.
- It does not explain the whole gap. Even with wage-indexed ordinary brackets,
  the long-run `2100` OASDI-only share remains well above the Trustees-style
  `~6.1%` path.
- Relative to Urban's public DYNASIM endpoint of `8.5%` in `2095`, the
  wage-indexed-brackets sensitivity lands in the same rough range by `2090`,
  but is still above that public number by `2100`.

## Broader core-threshold sensitivity

We also ran a broader but still targeted tax-side sensitivity that switches a
core set of federal thresholds from `C-CPI-U` to `NAWI` after `2034`:

- ordinary income-tax brackets
- standard deduction
- aged/blind additional standard deduction
- capital-gains rate thresholds
- AMT bracket threshold and exemption thresholds

This is broader than the minimum public Trustees approximation, but still
narrower than switching the entire `gov.irs.uprating` family to wages.

| Year | Baseline OASDI | Core-threshold OASDI | Trustees target | Remaining gap |
| --- | ---: | ---: | ---: | ---: |
| 2075 | 9.43% | 7.65% | 6.01% | +1.64 pp |
| 2090 | 10.52% | 7.31% | 6.08% | +1.22 pp |
| 2100 | 11.16% | 8.15% | 6.10% | +2.05 pp |

Interpretation:

- The broader threshold bundle explains more of the TOB gap than brackets
  alone.
- The additional movement is meaningful, especially in `2100`, where the
  OASDI-only share falls from `9.46%` under brackets-only to `8.15%` under the
  broader core-threshold sensitivity.
- Even this broader sensitivity still does not fully reconcile the modeled TOB
  path to the Trustees target, so some remaining gap likely reflects
  beneficiary income mix, filing composition, or other Treasury-ratio modeling
  differences.

## Full IRS-uprating upper bound

Finally, we ran an upper-bound sensitivity that rewrites every materialized IRS
parameter leaf that currently inherits from `gov.irs.uprating`, replacing
post-`2034` `C-CPI-U` growth with `NAWI` growth.

This is broader than the public Trustees text justifies, but it provides a
useful ceiling on how much of the TOB gap could plausibly be explained by the
IRS uprating family alone.

| Year | Baseline OASDI | Full IRS-uprating OASDI | Trustees target | Remaining gap |
| --- | ---: | ---: | ---: | ---: |
| 2075 | 9.43% | 7.46% | 6.01% | +1.45 pp |
| 2090 | 10.52% | 7.17% | 6.08% | +1.09 pp |
| 2100 | 11.16% | 8.16% | 6.10% | +2.06 pp |

Interpretation:

- The full IRS-uprating upper bound is only slightly lower than the narrower
  core-threshold bundle.
- That implies most of the tax-side movement is already coming from the core
  federal threshold families, not from the rest of the CPI-uprated IRS
  parameter tree.
- Even under this broad upper bound, the model still remains above the
  Trustees OASDI-only TOB path, especially in `2100`.

## DYNASIM public benchmark

Urban's 2024 appendix says DYNASIM's revenue from taxing Social Security
benefits rises from `5 percent` in `2027` to `8.5 percent` in `2095`, while
the Social Security actuaries' corresponding share rises from `5 percent` to
`5.6 percent` over the same period.

Important caveats:

- We have not found a public annual DYNASIM TOB series, only these endpoint
  shares.
- The Urban paper is tied to a 2023-vintage Trustees baseline. Comparing the
  2023 and 2025 Trustees reports, the long-run CPI and average-wage growth
  assumptions are effectively unchanged, so that vintage difference does not
  explain most of our remaining gap.

## Reproducing the table

The comparison script is:

- [compare_tob_shares.py](./compare_tob_shares.py)
- [benchmark_trustees_bracket_indexing.py](./benchmark_trustees_bracket_indexing.py)

Example:

```bash
uv run python policyengine_us_data/datasets/cps/long_term/compare_tob_shares.py \
  /path/to/2075-output-dir \
  /path/to/2090-output-dir \
  /path/to/2100-output-dir

uv run python policyengine_us_data/datasets/cps/long_term/benchmark_trustees_bracket_indexing.py \
  /path/to/2075-output-dir \
  /path/to/2090-output-dir \
  /path/to/2100-output-dir \
  --policyengine-us-path /path/to/local/policyengine-us
```

The script expects metadata sidecars containing:

- `calibration_audit.constraints.ss_total`
- `calibration_audit.benchmarks.oasdi_tob`
- `calibration_audit.benchmarks.hi_tob`

To regenerate the underlying sidecars from scratch, first ensure that
`policyengine-us` includes the Social Security wage-base extension from
[PR #7912](https://github.com/PolicyEngine/policyengine-us/pull/7912) or later.
Without that fix, late-year taxable-payroll calibration is materially wrong.
