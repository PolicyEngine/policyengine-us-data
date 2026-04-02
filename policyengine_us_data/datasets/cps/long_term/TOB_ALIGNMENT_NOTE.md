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

Example:

```bash
uv run python policyengine_us_data/datasets/cps/long_term/compare_tob_shares.py \
  /path/to/2075-output-dir \
  /path/to/2090-output-dir \
  /path/to/2100-output-dir
```

The script expects metadata sidecars containing:

- `calibration_audit.constraints.ss_total`
- `calibration_audit.benchmarks.oasdi_tob`
- `calibration_audit.benchmarks.hi_tob`

To regenerate the underlying sidecars from scratch, first ensure that
`policyengine-us` includes the Social Security wage-base extension from
[PR #7912](https://github.com/PolicyEngine/policyengine-us/pull/7912) or later.
Without that fix, late-year taxable-payroll calibration is materially wrong.
