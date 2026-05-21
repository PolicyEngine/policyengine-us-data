# Imputation

Use this guide when adding, changing, or reviewing donor-survey imputations.

## Source Provenance

Do not train an imputation target on donor rows whose source value for that
target is itself allocated, hot-decked, edited, or imputed by the source survey.
Wire source-survey allocation or quality flags into the training frame whenever
the donor file exposes them.

Apply this rule at the target-variable level, not the donor-row level. A donor
row with observed tip income but allocated bank-account assets can train
`tip_income`; the same row must be excluded from the `bank_account_assets`
training target. Use `policyengine_us_data.utils.source_quality` to build
target masks, then pass them to `microimpute` through `target_filters` or
`row_filter` so the filtering logic lives in the imputation library rather than
in one-off model wrappers.

Do not drop final CPS, ECPS, or calibration records solely because a donor
survey target was excluded from training. The exclusion applies to donor
training rows only; recipient datasets should remain complete.

When a donor source lacks target-level quality flags, document that limitation
near the imputation code and keep the training surface structured so flags can
be added later.

## Tests

Add focused regression tests when adding a donor imputation or a source-quality
flag:

- allocation flags are read from the donor source,
- allocated source values are excluded for the affected target,
- unrelated observed targets from the same row can still train, and
- legacy and current imputation surfaces use the same target provenance rule.
