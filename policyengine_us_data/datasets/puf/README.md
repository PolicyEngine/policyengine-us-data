# IRS Public Use File (PUF)

This folder contains the tooling that ingests the IRS Statistics of Income
Public Use File into PolicyEngine's US microdata pipeline
(`irs_puf.py`, `puf.py`, `disaggregate_puf.py`, `uprate_puf.py`, and
the supporting aggregate-record utilities).

The `$100M+` aggregate record (`RECID 999999`) now has an optional
Forbes-backed synthesis path. It pulls a public US rich-list backbone
from the `rtb-api` project, using the canonical `2024-09-01` snapshot
(the valuation date Forbes uses for the 2024 Forbes 400 list) from a
pinned `rtb-api` commit. The normalized top-400 snapshot and normalized
top-tail SCF donor pool are committed in this package so default builds
are reproducible offline; explicit refresh runs can still write local
cache files under [`policyengine_us_data/storage`](../../storage). The
builder then creates the top tail in two stages:

- `Forbes -> SCF`: selected Forbes units are expanded into replicate
  draws, and the same top-tail SCF donor model is used both to decide
  which Forbes units enter the `$100M+` bucket and to draw each unit's
  joint wealth-to-income regime.
- `SCF -> PUF`: those SCF draws are matched to top-tail PUF donors to
  fill tax-return detail that SCF does not directly observe.

The builder creates a staged artifact with the source Forbes snapshot,
selected Forbes units, SCF draws, PUF priors, calibrated synthetic rows,
and diagnostics. Only the final synthetic rows are upserted into the PUF
aggregate-record replacement path. If the Forbes snapshot or SCF donor
pool cannot be loaded in the production disaggregation entry point, the
code falls back to the existing donor-based disaggregation path.

For PR review or local validation, build the staged artifact and inspect
the deterministic diagnostics before running the full data pipeline:

```python
from policyengine_us_data.datasets.puf.forbes_backbone import (
    build_forbes_top_tail_artifact,
    build_forbes_top_tail_diagnostic_tables,
    format_forbes_top_tail_diagnostics,
)

artifact = build_forbes_top_tail_artifact(...)
tables = build_forbes_top_tail_diagnostic_tables(artifact, row, amount_columns)
print(format_forbes_top_tail_diagnostics(tables))
```

The diagnostic bundle includes a one-row summary, exact calibration
errors by PUF amount column, component composition comparing SCF priors,
PUF priors, calibrated synthetic totals, and target totals, selected
Forbes units, and SCF draw composition. The formatted summary includes
ASCII bar visuals so it can be pasted directly into a PR or CI log.

The PUF is an IRS SOI Division sample of individual income-tax returns,
stripped of direct identifiers, with top-coded amounts and
disclosure-avoidance perturbations applied. PolicyEngine uses the 2015
tax-year PUF as the tax-return backbone that is then merged into the
Enhanced CPS.

## Documentation

The IRS publishes two booklets describing the PUF file layout, field
definitions, and disclosure-avoidance methodology. Both are public
information and the canonical reference for anything this folder does:

- [2015 Public Use Booklet (main file documentation)](https://github.com/user-attachments/files/17535835/2015.Public.Use.Booklet.pdf)
- [2015 Public Use Booklet - demographic supplement](https://github.com/user-attachments/files/17535839/2015.Public.Use.Booklet.demographic.pdf)

See also:

- [IRS SOI Individual Tax Statistics (public landing page)](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics)
- [IRS SOI Tax Stats: Individual Public Use Microdata Files](https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files)

## Licensing and redistribution

The PUF itself is a sensitive dataset. IRS SOI distributes it under a
paid agreement that restricts redistribution; users must obtain their
own copy directly from the IRS. PolicyEngine does **not** redistribute
the raw PUF CSVs or derived row-level PUF data, and the H5 outputs that
include PUF-derived records are only published to access-controlled
locations.

The two booklets linked above are the only PUF-related artefacts that
are freely redistributable, and they are linked here only as
documentation of the source file format.

If you have access to the PUF, place the two source CSVs
(`puf_2015.csv` and `demographics_2015.csv`) in the local storage folder
referenced by `irs_puf.py` before running the build.
