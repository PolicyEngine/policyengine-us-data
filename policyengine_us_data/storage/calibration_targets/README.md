## Directory for storing calibration targets

This directory contains all data sources of the targets that will be calibrated for by the Enhanced CPS. Currently it stores all raw, or unprocessed targets as tracked csv files (for backward compatibility). Soon it will store scripts to pull data from each data source (one script per source) into long-formatted csv files that follow the column structure:

DATA_SOURCE,GEO_ID,GEO_NAME,VARIABLE,VALUE,IS_COUNT,BREAKDOWN_VARIABLE,LOWER_BOUND,UPPER_BOUND

To refresh the tracked SOI table targets from the latest IRS workbook release, run:

`make refresh-soi-targets SOI_TARGET_YEAR=2023`

This refresh path covers the tracked workbook-based SOI table targets in
`soi_targets.csv`. The separate state/district AGI pulls still rely on the IRS
`in54`, `in55cm`, and `incd` files.
