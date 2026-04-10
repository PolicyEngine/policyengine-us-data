## Directory for storing calibration targets

This directory contains all data sources of the targets that will be calibrated for by the Enhanced CPS. Currently it stores all raw, or unprocessed targets as tracked csv files (for backward compatibility). Soon it will store scripts to pull data from each data source (one script per source) into long-formatted csv files that follow the column structure:

DATA_SOURCE,GEO_ID,GEO_NAME,VARIABLE,VALUE,IS_COUNT,BREAKDOWN_VARIABLE,LOWER_BOUND,UPPER_BOUND

To refresh the tracked SOI table targets from the IRS Publication 1304
workbooks, run:

`make refresh-soi-targets SOI_SOURCE_YEAR=2021 SOI_TARGET_YEAR=2022`

`make refresh-soi-targets SOI_SOURCE_YEAR=2021 SOI_TARGET_YEAR=2023`

This refresh path covers the tracked workbook-based national SOI table targets
in `soi_targets.csv`. The refresh code now rewrites the active Table 1.4 /
Table 2.1 targets with explicit semantic mappings for the current Publication
1304 layouts instead of reusing stale stored column letters.

`get_soi()` now selects the best available tracked year per variable for the
requested simulation year, so TY2024 uses TY2023 where available, TY2022 uses
TY2022, and older variables can still fall back to prior tracked years instead
of disappearing.

The DB-backed IRS SOI ETL now overlays the national targets it can source from
the workbook path using the latest published national year, independently of
the geography-file release cycle.

The separate state/district AGI pulls still rely on the IRS `in54`, `in55cm`,
and `incd` geography files, which remain on the latest published geography year.
`aca_ptc`, `refundable_ctc`, and `non_refundable_ctc` also still stay on that
geography-backed path for now, because the published national workbook tables
do not line up cleanly with the current `incd` code definitions.
