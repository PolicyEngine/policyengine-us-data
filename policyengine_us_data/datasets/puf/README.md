# IRS Public Use File (PUF)

This folder contains the tooling that ingests the IRS Statistics of Income
Public Use File into PolicyEngine's US microdata pipeline
(`irs_puf.py`, `puf.py`, `disaggregate_puf.py`, `uprate_puf.py`, and
the supporting aggregate-record utilities).

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
