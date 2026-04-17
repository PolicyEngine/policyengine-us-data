# CPS Outgoing Rotation Group (ORG)

This folder contains the tooling that builds a labor-market donor frame
from the CPS basic monthly public-use files (`org.py`).

The CPS Outgoing Rotation Group (ORG) earnings questions are asked only
of the one-quarter of the sample that is rotating out in a given month.
Pooling the twelve monthly ORG samples for a year yields a donor frame
PolicyEngine uses to impute wage, hourly-pay, and union variables onto
the CPS ASEC records.

The checked-in code does not vendor the donor file itself. Instead,
`org.py` builds `census_cps_org_2024_wages.csv.gz` on demand by
downloading the twelve official CPS basic monthly public-use CSVs for
`ORG_YEAR` (currently 2024) directly from the Census Bureau and filtering
each file to the ORG rotations.

## Documentation

The Census Bureau and BLS publish a data dictionary and users' guide for
the CPS basic monthly public-use microdata. These are the canonical
reference for every variable name and earnings-recipiency code used by
the code in this folder:

- [2024 CPS basic monthly public-use record layout (TXT)](https://www2.census.gov/programs-surveys/cps/datasets/2024/basic/2024_Basic_CPS_Public_Use_Record_Layout_plus_IO_Code_list.txt)
- [CPS basic monthly documentation landing page](https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html)

See also:

- [CPS technical documentation](https://www.census.gov/programs-surveys/cps/technical-documentation.html)

## Data products in this folder

- `org.py` — downloads the twelve monthly CSVs, filters to the MIS-4 and
  MIS-8 outgoing rotations (`HRMIS`), and caches the combined ORG donor
  frame. Trains a QRF model to impute `wage_income`, `hourly_wage`, and
  union-coverage variables onto the CPS ASEC records used by the
  Enhanced CPS pipeline.
