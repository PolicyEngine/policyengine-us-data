# Current Population Survey (CPS ASEC)

This folder contains the tooling that ingests the Census Bureau's Current
Population Survey Annual Social and Economic Supplement (CPS ASEC) into
PolicyEngine's US microdata pipeline (`census_cps.py`, `cps.py`,
`enhanced_cps.py`, `extended_cps.py`, `small_enhanced_cps.py`, `takeup.py`,
and `tipped_occupation.py`).

The CPS ASEC is the Census Bureau / Bureau of Labor Statistics' primary
source of annual demographic and income data for the US civilian
noninstitutional population. PolicyEngine uses it as the demographic
backbone of the Enhanced CPS; tax-return detail from the IRS PUF is then
merged onto each CPS record.

## Documentation

The Census Bureau publishes a data dictionary and technical documentation
for each ASEC vintage. These are the canonical reference for every
variable name, code, and SPM/tax-unit construction used by the code in
this folder:

- [2023 ASEC data dictionary (full PDF)](https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asec2023_ddl_pub_full.pdf)
- [2024 ASEC data dictionary (full PDF)](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf)
- [2025 ASEC data dictionary (full PDF)](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)

See also:

- [CPS ASEC landing page](https://www.census.gov/programs-surveys/cps.html)
- [CPS ASEC technical documentation](https://www.census.gov/programs-surveys/cps/technical-documentation.html)
- [CPS ASEC public-use microdata datasets](https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html)

The exact Census URLs the pipeline downloads for each ASEC year are
enumerated in `CPS_URL_BY_YEAR` inside `census_cps.py`.

## Data products in this folder

- `census_cps.py` — downloads and stages the raw ASEC person/family/
  household tables from Census for a given ASEC year.
- `cps.py` — derives the PolicyEngine `CPS` dataset (PolicyEngine variable
  names, entity structure, SPM units, tax units) from the Census tables.
- `enhanced_cps.py`, `extended_cps.py`, `small_enhanced_cps.py` —
  downstream enhanced datasets that merge PUF-based tax-return detail and
  imputed variables onto the CPS backbone.
- `takeup.py` — program take-up anchoring against reported CPS recipiency.
- `tipped_occupation.py` — Treasury tipped-occupation code derivation.
- `imputation_parameters.yaml` — hyperparameters for QRF imputations used
  by the enhanced CPS pipeline.
