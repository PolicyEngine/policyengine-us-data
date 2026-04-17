# Survey of Income and Program Participation (SIPP)

This folder contains the tooling that uses the Census Bureau's Survey of
Income and Program Participation (SIPP) as a donor source for imputations
onto the CPS (`sipp.py`).

PolicyEngine currently uses SIPP to train QRF imputation models for
tip income (using the SIPP job-level tip-amount columns) and for
household-level asset categories (bank, stock, bond, vehicle). These
models are then applied to the CPS-based Enhanced CPS to obtain
person-level tip income and household-level countable resources that the
CPS itself does not capture.

## Documentation

The Census Bureau publishes a users' guide and data dictionary for each
SIPP panel wave. These are the canonical reference for every variable
name, value code, and weighting construct used by the code in this
folder:

- [SIPP 2023 public-use data dictionary (PDF)](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf)
- [SIPP 2023 users' guide (PDF, Aug 2026 revision)](https://www2.census.gov/programs-surveys/sipp/tech-documentation/methodology/2023_SIPP_Users_Guide_AUG26.pdf)

See also:

- [SIPP landing page](https://www.census.gov/programs-surveys/sipp.html)
- [SIPP technical documentation](https://www.census.gov/programs-surveys/sipp/tech-documentation.html)
- [SIPP public-use datasets](https://www.census.gov/programs-surveys/sipp/data/datasets.html)

## Data products in this folder

- `sipp.py` — trains and caches QRF imputation models (`get_tip_model`,
  `get_asset_model`, `get_vehicle_model`) from SIPP 2023 person-month
  data. The training frame is filtered to `MONTHCODE == 12` (December)
  so every row represents one person-year rather than twelve annualized
  months.

The raw SIPP CSVs (`pu2023.csv` and the slim variant `pu2023_slim.csv`)
are mirrored on the `PolicyEngine/policyengine-us-data` HuggingFace model
repo and downloaded on demand when a training run is needed. They are
not vendored in this Git repository.
