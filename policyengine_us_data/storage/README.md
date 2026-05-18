# storage/ datasets 

- **aca_spending_and_enrollment_2024.csv**  
  • Source: CMS Marketplace Public Use File, 2024 open-enrollment  
  • Date: 2024  
  • Location: https://www.cms.gov/files/document/health-insurance-exchanges-2024-open-enrollment-report-final.pdf

- **aca_spending_and_enrollment_2025.csv**  
  • Source: CMS “Effectuated Enrollment: Early 2025 Snapshot and Full Year 2024 Average”, Table 2 and Table 3  
  • Date: March 15, 2025 snapshot  
  • Location: https://www.cms.gov/files/document/effectuated-enrollment-early-snapshot-2025-and-full-year-2024-average.pdf  
  • Notes: `enrollment` is APTC enrollment by state; `spending` is monthly APTC enrollment multiplied by average monthly APTC for APTC recipients

- **aca_ptc_state.csv**
  • Source: IRS SOI Historical Table 2, total premium tax credit columns `N85770` and `A85770`
  • Date: tax year 2022
  • Created by: `policyengine_us_data/storage/calibration_targets/refresh_aca_ptc_state_targets.py`
  • Location: https://www.irs.gov/pub/irs-soi/22in55cmcsv.csv
  • Notes: used as the ACA PTC amount target because the model variable `aca_ptc` is total premium tax credit entitlement, not CMS advance PTC outlays. Target-year calibration applies checked-in 2022-to-target-year ACA PTC multipliers when available.

- **agi_state.csv**
  • Source: IRS SOI state data file used by legacy local calibration
  • Date: tax year 2022
  • Created by: `policyengine_us_data/storage/calibration_targets/refresh_local_agi_state_targets.py`
  • Location: https://www.irs.gov/pub/irs-soi/22in55cmcsv.csv
  • Notes: This file intentionally keeps the legacy `utils/loss.py` schema (`AL`, `DC`, etc.) instead of the newer `state_AL` geography naming used in `soi.csv`/database overlays. It is separate from `soi_targets.csv`, and it currently lags the national SOI refresh because IRS geographic state SOI files are only published through TY2022.

- **eitc_claim_controls.csv**
  • Source: IRS EITC Central, "2024 EITC tax returns by state processed in 2025"
  • Date: tax year 2024
  • Location: https://www.irs.gov/tax-professionals/eitc-central/statistics-for-tax-returns-with-the-earned-income-tax-credit-eitc
  • Notes: Used as the EITC claim-level control for legacy loss-matrix calibration. The detailed 2022 SOI EITC AGI and child-count table is retained as a shape source and scaled to this claim control. Treasury and CBO EITC outlay series are not used as EITC claim calibration targets.

- **medicaid_enrollment_2024.csv**  
  • Source: MACPAC Enrollment Tables, FFY 2024  
  • Date: 2024  
  • Location: https://www.medicaid.gov/resources-for-states/downloads/eligib-oper-and-enrol-snap-december2024.pdf#page=26

- **medicaid_enrollment_2025.csv**
  • Source: Medicaid.gov performance indicator dataset, updated December 2025 Applications, Eligibility, and Enrollment Data
  • Date: December 2025 final reports, last updated April 24, 2026
  • Location: https://data.medicaid.gov/dataset/6165f45b-ca93-5bb5-9d06-db29c692a360?conditions%5B0%5D%5Boperator%5D=%3D&conditions%5B0%5D%5Bproperty%5D=reporting_period&conditions%5B0%5D%5Bvalue%5D=202512&conditions%5B1%5D%5Boperator%5D=%3D&conditions%5B1%5D%5Bproperty%5D=preliminary_or_updated&conditions%5B1%5D%5Bvalue%5D=U
  • Notes: Uses `total_medicaid_enrollment`, not combined Medicaid and CHIP enrollment.

- **district_mapping.csv**
  • Source: created by the script `policyengine_us/storage/calibration_targets/make_district_mapping.py`
  • Notes: this script is not part of `make data` because of the length of time it takes to run and the
    likelhood of timeout errors. See the script for more notes, including an alternative source. Also,
    once the IRS SOI updates their data in 2026, this mapping will likely be unncessesary.

- **SSPopJul_TR2024.csv**
  • Source: SSA Single Year Age demographic projections, "Mid Year" file (latest published: 2024)
  • Date: 2024
  • Location: https://www.ssa.gov/oact/HistEst/Population/2024/Population2024.html
  • Related: Single Year supplementary tables available at https://www.ssa.gov/oact/tr/2025/lrIndex.html

- **social_security_aux.csv**
  • Source: SSA Single Year supplementary tables
  • Date: 2025 Trustees Report
  • Locations:
     - https://www.ssa.gov/oact/tr/2025/lrIndex.html
     - `https://www.ssa.gov/oact/solvency/provisions/tables/table_run133.html`
  • Notes: Contains OASDI cost projections and taxable payroll data (2025-2100)

- **long_term_target_sources/**
  • Source packages for long-term CPS calibration targets
  • Files:
     - `trustees_2025_current_law.csv`: explicit frozen copy of the legacy Trustees/current-law target path
     - `oact_2025_08_05_provisional.csv`: OACT-updated TOB path with provisional HI bridge
     - `oasdi_oact_20250805_nominal_delta.csv`: raw OASDI TOB deltas from the August 5, 2025 OACT letter
     - `sources.json`: provenance and source metadata for each named package
  • Notes: `run_household_projection.py --target-source ...` selects from these packages instead of relying on branch-specific data files

- **national_and_district_rents_2023.csv**
  • Source: Census ACS 5-year estimates (2023), median 2BR rent by congressional district
  • Created by: `fetch_cd_rents.py` (requires `CENSUS_API_KEY` environment variable)
  • Notes: Legacy source for SPM geographic adjustment factors; current calculations live in `policyengine-us`.
