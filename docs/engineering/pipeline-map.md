# Pipeline Map

Generated from `docs/pipeline_map.yaml` and `@pipeline_node` decorators.

## Canonical Stages

| Stage | Title | Manifest steps |
| --- | --- | --- |
| `1_build_datasets` Stage 1 | Build Datasets | `01_build_datasets`, `04_stage_base_datasets` |
| `2_build_calibration_package` Stage 2 | Build Calibration Package | `02_build_package` |
| `3_fit_weights` Stage 3 | Fit Weights | `03_fit_weights_regional`, `03_fit_weights_national` |
| `4_build_outputs` Stage 4 | Build Outputs | `04_build_h5_regional`, `04_build_h5_national`, `04_upload_diagnostics` |
| `5_validate_and_promote_release` Stage 5 | Validate and Promote Release | `05_promote_release` |

## Stage 1: Build Datasets

Produce raw, base, extended, enhanced, stratified, source-imputed, and staged base datasets.

### Substage 1a: Raw Data Download

Download raw survey data from Census, IRS, Federal Reserve, and HuggingFace

- Substage ID: `1a_raw_data_download`
- Canonical stage: `1_build_datasets`
- Legacy stage: `0`
- Manifest steps: `01_build_datasets`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `cps_url` Census CPS ASEC | `artifact` | `unknown` | `unknown` |  |
| `acs_url` Census ACS PUMS | `artifact` | `unknown` | `unknown` |  |
| `scf_url` Federal Reserve SCF | `artifact` | `unknown` | `unknown` |  |
| `hf_private` HuggingFace Private Repo | `external` | `unknown` | `unknown` |  |
| `hf_public` HuggingFace Public Repo | `external` | `unknown` | `unknown` |  |
| `download_http` HTTP Download + ZIP Extract | `process` | `unknown` | `unknown` |  |
| `download_hf` HuggingFace Hub Download | `process` | `unknown` | `unknown` |  |
| `csv_parse` CSV/Stata Parsing | `process` | `unknown` | `unknown` |  |
| `out_cps_raw` census_cps_2024.h5 | `artifact` | `unknown` | `unknown` |  |
| `out_acs_raw` census_acs_2022.h5 | `artifact` | `unknown` | `unknown` |  |
| `out_puf_raw` irs_puf_2015.h5 | `artifact` | `unknown` | `unknown` |  |
| `out_soi` soi.csv | `artifact` | `unknown` | `unknown` |  |
| `out_scf` SCF raw data | `artifact` | `unknown` | `unknown` |  |
| `out_sipp` pu2023_slim.csv | `artifact` | `unknown` | `unknown` |  |
| `out_block` block_cd_distributions.csv.gz | `artifact` | `unknown` | `unknown` |  |
| `out_pop` np2023_d5_mid.csv | `artifact` | `unknown` | `unknown` |  |
| `out_calibration_db` policy_data.db | `artifact` | `unknown` | `unknown` |  |
| `util_storage` STORAGE_FOLDER | `utility` | `unknown` | `unknown` |  |

#### Edges

- `cps_url` -> `download_http` `external_source` (CPS ASEC ZIP)
- `acs_url` -> `download_http` `external_source` (ACS PUMS CSV)
- `scf_url` -> `download_http` `external_source` (SCF .dta)
- `hf_private` -> `download_hf` `external_source` (PUF, demographics, SOI, pop)
- `hf_public` -> `download_hf` `external_source` (SIPP, block, policy_data.db)
- `download_http` -> `csv_parse` `data_flow` (raw files)
- `download_hf` -> `csv_parse` `data_flow` (raw files)
- `csv_parse` -> `out_cps_raw` `produces_artifact` (census_cps_2024.h5)
- `csv_parse` -> `out_acs_raw` `produces_artifact` (census_acs_2022.h5)
- `csv_parse` -> `out_puf_raw` `produces_artifact` (irs_puf_2015.h5)
- `csv_parse` -> `out_soi` `produces_artifact` (soi.csv)
- `download_http` -> `out_scf` `produces_artifact` (SCF raw data)
- `download_hf` -> `out_sipp` `produces_artifact` (pu2023_slim.csv)
- `download_hf` -> `out_block` `produces_artifact` (block_cd_distributions.csv.gz)
- `download_hf` -> `out_pop` `produces_artifact` (np2023_d5_mid.csv)
- `download_hf` -> `out_calibration_db` `produces_artifact` (policy_data.db)

### Substage 1b: Base Dataset Construction

Build CPS 2024 and PUF 2024 from raw survey data, donor-based labor-market imputations, and retirement contribution inference

- Substage ID: `1b_base_dataset_construction`
- Canonical stage: `1_build_datasets`
- Legacy stage: `1`
- Manifest steps: `01_build_datasets`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_census_cps` census_cps_2024.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_census_cps_prev` census_cps_2023.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_acs` ACS 2022 | `artifact` | `unknown` | `unknown` |  |
| `in_sipp` SIPP 2023 | `artifact` | `unknown` | `unknown` |  |
| `in_scf` SCF 2022 | `artifact` | `unknown` | `unknown` |  |
| `in_org` CPS Basic ORG 2024 | `external` | `unknown` | `unknown` |  |
| `in_uprating` uprating_factors.csv | `artifact` | `unknown` | `unknown` |  |
| `out_cps` cps_2024.h5 | `artifact` | `unknown` | `unknown` |  |
| `out_puf` puf_2024.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_irs_puf` irs_puf_2015.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_demographics` demographics_2015.csv | `artifact` | `unknown` | `unknown` |  |
| `in_cps_pension` CPS_2024 / CPS_2021 | `artifact` | `unknown` | `unknown` |  |
| `util_seeded_rng` seeded_rng() | `utility` | `unknown` | `unknown` |  |
| `util_qrf` microimpute QRF | `utility` | `unknown` | `unknown` |  |
| `util_retirement_limits` get_retirement_limits() | `utility` | `unknown` | `unknown` |  |
| `add_id_variables` Add ID Variables | `library` | `current` | `stable` | `policyengine_us_data.datasets.cps.cps.add_id_variables` |
| `add_personal_variables` Add Personal Variables | `library` | `current` | `moving` | `policyengine_us_data.datasets.cps.cps.add_personal_variables` |
| `add_personal_income_variables` Add Income Variables | `library` | `current` | `moving` | `policyengine_us_data.datasets.cps.cps.add_personal_income_variables` |
| `add_previous_year_income` Previous-Year Income | `library` | `current` | `moving` | `policyengine_us_data.datasets.cps.cps.add_previous_year_income` |
| `add_ssn_card_type` Add SSN Card Type | `library` | `current` | `moving` | `policyengine_us_data.datasets.cps.cps.add_ssn_card_type` |
| `add_spm_variables` Add SPM Variables | `library` | `current` | `moving` | `policyengine_us_data.datasets.cps.cps.add_spm_variables` |
| `add_household_variables` Add Household Variables | `library` | `current` | `stable` | `policyengine_us_data.datasets.cps.cps.add_household_variables` |
| `add_rent` Rent Imputation | `library` | `legacy` | `moving` | `policyengine_us_data.datasets.cps.cps.add_rent` |
| `add_tips` Tips And Asset Imputation | `library` | `legacy` | `moving` | `policyengine_us_data.datasets.cps.cps.add_tips` |
| `add_org_inputs` ORG Labor-Market Inputs | `library` | `current` | `moving` | `policyengine_us_data.datasets.cps.cps.add_org_labor_market_inputs` |
| `add_auto_loan` Auto Loan And Net Worth Imputation | `library` | `legacy` | `moving` | `policyengine_us_data.datasets.cps.cps.add_auto_loan_interest_and_net_worth` |
| `add_takeup` Benefit Takeup | `library` | `current` | `moving` | `policyengine_us_data.datasets.cps.cps.add_takeup` |
| `downsample` Downsample CPS | `library` | `current` | `stable` | `policyengine_us_data.datasets.cps.cps.CPS.downsample` |
| `preprocess_puf` Preprocess PUF | `library` | `current` | `moving` | `policyengine_us_data.datasets.puf.puf.preprocess_puf` |
| `simulate_qbi` QBI Simulation | `library` | `current` | `moving` | `policyengine_us_data.datasets.puf.puf.simulate_w2_and_ubia_from_puf` |
| `impute_puf_demographics` Impute PUF Demographics | `library` | `current` | `moving` | `policyengine_us_data.datasets.puf.puf.impute_missing_demographics` |
| `impute_puf_pension` Impute PUF Pension Contributions | `library` | `current` | `moving` | `policyengine_us_data.datasets.puf.puf.impute_pension_contributions_to_puf` |
| `mortgage_convert` Structural Mortgage Conversion | `library` | `current` | `moving` | `policyengine_us_data.utils.mortgage_interest.convert_mortgage_interest_to_structural_inputs` |

#### Edges

- `in_census_cps` -> `add_id_variables` `data_flow` (raw CPS tables)
- `add_id_variables` -> `add_personal_variables` `data_flow`
- `add_personal_variables` -> `add_personal_income_variables` `data_flow`
- `add_personal_income_variables` -> `add_previous_year_income` `data_flow`
- `in_census_cps_prev` -> `add_previous_year_income` `data_flow` (prior year PERIDNUM)
- `add_previous_year_income` -> `add_ssn_card_type` `data_flow`
- `add_ssn_card_type` -> `add_spm_variables` `data_flow`
- `add_spm_variables` -> `add_household_variables` `data_flow`
- `add_household_variables` -> `add_rent` `data_flow`
- `in_acs` -> `add_rent` `external_source` (ACS training data)
- `add_rent` -> `add_tips` `data_flow`
- `in_sipp` -> `add_tips` `external_source` (SIPP training data)
- `add_tips` -> `add_org_inputs` `data_flow`
- `in_org` -> `add_org_inputs` `external_source` (ORG donor data)
- `add_org_inputs` -> `add_auto_loan` `data_flow`
- `in_scf` -> `add_auto_loan` `external_source` (SCF training data)
- `add_auto_loan` -> `add_takeup` `data_flow`
- `add_takeup` -> `downsample` `data_flow`
- `downsample` -> `out_cps` `produces_artifact` (cps_2024.h5)
- `in_irs_puf` -> `preprocess_puf` `data_flow` (raw PUF records)
- `preprocess_puf` -> `simulate_qbi` `data_flow`
- `simulate_qbi` -> `impute_puf_demographics` `data_flow`
- `in_demographics` -> `impute_puf_demographics` `data_flow` (demographics_2015.csv)
- `impute_puf_demographics` -> `impute_puf_pension` `data_flow`
- `in_cps_pension` -> `impute_puf_pension` `data_flow` (CPS donor sample)
- `impute_puf_pension` -> `mortgage_convert` `data_flow`
- `mortgage_convert` -> `out_puf` `produces_artifact` (puf_2024.h5)
- `in_uprating` -> `out_puf` `data_flow` (SOI growth rates)
- `util_seeded_rng` -> `add_takeup` `uses_utility`
- `util_qrf` -> `add_rent` `uses_utility`
- `util_qrf` -> `add_tips` `uses_utility`
- `util_qrf` -> `add_org_inputs` `uses_utility`
- `util_qrf` -> `add_auto_loan` `uses_utility`
- `util_retirement_limits` -> `add_personal_income_variables` `uses_utility`
- `util_qrf` -> `impute_puf_demographics` `uses_utility`
- `util_qrf` -> `impute_puf_pension` `uses_utility`

### Substage 1c: Extended CPS (PUF Clone)

Merge CPS + PUF via cloning, rematch clone features, QRF-impute incomes and CPS-only vars, then finalize Extended CPS inputs

- Substage ID: `1c_extended_cps_puf_clone`
- Canonical stage: `1_build_datasets`
- Legacy stage: `2`
- Manifest steps: `01_build_datasets`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_cps_s2` CPS_2024_Full | `artifact` | `unknown` | `unknown` |  |
| `in_puf_s2` PUF_2024 | `artifact` | `unknown` | `unknown` |  |
| `in_blocks_s2` block_cd_distributions.csv.gz | `artifact` | `unknown` | `unknown` |  |
| `in_scf_s2` SCF_2022 | `artifact` | `unknown` | `unknown` |  |
| `geo_assign_s2` Geography Assignment | `process` | `unknown` | `unknown` |  |
| `out_ext` extended_cps_2024.h5 | `artifact` | `unknown` | `unknown` |  |
| `util_qrf_s2` microimpute QRF | `utility` | `unknown` | `unknown` |  |
| `util_knn_s2` sklearn NearestNeighbors | `utility` | `unknown` | `unknown` |  |
| `record_double` PUF Clone Dataset | `library` | `current` | `moving` | `policyengine_us_data.calibration.puf_impute.puf_clone_dataset` |
| `puf_qrf_pass` PUF QRF Imputation Pass | `library` | `current` | `moving` | `policyengine_us_data.calibration.puf_impute._run_qrf_imputation` |
| `retire_impute` Retirement Contribution Imputation | `library` | `current` | `moving` | `policyengine_us_data.calibration.puf_impute._impute_retirement_contributions` |
| `weeks_impute` Weeks Unemployed Imputation | `library` | `current` | `moving` | `policyengine_us_data.calibration.puf_impute._impute_weeks_unemployed` |
| `ss_reconcile` Social Security Subcomponent Reconciliation | `library` | `current` | `moving` | `policyengine_us_data.calibration.puf_impute.reconcile_ss_subcomponents` |
| `clone_features` Splice Clone Features | `process` | `transitional` | `moving` | `policyengine_us_data.datasets.cps.extended_cps._splice_clone_feature_predictions` |
| `cps_only` Impute CPS-Only Variables | `process` | `transitional` | `moving` | `policyengine_us_data.datasets.cps.extended_cps._impute_cps_only_variables` |
| `qrf_pass2` Splice CPS-Only Predictions | `process` | `transitional` | `moving` | `policyengine_us_data.datasets.cps.extended_cps._splice_cps_only_predictions` |
| `mortgage_hints` Mortgage Balance Hint Imputation | `library` | `current` | `moving` | `policyengine_us_data.utils.mortgage_interest.impute_tax_unit_mortgage_balance_hints` |
| `mortgage_convert` Structural Mortgage Conversion | `library` | `current` | `moving` | `policyengine_us_data.utils.mortgage_interest.convert_mortgage_interest_to_structural_inputs` |
| `computed_export_contract` Validate Leaf-Input Export | `process` | `transitional` | `moving` | `policyengine_us_data.datasets.cps.extended_cps.ExtendedCPS._assert_no_computed_variables_exported` |

#### Edges

- `in_cps_s2` -> `geo_assign_s2` `data_flow` (CPS records)
- `in_blocks_s2` -> `geo_assign_s2` `data_flow` (block populations)
- `in_puf_s2` -> `record_double` `data_flow` (PUF records)
- `in_cps_s2` -> `record_double` `data_flow` (CPS records)
- `geo_assign_s2` -> `record_double` `data_flow`
- `record_double` -> `puf_qrf_pass` `data_flow`
- `puf_qrf_pass` -> `retire_impute` `data_flow`
- `puf_qrf_pass` -> `weeks_impute` `data_flow`
- `retire_impute` -> `ss_reconcile` `data_flow`
- `weeks_impute` -> `ss_reconcile` `data_flow`
- `ss_reconcile` -> `clone_features` `data_flow`
- `clone_features` -> `cps_only` `data_flow`
- `cps_only` -> `qrf_pass2` `data_flow`
- `qrf_pass2` -> `mortgage_hints` `data_flow`
- `in_scf_s2` -> `mortgage_hints` `data_flow` (SCF donor sample)
- `mortgage_hints` -> `mortgage_convert` `data_flow`
- `mortgage_convert` -> `computed_export_contract` `data_flow`
- `computed_export_contract` -> `out_ext` `produces_artifact`
- `util_qrf_s2` -> `puf_qrf_pass` `uses_utility`
- `util_qrf_s2` -> `cps_only` `uses_utility`
- `util_qrf_s2` -> `mortgage_hints` `uses_utility`
- `util_knn_s2` -> `clone_features` `uses_utility`

### Substage 1d: Enhanced CPS Reweighting

Reweight Extended CPS to match national IRS/Census/CBO targets, then apply the 2025 ACA post-calibration override (deprecated ECPS pathway)

- Substage ID: `1d_enhanced_cps_reweighting`
- Canonical stage: `1_build_datasets`
- Legacy stage: `3a`
- Manifest steps: `01_build_datasets`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_ext_half` ExtendedCPS_2024_Half | `artifact` | `unknown` | `unknown` |  |
| `build_loss` build_loss_matrix() | `process` | `unknown` | `unknown` |  |
| `t_soi` IRS SOI | `external` | `unknown` | `unknown` |  |
| `t_census` Census Population | `external` | `unknown` | `unknown` |  |
| `t_cbo` CBO Budget | `external` | `unknown` | `unknown` |  |
| `t_state` State Targets | `process` | `unknown` | `unknown` |  |
| `t_jct` JCT Tax Expenditures | `external` | `unknown` | `unknown` |  |
| `weight_validate` Weight Validation | `process` | `unknown` | `unknown` |  |
| `out_enhanced` enhanced_cps_2024.h5 | `artifact` | `unknown` | `unknown` |  |
| `util_loss` build_loss_matrix() | `utility` | `unknown` | `unknown` |  |
| `util_l0_s3` HardConcrete L0 | `utility` | `unknown` | `unknown` |  |
| `reweight` Enhanced CPS Reweighting | `process` | `transitional` | `moving` | `policyengine_us_data.datasets.cps.enhanced_cps.reweight` |
| `aca_2025_override` ACA 2025 Take-Up Override | `process` | `transitional` | `moving` | `policyengine_us_data.datasets.cps.enhanced_cps.create_aca_2025_takeup_override` |

#### Edges

- `in_ext_half` -> `build_loss` `data_flow`
- `t_soi` -> `build_loss` `external_source`
- `t_census` -> `build_loss` `external_source`
- `t_cbo` -> `build_loss` `external_source`
- `t_jct` -> `build_loss` `external_source`
- `t_state` -> `build_loss` `external_source`
- `build_loss` -> `reweight` `data_flow` ((matrix, targets))
- `reweight` -> `weight_validate` `data_flow`
- `weight_validate` -> `aca_2025_override` `data_flow`
- `aca_2025_override` -> `out_enhanced` `produces_artifact`
- `util_loss` -> `build_loss` `uses_utility`
- `util_l0_s3` -> `reweight` `uses_utility`

### Substage 1e: Stratified CPS

Stratify Extended CPS by income - keep top 1%, sample remaining 99%

- Substage ID: `1e_stratified_cps`
- Canonical stage: `1_build_datasets`
- Legacy stage: `3b`
- Manifest steps: `01_build_datasets`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_ext_cps` extended_cps_2024.h5 | `artifact` | `unknown` | `unknown` |  |
| `calc_agi` Calculate AGI | `process` | `unknown` | `unknown` |  |
| `strat_top` Retain Top 1% by AGI | `process` | `unknown` | `unknown` |  |
| `strat_sample` Uniform Sample Remaining 99% | `process` | `unknown` | `unknown` |  |
| `out_strat` stratified_extended_cps_2024.h5 | `artifact` | `unknown` | `unknown` |  |

#### Edges

- `in_ext_cps` -> `calc_agi` `data_flow`
- `calc_agi` -> `strat_top` `data_flow`
- `strat_top` -> `strat_sample` `data_flow`
- `strat_top` -> `out_strat` `data_flow` (top 1%)
- `strat_sample` -> `out_strat` `data_flow` (sampled 99%)

### Substage 1f: Source Imputation (ACS + SIPP + SCF)

Impute wealth/assets from external surveys onto stratified CPS via QRF

- Substage ID: `1f_source_imputation`
- Canonical stage: `1_build_datasets`
- Legacy stage: `4`
- Manifest steps: `01_build_datasets`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_strat_s4` stratified_extended_cps_2024.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_acs_s4` ACS_2022 | `artifact` | `unknown` | `unknown` |  |
| `in_sipp_s4` SIPP 2023 | `external` | `unknown` | `unknown` |  |
| `in_scf_s4` SCF_2022 | `artifact` | `unknown` | `unknown` |  |
| `sipp_assets_qrf` SIPP Assets QRF | `process` | `unknown` | `unknown` |  |
| `out_imputed` source_imputed_stratified_extended_cps.h5 | `artifact` | `unknown` | `unknown` |  |
| `util_clone_assign` clone_and_assign.py | `utility` | `unknown` | `unknown` |  |
| `util_qrf_s4` microimpute QRF | `utility` | `unknown` | `unknown` |  |
| `geo_assign` Assign Random Geography | `library` | `current` | `moving` | `policyengine_us_data.calibration.clone_and_assign.assign_random_geography` |
| `acs_qrf` ACS QRF Imputation | `library` | `current` | `moving` | `policyengine_us_data.calibration.source_impute._impute_acs` |
| `sipp_qrf` SIPP QRF Imputation | `library` | `current` | `moving` | `policyengine_us_data.calibration.source_impute._impute_sipp` |
| `scf_qrf` SCF QRF Imputation | `library` | `current` | `moving` | `policyengine_us_data.calibration.source_impute._impute_scf` |

#### Edges

- `in_strat_s4` -> `geo_assign` `data_flow`
- `geo_assign` -> `acs_qrf` `data_flow` (state_fips)
- `in_acs_s4` -> `acs_qrf` `data_flow`
- `in_sipp_s4` -> `sipp_qrf` `external_source`
- `in_sipp_s4` -> `sipp_assets_qrf` `external_source`
- `in_scf_s4` -> `scf_qrf` `external_source`
- `acs_qrf` -> `sipp_qrf` `data_flow` (chain)
- `sipp_qrf` -> `sipp_assets_qrf` `data_flow` (chain)
- `sipp_assets_qrf` -> `scf_qrf` `data_flow` (chain)
- `scf_qrf` -> `out_imputed` `produces_artifact`
- `util_clone_assign` -> `geo_assign` `uses_utility`
- `util_qrf_s4` -> `acs_qrf` `uses_utility`
- `util_qrf_s4` -> `sipp_qrf` `uses_utility`
- `util_qrf_s4` -> `sipp_assets_qrf` `uses_utility`
- `util_qrf_s4` -> `scf_qrf` `uses_utility`

### Substage 1g: Stage Base Datasets

Stage base source-imputed datasets and policy database artifacts for the run

- Substage ID: `1g_stage_base_datasets`
- Canonical stage: `1_build_datasets`
- Legacy stage: `7`
- Manifest steps: `04_stage_base_datasets`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_source_imputed_s1g` source_imputed_*.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_policy_db_s1g` policy_data.db | `artifact` | `unknown` | `unknown` |  |
| `hf_staging_base_s1g` HuggingFace staging/{candidate_version}-{run_id} | `external` | `unknown` | `unknown` |  |
| `stage_base_datasets` stage base datasets | `process` | `current` | `moving` |  |
| `out_staged_base_s1g` staged base datasets | `artifact` | `unknown` | `unknown` |  |

#### Edges

- `in_source_imputed_s1g` -> `stage_base_datasets` `data_flow`
- `in_policy_db_s1g` -> `stage_base_datasets` `data_flow`
- `stage_base_datasets` -> `out_staged_base_s1g` `produces_artifact`
- `out_staged_base_s1g` -> `hf_staging_base_s1g` `data_flow` (uploaded to)

## Stage 2: Build Calibration Package

Build the calibration target package, geography tables, constraints, sparse matrices, and supporting metadata.

### Substage 2a: Matrix Build (Calibration Target Construction)

Build sparse calibration matrix (targets x households x clones)

- Substage ID: `2a_matrix_build_calibration_target_construction`
- Canonical stage: `2_build_calibration_package`
- Legacy stage: `5`
- Manifest steps: `02_build_package`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_stage1_contract_s2` dataset_build_output.json | `artifact` | `unknown` | `unknown` |  |
| `in_cps_s5` source_imputed_stratified_extended_cps.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_db_s5` policy_data.db | `external` | `unknown` | `unknown` |  |
| `in_config_s5` target_config.yaml | `artifact` | `unknown` | `unknown` |  |
| `in_blocks_s5` block_cd_distributions.csv.gz | `artifact` | `unknown` | `unknown` |  |
| `target_resolve` Target Resolution | `process` | `unknown` | `unknown` |  |
| `target_uprate` Target Uprating | `process` | `unknown` | `unknown` |  |
| `geo_build` Geography Index Build | `process` | `unknown` | `unknown` |  |
| `constraint_resolve` Constraint Resolution | `process` | `unknown` | `unknown` |  |
| `takeup_rerand` Block-Level Takeup Re-randomization | `process` | `unknown` | `unknown` |  |
| `sparse_build` Sparse Matrix Construction | `process` | `unknown` | `unknown` |  |
| `out_pkg` calibration_package.pkl | `artifact` | `unknown` | `unknown` |  |
| `out_contract` calibration_package_contract.json | `artifact` | `unknown` | `unknown` |  |
| `util_sql` sqlalchemy | `utility` | `unknown` | `unknown` |  |
| `util_pool` ProcessPoolExecutor | `utility` | `unknown` | `unknown` |  |
| `util_takeup_s5` compute_block_takeup_for_entities() | `utility` | `unknown` | `unknown` |  |
| `util_scipy` scipy.sparse | `utility` | `unknown` | `unknown` |  |
| `stage2_input_bundle` Stage 2 Input Bundle | `library` | `current` | `moving` | `policyengine_us_data.calibration_package.specs.stage2_input_bundle_from_artifacts_dir` |
| `stage2_build_context` Stage 2 Build Context | `library` | `current` | `moving` | `policyengine_us_data.calibration_package.specs.stage2_build_context_for_run` |
| `stage2_artifact_specs` Stage 2 Artifact Specs | `library` | `current` | `moving` | `policyengine_us_data.calibration_package.specs.calibration_package_artifact_paths` |
| `stage2_calibration_package_writer` Stage 2 Package Writer | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.save_calibration_package` |
| `stage2_target_config_identity` Stage 2 Target Config Identity | `library` | `current` | `moving` | `policyengine_us_data.calibration_package.specs.resolve_target_config_identity` |
| `stage2_target_config_load` Load Stage 2 Target Config | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.load_target_config` |
| `stage2_target_config_apply` Apply Stage 2 Target Config | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.apply_target_config_to_targets` |
| `state_precomp` Per-State Simulation Precomputation | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_matrix_builder._compute_single_state` |
| `clone_assembly` Clone Value Assembly | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_matrix_builder._assemble_clone_values_standalone` |
| `build_matrix` Build Calibration Matrix | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_matrix_builder.UnifiedMatrixBuilder.build_matrix` |
| `build_matrix_chunked` Build Calibration Matrix In Chunks | `library` | `current` | `experimental` | `policyengine_us_data.calibration.unified_matrix_builder.UnifiedMatrixBuilder.build_matrix_chunked` |
| `stage2_calibration_package_contract_writer` Stage 2 Contract Writer | `library` | `current` | `moving` | `policyengine_us_data.stage_contracts.calibration_package.write_calibration_package_contract` |
| `stage2_calibration_package_contract_validator` Stage 2 Contract Validator | `validation` | `current` | `moving` | `policyengine_us_data.stage_contracts.calibration_package.validate_calibration_package_contract` |

#### Edges

- `in_stage1_contract_s2` -> `stage2_input_bundle` `data_flow` (preferred input contract)
- `in_cps_s5` -> `stage2_input_bundle` `data_flow` (compatibility fallback)
- `in_db_s5` -> `stage2_input_bundle` `external_source` (compatibility fallback)
- `stage2_input_bundle` -> `stage2_build_context` `data_flow` (validated inputs)
- `stage2_artifact_specs` -> `stage2_build_context` `uses_utility` (output bundle paths)
- `stage2_build_context` -> `target_resolve` `data_flow` (dataset and database paths)
- `stage2_build_context` -> `stage2_calibration_package_writer` `uses_utility` (package output bundle)
- `in_db_s5` -> `target_resolve` `external_source` (SQL targets)
- `in_config_s5` -> `stage2_target_config_identity` `data_flow` (config file)
- `stage2_target_config_identity` -> `stage2_target_config_load` `data_flow` (resolved path and checksum)
- `stage2_target_config_load` -> `stage2_target_config_apply` `data_flow` (include/exclude rules)
- `target_resolve` -> `stage2_target_config_apply` `data_flow` (candidate targets)
- `stage2_target_config_apply` -> `target_uprate` `data_flow` (selected targets)
- `target_uprate` -> `geo_build` `data_flow`
- `geo_build` -> `constraint_resolve` `data_flow`
- `constraint_resolve` -> `state_precomp` `data_flow`
- `in_cps_s5` -> `state_precomp` `data_flow` (household data)
- `state_precomp` -> `clone_assembly` `data_flow`
- `in_blocks_s5` -> `clone_assembly` `data_flow` (block populations)
- `clone_assembly` -> `takeup_rerand` `data_flow`
- `takeup_rerand` -> `sparse_build` `data_flow`
- `sparse_build` -> `build_matrix` `uses_library` (non-chunked path)
- `sparse_build` -> `build_matrix_chunked` `uses_library` (chunked path)
- `build_matrix` -> `stage2_calibration_package_writer` `data_flow`
- `build_matrix_chunked` -> `stage2_calibration_package_writer` `data_flow`
- `stage2_artifact_specs` -> `stage2_calibration_package_writer` `uses_utility` (package path)
- `stage2_calibration_package_writer` -> `out_pkg` `produces_artifact`
- `out_pkg` -> `stage2_calibration_package_contract_writer` `data_flow`
- `stage2_artifact_specs` -> `stage2_calibration_package_contract_writer` `uses_utility` (contract path)
- `stage2_calibration_package_contract_writer` -> `out_contract` `produces_artifact`
- `out_pkg` -> `stage2_calibration_package_contract_validator` `validates`
- `out_contract` -> `stage2_calibration_package_contract_validator` `validates`
- `in_cps_s5` -> `stage2_calibration_package_contract_validator` `validates`
- `in_db_s5` -> `stage2_calibration_package_contract_validator` `validates`
- `util_sql` -> `target_resolve` `uses_utility`
- `util_pool` -> `state_precomp` `uses_utility`
- `util_takeup_s5` -> `takeup_rerand` `uses_utility`
- `util_scipy` -> `sparse_build` `uses_utility`

## Stage 3: Fit Weights

Fit calibration weights for regional and national output builds.

### Substage 3a: Weight Fitting - Regional

Fit regional log-weights using L0 HardConcrete gates on GPU

- Substage ID: `3a_weight_fitting_regional`
- Canonical stage: `3_fit_weights`
- Legacy stage: `6`
- Manifest steps: `03_fit_weights_regional`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_pkg_s6` calibration_package.pkl | `artifact` | `unknown` | `unknown` |  |
| `modal_gpu` Modal GPU Container | `external` | `unknown` | `unknown` |  |
| `fit_spec_regional` FittedWeightsSpec regional | `library` | `unknown` | `unknown` |  |
| `fit_artifacts_regional` ScopedFitArtifacts regional | `library` | `unknown` | `unknown` |  |
| `create_model` Create SparseCalibrationWeights | `process` | `unknown` | `unknown` |  |
| `extract_weights` Extract Weights | `process` | `unknown` | `unknown` |  |
| `out_weights` calibration_weights.npy | `artifact` | `unknown` | `unknown` |  |
| `out_geo_s6` geography_assignment.npz | `artifact` | `unknown` | `unknown` |  |
| `out_diag` unified_diagnostics.csv | `artifact` | `unknown` | `unknown` |  |
| `out_config_s6` unified_run_config.json | `artifact` | `unknown` | `unknown` |  |
| `util_l0` l0-python | `utility` | `unknown` | `unknown` |  |
| `util_pytorch` PyTorch | `utility` | `unknown` | `unknown` |  |
| `init_weights` Compute Initial Weights | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.compute_initial_weights` |
| `fit_model` Fit L0 Calibration Weights | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.fit_l0_weights` |

#### Edges

- `in_pkg_s6` -> `init_weights` `data_flow`
- `fit_spec_regional` -> `fit_model` `uses_library`
- `fit_artifacts_regional` -> `out_weights` `documents`
- `fit_artifacts_regional` -> `out_geo_s6` `documents`
- `fit_artifacts_regional` -> `out_diag` `documents`
- `fit_artifacts_regional` -> `out_config_s6` `documents`
- `init_weights` -> `create_model` `data_flow`
- `create_model` -> `fit_model` `data_flow`
- `modal_gpu` -> `fit_model` `runs_on_infra` (runs on)
- `fit_model` -> `extract_weights` `data_flow`
- `extract_weights` -> `out_weights` `produces_artifact`
- `extract_weights` -> `out_geo_s6` `produces_artifact`
- `fit_model` -> `out_diag` `produces_artifact`
- `fit_model` -> `out_config_s6` `produces_artifact`
- `util_l0` -> `create_model` `uses_utility`
- `util_pytorch` -> `fit_model` `uses_utility`

### Substage 3b: Weight Fitting - National

Fit national log-weights for the national H5 output without an L0 penalty

- Substage ID: `3b_weight_fitting_national`
- Canonical stage: `3_fit_weights`
- Legacy stage: `6`
- Manifest steps: `03_fit_weights_national`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_pkg_national_s6` calibration_package.pkl | `artifact` | `unknown` | `unknown` |  |
| `modal_gpu_national` Modal GPU Container | `external` | `unknown` | `unknown` |  |
| `fit_spec_national` FittedWeightsSpec national | `library` | `unknown` | `unknown` |  |
| `fit_artifacts_national` ScopedFitArtifacts national | `library` | `unknown` | `unknown` |  |
| `create_model_national` Create National SparseCalibrationWeights | `process` | `unknown` | `unknown` |  |
| `extract_national_weights` Extract National Weights | `process` | `unknown` | `unknown` |  |
| `out_national_weights` national_calibration_weights.npy | `artifact` | `unknown` | `unknown` |  |
| `out_national_geo_s6` national_geography_assignment.npz | `artifact` | `unknown` | `unknown` |  |
| `out_national_diag` national_unified_diagnostics.csv | `artifact` | `unknown` | `unknown` |  |
| `out_national_config_s6` national_unified_run_config.json | `artifact` | `unknown` | `unknown` |  |
| `util_l0_national` l0-python | `utility` | `unknown` | `unknown` |  |
| `util_pytorch_national` PyTorch | `utility` | `unknown` | `unknown` |  |
| `init_weights` Compute Initial Weights | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.compute_initial_weights` |
| `fit_model` Fit L0 Calibration Weights | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.fit_l0_weights` |

#### Edges

- `in_pkg_national_s6` -> `init_weights` `data_flow`
- `fit_spec_national` -> `fit_model` `uses_library`
- `fit_artifacts_national` -> `out_national_weights` `documents`
- `fit_artifacts_national` -> `out_national_geo_s6` `documents`
- `fit_artifacts_national` -> `out_national_diag` `documents`
- `fit_artifacts_national` -> `out_national_config_s6` `documents`
- `init_weights` -> `create_model_national` `data_flow`
- `create_model_national` -> `fit_model` `data_flow`
- `modal_gpu_national` -> `fit_model` `runs_on_infra` (runs on)
- `fit_model` -> `extract_national_weights` `data_flow`
- `extract_national_weights` -> `out_national_weights` `produces_artifact`
- `extract_national_weights` -> `out_national_geo_s6` `produces_artifact`
- `fit_model` -> `out_national_diag` `produces_artifact`
- `fit_model` -> `out_national_config_s6` `produces_artifact`
- `util_l0_national` -> `create_model_national` `uses_utility`
- `util_pytorch_national` -> `fit_model` `uses_utility`

## Stage 4: Build Outputs

Build local-area and national H5 outputs and upload diagnostics.

### Substage 4a: Local Area H5 - Regional

Build 51 state + 435 district + 1 city H5 files on Modal workers

- Substage ID: `4a_local_area_h5_regional`
- Canonical stage: `4_build_outputs`
- Legacy stage: `7`
- Manifest steps: `04_build_h5_regional`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_weights_s7` calibration_weights.npy | `artifact` | `unknown` | `unknown` |  |
| `in_dataset_s7` source_imputed_stratified_extended_cps.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_db_s7` policy_data.db | `external` | `unknown` | `unknown` |  |
| `modal_coord` Modal Coordinator | `external` | `unknown` | `unknown` |  |
| `partition` Partition Work | `process` | `unknown` | `unknown` |  |
| `worker_s7` Modal Worker Container | `external` | `unknown` | `unknown` |  |
| `takeup_apply` Takeup Re-application | `process` | `unknown` | `unknown` |  |
| `out_states` states/*.h5 | `artifact` | `unknown` | `unknown` |  |
| `out_districts` districts/*.h5 | `artifact` | `unknown` | `unknown` |  |
| `out_cities` cities/*.h5 | `artifact` | `unknown` | `unknown` |  |
| `out_manifest` manifest.json | `artifact` | `unknown` | `unknown` |  |
| `util_build_h5` publish_local_area.build_h5() | `utility` | `unknown` | `unknown` |  |
| `util_takeup_s7` apply_block_takeup_to_arrays() | `utility` | `unknown` | `unknown` |  |
| `build_states` Build State H5 Files | `library` | `current` | `moving` | `policyengine_us_data.calibration.publish_local_area.build_states` |
| `build_districts` Build District H5 Files | `library` | `current` | `moving` | `policyengine_us_data.calibration.publish_local_area.build_districts` |
| `build_cities` Build City H5 Files | `library` | `current` | `moving` | `policyengine_us_data.calibration.publish_local_area.build_cities` |
| `build_h5` Build Local Area H5 | `library` | `transitional` | `moving` | `policyengine_us_data.calibration.publish_local_area.build_h5` |
| `geo_derive` Derive Geography From Blocks | `library` | `current` | `moving` | `policyengine_us_data.calibration.block_assignment.derive_geography_from_blocks` |

#### Edges

- `in_weights_s7` -> `partition` `data_flow`
- `in_dataset_s7` -> `partition` `data_flow`
- `in_db_s7` -> `partition` `external_source` (CD list)
- `partition` -> `build_states` `data_flow`
- `build_states` -> `build_districts` `data_flow`
- `build_districts` -> `build_cities` `data_flow`
- `build_states` -> `build_h5` `data_flow` (calls)
- `build_districts` -> `build_h5` `data_flow` (calls)
- `build_cities` -> `build_h5` `data_flow` (calls)
- `build_h5` -> `geo_derive` `data_flow`
- `geo_derive` -> `takeup_apply` `data_flow`
- `modal_coord` -> `worker_s7` `runs_on_infra` (orchestrates)
- `worker_s7` -> `build_h5` `runs_on_infra` (runs)
- `build_states` -> `out_states` `produces_artifact`
- `build_districts` -> `out_districts` `produces_artifact`
- `build_cities` -> `out_cities` `produces_artifact`
- `build_h5` -> `out_manifest` `produces_artifact`
- `util_build_h5` -> `build_h5` `uses_utility`
- `util_takeup_s7` -> `takeup_apply` `uses_utility`

### Substage 4b: Local Area H5 - National

Build the national US.h5 output from national weights and national geography artifacts

- Substage ID: `4b_local_area_h5_national`
- Canonical stage: `4_build_outputs`
- Legacy stage: `7`
- Manifest steps: `04_build_h5_national`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_national_weights_s4b` national_calibration_weights.npy | `artifact` | `unknown` | `unknown` |  |
| `in_national_dataset_s4b` source_imputed_stratified_extended_cps.h5 | `artifact` | `unknown` | `unknown` |  |
| `in_national_geo_s4b` national_geography_assignment.npz | `artifact` | `unknown` | `unknown` |  |
| `in_national_config_s4b` national_unified_run_config.json | `artifact` | `unknown` | `unknown` |  |
| `national_h5_coord` National H5 Coordinator | `process` | `unknown` | `unknown` |  |
| `national_worker` National Modal Worker | `external` | `unknown` | `unknown` |  |
| `national_request` AreaBuildRequest(type=national) | `process` | `unknown` | `unknown` |  |
| `national_validation` National H5 Validation | `process` | `unknown` | `unknown` |  |
| `out_national_h5` national/US.h5 | `artifact` | `unknown` | `unknown` |  |
| `out_national_validation` national_validation.txt | `artifact` | `unknown` | `unknown` |  |
| `util_build_h5_national` publish_local_area.build_h5() | `utility` | `unknown` | `unknown` |  |
| `build_h5` Build Local Area H5 | `library` | `transitional` | `moving` | `policyengine_us_data.calibration.publish_local_area.build_h5` |

#### Edges

- `in_national_weights_s4b` -> `national_request` `data_flow`
- `in_national_dataset_s4b` -> `national_request` `data_flow`
- `in_national_geo_s4b` -> `national_request` `data_flow`
- `in_national_config_s4b` -> `national_request` `data_flow`
- `national_request` -> `build_h5` `data_flow`
- `national_h5_coord` -> `national_worker` `runs_on_infra` (spawns)
- `national_worker` -> `build_h5` `runs_on_infra` (runs)
- `build_h5` -> `out_national_h5` `produces_artifact`
- `out_national_h5` -> `national_validation` `data_flow`
- `national_validation` -> `out_national_validation` `produces_artifact`
- `util_build_h5_national` -> `build_h5` `uses_utility`

### Substage 4d: Upload Diagnostics

Collect calibration and validation diagnostics and upload them to run-scoped archival paths

- Substage ID: `4d_upload_diagnostics`
- Canonical stage: `4_build_outputs`
- Legacy stage: `7`
- Manifest steps: `04_upload_diagnostics`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_calibration_diag_s4d` unified_diagnostics.csv | `artifact` | `unknown` | `unknown` |  |
| `in_validation_diag_s4d` validation_results.csv / national_validation.txt | `artifact` | `unknown` | `unknown` |  |
| `upload_diagnostics_s4d` Upload Run Diagnostics | `process` | `unknown` | `unknown` |  |
| `out_hf_diagnostics_s4d` calibration/runs/{run_id}/diagnostics/ | `external` | `unknown` | `unknown` |  |
| `calibration_diagnostics` Compute Calibration Diagnostics | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.compute_diagnostics` |

#### Edges

- `in_calibration_diag_s4d` -> `upload_diagnostics_s4d` `data_flow`
- `in_validation_diag_s4d` -> `upload_diagnostics_s4d` `data_flow`
- `calibration_diagnostics` -> `in_calibration_diag_s4d` `produces_artifact`
- `upload_diagnostics_s4d` -> `out_hf_diagnostics_s4d` `produces_artifact`

## Stage 5: Validate and Promote Release

Validate staged artifacts, promote release outputs, and finalize publication manifests.

### Substage 5a: Validate Outputs

Validate staged H5 and base artifacts before any production promotion

- Substage ID: `5a_validate_outputs`
- Canonical stage: `5_validate_and_promote_release`
- Legacy stage: `8`
- Manifest steps: `05_promote_release`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_h5s` 51 state + 435 district + 1 city H5s | `artifact` | `unknown` | `unknown` |  |
| `in_db_s8` policy_data.db | `external` | `unknown` | `unknown` |  |
| `v1` Layer 1: Manifest Verification | `process` | `unknown` | `unknown` |  |
| `v4` Layer 4: Smoke Test | `process` | `unknown` | `unknown` |  |
| `v5` Layer 5: National H5 Validation | `process` | `unknown` | `unknown` |  |
| `v6` Layer 6: Pre-Upload Validation | `process` | `unknown` | `unknown` |  |
| `v7` Layer 7: Package Validation | `process` | `unknown` | `unknown` |  |
| `out_validated_candidates_s5a` validated release candidates | `artifact` | `unknown` | `unknown` |  |
| `util_manifest_s8` manifest.py | `utility` | `unknown` | `unknown` |  |
| `util_sanity` sanity_checks.py | `utility` | `unknown` | `unknown` |  |
| `util_validate` validate_staging.py | `utility` | `unknown` | `unknown` |  |
| `target_validation` Validate Area Against Targets | `validation` | `current` | `moving` | `policyengine_us_data.calibration.validate_staging.validate_area` |
| `sanity_checks` Run H5 Sanity Checks | `validation` | `current` | `moving` | `policyengine_us_data.calibration.sanity_checks.run_sanity_checks` |

#### Edges

- `in_h5s` -> `v1` `data_flow`
- `in_db_s8` -> `target_validation` `external_source` (targets)
- `v1` -> `sanity_checks` `data_flow`
- `sanity_checks` -> `target_validation` `data_flow`
- `target_validation` -> `v4` `data_flow`
- `v4` -> `v5` `data_flow`
- `v5` -> `v6` `data_flow`
- `v6` -> `v7` `data_flow`
- `v7` -> `out_validated_candidates_s5a` `produces_artifact` (all pass)
- `util_manifest_s8` -> `v1` `uses_utility`
- `util_sanity` -> `sanity_checks` `uses_utility`
- `util_validate` -> `target_validation` `uses_utility`

### Substage 5b: Promote HuggingFace

Promote validated staged artifacts to HuggingFace production paths

- Substage ID: `5b_promote_huggingface`
- Canonical stage: `5_validate_and_promote_release`
- Legacy stage: `8`
- Manifest steps: `05_promote_release`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_validated_candidates_s5b` validated release candidates | `artifact` | `unknown` | `unknown` |  |
| `hf_staging_s5b` HuggingFace staging/{candidate_version}-{run_id} | `external` | `unknown` | `unknown` |  |
| `out_hf_prod` HuggingFace Production | `external` | `unknown` | `unknown` |  |
| `util_upload_s5b` data_upload.py | `utility` | `unknown` | `unknown` |  |
| `staging_upload` Upload Local H5s To Staging | `entrypoint` | `current` | `moving` | `modal_app.local_area.upload_to_staging` |
| `atomic_promote` Atomic Promote Local H5 Files | `entrypoint` | `current` | `moving` | `policyengine_us_data.calibration.promote_local_h5s.promote` |
| `promote_pipeline_run` Promote Pipeline Run | `entrypoint` | `current` | `moving` | `modal_app.pipeline.promote_run` |

#### Edges

- `in_validated_candidates_s5b` -> `staging_upload` `data_flow`
- `hf_staging_s5b` -> `atomic_promote` `external_source`
- `staging_upload` -> `atomic_promote` `data_flow`
- `promote_pipeline_run` -> `atomic_promote` `data_flow` (orchestrates)
- `atomic_promote` -> `out_hf_prod` `produces_artifact`
- `util_upload_s5b` -> `staging_upload` `uses_utility`
- `util_upload_s5b` -> `atomic_promote` `uses_utility`

### Substage 5c: Promote GCS

Upload promoted datasets to Google Cloud Storage with version metadata

- Substage ID: `5c_promote_gcs`
- Canonical stage: `5_validate_and_promote_release`
- Legacy stage: `8`
- Manifest steps: `05_promote_release`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_hf_prod_s5c` HuggingFace production release | `external` | `unknown` | `unknown` |  |
| `gcs_upload` GCS Parallel Upload | `process` | `unknown` | `unknown` |  |
| `out_gcs` Google Cloud Storage | `external` | `unknown` | `unknown` |  |
| `util_upload_gcs_s5c` data_upload.py | `utility` | `unknown` | `unknown` |  |
| `atomic_promote` Atomic Promote Local H5 Files | `entrypoint` | `current` | `moving` | `policyengine_us_data.calibration.promote_local_h5s.promote` |

#### Edges

- `in_hf_prod_s5c` -> `gcs_upload` `data_flow`
- `atomic_promote` -> `gcs_upload` `data_flow` (release files)
- `gcs_upload` -> `out_gcs` `produces_artifact`
- `util_upload_gcs_s5c` -> `gcs_upload` `uses_utility`

### Substage 5d: Write Version Manifest

Finalize release manifests, record run diagnostics paths, and clean staging state

- Substage ID: `5d_write_version_manifest`
- Canonical stage: `5_validate_and_promote_release`
- Legacy stage: `8`
- Manifest steps: `05_promote_release`
- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `in_release_outputs_s5d` promoted release outputs | `artifact` | `unknown` | `unknown` |  |
| `version_manifest_write` version_manifest.json update | `process` | `unknown` | `unknown` |  |
| `staging_cleanup` Staging Cleanup | `process` | `unknown` | `unknown` |  |
| `out_version_manifest` version_manifest.json | `artifact` | `unknown` | `unknown` |  |
| `out_release_finalized` finalized release | `artifact` | `unknown` | `unknown` |  |
| `util_manifest_s5d` version_manifest.py / release_manifest.py | `utility` | `unknown` | `unknown` |  |
| `atomic_promote` Atomic Promote Local H5 Files | `entrypoint` | `current` | `moving` | `policyengine_us_data.calibration.promote_local_h5s.promote` |

#### Edges

- `in_release_outputs_s5d` -> `version_manifest_write` `data_flow`
- `atomic_promote` -> `version_manifest_write` `data_flow` (release manifest inputs)
- `version_manifest_write` -> `out_version_manifest` `produces_artifact`
- `version_manifest_write` -> `staging_cleanup` `data_flow`
- `staging_cleanup` -> `out_release_finalized` `produces_artifact`
- `util_manifest_s5d` -> `version_manifest_write` `uses_utility`

## Pydoc API Surface

### `modal_app.local_area.build_areas_worker`

```python
def build_areas_worker(branch: str, run_id: str, scope: str, work_items: List[Dict] | None = None, calibration_inputs: WorkerCalibrationInputs | Mapping[str, object] | None = None, validate: bool = True, scope_fingerprint: str | None = None, request_payloads: List[Dict] | None = None) -> Dict
```

Worker function that builds a subset of H5 files.

### `modal_app.data_build.build_datasets`

```python
def build_datasets(upload: bool = False, branch: str = 'main', sequential: bool = False, clear_checkpoints: bool = False, skip_tests: bool = False, skip_enhanced_cps: bool = False, skip_stage_5: bool = False, stage_only: bool = False, run_id: str = '', version: str = DATA_PACKAGE_VERSION)
```

Build all datasets with preemption-resilient checkpointing.

### `modal_app.local_area._build_publishing_input_bundle`

```python
def _build_publishing_input_bundle(*, weights_path: Path, dataset_path: Path, db_path: Path | None, geography_path: Path | None, calibration_package_path: Path | None, run_config_path: Path | None, run_id: str, version: str, n_clones: int | None, seed: int, legacy_blocks_path: Path | None = None) -> PublishingInputBundle
```

Build the normalized coordinator input bundle for one publish scope.

### `modal_app.local_area._build_worker_bootstrap`

```python
def _build_worker_bootstrap(*, inputs: PublishingInputBundle, scope: str, artifacts_dir: Path, scope_fingerprint: str | None = None)
```

Persist optional worker bootstrap artifacts for one local H5 scope.

### `policyengine_us_data.build_outputs.geography_loader.CalibrationGeographyLoader`

```python
class CalibrationGeographyLoader
```

Resolve, load, and checksum exact geography artifacts.

### `policyengine_us_data.build_outputs.weights.CloneWeightMatrix`

```python
class CloneWeightMatrix
```

Structured view of clone-level household weights.

### `modal_app.local_area.coordinate_publish`

```python
def coordinate_publish(branch: str = 'main', num_workers: int = 50, skip_upload: bool = False, n_clones: int = 430, validate: bool = True, run_id: str = '', candidate_version: str = '', expected_fingerprint: str = '', work_items_override: List[Dict] | None = None) -> Dict
```

Coordinate the full publishing workflow.

### `modal_app.local_area.partition_work`

```python
def partition_work(work_items: List[Dict], num_workers: int, completed: set) -> List[List[Dict]]
```

Compatibility wrapper over the extracted pure partitioning seam.

### `modal_app.data_build.run_cps_then_puf_phase`

```python
def run_cps_then_puf_phase(branch: str, volume: modal.Volume, *, env: dict, log_file: IO = None, checkpoint_stats: CheckpointStats | None = None, coordinator: Stage1Coordinator | None = None) -> None
```

Build CPS before PUF because PUF pension imputation loads CPS_2024.

### `policyengine_us_data.calibration.create_stratified_cps.create_stratified_cps_dataset`

```python
def create_stratified_cps_dataset(target_households = 30000, oversample_poor = False, seed = None, base_dataset = None, output_path = None, high_agi_brackets = None)
```

Create a stratified sample of CPS data preserving high-income households

### `policyengine_us_data.fit_weights.artifacts.fit_artifacts_for_scope`

```python
def fit_artifacts_for_scope(scope: FitScope | str) -> ScopedFitArtifacts
```

Return canonical fitted-weight artifacts for a regional or national scope.

### `policyengine_us_data.fit_weights.bundles.FittedWeightsOutputBundle`

```python
class FittedWeightsOutputBundle
```

Scoped output bundle created before Stage 3 bytes become files.

### `policyengine_us_data.fit_weights.specs.fitted_weights_spec_for_scope`

```python
def fitted_weights_spec_for_scope(scope: FitScope | str) -> FittedWeightsSpec
```

Return the current fitted-weight spec for a regional or national scope.

### `policyengine_us_data.release_promotion.results.full.FullPromotionResult`

```python
class FullPromotionResult
```

Typed result for a full Stage 5 release promotion transaction.

### `policyengine_us_data.datasets.cps.extended_cps.ExtendedCPS._validate_housing_assistance_microsimulation`

```python
def _validate_housing_assistance_microsimulation(cls, data, time_period, microsimulation_cls = None)
```

Check formula-reconstructed housing assistance before export.

### `policyengine_us_data.release_promotion.candidate_builders.build_legacy_release_candidate_bundle`

```python
def build_legacy_release_candidate_bundle(*, context: ReleasePromotionContext, rel_paths: Sequence[str], artifact_metadata_by_path: Mapping[str, Mapping[str, Any]] | None = None, validation_report_paths: Sequence[str] = (), validation_report_refs: Sequence[DiagnosticRef] = (), source_output_contract_path: str | None = None, diagnostics_manifest_path: str | None = None) -> ReleaseCandidateInputBundle
```

Build a candidate bundle from the current legacy staged relative paths.

### `policyengine_us_data.calibration.publish_local_area.load_calibration_geography`

```python
def load_calibration_geography(weights_path: Path, n_records: int, n_clones: Optional[int] = None, geography_path: Optional[Path] = None, blocks_path: Optional[Path] = None, calibration_package_path: Optional[Path] = None)
```

Resolve exact geography from saved bundles, package metadata, or legacy block artifacts.

### `policyengine_us_data.build_outputs.area_catalog.USAreaCatalog`

```python
class USAreaCatalog
```

Construct typed H5 build requests for supported US geographies.

### `policyengine_us_data.build_outputs.requests.AreaFilter`

```python
class AreaFilter
```

Predicate used to select calibrated clones for one H5 output.

### `policyengine_us_data.build_outputs.requests.AreaBuildRequest`

```python
class AreaBuildRequest
```

Complete request for one local-area or national H5 file.

### `policyengine_us_data.build_outputs.selection.AreaSelector`

```python
class AreaSelector
```

Apply request geography filters to clone-level calibration weights.

### `policyengine_us_data.build_outputs.validation.AreaValidationService`

```python
class AreaValidationService
```

Build validation state for all H5 requests handled by one worker.

### `policyengine_us_data.build_outputs.fingerprinting.ArtifactIdentity`

```python
class ArtifactIdentity
```

Stable identity for an input artifact used by traceability.

### `policyengine_us_data.build_outputs.builder.LocalAreaBuildResult`

```python
class LocalAreaBuildResult
```

In-memory output from building one local H5 area.

### `policyengine_us_data.build_outputs.geography_loader.CalibrationGeographyIndex`

```python
class CalibrationGeographyIndex
```

Clone geography fields needed for coordinator-side request planning.

### `policyengine_us_data.build_outputs.selection.CloneSelection`

```python
class CloneSelection
```

Active clone rows selected for one H5 output.

### `policyengine_us_data.build_outputs.worker_responses.CoordinatorWorkerResult`

```python
class CoordinatorWorkerResult
```

Normalized worker response with explicit fatal and nonfatal issue classes.

### `policyengine_us_data.build_outputs.builder.LocalAreaDatasetBuilder`

```python
class LocalAreaDatasetBuilder
```

Coordinate clone selection, reindexing, variable cloning, and postprocessing.

### `policyengine_us_data.build_outputs.source_dataset.EntityGraph`

```python
class EntityGraph
```

Structural relationships between source dataset entities.

### `policyengine_us_data.build_outputs.reindexing.EntityReindexer`

```python
class EntityReindexer
```

Build sequential entity IDs and relationship arrays after clone selection.

### `policyengine_us_data.calibration.publish_local_area.compute_input_fingerprint`

```python
def compute_input_fingerprint(weights_path: Path, dataset_path: Path, n_clones: Optional[int] = None, seed: int = 42, geography_path: Optional[Path] = None, blocks_path: Optional[Path] = None, target_db_path: Optional[Path] = None, run_config_path: Optional[Path] = None, calibration_package_path: Optional[Path] = None, scope: str = 'regional') -> str
```

Compute a scope fingerprint for local H5 checkpoint and resume decisions.

### `policyengine_us_data.build_outputs.partitioning.partition_weighted_work_items`

```python
def partition_weighted_work_items(work_items: WorkItems, num_workers: int, completed: set[str] | None = None) -> WorkChunks
```

Partition remaining H5 work across worker chunks.

### `policyengine_us_data.build_outputs.source_dataset.MicrosimulationVariableProvider`

```python
class MicrosimulationVariableProvider
```

Lazy holder-backed variable reader for a source microsimulation.

### `policyengine_us_data.build_outputs.partitioning.partition_weighted_area_requests`

```python
def partition_weighted_area_requests(requests: Sequence[WeightedAreaRequest], num_workers: int, completed: set[str] | None = None) -> WeightedAreaRequestChunks
```

Partition remaining typed H5 requests across worker chunks.

### `policyengine_us_data.build_outputs.payload.H5Payload`

```python
class H5Payload
```

Period-grouped arrays ready to write to a local-area H5 file.

### `policyengine_us_data.build_outputs.payload.PayloadBuildContext`

```python
class PayloadBuildContext
```

Context available to country-specific local H5 payload postprocessors.

### `policyengine_us_data.build_outputs.source_dataset.PolicyEngineDatasetReader`

```python
class PolicyEngineDatasetReader
```

Read PolicyEngine source H5 files into `SourceDatasetSnapshot` objects.

### `policyengine_us_data.build_outputs.fingerprinting.PublishingInputBundle`

```python
class PublishingInputBundle
```

Input artifact bundle for one local H5 publication scope.

### `policyengine_us_data.build_outputs.target_universe.RegionalTargetUniverse`

```python
class RegionalTargetUniverse
```

Congressional district target universe for regional H5 outputs.

### `policyengine_us_data.build_outputs.reindexing.ReindexedEntities`

```python
class ReindexedEntities
```

Entity IDs, relationship arrays, and source indices for one H5 output.

### `policyengine_us_data.build_outputs.geography_loader.ResolvedGeographySource`

```python
class ResolvedGeographySource
```

Resolved physical source used to recover calibration geography.

### `policyengine_us_data.build_outputs.source_dataset.SourceDatasetSnapshot`

```python
class SourceDatasetSnapshot
```

Explicit in-memory worker view of a source H5 dataset.

### `policyengine_us_data.build_outputs.target_universe.TargetUniverseReader`

```python
class TargetUniverseReader
```

Adapter from the Stage 1 target database artifact to H5 target contracts.

### `policyengine_us_data.build_outputs.fingerprinting.FingerprintingService`

```python
class FingerprintingService
```

Build traceability bundles and derive deterministic scope fingerprints.

### `policyengine_us_data.build_outputs.fingerprinting.TraceabilityBundle`

```python
class TraceabilityBundle
```

Full provenance record for one local H5 publish scope.

### `policyengine_us_data.build_outputs.us_augmentations.USEntityPostProcessor`

```python
class USEntityPostProcessor
```

Apply US entity IDs and calibrated household weights.

### `policyengine_us_data.build_outputs.us_augmentations.USEntityPostProcessorResult`

```python
class USEntityPostProcessorResult
```

Payload after US entity ID and household-weight fields are applied.

### `policyengine_us_data.build_outputs.us_augmentations.USGeographyPostProcessor`

```python
class USGeographyPostProcessor
```

Apply block-derived US geography overrides.

### `policyengine_us_data.build_outputs.us_augmentations.USGeographyPostProcessorResult`

```python
class USGeographyPostProcessorResult
```

Payload after US geography fields are applied.

### `policyengine_us_data.build_outputs.us_augmentations.USTakeupPostProcessor`

```python
class USTakeupPostProcessor
```

Apply US take-up draws after entity and geography postprocessing.

### `policyengine_us_data.build_outputs.us_augmentations.USTakeupPostProcessorResult`

```python
class USTakeupPostProcessorResult
```

Payload after US take-up fields are applied.

### `policyengine_us_data.build_outputs.validation.ValidationContext`

```python
class ValidationContext
```

Prepared validation data reused across all requests in one worker.

### `policyengine_us_data.build_outputs.validation.ValidationPolicy`

```python
class ValidationPolicy
```

Validation switch for a local H5 worker session.

### `policyengine_us_data.build_outputs.variables.VariableClonePayload`

```python
class VariableClonePayload
```

Cloned source variable arrays before H5-specific overrides.

### `policyengine_us_data.build_outputs.variables.VariableCloner`

```python
class VariableCloner
```

Clone source variable arrays using selected and reindexed entity rows.

### `policyengine_us_data.build_outputs.partitioning.WeightedAreaRequest`

```python
class WeightedAreaRequest
```

Area build request plus scheduling weight for coordinator partitioning.

### `policyengine_us_data.build_outputs.worker_service.WorkerAreaResult`

```python
class WorkerAreaResult
```

Structured result for one area handled by a worker.

### `policyengine_us_data.build_outputs.bootstrap.WorkerBootstrapBuilder`

```python
class WorkerBootstrapBuilder
```

Build and persist one scope's local H5 worker bootstrap artifacts.

### `policyengine_us_data.build_outputs.bootstrap.WorkerBootstrapBundle`

```python
class WorkerBootstrapBundle
```

Manifest-backed bootstrap bundle for one worker setup scope.

### `policyengine_us_data.build_outputs.bootstrap.WorkerBootstrapStore`

```python
class WorkerBootstrapStore
```

Filesystem adapter for scope-specific bootstrap bundle paths.

### `policyengine_us_data.build_outputs.worker_inputs.WorkerCalibrationInputs`

```python
class WorkerCalibrationInputs
```

Input artifact paths and runtime settings for one H5 worker batch.

### `policyengine_us_data.build_outputs.worker_service.WorkerExecutionConfig`

```python
class WorkerExecutionConfig
```

Execution policy for one worker chunk.

### `policyengine_us_data.build_outputs.worker_service.WorkerIssue`

```python
class WorkerIssue
```

Structured worker issue for request, build, write, or validation failures.

### `policyengine_us_data.build_outputs.worker_service.WorkerResult`

```python
class WorkerResult
```

Structured result for a worker chunk.

### `policyengine_us_data.build_outputs.worker_service.LocalH5WorkerService`

```python
class LocalH5WorkerService
```

Execute typed local H5 requests for one prepared worker session.

### `policyengine_us_data.build_outputs.worker_session.WorkerSession`

```python
class WorkerSession
```

Prepared local H5 state for one worker process.

### `policyengine_us_data.build_outputs.worker_session.WorkerSessionFactory`

```python
class WorkerSessionFactory
```

Build worker-scoped setup from raw inputs or persisted bootstrap facts.

### `policyengine_us_data.build_outputs.writer.H5WriteResult`

```python
class H5WriteResult
```

Summary of one H5 write and lightweight verification pass.

### `policyengine_us_data.build_outputs.writer.H5Writer`

```python
class H5Writer
```

Write period-grouped local H5 payloads and verify key output counts.

### `policyengine_us_data.calibration.promote_local_h5s.stage`

```python
def stage(files: list, version: str, run_id: str = '')
```

Upload locally built H5 files into Hugging Face staging paths.

### `policyengine_us_data.build_outputs.worker_responses.normalize_worker_response`

```python
def normalize_worker_response(*, worker_index: int, result: object) -> CoordinatorWorkerResult
```

Normalize worker JSON into explicit fatal and nonfatal coordinator issues.

### `policyengine_us_data.release_promotion.published_index.build_published_artifact_index`

```python
def build_published_artifact_index(*, candidate_bundle: ReleaseCandidateInputBundle, promotion_result: FullPromotionResult, release_manifest: Mapping[str, Any] | None = None, diagnostic_artifacts: Sequence[ArtifactRef] = ()) -> tuple[PublishedArtifactIndexRow, ...]
```

Build deterministic published artifact rows for a promoted release.

### `policyengine_us_data.release_promotion.published_index.PublishedArtifactIndexRow`

```python
class PublishedArtifactIndexRow
```

One row in the Stage 5 published artifact JSONL index.

### `policyengine_us_data.release_promotion.artifacts.ReleaseArtifactSpec`

```python
class ReleaseArtifactSpec
```

Normalized identity for one artifact in a Stage 5 release candidate.

### `policyengine_us_data.release_promotion.candidate.ReleaseCandidateInputBundle`

```python
class ReleaseCandidateInputBundle
```

Typed Stage 5 input bundle describing a candidate ready for promotion.

### `policyengine_us_data.release_promotion.validation.build_release_candidate_shape_report`

```python
def build_release_candidate_shape_report(bundle: ReleaseCandidateInputBundle) -> ValidationReport
```

Describe candidate-bundle shape using the shared validation schema.

### `policyengine_us_data.release_promotion.validation.ReleaseCandidateValidator`

```python
class ReleaseCandidateValidator
```

Validate a Stage 5 release candidate before public release writes.

### `policyengine_us_data.release_promotion.context.ReleasePromotionContext`

```python
class ReleasePromotionContext
```

Canonical run, candidate, release, and destination identity for Stage 5.

### `policyengine_us_data.release_promotion.contract.ReleasePromotionContractBuilder`

```python
class ReleasePromotionContractBuilder
```

Build a Stage 5 contract from candidate identity and promotion results.

### `modal_app.local_area._resolve_scope_fingerprint`

```python
def _resolve_scope_fingerprint(*, inputs: PublishingInputBundle, scope: str, expected_fingerprint: str = '') -> str
```

Compute the scope fingerprint while preserving pinned resume values.

### `policyengine_us_data.calibration.unified_calibration.run_calibration`

```python
def run_calibration(dataset_path: str, db_path: str, n_clones: int = DEFAULT_N_CLONES, lambda_l0: float = 1e-08, epochs: int = DEFAULT_EPOCHS, device: str = 'cpu', seed: int = 42, domain_variables: list = None, hierarchical_domains: list = None, skip_takeup_rerandomize: bool = False, skip_source_impute: bool = True, skip_county: bool = True, target_config: dict = None, target_config_path: str = None, target_config_identity: TargetConfigIdentity | None = None, build_only: bool = False, package_path: str = None, package_output_path: str = None, beta: float = BETA, lambda_l2: float = LAMBDA_L2, learning_rate: float = LEARNING_RATE, log_freq: int = None, log_path: str = None, workers: int = 1, resume_from: str = None, checkpoint_path: str = None, chunked_matrix: bool = False, chunk_size: int = 25000, chunk_dir: str = None, keep_chunks: bool = False, resume_chunks: bool = False, parallel: bool = False, num_matrix_workers: int = 50, run_id: str = '')
```

Run unified calibration pipeline.

### `modal_app.local_area.run_phase`

```python
def run_phase(phase_name: str, weighted_requests: Sequence[WeightedAreaRequest] | None, num_workers: int, completed: set, branch: str, run_id: str, calibration_inputs: WorkerCalibrationInputs | Mapping[str, object], run_dir: Path, validate: bool = True, scope_fingerprint: str | None = None, work_items: List[Dict] | None = None) -> tuple
```

Run a single build phase, spawning workers and collecting results.

### `modal_app.pipeline.run_pipeline`

```python
def run_pipeline(branch: str = 'main', gpu: str = 'T4', epochs: int = 1000, national_gpu: str = 'T4', national_epochs: int = 1000, num_workers: int = 50, n_clones: int = 430, skip_national: bool = False, resume_run_id: str = None, clear_checkpoints: bool = False, candidate_version: str = '', release_version: str = '', base_release_version: str = '', release_bump: str = '', sha_override: str = '', run_id: str = '', run_context: dict | None = None, modal_app_name: str = '', modal_environment: str = '', chunked_matrix: bool = False, chunk_size: int = 25000, parallel_matrix: bool = False, num_matrix_workers: int = 50) -> str
```

Run the full pipeline end-to-end.

### `policyengine_us_data.calibration.source_impute.impute_source_variables`

```python
def impute_source_variables(data: Dict[str, Dict[int, np.ndarray]], state_fips: np.ndarray, time_period: int = 2024, dataset_path: Optional[str] = None, skip_acs: bool = False, skip_sipp: bool = False, skip_org: bool = False, skip_scf: bool = False) -> Dict[str, Dict[int, np.ndarray]]
```

Re-impute ACS/SIPP/ORG/SCF variables from donor surveys.

### `policyengine_us_data.release_promotion.stage4_reader.build_release_candidate_bundle_from_stage4_contract`

```python
def build_release_candidate_bundle_from_stage4_contract(*, context: ReleasePromotionContext, output_contract: StageContract, inventory_records: Iterable[Mapping[str, Any]] = (), source_output_contract_path: str | None = None, validation_report_paths: Sequence[str] = (), validation_report_refs: Sequence[DiagnosticRef] = (), diagnostics_manifest_path: str | None = None) -> ReleaseCandidateInputBundle
```

Build a candidate bundle from a Stage 4 output contract shape.

### `policyengine_us_data.release_promotion.stage4_reader.read_stage4_release_candidate_bundle`

```python
def read_stage4_release_candidate_bundle(*, context: ReleasePromotionContext, output_contract_path: str | Path, output_inventory_path: str | Path | None = None, source_output_contract_path: str | None = None, validation_report_paths: Sequence[str] = (), validation_report_refs: Sequence[DiagnosticRef] = (), diagnostics_manifest_path: str | None = None) -> ReleaseCandidateInputBundle
```

Read a candidate bundle from Stage 4 contract and optional inventory files.

### `policyengine_us_data.build_datasets.artifacts.stage_1_artifact_specs`

```python
def stage_1_artifact_specs() -> tuple[DatasetArtifactSpec, ...]
```

Return all artifact specs known to the Stage 1 dataset build.

### `policyengine_us_data.build_datasets.specs.stage_1_step_specs`

```python
def stage_1_step_specs() -> tuple[DatasetBuildStepSpec, ...]
```

Return the canonical Stage 1 dataset-build substage specs.

### `policyengine_us_data.utils.release_promotion.promote_full_release_with_result`

```python
def promote_full_release_with_result(config: FullReleasePromotionConfig, deps: FullReleasePromotionDependencies) -> 'FullPromotionResult'
```

Run the existing transaction engine and wrap its output in a typed result.

### `policyengine_us_data.calibration.unified_matrix_builder.UnifiedMatrixBuilder`

```python
class UnifiedMatrixBuilder
```

Build sparse calibration matrix for cloned CPS records.

### `modal_app.local_area.validate_staging`

```python
def validate_staging(branch: str, run_id: str, version: str = '') -> Dict
```

Validate all expected files and generate manifest.

### `policyengine_us_data.validation_core.context.ValidationArtifactResolver`

```python
class ValidationArtifactResolver
```

Resolve logical validation artifact names to stage-contract references.

### `policyengine_us_data.validation_core.checks.ValidationCheck`

```python
class ValidationCheck
```

One executable validation check with stable identity and dependencies.

### `policyengine_us_data.validation_core.context.ValidationContext`

```python
class ValidationContext
```

Read-only context passed to validation checks.

### `policyengine_us_data.validation_core.writers.ValidationReportWriter`

```python
class ValidationReportWriter
```

Write validation report outputs generated by output strategies.

### `policyengine_us_data.validation_core.runner.ValidationRunner`

```python
class ValidationRunner
```

Run validation suites and aggregate canonical stage-contract reports.

### `policyengine_us_data.validation_core.checks.ValidationSuite`

```python
class ValidationSuite
```

Ordered validation checks for one stage or substage boundary.

### `modal_app.pipeline.verify_runtime_seams`

```python
def verify_runtime_seams() -> dict
```

Verify deployed-image imports and subprocess seams.
