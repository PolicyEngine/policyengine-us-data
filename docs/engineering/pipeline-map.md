# Pipeline Map

Generated from `docs/pipeline_map.yaml` and `@pipeline_node` decorators.

## Data Build And Source-Imputed Dataset

Build base CPS/PUF artifacts and the source-imputed stratified CPS input.

- Status: `transitional`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `raw_sources` Raw survey and administrative sources | `external` | `current` | `moving` |  |
| `source_imputed_h5` source_imputed_stratified_extended_cps*.h5 | `artifact` | `current` | `moving` |  |
| `build_datasets` Build Datasets On Modal | `entrypoint` | `current` | `moving` | `modal_app.data_build.build_datasets` |
| `create_stratified` Create Stratified CPS Dataset | `entrypoint` | `current` | `moving` | `policyengine_us_data.calibration.create_stratified_cps.create_stratified_cps_dataset` |
| `source_impute` Source-Impute Stratified CPS | `entrypoint` | `current` | `moving` | `policyengine_us_data.calibration.source_impute.impute_source_variables` |

### Edges

- `raw_sources` -> `build_datasets` `external_source`
- `build_datasets` -> `create_stratified` `data_flow`
- `create_stratified` -> `source_impute` `data_flow`
- `source_impute` -> `source_imputed_h5` `produces_artifact`

## Calibration Matrix And Package Build

Resolve targets, assign geography, compute simulated values, and assemble the sparse matrix package.

- Status: `transitional`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `policy_data_db` policy_data.db | `artifact` | `current` | `moving` |  |
| `calibration_package` calibration_package.pkl | `artifact` | `current` | `moving` |  |
| `source_imputed_h5` source_imputed_stratified_extended_cps*.h5 | `artifact` | `current` | `moving` |  |
| `run_calibration` Run Unified Calibration | `entrypoint` | `transitional` | `moving` | `policyengine_us_data.calibration.unified_calibration.run_calibration` |
| `state_precomp` Per-State Simulation Precomputation | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_matrix_builder._compute_single_state` |
| `clone_assembly` Clone Value Assembly | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_matrix_builder._assemble_clone_values_standalone` |

### Edges

- `source_imputed_h5` -> `run_calibration` `data_flow`
- `policy_data_db` -> `run_calibration` `external_source`
- `run_calibration` -> `state_precomp` `uses_library`
- `state_precomp` -> `clone_assembly` `data_flow`
- `clone_assembly` -> `calibration_package` `produces_artifact`

## Sparse Weight Fitting

Fit L0-sparse calibration weights and write diagnostics.

- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `calibration_weights` calibration_weights.npy | `artifact` | `current` | `moving` |  |
| `diagnostics` diagnostics | `artifact` | `current` | `moving` |  |
| `calibration_package` calibration_package.pkl | `artifact` | `current` | `moving` |  |
| `init_weights` Compute Initial Weights | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.compute_initial_weights` |
| `fit_model` Fit L0 Calibration Weights | `library` | `current` | `moving` | `policyengine_us_data.calibration.unified_calibration.fit_l0_weights` |

### Edges

- `calibration_package` -> `init_weights` `data_flow`
- `init_weights` -> `fit_model` `data_flow`
- `fit_model` -> `calibration_weights` `produces_artifact`
- `fit_model` -> `diagnostics` `produces_artifact`

## Local Area H5 Build And Staging

Partition area work, resolve traceability, build H5s, validate, stage, and promote.

- Status: `transitional`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `staged_h5s` staged local-area H5 files | `artifact` | `current` | `moving` |  |
| `calibration_weights` calibration_weights.npy | `artifact` | `current` | `moving` |  |
| `load_calibration_geography` Load Calibration Geography | `library` | `legacy` | `moving` | `policyengine_us_data.calibration.local_h5.geography_loader.CalibrationGeographyLoader`, `policyengine_us_data.calibration.publish_local_area.load_calibration_geography` |
| `local_h5_traceability` FingerprintingService | `library` | `current` | `moving` | `policyengine_us_data.calibration.local_h5.fingerprinting.FingerprintingService` |
| `local_h5_partition` Partition Local H5 Work | `library` | `current` | `stable` | `policyengine_us_data.calibration.local_h5.partitioning.partition_weighted_work_items` |
| `build_h5` Build Local Area H5 | `library` | `transitional` | `moving` | `policyengine_us_data.calibration.publish_local_area.build_h5` |
| `validate_staging` Validate Staged H5 Files | `validation` | `current` | `moving` | `modal_app.local_area.validate_staging` |
| `staging_upload` Upload Local H5s To Staging | `entrypoint` | `current` | `moving` | `modal_app.local_area.upload_to_staging` |
| `atomic_promote` Atomic Promote Local H5 Files | `entrypoint` | `current` | `moving` | `policyengine_us_data.calibration.promote_local_h5s.promote` |

### Edges

- `calibration_weights` -> `load_calibration_geography` `data_flow`
- `load_calibration_geography` -> `local_h5_traceability` `data_flow`
- `local_h5_traceability` -> `local_h5_partition` `data_flow`
- `local_h5_partition` -> `build_h5` `data_flow`
- `build_h5` -> `staged_h5s` `produces_artifact`
- `staged_h5s` -> `validate_staging` `validates`
- `validate_staging` -> `staging_upload` `data_flow`
- `staging_upload` -> `atomic_promote` `data_flow`

## Modal Pipeline Orchestration

The Modal run controller that ties Stage 1-5 artifacts together with resume and promotion state.

- Status: `current`
- Stability: `moving`

| Node | Type | Status | Stability | API refs |
| --- | --- | --- | --- | --- |
| `pipeline_run` pipeline run metadata | `artifact` | `current` | `moving` |  |
| `run_modal_pipeline` Run Modal Pipeline | `entrypoint` | `current` | `moving` | `modal_app.pipeline.run_pipeline` |
| `build_datasets` Build Datasets On Modal | `entrypoint` | `current` | `moving` | `modal_app.data_build.build_datasets` |
| `run_calibration` Run Unified Calibration | `entrypoint` | `transitional` | `moving` | `policyengine_us_data.calibration.unified_calibration.run_calibration` |
| `coordinate_publish` Coordinate Local H5 Publish | `entrypoint` | `current` | `moving` | `modal_app.local_area.coordinate_publish` |

### Edges

- `run_modal_pipeline` -> `build_datasets` `data_flow`
- `run_modal_pipeline` -> `run_calibration` `data_flow`
- `run_modal_pipeline` -> `coordinate_publish` `data_flow`
- `run_modal_pipeline` -> `pipeline_run` `produces_artifact`

## Pydoc API Surface

### `policyengine_us_data.datasets.cps.enhanced_cps.create_aca_2025_takeup_override`

```python
def create_aca_2025_takeup_override(base_takeup: np.ndarray, person_enrolled_if_takeup: np.ndarray, person_weights: np.ndarray, person_tax_unit_ids: np.ndarray, tax_unit_ids: np.ndarray, target_people: float = ACA_POST_CALIBRATION_PERSON_TARGETS[2025]) -> np.ndarray
```

Add 2025 ACA takers until weighted APTC enrollment hits target.

### `policyengine_us_data.calibration.source_impute._impute_acs`

```python
def _impute_acs(data: Dict[str, Dict[int, np.ndarray]], state_fips: np.ndarray, time_period: int, dataset_path: Optional[str] = None) -> Dict[str, Dict[int, np.ndarray]]
```

Impute rent and real_estate_taxes from ACS with state.

### `policyengine_us_data.datasets.cps.cps.add_auto_loan_interest_and_net_worth`

```python
def add_auto_loan_interest_and_net_worth(self, cps: h5py.File) -> None
```

"Add auto loan balance, interest and net_worth variable.

### `policyengine_us_data.datasets.cps.cps.add_household_variables`

```python
def add_household_variables(cps: h5py.File, household: DataFrame) -> None
```

Populate household geography variables including state, county, and NYC flag.

### `policyengine_us_data.datasets.cps.cps.add_id_variables`

```python
def add_id_variables(cps: h5py.File, person: DataFrame, tax_unit: DataFrame, family: DataFrame, spm_unit: DataFrame, household: DataFrame) -> None
```

Add basic ID and weight variables.

### `policyengine_us_data.datasets.cps.cps.add_org_labor_market_inputs`

```python
def add_org_labor_market_inputs(cps: h5py.File) -> None
```

Impute ORG-derived wage and union inputs onto CPS persons.

### `policyengine_us_data.datasets.cps.cps.add_personal_income_variables`

```python
def add_personal_income_variables(cps: h5py.File, person: DataFrame, year: int)
```

Add income variables.

### `policyengine_us_data.datasets.cps.cps.add_personal_variables`

```python
def add_personal_variables(cps: h5py.File, person: DataFrame) -> None
```

Add personal demographic variables.

### `policyengine_us_data.datasets.cps.cps.add_previous_year_income`

```python
def add_previous_year_income(self, cps: h5py.File) -> None
```

Link CPS records across adjacent years and populate prior-year income inputs.

### `policyengine_us_data.datasets.cps.cps.add_rent`

```python
def add_rent(self, cps: h5py.File, person: DataFrame, household: DataFrame)
```

Impute rent and real estate taxes using ACS donor data.

### `policyengine_us_data.datasets.cps.cps.add_spm_variables`

```python
def add_spm_variables(self, cps: h5py.File, spm_unit: DataFrame) -> None
```

Populate CPS supplemental poverty measure variables and thresholds.

### `policyengine_us_data.datasets.cps.cps.add_ssn_card_type`

```python
def add_ssn_card_type(cps: h5py.File, person: pd.DataFrame, spm_unit: pd.DataFrame, time_period: int, undocumented_target: float = 13000000.0, undocumented_workers_target: float = 8300000.0, undocumented_students_target: float = 0.21 * 1900000.0) -> np.ndarray
```

Assign SSN card type using PRCITSHP, employment status, and ASEC-UA conditions.

### `policyengine_us_data.datasets.cps.cps.add_takeup`

```python
def add_takeup(self)
```

Apply stochastic takeup and reported-anchor alignment for benefit programs.

### `policyengine_us_data.datasets.cps.cps.add_tips`

```python
def add_tips(self, cps: h5py.File)
```

Impute tip income and household asset inputs from SIPP donor data.

### `modal_app.local_area.build_areas_worker`

```python
def build_areas_worker(branch: str, run_id: str, work_items: List[Dict], calibration_inputs: Dict[str, str], validate: bool = True) -> Dict
```

Worker function that builds a subset of H5 files.

### `policyengine_us_data.calibration.publish_local_area.build_cities`

```python
def build_cities(weights_path: Path, dataset_path: Path, geography, output_dir: Path, completed_cities: set, hf_batch_size: int = 10, takeup_filter: List[str] = None, upload: bool = False)
```

Build city H5 files with checkpointing, optionally uploading.

### `policyengine_us_data.calibration.publish_local_area.build_districts`

```python
def build_districts(weights_path: Path, dataset_path: Path, geography, output_dir: Path, completed_districts: set, hf_batch_size: int = 10, takeup_filter: List[str] = None, upload: bool = False)
```

Build district H5 files with checkpointing, optionally uploading.

### `policyengine_us_data.calibration.unified_matrix_builder.UnifiedMatrixBuilder.build_matrix`

```python
def build_matrix(self, geography, sim, target_filter: Optional[dict] = None, hierarchical_domains: Optional[List[str]] = None, cache_dir: Optional[str] = None, sim_modifier = None, rerandomize_takeup: bool = True, county_level: bool = True, workers: int = 1) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]
```

Build sparse calibration matrix.

### `policyengine_us_data.calibration.unified_matrix_builder.UnifiedMatrixBuilder.build_matrix_chunked`

```python
def build_matrix_chunked(self, geography, sim, target_filter: Optional[dict] = None, hierarchical_domains: Optional[List[str]] = None, chunk_size: int = 25000, chunk_dir: Optional[str] = None, keep_chunks: bool = False, resume_chunks: bool = False, rerandomize_takeup: bool = True) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]
```

Build a sparse matrix by materializing mixed-geography chunks.

### `modal_app.local_area._build_publishing_input_bundle`

```python
def _build_publishing_input_bundle(*, weights_path: Path, dataset_path: Path, db_path: Path | None, geography_path: Path | None, calibration_package_path: Path | None, run_config_path: Path | None, run_id: str, version: str, n_clones: int | None, seed: int, legacy_blocks_path: Path | None = None) -> PublishingInputBundle
```

Build the normalized coordinator input bundle for one publish scope.

### `policyengine_us_data.calibration.publish_local_area.build_states`

```python
def build_states(weights_path: Path, dataset_path: Path, geography, output_dir: Path, completed_states: set, hf_batch_size: int = 10, takeup_filter: List[str] = None, upload: bool = False, state_filter: str = None)
```

Build state H5 files with checkpointing, optionally uploading.

### `policyengine_us_data.calibration.unified_calibration.compute_diagnostics`

```python
def compute_diagnostics(weights: np.ndarray, X_sparse, targets_df, target_names: list) -> 'pd.DataFrame'
```

Compare fitted weighted sums to calibration targets and summarize error.

### `policyengine_us_data.calibration.local_h5.geography_loader.CalibrationGeographyLoader`

```python
class CalibrationGeographyLoader
```

Resolve and load exact geography artifacts for publication flows.

### `policyengine_us_data.datasets.cps.extended_cps._splice_clone_feature_predictions`

```python
def _splice_clone_feature_predictions(data: dict, predictions: pd.DataFrame, time_period: int) -> dict
```

Replace clone-half person-level feature variables with donor matches.

### `modal_app.local_area.partition_work`

```python
def partition_work(work_items: List[Dict], num_workers: int, completed: set) -> List[List[Dict]]
```

Compatibility wrapper over the extracted pure partitioning seam.

### `policyengine_us_data.datasets.cps.extended_cps._impute_cps_only_variables`

```python
def _impute_cps_only_variables(data: dict, time_period: int, dataset_path: str) -> pd.DataFrame
```

Second-stage QRF: train on CPS, predict for PUF clones.

### `modal_app.data_build.run_cps_then_puf_phase`

```python
def run_cps_then_puf_phase(branch: str, volume: modal.Volume, *, env: dict, log_file: IO = None) -> None
```

Build CPS before PUF because PUF pension imputation loads CPS_2024.

### `policyengine_us_data.datasets.cps.cps.CPS.downsample`

```python
def downsample(self, frac: float)
```

Subsample CPS arrays for released CPS vintages while full variants skip this step.

### `policyengine_us_data.datasets.cps.extended_cps.ExtendedCPS._drop_formula_variables`

```python
def _drop_formula_variables(cls, data)
```

Remove variables that are computed by policyengine-us.

### `policyengine_us_data.calibration.clone_and_assign.assign_random_geography`

```python
def assign_random_geography(n_records: int, n_clones: int = 10, seed: int = 42, household_agi: np.ndarray = None, cd_agi_targets: dict = None, agi_threshold_pctile: float = 90.0) -> GeographyAssignment
```

Assign random census block geography to cloned

### `policyengine_us_data.calibration.block_assignment.derive_geography_from_blocks`

```python
def derive_geography_from_blocks(block_geoids: np.ndarray) -> Dict[str, np.ndarray]
```

Derive all geography from pre-assigned block GEOIDs.

### `policyengine_us_data.datasets.puf.puf.impute_missing_demographics`

```python
def impute_missing_demographics(puf: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame
```

Impute missing PUF demographics from demographic donor records.

### `policyengine_us_data.datasets.puf.puf.impute_pension_contributions_to_puf`

```python
def impute_pension_contributions_to_puf(puf_df)
```

Impute pre-tax retirement contributions onto PUF tax units from CPS donors.

### `policyengine_us_data.calibration.local_h5.area_catalog.USAreaCatalog`

```python
class USAreaCatalog
```

Construct typed local H5 requests for the current US publication flow.

### `policyengine_us_data.calibration.local_h5.requests.AreaFilter`

```python
class AreaFilter
```

A single geography predicate used to select rows for one output area.

### `policyengine_us_data.calibration.local_h5.requests.AreaBuildRequest`

```python
class AreaBuildRequest
```

A complete request describing one local or national H5 to build.

### `policyengine_us_data.calibration.publish_local_area.compute_input_fingerprint`

```python
def compute_input_fingerprint(weights_path: Path, dataset_path: Path, n_clones: Optional[int] = None, seed: int = 42, geography_path: Optional[Path] = None, blocks_path: Optional[Path] = None, target_db_path: Optional[Path] = None, run_config_path: Optional[Path] = None, calibration_package_path: Optional[Path] = None, scope: str = 'regional') -> str
```

Compute a scope fingerprint for local H5 checkpoint and resume decisions.

### `policyengine_us_data.calibration.local_h5.fingerprinting.PublishingInputBundle`

```python
class PublishingInputBundle
```

File-system and run metadata needed to publish one H5 scope.

### `policyengine_us_data.calibration.local_h5.fingerprinting.TraceabilityBundle`

```python
class TraceabilityBundle
```

Full provenance record for one publish scope.

### `policyengine_us_data.calibration.promote_local_h5s.stage`

```python
def stage(files: list, version: str, run_id: str = '')
```

Upload locally built H5 files into Hugging Face staging paths.

### `policyengine_us_data.utils.mortgage_interest.convert_mortgage_interest_to_structural_inputs`

```python
def convert_mortgage_interest_to_structural_inputs(data: Dict[str, Dict[int, np.ndarray]], time_period: int) -> Dict[str, Dict[int, np.ndarray]]
```

Replace formula-level mortgage inputs with structural mortgage data.

### `policyengine_us_data.utils.mortgage_interest.impute_tax_unit_mortgage_balance_hints`

```python
def impute_tax_unit_mortgage_balance_hints(data: Dict[str, Dict[int, np.ndarray]], time_period: int) -> Dict[str, Dict[int, np.ndarray]]
```

Impute tax-unit mortgage balance hints from SCF data.

### `policyengine_us_data.datasets.puf.puf.preprocess_puf`

```python
def preprocess_puf(puf: pd.DataFrame) -> pd.DataFrame
```

Rename IRS variables and derive PolicyEngine-ready PUF tax inputs.

### `modal_app.pipeline.promote_run`

```python
def promote_run(run_id: str, version: str = None) -> str
```

Promote a completed pipeline run to production.

### `policyengine_us_data.calibration.puf_impute._run_qrf_imputation`

```python
def _run_qrf_imputation(data: Dict[str, Dict[int, np.ndarray]], time_period: int, puf_dataset, dataset_path: Optional[str] = None) -> tuple
```

Run QRF imputation for PUF variables.

### `policyengine_us_data.datasets.cps.extended_cps._splice_cps_only_predictions`

```python
def _splice_cps_only_predictions(data: dict, predictions: pd.DataFrame, time_period: int, dataset_path: str) -> dict
```

Replace PUF clone half of CPS-only variables with QRF predictions.

### `policyengine_us_data.calibration.puf_impute.puf_clone_dataset`

```python
def puf_clone_dataset(data: Dict[str, Dict[int, np.ndarray]], state_fips: np.ndarray, time_period: int = 2024, puf_dataset = None, skip_qrf: bool = False, dataset_path: Optional[str] = None) -> Dict[str, Dict[int, np.ndarray]]
```

Clone CPS data 2x and impute PUF variables on one half.

### `modal_app.local_area._resolve_scope_fingerprint`

```python
def _resolve_scope_fingerprint(*, inputs: PublishingInputBundle, scope: str, expected_fingerprint: str = '') -> str
```

Compute the scope fingerprint while preserving pinned resume values.

### `policyengine_us_data.calibration.puf_impute._impute_retirement_contributions`

```python
def _impute_retirement_contributions(data: Dict[str, Dict[int, np.ndarray]], puf_imputations: Dict[str, np.ndarray], time_period: int, dataset_path: str) -> Dict[str, np.ndarray]
```

Impute retirement contributions for the PUF half using QRF.

### `policyengine_us_data.datasets.cps.enhanced_cps.reweight`

```python
def reweight(original_weights, loss_matrix, targets_array, log_path = 'calibration_log.csv', epochs = 500, l0_lambda = 2.6445e-07, init_mean = 0.999, temperature = 0.25, seed = 1456)
```

Fits enhanced CPS weights against calibration targets with the hard-concrete loss machinery.

### `modal_app.local_area.run_phase`

```python
def run_phase(phase_name: str, work_items: List[Dict], num_workers: int, completed: set, branch: str, run_id: str, calibration_inputs: Dict[str, str], run_dir: Path, validate: bool = True) -> tuple
```

Run a single build phase, spawning workers and collecting results.

### `policyengine_us_data.calibration.sanity_checks.run_sanity_checks`

```python
def run_sanity_checks(h5_path: str, period: int = 2024) -> List[dict]
```

Run structural integrity checks on an H5 file.

### `policyengine_us_data.calibration.source_impute._impute_scf`

```python
def _impute_scf(data: Dict[str, Dict[int, np.ndarray]], state_fips: np.ndarray, time_period: int, dataset_path: Optional[str] = None) -> Dict[str, Dict[int, np.ndarray]]
```

Impute net_worth and auto_loan from SCF.

### `policyengine_us_data.datasets.puf.puf.simulate_w2_and_ubia_from_puf`

```python
def simulate_w2_and_ubia_from_puf(puf, *, seed = None, diagnostics = True)
```

Simulate two Section 199A guard-rail quantities for every record

### `policyengine_us_data.calibration.source_impute._impute_sipp`

```python
def _impute_sipp(data: Dict[str, Dict[int, np.ndarray]], state_fips: np.ndarray, time_period: int, dataset_path: Optional[str] = None) -> Dict[str, Dict[int, np.ndarray]]
```

Impute tip_income, liquid assets, and vehicle signals from SIPP.

### `policyengine_us_data.calibration.puf_impute.reconcile_ss_subcomponents`

```python
def reconcile_ss_subcomponents(data: Dict[str, Dict[int, np.ndarray]], n_cps: int, time_period: int) -> None
```

Predict SS sub-components for PUF half from demographics.

### `modal_app.pipeline.stage_base_datasets`

```python
def stage_base_datasets(run_id: str, version: str, branch: str) -> None
```

Upload source_imputed + policy_data.db from pipeline

### `policyengine_us_data.calibration.validate_staging.validate_area`

```python
def validate_area(sim, targets_df: pd.DataFrame, engine, area_type: str, area_id: str, display_id: str, dataset_path: str, period: int, training_mask: np.ndarray, variable_entity_map: dict, constraints_map: Optional[dict] = None) -> list
```

Run microsimulation target comparisons for one staged area.

### `policyengine_us_data.calibration.unified_matrix_builder.UnifiedMatrixBuilder`

```python
class UnifiedMatrixBuilder
```

Build sparse calibration matrix for cloned CPS records.

### `modal_app.pipeline.verify_runtime_seams`

```python
def verify_runtime_seams() -> dict
```

Verify deployed-image imports and subprocess seams.

### `policyengine_us_data.calibration.puf_impute._impute_weeks_unemployed`

```python
def _impute_weeks_unemployed(data: Dict[str, Dict[int, np.ndarray]], puf_imputations: Dict[str, np.ndarray], time_period: int, dataset_path: str) -> np.ndarray
```

Impute weeks_unemployed for the PUF half using QRF.
