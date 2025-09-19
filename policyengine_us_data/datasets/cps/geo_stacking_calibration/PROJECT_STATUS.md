# Geo-Stacking Calibration: Project Status

### In Progress 🚧

### Congressional District Target Hierarchy Issue (Critical)

After careful analysis, the correct target count **for congressional district calibration** should be:

| Target Type | Count | Calculation | Notes |
|-------------|-------|-------------|-------|
| National | 5 | From etl_national_targets | All 5 confirmed present |
| CD Age | 7,848 | 18 bins × 436 CDs | Survey source |
| CD Medicaid | 436 | 1 × 436 CDs | Survey (state admin exists but not used) |
| SNAP Hybrid | 487 | 436 CD household_count + 51 state cost | Mixed admin sources |
| CD IRS SOI | 21,800 | 50 × 436 CDs | See breakdown below |
| **TOTAL** | **30,576** | | **For CD calibration only** |

**IRS SOI Breakdown (50 variables per CD)**:
- 20 straightforward targets with tax_unit_count and amount (20 × 2 = 40)
  - Includes 4 EITC categories (eitc_qualifying_children_0 through 3)
- 9 AGI histogram bins with ONE count variable (9 × 1 = 9)
  - Must choose between person_count or tax_unit_count for consistency
  - NOT including adjusted_gross_income amounts in bins (would double-count)
- 1 AGI total amount scalar
- Total: 40 + 9 + 1 = 50 per CD

**Key Design Decision for CD Calibration**: State SNAP cost targets (51 total) apply to households within each state but remain state-level constraints. Households in CDs within a state have non-zero values in the design matrix for their state's SNAP cost target.

**Note**: This target accounting is specific to congressional district calibration. State-level calibration will have a different target structure and count.

#### What Should Happen (Hierarchical Target Selection)
For each target concept (e.g., "age 25-30 population in Texas"):
1. **If CD-level target exists** → use it for that CD only
2. **If no CD target but state target exists** → use state target for all CDs in that state  
3. **If neither CD nor state target exists** → use national target

For administrative data (e.g., SNAP):
- **Always prefer administrative over survey data**, even if admin is less granular
- State-level SNAP admin data should override CD-level survey estimates

## Analysis

#### State Activation Patterns

Clear inverse correlation between activation rate and error:

| State | Active Weights | Activation Rate | Mean Error |
|-------|---------------|-----------------|------------|
| Texas | 40 | 0.2% | 26.1% |
| Alaska | 35 | 0.2% | 21.8% |
| Tennessee | 39 | 0.2% | 18.3% |
| **vs** | | | |
| DC | 1,177 | 5.5% | 7.1% |
| Connecticut | 1,095 | 5.2% | 4.1% |
| Maryland | 1,062 | 5.0% | 3.6% |

#### Population Target Achievement

| State | Target Pop | Sum of Weights | Achievement |
|-------|------------|----------------|-------------|
| Texas | 30,503,301 | 7,484,589 | 24.5% |
| California | 38,965,193 | 14,532,248 | 37.3% |
| North Carolina | 10,835,491 | 3,609,763 | 33.3% |
| Florida | 22,610,726 | 7,601,966 | 33.6% |
| New York | 19,571,216 | 7,328,156 | 37.4% |

## Implementation History

### December 2024: SNAP Integration
- Successfully integrated SNAP administrative targets from USDA FNS data
- Using state-level administrative data only
- Two variables per state: `household_count` and `snap` (benefit costs)
- Fixed constraint handling for SNAP > 0 with explicit `.astype(bool)` conversion
- SNAP targets form their own group (Group 6) in group-wise loss averaging

### 2025-09-04: Sparse Matrix Implementation ✅
- Eliminated dense matrix creation achieving **99% memory reduction**
- 51 states: 23 GB dense → 166 MB sparse
- Created `metrics_matrix_geo_stacking_sparse.py` and `calibrate_states_sparse.py`
- Memory is solved! Bottleneck is now computation time

### 2025-09-07: L0 Calibration API Improvements ✅
- Replaced `init_weight_scale` with intuitive `init_weights` parameter
- Added per-feature gate initialization via arrays
- State-aware initialization now first-class feature
- Clean separation between calibration weights and sparsity gates

### 2025-09-07: Population-Based Weight Initialization ✅
- Fixed critical initialization where all weights started at 1.0
- Base weight = state_population / n_households_per_state
- Sparsity adjustment = 1/sqrt(keep_probability)
- Texas households now start at ~20,000 instead of 1.0

### 2025-09-08: Weight-to-Reality Mapping ✅
- Verified lossless weight mapping structure
- Documented weight vector indexing formula
- Created `weight_diagnostics.py` for verification
- Established Microsimulation as ground truth for household ordering

### 2025-09-09: Sparse State-Stacked Dataset Creation ✅
- Created `create_sparse_state_stacked.py` to build reality-linked dataset
- Successfully reduced 5.7M household dataset (would crash system) to 64K households
- Achieved **97% memory reduction** while preserving calibrated weights
- Used DataFrame approach to handle all entity types correctly (households, persons, tax units, SPM units, marital units)
- Dataset loads successfully in Microsimulation with all relationships intact
- Key findings:
  - Florida has only 906 active households but achieves 10M population through high weights
  - All state_fips values correctly assigned and consistent across entities
  - Total population achieved: 136M across all states

#### Technical Implementation
- Leveraged `Dataset.from_dataframe()` for automatic entity relationship handling
- **Critical**: Added household-to-state assignment logic - each household assigned to state with maximum weight
- Modified entity IDs using encoding scheme:
  - Household IDs: `state_idx * 10_000_000 + original_id`
  - Person/Tax/SPM/Marital IDs: `state_idx * 100_000_000 + original_id`
- Added complete reindexing after combination to prevent overflow
- Processed each state separately to manage memory, then concatenated DataFrames
- Validated against original `extended_cps_2023.h5` (112,502 households)
- Output: `/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/sparse_state_stacked_2023.h5`

### 2025-09-11: Stratified CPS Sampling for Congressional Districts ✅

Created `create_stratified_cps.py` to subsample extended_cps_2023.h5 while preserving high-income households for congressional district calibration.

#### The Problem
- Full dataset: 436 CDs × 112,502 households = 49M matrix columns (32+ GB memory)
- Even sparse matrices hit memory limits on 32GB machines and 15GB GPUs
- Random sampling would lose critical high-income households

#### The Solution: Income-Based Stratified Sampling
- **Preserves ALL households above 99th percentile** (AGI > $797,706)
- Progressive sampling rates by income strata:
  - Top 0.1%: 100% kept
  - 99-99.5%: 100% kept  
  - 95-99%: 80% kept
  - 90-95%: 60% kept
  - Lower strata: 10-40% kept
- Flexible target sizing (10k-30k households)

#### Results
- **10k target → 13k actual** (due to preserving all high earners)
- **30k target → 29k actual** (well-balanced across strata)
- **Maximum AGI preserved**: $2,276,370 in both samples
- **Memory reduction**: 436 CDs × 13k = 5.7M columns (88% reduction)
- Successfully handles tricky `county_fips` and enum types

#### Technical Notes
- Uses same DataFrame approach as `create_sparse_state_stacked.py`
- Reproducible with seed=42 for random sampling within strata
- Output: `/storage/stratified_extended_cps_2023.h5`

### 2025-09-09: Sparse Dataset Creation - FULLY RESOLVED ✅

#### The Conceptual Breakthrough
**Key Insight**: In geo-stacking, each household-state pair with non-zero weight should be treated as a **separate household** in the final dataset. 

Example:
- Household 6 has weight 32.57 in Hawaii and weight 0.79 in South Dakota
- This becomes TWO separate households in the sparse dataset:
  - One household assigned to Hawaii with weight 32.57
  - Another household assigned to South Dakota with weight 0.79

#### Final Implementation ✅
Modified `create_sparse_state_stacked.py` to:
1. Keep ALL household-state pairs where weight > 0 (not just max weight)
2. Process each state independently, keeping all active households
3. After concatenation, reindex all entities to handle duplicates:
   - Each household occurrence gets unique ID
   - Person/tax/SPM/marital units properly linked to new household IDs
4. Sequential reindexing keeps IDs small to prevent overflow

## Pipeline Control Mechanism (2025-01-10) ✅

### Environment Variable Control
The geo-stacking pipeline is now controlled via the `GEO_STACKING_MODE` environment variable:

```bash
# Run the geo-stacking pipeline (generates BOTH 2023 and 2024)
GEO_STACKING_MODE=true make data

# Run the regular pipeline (only 2024)
make data
```

This mechanism:
- When `GEO_STACKING_MODE=true`:
  - Generates `ExtendedCPS_2023` using `CPS_2023_Full` (non-downsampled) for geo-stacking
  - Also generates `ExtendedCPS_2024` to satisfy downstream dependencies
  - All downstream scripts (enhanced_cps, small_enhanced_cps) run normally
- When not set (default):
  - Only generates `ExtendedCPS_2024` as usual
- Provides clear logging to indicate which mode is active
- Ready for future workflow integration but not yet added to CI/CD

### Implementation Details
- Modified only `extended_cps.py` - no changes needed to other pipeline scripts
- Generates both datasets in geo-stacking mode to avoid breaking downstream dependencies
- Extra compute cost is acceptable for the simplicity gained

## Variable Coverage Analysis (2025-01-16) ✅

### Analysis Scripts Created
Seven diagnostic scripts were created to analyze variable coverage:

1. **`analyze_missing_variables.py`** - Initial legacy column analysis
2. **`analyze_missing_actionable.py`** - Tests PolicyEngine variable availability  
3. **`compare_legacy_vs_new.py`** - Direct legacy vs new comparison
4. **`analyze_calibration_coverage.py`** - Checks what's actually in calibration matrix
5. **`missing_irs_variables.py`** - Compares IRS SOI documentation to database
6. **`irs_variables_final_analysis.py`** - Final IRS variable analysis with ETL check
7. **`missing_national_targets.py`** - Identifies missing national-level targets

### Key Findings

#### ✅ Variables We Have (Confirmed)
- **IRS SOI Variables** (19 total at CD level):
  - Income tax, EITC (by children), qualified dividends, capital gains
  - SALT payments, medical expense deductions, QBI deductions
  - Unemployment compensation, taxable social security/pensions
  - Real estate taxes, partnership/S-corp income
- **Demographics**: Age bins (18 categories)
- **Benefits**: SNAP (hybrid state/CD), Medicaid enrollment
- **National Targets**: 5 hardcoded from database

#### ❌ Critical Missing Variables

**1. Self-Employment Income (A00900)** - **CONFIRMED MISSING**
- Boss was correct - this is NOT in the database
- IRS provides it at CD level (Schedule C business income)
- Added to `etl_irs_soi.py` line 227 but database needs update
- PolicyEngine variable: `self_employment_income` ($444B total)

**2. Major Benefits Programs**
- **Social Security benefits** (~$1.5T) - Have taxable portion, missing total
- **SSI** (~$60B) - Completely missing
- **TANF** ($9B) - Hardcoded in loss.py, missing from our calibration

**3. Tax Expenditures vs Deductions**
- We have deduction AMOUNTS (what people claimed)
- Missing tax EXPENDITURES (federal revenue loss)
- Example: Have SALT payments, missing SALT revenue impact

**4. Other IRS Variables Available but Not Extracted**
- A25870: Rental and royalty income
- A19700: Charitable contributions  
- A19300: Mortgage interest
- A09400: Self-employment tax

### Understanding Variable Naming

**Legacy System Structure**:
- Format: `geography/source/variable/details`
- Example: `nation/irs/business net profits/total/AGI in -inf-inf/taxable/All`

**Key Mappings**:
- `business_net_profits` = PolicyEngine's `self_employment_income` (positive values)
- `rent_and_royalty_net_income` = PolicyEngine's `rental_income`
- These are split into positive/negative in legacy for IRS alignment

**Geographic Levels**:
- National: Authoritative totals (CBO, Treasury)
- State: Some admin data (SNAP costs)
- CD: Primarily IRS SOI and survey data

### Action Items

**Immediate** (Database Updates Needed):
1. Run ETL with self_employment_income (A00900) added
2. Add Social Security benefits, SSI, TANF as national targets
3. Consider adding filing status breakdowns

**Future Improvements**:
- Add more IRS variables (rental, charitable, mortgage interest)
- Implement hierarchical target selection (prefer admin over survey)
- Add tax expenditure targets for better high-income calibration

## ETL and Uprating Refactoring (2025-09-18) ✅

### Major Refactoring of National Targets ETL

Refactored `etl_national_targets.py` to follow proper ETL pattern and moved uprating logic to calibration pipeline:

#### Key Changes Made:

1. **Proper ETL Structure**:
   - Separated into `extract_national_targets()`, `transform_national_targets()`, and `load_national_targets()` functions
   - Fixed code ordering bug where `sim` was used before being defined
   - Removed unnecessary variable group metadata creation (not used by calibration system)

2. **Enrollment Count Handling**:
   - Split targets into direct sum targets (dollar amounts) and conditional count targets (enrollments)
   - Created proper strata with constraints for enrollment counts (e.g., `medicaid > 0` with target `person_count`)
   - Follows pattern established in `etl_snap.py`

3. **Uprating Moved to Calibration**:
   - **Database now stores actual source years**: 2024 for hardcoded values from loss.py, 2023 for CBO/Treasury
   - Added `uprate_target_value()` and `uprate_targets_df()` to `calibration_utils.py`
   - All `get_*_targets()` methods in `SparseGeoStackingMatrixBuilder` now apply uprating
   - Uses CPI-U for monetary values, population growth for count variables

#### Important Notes:

⚠️ **Database Recreation Required**: After ETL changes, must delete and recreate `policy_data.db`:
```bash
rm policyengine_us_data/storage/policy_data.db
python policyengine_us_data/db/create_database_tables.py
python policyengine_us_data/db/create_initial_strata.py
python policyengine_us_data/db/etl_national_targets.py
```

⚠️ **Import Issues**: Added fallback imports in `metrics_matrix_geo_stacking_sparse.py` due to `microimpute` dependency issues

⚠️ **Years in Database**: Targets now show their actual source years (2023/2024 mix) rather than all being 2023

#### Benefits of New Approach:

- **Transparency**: Database shows actual source years
- **Flexibility**: Can calibrate to any dataset year without re-running ETL
- **Auditability**: Uprating happens explicitly with logging (shows when >1% change)
- **Correctness**: Each target type uses appropriate uprating method

#### Uprating Factors (2024→2023):
- CPI-U: 0.970018 (3% reduction for monetary values)
- Population: 0.989172 (1.1% reduction for enrollment counts)

### Redundant Uprating Issue (2025-09-19) ⚠️

Discovered redundant uprating calculations causing excessive console output and wasted computation:

#### The Problem:
- National targets are fetched and uprated **for each geographic unit** (state or CD)
- With 436 CDs, the same 33 national targets get uprated 436 times redundantly
- Each uprating with >1% change prints a log message to console
- Results in thousands of repetitive console messages and unnecessary computation

#### Uprating Details:
- **National variables** (2024→2023): Downrated using CPI factor 0.9700
  - Examples: interest_deduction, medicaid, rent, tanf
- **IRS scalar variables** (2022→2023): Uprated using CPI factor 1.0641  
  - Examples: income_tax, qualified_business_income_deduction, taxable_ira_distributions
- **IRS AGI distribution** (2022→2023): Uprated using **population growth** factor 1.0641
  - These are `person_count` variables counting people in each AGI bin
  - Correctly uses population growth, not CPI, for demographic counts

#### Impact:
- **Performance**: ~436x more uprating calculations than necessary for national targets
- **Console output**: Thousands of redundant log messages making progress hard to track
- **User experience**: Appears frozen due to console spam, though actually progressing

#### Solution Needed:
- Cache uprated national targets since they're identical for all geographic units
- Consider caching other repeatedly uprated target sets
- Would reduce uprating calls from O(n_geographic_units) to O(1) for shared targets

## Next Priority Actions

### TODOs 

1. **Add epoch-by-epoch logging for calibration dashboard** - Enable loss curve visualization
2. **Update database with self_employment_income** - Re-run ETL with A00900 added
3. **Add missing benefit programs** - Social Security total, SSI, TANF at national level (Note: TANF was added in the refactoring)
4. **Add filing status breakdowns for IRS variables** - The legacy system segments many IRS variables by filing status (Single, MFJ/Surviving Spouse, MFS, Head of Household). This should be added as stratum constraints to improve calibration accuracy.

### Epoch Logging Implementation Plan

To enable loss curve visualization in the calibration dashboard (https://microcalibrate.vercel.app), we need to capture metrics at regular intervals during training. The dashboard expects a CSV with columns: `target_name`, `estimate`, `target`, `epoch`, `error`, `rel_error`, `abs_error`, `rel_abs_error`, `loss`.

**Recommended approach (without modifying L0):**

Train in chunks of epochs and capture metrics between chunks:

```python
# In calibrate_cds_sparse.py or calibrate_states_sparse.py
epochs_per_chunk = 50
total_epochs = 1000
epoch_data = []

for chunk in range(0, total_epochs, epochs_per_chunk):
    # Train for a chunk of epochs
    model.fit(
        M=X_sparse,
        y=targets,
        lambda_l0=0.01,
        epochs=epochs_per_chunk,
        loss_type="relative",
        verbose=True,
        verbose_freq=epochs_per_chunk,
        target_groups=target_groups
    )
    
    # Capture metrics after this chunk
    with torch.no_grad():
        y_pred = model.forward(X_sparse, deterministic=True).cpu().numpy()
        
        for i, (idx, row) in enumerate(targets_df.iterrows()):
            # Create hierarchical target name
            if row['geographic_id'] == 'US':
                target_name = f"nation/{row['variable']}/{row['description']}"
            else:
                target_name = f"CD{row['geographic_id']}/{row['variable']}/{row['description']}"
            
            # Calculate all metrics
            estimate = y_pred[i]
            target = row['value']
            error = estimate - target
            rel_error = error / target if target != 0 else 0
            
            epoch_data.append({
                'target_name': target_name,
                'estimate': estimate,
                'target': target,
                'epoch': chunk + epochs_per_chunk,
                'error': error,
                'rel_error': rel_error,
                'abs_error': abs(error),
                'rel_abs_error': abs(rel_error),
                'loss': rel_error ** 2
            })

# Save to CSV
calibration_log = pd.DataFrame(epoch_data)
calibration_log.to_csv('calibration_log.csv', index=False)
```

This approach:
- Trains efficiently in 50-epoch chunks (avoiding single-epoch overhead)
- Captures full metrics every 50 epochs for the loss curve
- Produces the exact CSV format expected by the dashboard
- Works without any modifications to the L0 package

## Project Files

### Core Implementation
- `metrics_matrix_geo_stacking_sparse.py` - Sparse matrix builder
- `calibrate_states_sparse.py` - Main calibration script with diagnostics
- `calibrate_cds_sparse.py` - Congressional district calibration script
- `calibration_utils.py` - Shared utilities (target grouping)
- `weight_diagnostics.py` - State-level weight analysis tool with CSV export
- `cd_weight_diagnostics.py` - CD-level weight analysis tool with CSV export
- `create_sparse_state_stacked.py` - Creates sparse state-stacked dataset from calibrated weights
- `create_stratified_cps.py` - Creates stratified sample preserving high-income households

### Diagnostic Scripts (Can be cleaned up later)
- `analyze_cd_exclusions.py` - Analysis of excluded CD targets in dashboard
- `check_duplicates.py` - Investigation of duplicate targets in CSV output

### L0 Package (~/devl/L0)
- `l0/calibration.py` - Core calibration class
- `tests/test_calibration.py` - Test coverage

### Documentation
- `GEO_STACKING_TECHNICAL.md` - Technical documentation and architecture
- `PROJECT_STATUS.md` - This file (active project management)
