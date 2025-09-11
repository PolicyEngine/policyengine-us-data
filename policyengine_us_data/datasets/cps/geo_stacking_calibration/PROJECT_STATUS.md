# Geo-Stacking Calibration: Project Status

## Current Issues & Analysis

### The Texas Problem (Critical)

Analysis of L0 sparse calibration weights (97.8% sparsity) reveals severe underfitting for specific states, particularly Texas, which achieves only 24.5% of its population target.

#### Performance Metrics
- **Overall mean relative error**: 6.27% across all 5,717 targets
- **National targets**: Excellent performance (<0.03% error)
- **State targets**: Highly variable (0% to 88% error)
- **Active weights**: 24,331 out of 1,083,801 (2.24% active)

#### Texas-Specific Issues
- **Mean error**: 26.1% (highest of all states)
- **Max error**: 88.1% (age group 60-64)
- **Active weights**: Only 40 out of 21,251 available (0.2% activation rate)
- **Population coverage**: 7.5M out of 30.5M target (24.5% achievement)

Paradoxically, Texas is the second-most represented state in the underlying CPS data (1,365 households, 6.4% of dataset).

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

### Root Cause Analysis

1. **Extreme Sparsity Constraint**: The 97.8% sparsity constraint forces selection of only 2.2% of available household weights, creating competition for "universal donor" households.

2. **Texas Household Characteristics**: Despite good representation in base data, Texas households appear to be poor universal donors. The optimizer sacrifices Texas accuracy for better overall performance.

3. **Weight Magnitude Constraints**: With only 40 active weights for 30.5M people, each weight would need to average 763K - approximately 500x larger than typical survey weights.

### Recommendations

#### Short-term Solutions
1. **Reduce sparsity constraint**: Target 95-96% sparsity instead of 97.8%
2. **State-specific minimum weights**: Enforce minimum 1% activation per state
3. **Population-proportional sparsity**: Allocate active weights proportional to state populations

#### Long-term Solutions
1. **Hierarchical calibration**: Calibrate national targets first, then state targets
2. **State-specific models**: Separate calibration for problematic states
3. **Adaptive sparsity**: Allow sparsity to vary by state based on fit quality

## In Progress 🚧

### Congressional District Support
- Functions are stubbed out but need testing
- Will create even sparser matrices (436 CDs)
- ~~Memory feasible but computation time is the bottleneck~~ **RESOLVED with stratified sampling**
- Stratified dataset reduces matrix from 49M to 5.7M columns (88% reduction)

## To Do 📋

### 1. Scale to All States
- [ ] Test with all 51 states (including DC)
- [ ] Monitor memory usage and performance
- [ ] Verify group-wise loss still converges well

### 2. Add Remaining Demographic Groups
- [x] Age targets (stratum_group_id = 2) - COMPLETED
- [x] SNAP targets (stratum_group_id = 4) - COMPLETED  
- [x] Medicaid targets (stratum_group_id = 5) - COMPLETED (person_count only)
- [ ] Income/AGI targets (stratum_group_id = 3) - TODO
- [ ] EITC targets (stratum_group_id = 6) - TODO

### 3. Optimization & Performance
- [ ] Parallelize matrix construction for speed
- [ ] Implement chunking strategies for very large matrices
- [ ] Consider GPU acceleration for L0 optimization

### 4. Production Readiness
- [ ] Address temporal mismatch between CPS data (2024) and targets (various years)
- [ ] Implement proper uprating for temporal consistency
- [ ] Create validation suite for calibration quality
- [ ] Build monitoring/diagnostics dashboard

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

### 2025-09-10: Congressional District Target Filtering Attempt - FAILED ❌

#### The Problem
When trying to build calibration matrix for 436 congressional districts, memory usage was projected to reach 32+ GB for full target set. Attempted to reduce memory by filtering out specific target groups (EITC and IRS scalars).

#### What We Tried
Created `build_stacked_matrix_sparse_filtered()` method to selectively include target groups:
- Planned to exclude EITC (group 6) and IRS scalars (group 7) 
- Keep national, age, AGI distribution, SNAP, and Medicaid targets

#### Why It Failed
1. **Indexing Error**: Method incorrectly tried to use original simulation indices (112,502) on stacked matrix (1,125,020 columns for 10 CDs)
2. **Multiplicative Effect Underestimated**: EITC has 6 targets × 436 CDs = 2,616 targets total (not just 6)
3. **Target Interdependencies**: National targets need to sum correctly across all geographies; removing groups breaks validation
4. **Column Index Out of Bounds**: Got errors like "column index 112607 out of bounds" - corrupted matrix construction

#### Lessons Learned
- Target filtering is much harder than it seems due to interdependencies
- Each target group scales by number of geographies (multiplicative, not additive)
- **Household subsampling is likely superior approach** - preserves all targets while reducing memory proportionally

#### Recommendation
For memory reduction, use household subsampling instead:
```python
sample_rate = 0.3  # Use 30% of households
household_mask = np.random.random(n_households) < sample_rate
X_sparse_sampled = X_sparse[:, household_mask]
```

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

#### Original Issues
1. **ID Overflow Warnings**: PolicyEngine multiplies person IDs by 100 for RNG seeds
2. **Duplicate Persons**: Same household appearing in multiple states
3. **Household Count Mismatch**: Only 64,522 households instead of 167,089 non-zero weights

#### Root Cause Discovery
- L0 sparse calibration creates "universal donor" households active in multiple states
- 33,484 households (30%) had weights in multiple states  
- Some households active in up to 50 states!
- Original approach incorrectly assigned each household to only ONE state (max weight)

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

#### Results
- **167,089 households** in final dataset (matching non-zero weights exactly)
- **495,170 persons** with max ID well below int32 limit
- **No overflow** when PolicyEngine multiplies by 100
- **No duplicate persons** - each household-state combo is unique
- **Proper state assignments** - each household has correct state_fips
- **Total population**: 136M across all states

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

## Next Priority Actions

1. **Run full 51-state calibration** - The system is ready, test at scale
2. **Experiment with sparsity relaxation** - Try 95% instead of 97.8% to improve Texas
3. **Add income demographic targets** - Next logical variable type to include
4. **Parallelize matrix construction** - Address the computation bottleneck

## Project Files

### Core Implementation
- `metrics_matrix_geo_stacking_sparse.py` - Sparse matrix builder
- `calibrate_states_sparse.py` - Main calibration script with diagnostics
- `calibration_utils.py` - Shared utilities (target grouping)
- `weight_diagnostics.py` - Standalone weight analysis tool
- `create_sparse_state_stacked.py` - Creates sparse state-stacked dataset from calibrated weights
- `create_stratified_cps.py` - Creates stratified sample preserving high-income households

### L0 Package (~/devl/L0)
- `l0/calibration.py` - Core calibration class
- `tests/test_calibration.py` - Test coverage

### Documentation
- `GEO_STACKING_TECHNICAL.md` - Technical documentation and architecture
- `PROJECT_STATUS.md` - This file (active project management)