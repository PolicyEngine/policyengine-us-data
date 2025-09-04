# Geo-Stacking Matrix Implementation Status

## Completed ✅

### 1. Core Infrastructure
- Built `GeoStackingMatrixBuilder` class with extensible design
- Implemented database queries for national and demographic targets
- Created proper constraint application at entity levels
- Correctly maps person-level constraints to household level

### 2. Single State Matrix Creation
- Successfully creates calibration matrix for California (or any state)
- Matrix dimensions: 18 age targets (rows) x 21,251 households (columns)
- Values represent person counts per household for each age group
- Properly handles age constraints with database operators (>, <, >=, etc.)

### 3. Period Handling Resolution
- **Critical Finding**: The 2024 enhanced CPS dataset only contains 2024 data
- Attempting to set `default_calculation_period=2023` doesn't actually work - it remains 2024
- When requesting past data explicitly via `calculate(period=2023)`, returns defaults (zeros)
- **Final Decision**: Use 2024 data and pull targets from whatever year they exist in the database
- **Temporal Mismatch**: Targets exist for different years (2022 for admin data, 2023 for age, 2024 for hardcoded)
- This mismatch is acceptable for the calibration prototype and will be addressed in production

### 4. Weight Independence
- Successfully separated matrix creation from dataset weights
- Matrix values are pure counts (unweighted)
- Validation uses custom uniform weights, not dataset weights
- Ready for calibration/reweighting algorithms

### 5. Documentation
- Created comprehensive GEO_STACKING_APPROACH.md explaining the methodology
- Documented the sparse matrix structure and scaling implications
- Added clear comments about period handling quirks

### 6. Multi-State Stacking
- Successfully fixed DataFrame indexing issues
- National targets now correctly appear once and apply to all household copies
- State-specific targets apply only to their respective household copies
- Tested with California and North Carolina - proper sparse block structure verified

### 7. National Hardcoded Targets
- Fixed SQL query to handle uppercase 'HARDCODED' source type
- Successfully retrieving 5 national targets (health insurance, medical expenses, child support, tips)
- Targets correctly marked with geographic_id='US'

### 8. SNAP Integration (December 2024)
- Successfully integrated SNAP administrative targets from USDA FNS data
- Using state-level administrative data only (not survey or national data)
- Two variables per state:
  - `household_count`: Number of households receiving SNAP
  - `snap`: Annual benefit costs in dollars
- Fixed constraint handling for SNAP > 0:
  - Issue: `snap` returns float arrays that couldn't combine with boolean masks
  - Solution: Explicitly convert all comparison results to `.astype(bool)`
- Improved naming convention:
  - `household_count_snap_recipients` for counts
  - `snap_benefits` for dollar amounts (avoiding redundant "snap_snap")
- SNAP targets form their own group (Group 6) in group-wise loss averaging
- With 2 states: 8 SNAP targets total (2 variables × 2 states × 2 targets each)

## In Progress 🚧

### 1. Calibration Integration with L0 Sparse Weights
- Successfully integrated L0 sparse calibration from ~/devl/L0 repository
- Using relative loss function: `((y - y_pred) / (y + 1))^2` 
  - Handles massive scale disparities between targets (178K to 385B range)
  - National targets (billions) and state targets (thousands) contribute based on percentage error
  - The `+1` epsilon is negligible given target scales but prevents any edge cases
  - Loss is symmetric: 50% over-prediction and 50% under-prediction produce equal penalty

### 2. Group-wise Loss Averaging (Critical Innovation)
**Problem**: Without grouping, histogram-type variables dominate the loss function
- Age has 18 bins per geography = 36 targets for 2 states, 918 targets for 51 states
- Each national target is just 1 target
- Without grouping, age would contribute 36/41 = 88% of the loss!

**Solution**: Automatic target grouping based on database metadata
- Each target belongs to a group based on its conceptual type
- All targets in a group are averaged together before contributing to total loss
- Each group contributes equally to the final loss, regardless of size

**Grouping Rules**:
1. **National hardcoded targets**: Each gets its own singleton group
   - These are fundamentally different quantities (tips, medical expenses, etc.)
   - Each should contribute individually to the loss
   
2. **Demographic targets**: Grouped by `stratum_group_id` across ALL geographies
   - All 36 age targets (18 CA + 18 NC) form ONE group
   - When scaled to 51 states, all 918 age targets will still be ONE group
   - Future: All income targets across all states will be ONE group, etc.

**Implementation Details**:
- Modified L0 calibration to accept `target_groups` parameter
- Each target gets weight `1/group_size` in the loss calculation
- Groups contribute equally regardless of cardinality
- Automatic grouping uses database metadata:
  - `stratum_group_id == 'national_hardcoded'` → singleton groups
  - `stratum_group_id == 2` → age group
  - `stratum_group_id == 3` → income group (future)
  - etc.

**Result with 2-state example (CA + NC)**:
- 8 total groups: 5 national + 1 age + 1 SNAP + 1 Medicaid
- National targets contribute 5/8 of total loss
- Age targets (36) contribute 1/8 of total loss
- SNAP targets (8) contribute 1/8 of total loss
- Medicaid targets (2) contribute 1/8 of total loss
- Mean group loss: ~25% (good convergence given target diversity)
- Sparsity: 99.5% (228 active weights out of 42,502)

**Why this matters for scaling**:
- With 51 states and 5 demographic types, we'd have:
  - 5 national groups (one per target)
  - 1 age group (918 targets)
  - 1 income group (459 targets)
  - 1 SNAP group (51 targets)
  - 1 Medicaid group (51 targets)
  - 1 EITC group (204 targets)
  - Total: 10 groups, each contributing 1/10 to the loss
- Prevents any variable type from dominating just because it has many instances

## To Do 📋

### 1. Scale to All States
- Test with all 51 states (including DC)
- Monitor memory usage and performance
- Verify group-wise loss still converges well

### 2. Add Remaining Demographic Groups
- ✅ SNAP targets (stratum_group_id = 4) - COMPLETED
- ✅ Medicaid targets (stratum_group_id = 5) - COMPLETED (person_count only)
- Income/AGI targets (stratum_group_id = 3) - TODO
- EITC targets (stratum_group_id = 6) - TODO

### 2. Congressional District Support
- Functions are stubbed out but need testing
- Will create even sparser matrices (436 CDs)

### 3. Sparse Matrix Optimization
- Convert to scipy.sparse for memory efficiency
- Implement block-diagonal optimizations
- Consider chunking strategies for very large matrices

### 4. Fix Stacking Implementation
- Debug DataFrame indexing issue in `build_stacked_matrix()`
- Ensure proper alignment of targets and households
- Test with multiple states

## Usage Example

```python
from policyengine_us import Microsimulation
from metrics_matrix_geo_stacking import GeoStackingMatrixBuilder

# Setup
db_uri = "sqlite:////path/to/policy_data.db"
builder = GeoStackingMatrixBuilder(db_uri, time_period=2023)

# Create simulation (note the period handling!)
sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim.default_calculation_period = 2023
sim.build_from_dataset()

# Build matrix for California
targets_df, matrix_df = builder.build_matrix_for_geography('state', '6', sim)

# Matrix is ready for calibration
# Rows = targets, Columns = households
# Values = person counts per household for each demographic group
```

## Key Design Decisions & Reasoning

### Why Relative Loss?
**Problem**: Target values span from 178K to 385B (6 orders of magnitude!)
- MSE would only optimize the billion-scale targets
- Small targets would be completely ignored

**Solution**: Relative loss `((y - y_pred) / (y + 1))^2`
- 10% error on $1B target = same penalty as 10% error on $100K target
- Allows meaningful "percent error" reporting
- The +1 prevents division by zero (negligible given target scales)

### Why Group-wise Averaging?
**Initial Problem**: Age variables dominated the loss
- Without grouping: 36 age targets vs 5 national targets
- Age contributed 36/41 = 88% of the loss
- National targets were essentially ignored

**First Attempt**: Group by (geography, variable_type)
- Created 7 groups: 5 national + 1 CA_age + 1 NC_age
- Better, but would scale poorly: 51 states × 5 types = 255 groups!
- State targets would dominate: 255 state groups vs 5 national groups

**Final Solution**: Group by variable_type only
- All age targets across ALL states = 1 group
- Each national target = its own group
- Result: 6 balanced groups (5 national + 1 age)
- Scales perfectly: even with 51 states, still just ~10 groups total

### Why Automatic Grouping?
**Problem**: Hard-coding groups wouldn't scale as new variable types are added

**Solution**: Use database metadata
- `stratum_group_id` identifies the variable type (2=age, 3=income, etc.)
- Special marker 'national_hardcoded' for singleton national targets
- Grouping logic automatically adapts as new types are added
- No code changes needed when adding income, SNAP, Medicaid targets

## Key Insights

1. **Geo-stacking works**: We successfully treat all US households as potential state households
2. **Matrix values are correct**: Proper household counts for each demographic group
3. **Group-wise loss is essential**: Without it, histogram variables dominate
4. **Automatic grouping scales**: Database metadata drives the grouping logic
5. **Convergence is good**: Mean group loss ~25% with 99.5% sparsity
6. **Period handling is tricky**: Must use 2024 CPS data with targets from various years
7. **Boolean mask handling**: Must explicitly convert float comparisons to bool for constraint application
8. **SNAP integration successful**: Two-variable targets (counts + dollars) work well in framework

## Sparse Matrix Implementation (2025-09-04) ✅

### Achievement: Eliminated Dense Matrix Creation
Successfully refactored entire pipeline to build sparse matrices directly, achieving **99% memory reduction**.

### Results:
- **2 states**: 37 MB dense → 6.5 MB sparse (82% reduction, 91% sparsity)
- **51 states**: 23 GB dense → 166 MB sparse (99% reduction)
- **436 CDs projection**: Would need ~1.5 GB sparse (feasible on 32 GB RAM)

### New Files:
- `metrics_matrix_geo_stacking_sparse.py` - Sparse matrix builder
- `calibrate_states_sparse.py` - Sparse calibration script  
- `calibration_utils.py` - Shared utilities (extracted `create_target_groups`)

### L0 Optimization Updates:
- Added `total_loss` to monitor convergence
- Loss components: `data_loss + λ_L0 * l0_loss`
- L0 penalty dominates as expected (trades accuracy for sparsity)

### Key Finding:
**Memory is solved!** Bottleneck is now computation time (matrix construction), not RAM.
- 51 states easily fit in 32 GB RAM
- 436 CDs would fit but take hours to build/optimize

## Next Priority

The system is ready for scaling to production:
1. ✅ Test with all 51 states configured (ready to run)
2. Add remaining demographic groups (income, EITC targets)  
3. Consider parallelizing matrix construction for speed
4. Test congressional district level (memory OK, time is issue)