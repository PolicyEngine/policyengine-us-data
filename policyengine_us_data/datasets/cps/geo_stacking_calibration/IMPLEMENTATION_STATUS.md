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

### 3. Period Handling Discovery
- **Critical Finding**: The 2024 enhanced CPS dataset only contains 2024 data
- When requesting 2023 data explicitly via `calculate(period=2023)`, returns defaults (age=40, weight=0)
- **Solution**: Set `default_calculation_period=2023` BEFORE `build_from_dataset()`, then DON'T pass period to `calculate()`
- This triggers a fallback mechanism that uses the 2024 data for 2023 calculations

### 4. Weight Independence
- Successfully separated matrix creation from dataset weights
- Matrix values are pure counts (unweighted)
- Validation uses custom uniform weights, not dataset weights
- Ready for calibration/reweighting algorithms

### 5. Documentation
- Created comprehensive GEO_STACKING_APPROACH.md explaining the methodology
- Documented the sparse matrix structure and scaling implications
- Added clear comments about period handling quirks

## In Progress 🚧

### 1. Multi-State Stacking
- Basic structure implemented but has DataFrame indexing issues
- Need to fix the combined matrix assembly in `build_stacked_matrix()`
- The sparse block structure is conceptually correct

### 2. National Hardcoded Targets
- Query is in place but returns 0 targets currently
- Need to verify why hardcoded national targets aren't being found
- May need to adjust the query conditions

## To Do 📋

### 1. Add Other Demographic Groups
- Income/AGI targets (stratum_group_id = 3)
- SNAP targets (stratum_group_id = 4)
- Medicaid targets (stratum_group_id = 5)
- EITC targets (stratum_group_id = 6)

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

## Key Insights

1. **Geo-stacking works**: We successfully treat all US households as potential California households
2. **Matrix values are correct**: ~2,954 children age 0-4 across 21,251 households
3. **Scaling makes sense**: With uniform weights, estimates are ~2.5x California targets (US is larger)
4. **Ready for calibration**: The matrix structure supports finding optimal weights to match targets
5. **Period handling is tricky**: Must use the workaround documented above for 2024 data with 2023 targets

## Next Steps

1. Fix the multi-state stacking bug
2. Add national hardcoded targets
3. Test with congressional districts
4. Implement sparse matrix optimizations
5. Add other demographic groups beyond age