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
- Memory feasible but computation time is the bottleneck

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

### L0 Package (~/devl/L0)
- `l0/calibration.py` - Core calibration class
- `tests/test_calibration.py` - Test coverage

### Documentation
- `GEO_STACKING_TECHNICAL.md` - Technical documentation and architecture
- `PROJECT_STATUS.md` - This file (active project management)