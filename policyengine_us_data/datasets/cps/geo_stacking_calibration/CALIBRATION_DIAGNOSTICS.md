# Calibration Diagnostics: L0 Sparse Weight Analysis

## Executive Summary

Analysis of the L0 sparse calibration weights (97.8% sparsity) reveals severe underfitting for specific states, particularly Texas, which achieves only 24.5% of its population target. The root cause is insufficient active weights allocated to high-population states under extreme sparsity constraints.

## Key Findings

### Overall Performance
- **Mean relative error**: 6.27% across all 5,717 targets
- **National targets**: Excellent performance (<0.03% error)
- **State targets**: Highly variable (0% to 88% error)
- **Active weights**: 24,331 out of 1,083,801 (2.24% active)

### The Texas Problem

Texas exhibits the worst performance among all states:
- **Mean error**: 26.1% (highest of all states)
- **Max error**: 88.1% (age group 60-64)
- **Active weights**: Only 40 out of 21,251 available (0.2% activation rate)
- **Population coverage**: 7.5M out of 30.5M target (24.5% achievement)

This is paradoxical because Texas is the second-most represented state in the underlying CPS data (1,365 households, 6.4% of dataset).

### State Activation Patterns

Clear inverse correlation between activation rate and error:

| State | Active Weights | Activation Rate | Mean Error |
|-------|---------------|-----------------|------------|
| Texas | 40 | 0.2% | 26.1% |
| Alaska | 35 | 0.2% | 21.8% |
| Tennessee | 39 | 0.2% | 18.3% |
| S. Dakota | 39 | 0.2% | 14.4% |
| Washington | 43 | 0.2% | 13.6% |
| **vs** | | | |
| DC | 1,177 | 5.5% | 7.1% |
| Connecticut | 1,095 | 5.2% | 4.1% |
| Maryland | 1,062 | 5.0% | 3.6% |
| Utah | 962 | 4.5% | 3.3% |
| California | 247 | 1.2% | 4.2% |

### Weight Distribution Analysis

#### Expected vs Actual Weights

For proper survey representation, weights should approximate:
- **Texas**: ~1,435 per household (30.5M / 21,251 slots)
- **California**: ~1,834 per household (39M / 21,251 slots)
- **North Carolina**: ~510 per household (10.8M / 21,251 slots)

Given actual sparsity, required average weights would be:
- **Texas**: 762,583 (30.5M / 40 active weights)
- **California**: 157,754 (39M / 247 active weights)
- **North Carolina**: 24,682 (10.8M / 439 active weights)

Actual average weights achieved:
- **Texas**: 187,115 (25% of required)
- **California**: 58,835 (37% of required)
- **North Carolina**: 8,223 (33% of required)

### Population Target Achievement

| State | Target Pop | Sum of Weights | Achievement |
|-------|------------|----------------|-------------|
| Texas | 30,503,301 | 7,484,589 | 24.5% |
| California | 38,965,193 | 14,532,248 | 37.3% |
| North Carolina | 10,835,491 | 3,609,763 | 33.3% |
| Florida | 22,610,726 | 7,601,966 | 33.6% |
| New York | 19,571,216 | 7,328,156 | 37.4% |
| DC | 678,972 | 263,949 | 38.9% |

## Root Cause Analysis

### 1. Extreme Sparsity Constraint
The 97.8% sparsity constraint (L0 regularization) forces the model to select only 2.2% of available household weights. This creates a competition where the optimizer must choose "universal donor" households that work well across multiple states.

### 2. Texas Household Characteristics
Despite Texas being well-represented in the base data, Texas households appear to be poor universal donors. The optimizer finds it more efficient to:
- Use California/NY households for multiple states
- Sacrifice Texas accuracy to maintain better overall performance
- Accept massive undercounting rather than use unrealistic weight magnitudes

### 3. Weight Magnitude Constraints
With only 40 active weights for 30.5M people, each weight would need to average 763K - approximately 500x larger than typical survey weights. The model appears to prefer underrepresentation over such extreme weights.

## Recommendations

### Short-term Solutions
1. **Reduce sparsity constraint**: Target 95-96% sparsity instead of 97.8%
2. **State-specific minimum weights**: Enforce minimum 1% activation per state
3. **Population-proportional sparsity**: Allocate active weights proportional to state populations

### Long-term Solutions
1. **Hierarchical calibration**: Calibrate national targets first, then state targets
2. **State-specific models**: Separate calibration for problematic states
3. **Adaptive sparsity**: Allow sparsity to vary by state based on fit quality

## Technical Details

### Diagnostic Code Location
Full diagnostic analysis implemented in `calibrate_states_sparse.py`:
- Lines 456-562: Active weights analysis by state
- Lines 559-663: Weight distribution analysis
- Lines 193-369: Error analysis by various dimensions

### Key Metrics Tracked
- Per-target relative and absolute errors
- State-level activation rates
- Weight distribution quantiles
- Population target achievement ratios
- Error patterns by demographic groups

## Conclusion

The current L0 sparse calibration with 97.8% sparsity is too aggressive for proper multi-state representation. States requiring unique demographic patterns (like Texas) are severely underrepresented, leading to massive errors in age distribution targets. The solution requires either relaxing the sparsity constraint or implementing a more sophisticated hierarchical approach that ensures minimum representation for each state.