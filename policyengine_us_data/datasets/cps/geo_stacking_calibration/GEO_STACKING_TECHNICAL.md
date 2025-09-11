# Geo-Stacking Calibration: Technical Documentation

## Overview

The geo-stacking approach treats the same household dataset as existing in multiple geographic areas simultaneously. This creates an "empirical superpopulation" where each household can represent itself in different locations with different weights.

## Conceptual Framework

### Matrix Structure

**Dimensions:**
- **Rows = Targets** (the "observations" in our regression problem)
- **Columns = Households** (the "variables" whose weights we're estimating)

This creates a "small n, large p" problem where:
- n = number of targets (rows)
- p = number of households × number of geographic areas (columns)

**Key Insight:** In traditional regression, we estimate parameters (coefficients) for variables using observations. Here:
- Household weights are the parameters we estimate
- Calibration targets are the observations
- Each household's characteristics are the "variables"

### Why Stack?

When calibrating to multiple geographic areas, we need to:
1. Respect national-level targets that apply to all households
2. Respect state-specific (or CD-specific) targets that only apply to households in that geography
3. Allow the same household to have different weights when representing different geographies

### Sparsity Pattern

Consider two states (California and Texas) with households H1, H2, H3:

```
                     H1_CA  H2_CA  H3_CA  H1_TX  H2_TX  H3_TX
national_employment    X      X      X      X      X      X
national_tax_revenue   X      X      X      X      X      X
CA_age_0_5            X      X      X      0      0      0
CA_age_5_10           X      X      X      0      0      0
CA_age_10_15          X      X      X      0      0      0
TX_age_0_5            0      0      0      X      X      X
TX_age_5_10           0      0      0      X      X      X
TX_age_10_15          0      0      0      X      X      X
```

Where:
- X = non-zero value (household contributes to this target)
- 0 = zero value (household doesn't contribute to this target)

## Implementation Architecture

### Core Infrastructure

Built `GeoStackingMatrixBuilder` class with extensible design:
- Database queries for national and demographic targets
- Proper constraint application at entity levels
- Correctly maps person-level constraints to household level
- Weight independence: matrix values are pure counts (unweighted)

### Target Types and Database Structure

The database uses stratum_group_id to categorize target types:
- 1 = Geographic boundaries
- 2 = Age-based strata (18 age bins)
- 3 = Income/AGI-based strata (9 brackets)
- 4 = SNAP recipient strata
- 5 = Medicaid enrollment strata
- 6 = EITC recipient strata (4 categories by qualifying children)

### Geographic Hierarchy

The approach respects the geographic hierarchy:
1. **National targets**: Apply to all household copies
2. **State targets**: Apply only to households in that state's copy
3. **Congressional District targets**: Apply only to households in that CD's copy

When more precise geographic data is available, it overrides less precise data.

## Sparse Matrix Implementation

### Achievement: 99% Memory Reduction

Successfully refactored entire pipeline to build sparse matrices directly:
- **2 states**: 37 MB dense → 6.5 MB sparse (82% reduction, 91% sparsity)
- **51 states**: 23 GB dense → 166 MB sparse (99% reduction)
- **436 CDs projection**: Would need ~1.5 GB sparse (feasible on 32 GB RAM)

**Key Finding:** Memory is solved! Bottleneck is now computation time (matrix construction), not RAM.

### Files
- `metrics_matrix_geo_stacking_sparse.py` - Sparse matrix builder
- `calibrate_states_sparse.py` - Sparse calibration script  
- `calibration_utils.py` - Shared utilities (extracted `create_target_groups`)

## L0 Calibration Integration

### Relative Loss Function

Using relative loss function: `((y - y_pred) / (y + 1))^2` 
- Handles massive scale disparities between targets (178K to 385B range)
- National targets (billions) and state targets (thousands) contribute based on percentage error
- The `+1` epsilon is negligible given target scales but prevents edge cases
- Loss is symmetric: 50% over-prediction and 50% under-prediction produce equal penalty

### Group-wise Loss Averaging (Critical Innovation)

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
2. **Demographic targets**: Grouped by `stratum_group_id` across ALL geographies

**Result with 2-state example (CA + NC)**:
- 8 total groups: 5 national + 1 age + 1 SNAP + 1 Medicaid
- National targets contribute 5/8 of total loss
- Age targets (36) contribute 1/8 of total loss
- Mean group loss: ~25% (good convergence given target diversity)
- Sparsity: 99.5% (228 active weights out of 42,502)

### L0 API Improvements

Successfully refactored `SparseCalibrationWeights` class for cleaner API:

**Key Changes**:
1. Replaced `init_weight_scale` with `init_weights` - accept actual weight values
2. Per-feature gate initialization via arrays in `init_keep_prob`
3. Clarified jitter parameters for symmetry breaking

**Clean API Example**:
```python
# Calculate per-household keep probabilities based on state
keep_probs = np.zeros(n_households)
keep_probs[ca_households] = 0.15  # CA more likely to stay
keep_probs[nc_households] = 0.05  # NC more likely to drop

model = SparseCalibrationWeights(
    n_features=n_households,
    init_weights=10.0,           # Natural survey weight
    init_keep_prob=keep_probs,   # Per-household probabilities
    weight_jitter_sd=0.5,        # Symmetry breaking
)
```

## Weight Initialization and Mapping

### Population-Based Weight Initialization

Fixed critical initialization issue with population-proportional weights:
- Base weight = state_population / n_households_per_state
- Sparsity adjustment = 1/sqrt(keep_probability) to compensate for dropout
- Final weight clipped to [100, 100,000] range for stability

Example initial weights:
- **Texas** (pop 30.5M): ~20,000 per household
- **California** (pop 39M): ~6,400 per household  
- **North Carolina** (pop 10.8M): ~2,500 per household
- **DC** (pop 679K): ~500 per household

### Weight-to-Reality Mapping

Verified lossless weight mapping with completely predictable structure:

**Weight Vector Structure**:
- Length: `n_states × n_households = 51 × 112,502 = 5,737,602`
- Ordering: Sequential by state FIPS codes, same household order within each state
- Mapping: For weight at index `i`:
  - State: `states_to_calibrate[i // 112502]`
  - Household: `household_ids[i % 112502]`

**Microsimulation as Ground Truth**:
```python
sim = Microsimulation(dataset="hf://policyengine/test/extended_cps_2023.h5")
sim.build_from_dataset()
household_ids = sim.calculate("household_id", map_to="household").values
# household_ids[586] is ALWAYS household 1595 across ALL states
```

### Universal Donor Households

L0 sparse calibration creates "universal donor" households that contribute to multiple states:
- **64,522 unique households** have non-zero weights
- These households appear in **167,089 household-state pairs**
- Average: 2.59 states per active household
- Distribution:
  - 31,038 households in only 1 state
  - 15,047 households in 2 states
  - 2,095 households in 10+ states
  - Maximum: One household active in 50 states!

## Stratified CPS Sampling for Congressional Districts

### The Memory Challenge

Congressional district calibration with full CPS data creates intractable memory requirements:
- 436 CDs × 112,502 households = 49M matrix columns
- Even sparse matrices exceed 32GB RAM and 15GB GPU limits
- Random sampling would lose critical high-income households essential for tax policy simulation

### Stratified Sampling Solution

Created `create_stratified_cps.py` implementing income-based stratified sampling that:

1. **Preserves ALL high-income households** (top 1% by AGI)
2. **Progressively samples lower income strata** with decreasing rates
3. **Maintains income distribution integrity** while reducing size by ~75%

#### Sampling Strategy

| Income Percentile | Sampling Rate | Rationale |
|------------------|---------------|-----------|
| 99.9-100% | 100% | Ultra-high earners critical for tax revenue |
| 99-99.9% | 100% | High earners essential for policy analysis |
| 95-99% | 80% | Upper middle class well-represented |
| 90-95% | 60% | Professional class adequately sampled |
| 75-90% | 40% | Middle class proportionally represented |
| 50-75% | 25% | Lower middle class sampled |
| 25-50% | 15% | Working class represented |
| 0-25% | 10% | Lower income maintained for completeness |

#### Results

- **10k target**: Yields 13k households (preserving all high earners)
- **30k target**: Yields 29k households (balanced across strata)
- **Maximum AGI preserved**: $2,276,370 (identical to original)
- **Memory reduction**: 88% (5.7M vs 49M matrix columns for CDs)

## Sparse State-Stacked Dataset Creation

### Conceptual Model

Each household-state pair with non-zero weight becomes a **separate household** in the final dataset:

```
Original: Household 6 with weights in multiple states
- Hawaii: weight = 32.57
- South Dakota: weight = 0.79

Sparse Dataset: Two separate households
- Household_A: state_fips=15 (HI), weight=32.57, all characteristics of HH 6
- Household_B: state_fips=46 (SD), weight=0.79, all characteristics of HH 6
```

### Implementation (`create_sparse_state_stacked.py`)

1. **State Processing**: For each state, extract ALL households with non-zero weight
2. **DataFrame Creation**: Use `sim.to_input_dataframe()` to preserve entity relationships
3. **State Assignment**: Set `state_fips` to the target state for all entities
4. **Concatenation**: Combine all state DataFrames (creates duplicate IDs)
5. **Reindexing**: Sequential reindexing to handle duplicates and prevent overflow:
   - Each household occurrence gets unique ID
   - Person/tax/SPM/marital units properly linked to new household IDs
   - Max person ID kept below 500K (prevents int32 overflow)

### Results

- **Input**: 5,737,602 weights (51 states × 112,502 households)
- **Active weights**: 167,089 non-zero weights
- **Output dataset**:
  - 167,089 households (one per non-zero weight)
  - 495,170 persons
  - Total population: 136M
  - No ID overflow issues
  - No duplicate persons
  - Correct state assignments

## Period Handling

**Critical Finding**: The 2024 enhanced CPS dataset only contains 2024 data
- Attempting to set `default_calculation_period=2023` doesn't actually work - it remains 2024
- When requesting past data explicitly via `calculate(period=2023)`, returns defaults (zeros)
- **Final Decision**: Use 2024 data and pull targets from whatever year they exist in the database
- **Temporal Mismatch**: Targets exist for different years (2022 for admin data, 2023 for age, 2024 for hardcoded)
- This mismatch is acceptable for the calibration prototype and will be addressed in production

## Usage Example

```python
from policyengine_us import Microsimulation
from metrics_matrix_geo_stacking import GeoStackingMatrixBuilder

# Setup
db_uri = "sqlite:////path/to/policy_data.db"
builder = GeoStackingMatrixBuilder(db_uri, time_period=2023)

# Create simulation
sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim.default_calculation_period = 2023
sim.build_from_dataset()

# Build matrix for California
targets_df, matrix_df = builder.build_matrix_for_geography('state', '6', sim)

# Matrix is ready for calibration
# Rows = targets, Columns = households
# Values = person counts per household for each demographic group
```

## Key Design Decisions

### Why Relative Loss?
Target values span from 178K to 385B (6 orders of magnitude!). MSE would only optimize the billion-scale targets. Relative loss ensures 10% error on $1B target = same penalty as 10% error on $100K target.

### Why Group-wise Averaging?
Prevents any variable type from dominating just because it has many instances. All age targets across ALL states = 1 group. Each national target = its own group. Scales perfectly: even with 51 states, still just ~10 groups total.

### Why Automatic Grouping?
Uses database metadata (`stratum_group_id`) to automatically adapt as new types are added. No code changes needed when adding income, SNAP, Medicaid targets.

## Technical Notes

### Scaling Considerations

For full US implementation:
- 51 states (including DC) × ~100,000 households = 5.1M columns
- 436 congressional districts × ~100,000 households = 43.6M columns

**With stratified sampling:**
- 51 states × 30,000 households = 1.5M columns (manageable)
- 436 CDs × 13,000 households = 5.7M columns (feasible on 32GB RAM)

With targets:
- National: ~10-20 targets
- Per state: 18 age bins + future demographic targets
- Per CD: 18 age bins + future demographic targets

This creates extremely sparse matrices requiring specialized solvers.

### Constraint Handling
Constraints are applied hierarchically:
1. Geographic constraints determine which targets apply
2. Demographic constraints (age, income, etc.) determine which individuals/households contribute
3. Masks are created at appropriate entity levels and mapped to household level

### Files and Diagnostics
- `weight_diagnostics.py` - Standalone weight analysis using Microsimulation ground truth
- `calibrate_states_sparse.py` - Main calibration script with extensive diagnostics
- `calibration_utils.py` - Shared utilities for target grouping

## Advantages

1. **Diversity**: Access to full household diversity even in small geographic areas
2. **Consistency**: Same households across geographies ensures coherent microsimulation
3. **Flexibility**: Can add new geographic levels or demographic targets easily
4. **Reweighting**: Each geography gets appropriate weights for its households
5. **Memory Efficient**: Sparse implementation makes national-scale calibration feasible
6. **Balanced Optimization**: Group-wise loss ensures all target types contribute fairly