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

### Hierarchical Fallback for Target Selection

When building calibration matrices for a specific geographic level (e.g., congressional districts or states), the system implements a **hierarchical fallback** strategy to select the most appropriate target for each concept.

#### The Problem
With the introduction of filer strata (tax_unit_is_filer == 1) as an intermediate layer between geographic and IRS-specific strata, targets now exist at multiple levels of geographic specificity:
- National filer level → IRS-specific strata
- State filer level → IRS-specific strata  
- CD filer level → IRS-specific strata

For example, `qualified_business_income_deduction` might exist at the national level but not at state or CD levels. Without proper handling, this could lead to:
1. Missing targets (if only looking at the CD level)
2. Duplicate targets (if including all levels)
3. Incorrect calibration (using less specific targets when more specific ones exist)

#### The Solution: Hierarchical Fallback
For each target concept, the system follows this priority order:

**For Congressional District Calibration:**
1. Check if target exists at CD level → Use it
2. If not, check if target exists at State level → Use it
3. If not, use National level target

**For State Calibration:**
1. Check if target exists at State level → Use it
2. If not, use National level target

#### Important Distinctions
- Each **target concept** is evaluated independently
- A "concept" is defined by the combination of variable name and constraint pattern
- Different concepts can resolve at different levels

**Example:** For California CD 1 calibration:
- `SNAP person_count` → Found at CD level (use CD target)
- `SNAP cost` → Not at CD level, found at State level (use state target)
- `qualified_business_income_deduction` → Not at CD or State, found at National (use national target)

#### Implementation Considerations

**Query Strategy:**
Instead of querying only direct children of geographic strata, the system must:
1. Query the entire subtree rooted at each geographic level
2. Traverse through filer strata to reach IRS-specific strata
3. Deduplicate targets based on concept and geographic specificity

**For IRS Targets specifically:**
- Geographic stratum (e.g., CD 601)
  - → Filer stratum (CD 601 filers, tax_unit_is_filer == 1)
    - → IRS variable stratum (CD 601 filers with salt > 0)

The system needs to traverse this full hierarchy, checking at each geographic level (CD → State → National) before falling back.

**Constraint Inheritance:**
When a target is selected from a higher geographic level (e.g., using a national target for CD calibration), the constraints from that target's stratum still apply, ensuring the target is calculated correctly for the subset of households it represents.

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

### Gate-Induced Sparsity (Important Finding)

The L0 regularization framework induces sparsity through **stochastic gates** even when `lambda_l0=0`:

**Gate Mechanism**:
- Gates control which weights are active: `weight = exp(log_weight) * gate`
- Gate formula: `gate = sigmoid(log_alpha/beta) * (zeta - gamma) + gamma`
- With default parameters: `gamma = -0.1`, `zeta = 1.1`, `beta = 2/3`

**Implicit Sparsity Creation**:
- The gate formula becomes: `gate = s * 1.2 - 0.1` where `s = sigmoid(log_alpha/beta)`
- When `sigmoid(log_alpha/beta) < 0.0833`, the gate becomes negative
- Negative gates are clamped to 0, creating **exact zeros** in weights
- This happens even with `lambda_l0=0` (no explicit sparsity penalty)

**Practical Implications**:
- Sparsity emerges naturally during optimization as the model learns
- The `gamma` parameter creates a "hard concrete" distribution with mass at exactly 0
- To prevent any sparsity, would need `gamma=0` or a very small negative value
- The L0 penalty (`lambda_l0 > 0`) encourages more weights to hit this zero threshold
- Default parameters typically achieve 5-40% sparsity even without L0 penalty

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

**Simplified Example Result with 2-state example (CA + NC)**:
- 8 total groups: 5 national + 1 age + 1 SNAP + 1 Medicaid
- National targets contribute 5/8 of total loss
- Age targets (36) contribute 1/8 of total loss
- Mean group loss: ~25% (good convergence given target diversity)
- Sparsity: 99.5% (228 active weights out of 42,502)

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

### Finding Targets in the Database

#### 1. National Targets (5 total)
These are pulled directly from the database (not hardcoded in Python):
```sql
-- National targets from the database
SELECT t.variable, t.value, t.period, s.notes
FROM targets t
JOIN strata s ON t.stratum_id = s.stratum_id
WHERE t.variable IN ('child_support_expense', 
                     'health_insurance_premiums_without_medicare_part_b',
                     'medicare_part_b_premiums', 
                     'other_medical_expenses', 
                     'tip_income')
  AND s.notes = 'United States';
```

#### 2. Age Targets (18 bins per CD)
```sql
-- Find age targets for a specific CD (e.g., California CD 1)
SELECT t.variable, t.value, sc.constraint_variable, sc.value as constraint_value
FROM targets t
JOIN strata s ON t.stratum_id = s.stratum_id
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE s.stratum_group_id = 2  -- Age group
  AND s.parent_stratum_id IN (
    SELECT stratum_id FROM strata WHERE stratum_group_id = 1
    AND stratum_id IN (
      SELECT stratum_id FROM stratum_constraints
      WHERE constraint_variable = 'congressional_district_geoid'
      AND value = '601'  -- California CD 1
    )
  )
  AND t.period = 2023;
```

#### 3. AGI Distribution Targets (9 bins per CD)
**Important:** These appear as `person_count` with AGI ranges in the description. They're in stratum_group_id=3 but only exist for period=2022 in the database:

```python
# After loading targets_df
agi_targets = targets_df[
    (targets_df['description'].str.contains('adjusted_gross_income', na=False)) &
    (targets_df['variable'] == 'person_count')
]
# Example descriptions:
# - person_count_adjusted_gross_income<1_adjusted_gross_income>=-inf
# - person_count_adjusted_gross_income<10000_adjusted_gross_income>=1
# - person_count_adjusted_gross_income<inf_adjusted_gross_income>=500000
```

Note: AGI distribution targets exist in the database but only for states (not CDs) and only for period=2022. The CD-level AGI targets are likely being generated programmatically.

#### 4. SNAP Targets (Hierarchical)
- **CD-level**: `household_count` for SNAP>0 households (survey data)
- **State-level**: `snap` cost in dollars (administrative data)

```sql
-- CD-level SNAP household count (survey) for California CD 1
SELECT t.variable, t.value, sc.constraint_variable, sc.value as constraint_value
FROM targets t
JOIN strata s ON t.stratum_id = s.stratum_id
LEFT JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE s.stratum_group_id = 4  -- SNAP
  AND t.variable = 'household_count'
  AND s.parent_stratum_id IN (
    SELECT stratum_id FROM strata WHERE stratum_group_id = 1
    AND stratum_id IN (
      SELECT stratum_id FROM stratum_constraints
      WHERE constraint_variable = 'congressional_district_geoid'
      AND value = '601'
    )
  )
  AND t.period = 2023;

-- State SNAP cost for California (administrative)
SELECT t.variable, t.value
FROM targets t
JOIN strata s ON t.stratum_id = s.stratum_id
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE s.stratum_group_id = 4  -- SNAP
  AND t.variable = 'snap'  -- Cost variable
  AND sc.constraint_variable = 'state_fips'
  AND sc.value = '6'  -- California
  AND t.period = 2023;
```

The state SNAP costs cascade to all CDs within that state in the calibration matrix.

#### 5. IRS SOI Targets (50 per CD)
These include various tax-related variables stored with stratum_group_id=115 and period=2022:

```sql
-- Example: Income tax for California CD 601
SELECT t.variable, t.value, t.period, s.notes
FROM targets t
JOIN strata s ON t.stratum_id = s.stratum_id
WHERE t.variable = 'income_tax'
  AND s.notes = 'CD 601 with income_tax > 0'
  AND t.period = 2022;
-- Returns: income_tax = $2,802,681,423
```

```python
# In Python targets_df, find income_tax for CD 601
income_tax = targets_df[
    (targets_df['variable'] == 'income_tax') &
    (targets_df['geographic_id'] == '601')
]
# Shows: income_tax with stratum_group_id='irs_scalar_income_tax'

# Common IRS variables (many have both tax_unit_count and amount versions)
irs_variables = [
    'income_tax',
    'qualified_business_income_deduction', 
    'salt_refundable_credits',
    'net_capital_gain',
    'taxable_ira_distributions',
    'taxable_interest_income',
    'tax_exempt_interest_income',
    'dividend_income',
    'qualified_dividend_income',
    'partnership_s_corp_income',
    'taxable_social_security',
    'unemployment_compensation',
    'real_estate_taxes',
    'eitc_qualifying_children_0',  # through _3
    'adjusted_gross_income'  # scalar total
]
```

### IRS Target Deduplication (Critical Implementation Detail)

**Problem Discovered (2024-12)**: The AGI histogram bins have overlapping boundary constraints that were being incorrectly deduplicated:
- Each AGI bin has TWO constraints: `adjusted_gross_income >= lower` AND `adjusted_gross_income < upper`
- The `get_all_descendant_targets` query returns only the FIRST non-geographic constraint for backward compatibility
- The deduplication logic was creating concept IDs without the operation, causing collisions

**Example of the Issue**:
- Bin 3: `adjusted_gross_income >= 10000` AND `adjusted_gross_income < 25000`
- Bin 4: `adjusted_gross_income >= 25000` AND `adjusted_gross_income < 50000`
- Both would return first constraint with value 10000/25000
- Without operation in concept ID: both become `person_count_agi_25000` → collision!

**Solution**: Include the operation in concept IDs:
- `person_count_agi_lt_25000` (for bin 3's upper bound)
- `person_count_agi_gte_25000` (for bin 4's lower bound)
- Now properly distinguished → all 58 targets per CD preserved

This fix recovered 872 missing targets (2 per CD × 436 CDs) and brought the matrix to its correct dimensions.

### Debugging Target Counts

If your target count doesn't match expectations:

```python
# Load the calibration results
import pickle
with open('/path/to/cd_targets_df.pkl', 'rb') as f:
    targets_df = pickle.load(f)

# Check breakdown by geographic level
print("National:", len(targets_df[targets_df['geographic_level'] == 'national']))
print("State:", len(targets_df[targets_df['geographic_level'] == 'state']))
print("CD:", len(targets_df[targets_df['geographic_level'] == 'congressional_district']))

# Check by stratum_group_id
for group_id in targets_df['stratum_group_id'].unique():
    count = len(targets_df[targets_df['stratum_group_id'] == group_id])
    print(f"Group {group_id}: {count} targets")

```

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

## Sparse Dataset Creation - Implementation Details

### Critical Dataset Requirements
- **Congressional Districts**: Must use `stratified_extended_cps_2023.h5` (13,089 households)
- **States**: Must use standard `extended_cps_2023.h5` (112,502 households)  
- **IMPORTANT**: The dataset used for stacking MUST match what was used during calibration

### The DataFrame Approach (Essential for Entity Relationships)
The DataFrame approach preserves all entity relationships automatically:

```python
# Pattern that works:
sim = Microsimulation(dataset=dataset_path)
sim.set_input("household_weight", period, calibrated_weights)
df = sim.to_input_dataframe()  # This preserves ALL relationships
# ... filter and process df ...
sparse_dataset = Dataset.from_dataframe(combined_df, period)
```

Direct array manipulation will break household-person-tax unit relationships.

### ID Overflow Prevention Strategy
With large geo-stacked datasets (e.g., 436 CDs × 13,089 households):
- Person IDs can overflow int32 when multiplied by 100 (PolicyEngine internal)
- Solution: Complete reindexing of ALL entity IDs after combining DataFrames
- Start from 0 and assign sequential IDs to prevent overflow

### EnumArray Handling for h5 Serialization
When saving to h5, handle PolicyEngine's EnumArray objects:
```python
if hasattr(values, 'decode_to_str'):
    values = values.decode_to_str().astype("S")
else:
    # Already numpy array
    values = values.astype("S")
```

### Geographic Code Formats
- State FIPS: String format ('1', '2', ..., '56')
- Congressional District GEOIDs: String format ('601', '3601', '4801')
  - First 1-2 digits = state FIPS
  - Last 2 digits = district number

### File Organization
- `create_sparse_state_stacked.py` - Self-contained state stacking (function + runner)
- `create_sparse_cd_stacked.py` - Self-contained CD stacking (function + runner)
- Both follow identical patterns for consistency

### Common Pitfalls to Avoid
1. Using the wrong dataset (extended vs stratified)
2. Not reindexing IDs after combining geographic units
3. Trying to modify arrays directly instead of using DataFrames
4. Not checking for integer overflow with large datasets
5. Forgetting that the same household appears in multiple geographic units
6. Progress indicators - use appropriate intervals (every 10 CDs, not 50)

### Testing Strategy
Always test with subsets first:
- Single geographic unit
- Small diverse set (10 units)
- Regional subset (e.g., all California CDs)
- Full dataset only after smaller tests pass

## Tax Unit Count Aggregation (Investigation 2024-12-25)

### Initial Concern

There was initial concern that `tax_unit_count` variables were being double-counted when aggregated from tax unit to household level, potentially causing over-prediction.

### Investigation Results

After thorough testing, it was determined that the original implementation was correct:

1. **29% of households have multiple tax units** - this is real structure in the CPS data
2. **Tax unit weights = household weights** - when a household has 2 tax units, both inherit the household weight
3. **Summing is the correct operation** - when we sum tax unit counts to household level and multiply by household weights, we get the correct total

Testing showed:
- Original method (summing): 0.0% error
- Alternative method (scaled binary): 0.4% error

The original approach of summing tax unit counts to household level produces virtually perfect results.

## Dashboard Integration and Target Accounting

### Understanding "Excluded Targets" in the Calibration Dashboard

The calibration dashboard (https://microcalibrate.vercel.app) may show fewer targets than expected due to its "excluded targets" logic.

#### What Are Excluded Targets?
The dashboard identifies targets as "excluded" when their estimates remain constant across all training epochs. Specifically:
- Targets with multiple epoch data points where all estimates are within 1e-6 tolerance
- Most commonly: targets that remain at 0.0 throughout training
- These targets are effectively not participating in the calibration

#### Example: Congressional District Calibration
- **Total targets in matrix**: 30,576
- **Targets shown in dashboard**: 24,036  
- **"Excluded" targets**: 6,540

This discrepancy occurs when ~6,540 targets have zero estimates throughout training, indicating they're not being actively calibrated. Common reasons:
- Very sparse targets with no qualifying households in the sample
- Targets for rare demographic combinations
- Early training epochs where the model hasn't activated weights for certain targets

#### Target Group Accounting

The 30,576 CD calibration targets break down into 28 groups:

**National Targets (5 singleton groups)**:
- Group 0-4: Individual national targets (tip_income, medical expenses, etc.)

**Demographic Targets (23 groups)**:
- Group 5: Age (7,848 targets - 18 bins × 436 CDs)
- Group 6: AGI Distribution (3,924 targets - 9 bins × 436 CDs)  
- Group 7: SNAP household counts (436 targets - 1 × 436 CDs)
- Group 8: Medicaid (436 targets - 1 × 436 CDs)
- Group 9: EITC (3,488 targets - 4 categories × 436 CDs, some CDs missing categories)
- Groups 10-25: IRS SOI variables (16 groups × 872 targets each)
- Group 26: AGI Total Amount (436 targets - 1 × 436 CDs)
- Group 27: State SNAP Cost Administrative (51 targets - state-level constraints)

**Important**: The state SNAP costs (Group 27) have `stratum_group_id = 'state_snap_cost'` rather than `4`, keeping them separate from CD-level SNAP household counts. This is intentional as they represent different constraint types (counts vs. dollars).

#### Verifying Target Counts

To debug target accounting issues:

```python
# Check what's actually in the targets dataframe
import pandas as pd
targets_df = pd.read_pickle('cd_targets_df.pkl')

# Total should be 30,576
print(f"Total targets: {len(targets_df)}")

# Check for state SNAP costs specifically
state_snap = targets_df[targets_df['stratum_group_id'] == 'state_snap_cost']
print(f"State SNAP cost targets: {len(state_snap)}")  # Should be 51

# Check for CD SNAP household counts
cd_snap = targets_df[targets_df['stratum_group_id'] == 4]
print(f"CD SNAP household targets: {len(cd_snap)}")  # Should be 436

# Total SNAP-related targets
print(f"Total SNAP targets: {len(state_snap) + len(cd_snap)}")  # Should be 487
```
