# Geo-Stacking Matrix Audit and Validation

## Overview

This document describes the audit process and validation methodology for the geo-stacking calibration matrix used in the PolicyEngine US data pipeline. The matrix is a critical component that enables calibration of household weights to match IRS Statistics of Income (SOI) targets across all US Congressional Districts.

## Matrix Structure

### Dimensions (Full Matrix)
- **Rows**: 34,089 targets (demographic and economic variables for each geography)
- **Columns**: 4,602,300 (10,580 households × 435 Congressional Districts)
- **Type**: Sparse CSR matrix (most values are zero)

### Column Organization (Geo-Stacking)
Each household appears in EVERY Congressional District's column block:
```
Columns 0-10,579:        CD '1001' (Delaware at-large) - All households
Columns 10,580-21,159:   CD '101'  (Alabama 1st) - All households
Columns 21,160-31,739:   CD '102'  (Alabama 2nd) - All households
...
Columns 4,591,720-4,602,299: CD '5600' (Wyoming at-large) - All households
```

### Row Organization
Targets are interleaved by geography:
- Each CD has its own row for each target variable
- National targets appear once
- Pattern: CD1_target1, CD2_target1, ..., CD435_target1, CD1_target2, ...

### Key Insight: No Geographic Assignment
- `congressional_district_geoid` is NOT set in the simulation
- Every household potentially contributes to EVERY CD
- Geographic constraints are handled through matrix structure, not data filtering
- Calibration weights later determine actual geographic assignment

## Household Tracer Utility

The `household_tracer.py` utility was created to navigate this complex structure.

### Setup Code (Working Example)

```python
from policyengine_us import Microsimulation
from metrics_matrix_geo_stacking_sparse import SparseGeoStackingMatrixBuilder
from household_tracer import HouseholdTracer
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

# Initialize
db_uri = "sqlite:////home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)
sim = Microsimulation(dataset="/home/baogorek/devl/stratified_10k.h5")

# For testing, use a subset of CDs (full matrix takes ~15 minutes to build)
test_cd_geoids = ['101', '601', '3910', '1001']  # Alabama 1st, CA 1st, Ohio 10th, Delaware

print(f"Building matrix for {len(test_cd_geoids)} CDs (demo mode)...")
targets_df, matrix, household_mapping = builder.build_stacked_matrix_sparse(
    'congressional_district', test_cd_geoids, sim
)

# Create tracer
tracer = HouseholdTracer(targets_df, matrix, household_mapping, test_cd_geoids, sim)
print(f"Matrix shape: {matrix.shape}")
```

Note: For full analysis, replace `test_cd_geoids` with all 436 CDs retrieved from the database.

### Essential Methods

```python
# Find where a household appears
household_id = 565
positions = tracer.get_household_column_positions(household_id)
print(f"Household {household_id} appears at columns: {positions}")

# Look up any cell
row_idx, col_idx = 10, 500
cell_info = tracer.lookup_matrix_cell(row_idx, col_idx)
print(f"Cell [{row_idx}, {col_idx}]: value = {cell_info['matrix_value']}")
print(f"  Variable: {cell_info['target']['variable']}")
print(f"  Household: {cell_info['household']['household_id']}")

# View matrix structure
tracer.print_matrix_structure()

# Get targets by group
from calibration_utils import create_target_groups
tracer.target_groups, _ = create_target_groups(tracer.targets_df)
group_31 = tracer.get_group_rows(31)  # Person count targets
print(f"Group 31 has {len(group_31)} targets")
```

## Validation Tests

### Test 1: Single Person Household (AGI Bracket Validation)

```python
# Test household 565: 1 person, AGI = $87,768
test_household = 565
positions = tracer.get_household_column_positions(test_household)

# Get household info
df = sim.calculate_dataframe(['household_id', 'person_count', 'adjusted_gross_income'],
                             map_to="household")
hh_data = df[df['household_id'] == test_household]
print(f"Household {test_household}:")
print(f"  People: {hh_data['person_count'].values[0]}")
print(f"  AGI: ${hh_data['adjusted_gross_income'].values[0]:,.0f}")

# Find AGI 75k-100k bracket targets
from calibration_utils import create_target_groups
target_groups, _ = create_target_groups(targets_df)
group_mask = target_groups == 31  # Person count group
group_31_full = targets_df[group_mask].copy()
group_31_full['row_index'] = np.where(group_mask)[0]

agi_targets = group_31_full[
    group_31_full['variable_desc'].str.contains('adjusted_gross_income<100000') &
    group_31_full['variable_desc'].str.contains('>=75000')
]

# Check value for CD 101
cd_101_target = agi_targets[agi_targets['geographic_id'] == '101']
if not cd_101_target.empty:
    row_idx = cd_101_target['row_index'].values[0]
    col_idx = positions['101']
    value = matrix[row_idx, col_idx]
    print(f"\nCD 101 AGI 75k-100k bracket:")
    print(f"  Row {row_idx}, Column {col_idx}")
    print(f"  Matrix value: {value} (should be 1.0 for 1 person)")
```

### Test 2: Multi-Person Household Size Validation

```python
# Test households of different sizes
df = sim.calculate_dataframe(['household_id', 'person_count', 'adjusted_gross_income'],
                             map_to="household")
agi_bracket_hh = df[(df['adjusted_gross_income'] >= 75000) &
                    (df['adjusted_gross_income'] < 100000)]

print("Testing household sizes in 75k-100k AGI bracket:")
for size in [1, 2, 3, 4]:
    size_hh = agi_bracket_hh[agi_bracket_hh['person_count'] == size]
    if len(size_hh) > 0:
        hh = size_hh.iloc[0]
        hh_id = hh['household_id']
        positions = tracer.get_household_column_positions(hh_id)

        # Find the AGI bracket row for CD 101
        if not cd_101_target.empty:
            row_idx = cd_101_target['row_index'].values[0]
            col_idx = positions['101']
            value = matrix[row_idx, col_idx]
            print(f"  HH {hh_id}: {size} people, matrix value = {value}")
```

### Test 3: Tax Unit Level Constraints

```python
# Investigate households where person_count might not match matrix value
# This occurs when households have multiple tax units with different AGIs

# Create person-level dataframe
person_df = pd.DataFrame({
    'household_id': sim.calculate('household_id', map_to="person").values,
    'person_id': sim.calculate('person_id').values,
    'tax_unit_id': sim.calculate('tax_unit_id', map_to="person").values,
    'age': sim.calculate('age', map_to="person").values,
    'is_tax_unit_dependent': sim.calculate('is_tax_unit_dependent', map_to="person").values
})

# Example: Check household 8259 (if it exists in the dataset)
test_hh = 8259
if test_hh in df['household_id'].values:
    hh_persons = person_df[person_df['household_id'] == test_hh]
    print(f"\nHousehold {test_hh} structure:")
    print(f"  Total people: {len(hh_persons)}")
    print(f"  Tax units: {hh_persons['tax_unit_id'].nunique()}")

    # Check AGI for each tax unit
    for tu_id in hh_persons['tax_unit_id'].unique():
        tu_members = hh_persons[hh_persons['tax_unit_id'] == tu_id]
        tu_agi = sim.calculate('adjusted_gross_income', map_to="tax_unit")
        tu_mask = sim.calculate('tax_unit_id', map_to="tax_unit") == tu_id
        if tu_mask.any():
            agi_value = tu_agi[tu_mask].values[0]
            print(f"  Tax unit {tu_id}: {len(tu_members)} members, AGI = ${agi_value:,.0f}")
```

## Key Findings

### 1. Matrix Construction is Correct
- Values accurately reflect household/tax unit characteristics
- Constraints properly applied at appropriate entity levels
- Sparse structure efficiently handles 4.6M columns
- All test cases validate correctly once tax unit logic is understood

### 2. Person Count Interpretation
The IRS SOI data counts **people per tax return**, not households:
- Average of 1.67 people per tax return in our test case
- Includes filers + spouses + dependents
- Explains seemingly high person_count targets (56,654 people for Alabama CD1's 75k-100k bracket)

### 3. Tax Unit vs Household Distinction (Critical)
- AGI constraints apply at **tax unit** level
- Multiple tax units can exist in one household
- Only people in qualifying tax units are counted
- This is the correct implementation for matching IRS data

Example from testing:
```
Household 8259: 5 people total
  Tax unit 825901: 3 members, AGI = $92,938 (in 75k-100k range) ✓
  Tax unit 825904: 1 member, AGI = $0 (not in range) ✗
  Tax unit 825905: 1 member, AGI = $0 (not in range) ✗
Matrix value: 3.0 (correct - only counts the 3 people in qualifying tax unit)
```

### 4. Geographic Structure Validation

Column positions follow a predictable pattern:
```python
# Formula: cd_block_number × n_households + household_index
# Example: Household 565 (index 12) in CD 601 (block 371)
column = 371 * 10580 + 12  # = 3,925,192

# Verify:
col_info = tracer.get_column_info(3925192)
print(f"CD: {col_info['cd_geoid']}, Household: {col_info['household_id']}")
# Output: CD: 601, Household: 565
```

## Full CD List Generation

To work with all 436 Congressional Districts:

```python
# Get all CDs from database
engine = create_engine(db_uri)
query = """
SELECT DISTINCT sc.value as cd_geoid
FROM strata s
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE s.stratum_group_id = 1
  AND sc.constraint_variable = 'congressional_district_geoid'
ORDER BY sc.value
"""
with engine.connect() as conn:
    result = conn.execute(text(query)).fetchall()
    all_cd_geoids = [row[0] for row in result]

print(f"Found {len(all_cd_geoids)} Congressional Districts")
# Note: Building full matrix takes ~15 minutes
```

## Target Grouping for Loss Function

### Overview
Targets are grouped to ensure each distinct measurement contributes equally to the calibration loss, regardless of how many individual targets represent it.

### Target Group Breakdown (81 groups total)

**National targets (Groups 0-29)**: 30 singleton groups
- Each national hardcoded target gets its own group
- Examples: tip income, medical expenses, Medicaid enrollment, ACA PTC recipients

**Geographic targets (Groups 30-80)**: 51 groups
- Age Distribution (Group 30): 7,848 targets (18 age bins × 436 CDs)
- Person Income Distribution (Group 31): 3,924 targets (9 AGI bins × 436 CDs)
- Medicaid Enrollment (Group 32): 436 targets (1 per CD)
- Tax Unit groups (Groups 33-56): Various IRS variables with constraints
  - 24 IRS SOI variable groups (amount + count for each)
  - Examples: QBI deduction, self-employment income, capital gains
- AGI Total Amount (Group 57): 436 targets (total AGI per CD)
- SNAP Household Count (Group 60): 436 targets (CD-level household counts)
- EITC groups (Groups 34-37): 4 child count brackets × 436 CDs
- SNAP Cost (State) (Group 73): 51 targets (state-level dollar amounts)

### Labeling Strategy
Labels are generated from variable names + stratum_group_id context:

**Ambiguous cases handled explicitly:**
- `household_count` + `stratum_group_id=4` → "SNAP Household Count"
- `snap` + `stratum_group_id='state_snap_cost'` → "SNAP Cost (State)"
- `adjusted_gross_income` + `stratum_group_id=2` → "AGI Total Amount"

**Default:** Variable name with underscores replaced by spaces and title-cased
- Most IRS variables are self-documenting (e.g., "Qualified Business Income Deduction")

### Key Insight
Previously, hardcoded labels caused confusion:
- "SNAP Recipients" was actually SNAP cost (dollars, not people)
- "Household Count" was ambiguous (didn't specify SNAP)
- "AGI Distribution" was misleading (it's total AGI amount, not distribution)

New approach uses variable names directly, only adding context where truly ambiguous.

## Medicaid Target Investigation

### Background
Initial concerns arose when observing identical Medicaid values for household members:
```python
person_medicaid_df.loc[person_medicaid_df.person_id.isin([56001, 56002])]
# Output:
#     person_id    medicaid  medicaid_enrolled
# 41      56001  18248.0625               True
# 42      56002  18248.0625               True
```

### Key Findings

#### 1. Correct Target Configuration
The ETL correctly uses `person_count` with `medicaid_enrolled==True` constraint:
- **Target variable**: `person_count` (always 1.0 per person)
- **Constraint**: `medicaid_enrolled==True` filters which people count
- **Aggregation**: Sums to household level (2 enrolled people = 2.0)
- **Metadata**: Fixed to reflect actual implementation

#### 2. Medicaid Cost Pattern Explanation
The identical values are **expected behavior**, not broadcasting:
- `medicaid_cost_if_enrolled` calculates state/group averages
- Groups: AGED_DISABLED, CHILD, EXPANSION_ADULT, NON_EXPANSION_ADULT
- Everyone in same state + group gets identical per-capita cost
- Example: All AGED_DISABLED in Maine get $18,248.0625

#### 3. Cost Variation Across Groups
Costs DO vary when household members are in different groups:
```
Household 113137 in Minnesota:
- 8-year-old child: $3,774.96 (CHILD group)
- 45-year-old disabled: $40,977.58 (AGED_DISABLED group)
- Difference: $37,202.62

Household 99593 in New York (7 people):
- Children (ages 6,8,18): $3,550.02 each
- Adults (ages 19,43): $6,465.34 each
- Elderly (age 72): $31,006.63
```

#### 4. Implications
- **For enrollment counting**: Working correctly, no issues
- **For cost calibration**: State/group averages may be too coarse
- **For realistic simulation**: Lacks individual variation within groups

## Hierarchical Target Consistency

### Qualified Business Income Deduction (QBID) Validation
Verified that QBID targets maintain perfect hierarchical consistency across geographic levels:

- **National level**: 1 target = $208,335,245,000
- **State level**: 51 targets (all states + DC) sum to $208,335,245,000
- **CD level**: 436 targets sum to $208,335,245,000

**Key findings:**
- CD-level targets sum exactly to their respective state totals
- State-level targets sum exactly to the national total
- Zero discrepancies found across all geographic aggregations

Example state validations:
- California: 52 CDs sum to exactly $25,340,115,000 (matches state target)
- Texas: 38 CDs sum to exactly $17,649,733,000 (matches state target)
- New York: 26 CDs sum to exactly $11,379,223,000 (matches state target)

This confirms the calibration targets are designed with perfect hierarchical consistency, where CDs aggregate to states and states aggregate to national totals.

**Technical note**: CD GEOIDs in the database are stored as integers (e.g., 601 for CA-1), requiring integer division by 100 to extract state FIPS codes.

## Conclusions

1. **Matrix is correctly constructed**: All tested values match expected behavior when tax unit logic is considered
2. **Geo-stacking approach is valid**: Households correctly appear in all CD columns
3. **Tax unit level constraints work properly**: Complex households with multiple tax units are handled correctly
4. **Medicaid targets are correct**: Using `person_count` with constraints properly counts enrolled individuals
5. **Hierarchical consistency verified**: Targets sum correctly from CD → State → National levels
6. **No errors found**: What initially appeared as errors were correct implementations of IRS data grouping logic and Medicaid cost averaging
7. **Tracer utility is effective**: Successfully navigates 4.6M column matrix and helped identify the tax unit logic
8. **Target grouping is transparent**: Labels now accurately describe what each group measures

## Recommendations

1. **Document tax unit vs household distinction prominently** - this is the most common source of confusion
2. **Add validation tests** to the build pipeline using patterns from this audit
3. **Include tax unit analysis** in any future debugging of person_count discrepancies
4. **Preserve household_tracer.py** as a debugging tool for future issues
5. **Consider caching** the full matrix build for development (takes ~15 minutes)

## Files Created/Modified

- `household_tracer.py`: Complete utility for matrix navigation and debugging
- `AUDIT.md`: This documentation
- Enhanced `print_matrix_structure()` method to show subgroups within large target groups

## Key Learning

The most important finding is that apparent "errors" in person counting were actually correct implementations. The matrix properly applies AGI constraints at the tax unit level, matching how IRS SOI data is structured. This tax unit vs household distinction is critical for understanding the calibration targets.

## Authors

Generated through collaborative debugging session, documenting the validation of geo-stacking sparse matrix construction for Congressional District calibration.