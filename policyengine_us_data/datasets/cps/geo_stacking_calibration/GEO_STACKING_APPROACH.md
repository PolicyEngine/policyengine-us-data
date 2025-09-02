# Geo-Stacking Calibration Approach

## Overview

The geo-stacking approach treats the same household dataset as existing in multiple geographic areas simultaneously. This creates an "empirical superpopulation" where each household can represent itself in different locations with different weights.

## Matrix Structure

### Dimensions
- **Rows = Targets** (the "observations" in our regression problem)
- **Columns = Households** (the "variables" whose weights we're estimating)

This creates a "small n, large p" problem where:
- n = number of targets (rows)
- p = number of households × number of geographic areas (columns)

### Key Insight
In traditional regression, we estimate parameters (coefficients) for variables using observations. Here:
- Household weights are the parameters we estimate
- Calibration targets are the observations
- Each household's characteristics are the "variables"

## Stacking Logic

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

### Geographic Hierarchy

The approach respects the geographic hierarchy:
1. **National targets**: Apply to all household copies
2. **State targets**: Apply only to households in that state's copy
3. **Congressional District targets**: Apply only to households in that CD's copy

When more precise geographic data is available, it overrides less precise data:
- If we have CD-level age distributions, use those instead of state-level
- If we have state-level age distributions, use those instead of national

## Implementation Details

### Target Types

Currently implemented:
- **National hardcoded targets**: Simple scalar values (employment_income, tax_revenue, etc.)
- **Age distribution targets**: 18 age bins per geography

Future additions:
- **Income/AGI targets**: 9 income brackets per geography (stratum_group_id = 3)
- **SNAP targets**: 1 boolean per geography (stratum_group_id = 4)
- **Medicaid targets**: 1 boolean per geography (stratum_group_id = 5)
- **EITC targets**: 4 categories by qualifying children (stratum_group_id = 6)

### Database Structure

The database uses stratum_group_id to categorize target types:
- 1 = Geographic boundaries
- 2 = Age-based strata
- 3 = Income/AGI-based strata
- 4 = SNAP recipient strata
- 5 = Medicaid enrollment strata
- 6 = EITC recipient strata

### Scaling Considerations

For full US implementation:
- 51 states (including DC) × ~100,000 households = 5.1M columns
- 436 congressional districts × ~100,000 households = 43.6M columns

With targets:
- National: ~10-20 targets
- Per state: 18 age bins + future demographic targets
- Per CD: 18 age bins + future demographic targets

This creates extremely sparse matrices requiring specialized solvers.

## Advantages

1. **Diversity**: Access to full household diversity even in small geographic areas
2. **Consistency**: Same households across geographies ensures coherent microsimulation
3. **Flexibility**: Can add new geographic levels or demographic targets easily
4. **Reweighting**: Each geography gets appropriate weights for its households

## Technical Notes

### Sparse Matrix Handling
The matrix becomes increasingly sparse as we add geographic areas. Future optimizations:
- Use scipy.sparse matrices for memory efficiency
- Implement specialized sparse solvers
- Consider block-diagonal structure for some operations

### Constraint Handling
Constraints are applied hierarchically:
1. Geographic constraints determine which targets apply
2. Demographic constraints (age, income, etc.) determine which individuals/households contribute
3. Masks are created at appropriate entity levels and mapped to household level

### Period Consistency
All calculations use explicit period (year) arguments to ensure:
- Target values match the correct year
- Microsimulation calculations use consistent time periods
- Future uprating can adjust for temporal mismatches