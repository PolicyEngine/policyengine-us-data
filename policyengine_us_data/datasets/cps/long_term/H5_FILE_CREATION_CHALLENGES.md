# Challenges in Creating Year-Specific .h5 Files for PolicyEngine

## Executive Summary - Reproduce the Problem

```bash
# Create calibrated year-specific .h5 files
python run_full_projection.py 2026 --greg --save-h5
# Output: 2026 income tax projection = $2233.8B
```

```python
# Test the created file
from policyengine_us import Microsimulation
import numpy as np

sim_base = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim26 = Microsimulation(dataset="./projected_datasets/2026.h5")

# Baseline weights comparison
baseline_weights = sim_base.calculate("income_tax", period=2026, map_to="household").weights
print(f"Base 2026: ${sim_base.calculate('income_tax', period=2026, map_to='household').sum()/1e9:.1f}B")
# Expected: $2174.7B
print(f"From h5:   ${np.sum(sim26.calculate('income_tax', map_to='household').values * baseline_weights)/1e9:.1f}B")
# Actual: $2132.9B ❌ (-$41.8B error)

# Calibrated weights
print(f"Calibrated: ${sim26.calculate('income_tax', map_to='household').sum()/1e9:.1f}B")
# Expected: $2233.8B, Actual: $2190.5B ❌ (-$43.3B error)
```

**Root Cause**: 20 tax-unit level variables (deductions/credits) not uprated → keep 2024 values → income tax $40B too low.

**Fix**: Add tax-unit/SPM-unit/marital-unit level mapping in `create_year_h5()` uprating loop.

---

## Overview

This document describes the challenges encountered when creating year-specific .h5 files (2025.h5, 2026.h5, ..., 2100.h5) from a base 2024 CPS microdata file with demographic calibration weights and economic uprating.

**Goal**: Create datasets where:
- Each file represents a specific future year (e.g., 2026.h5)
- Contains demographically-calibrated weights matching SSA population projections
- Contains uprated economic values (incomes, benefits) for that year
- Can be loaded by PolicyEngine and produce accurate tax calculations

**Current Status**: Partial success with $40B+ discrepancy in income tax calculations.

---

## The PolicyEngine .h5 Architecture

### What PolicyEngine Expects

PolicyEngine .h5 files store microdata with this structure:
```
file.h5
├── variable_name_1/
│   └── "2026" (dataset)
├── variable_name_2/
│   └── "2026" (dataset)
└── ...
```

Each variable group contains datasets keyed by **period** (year). The `default_calculation_period` is inferred from which periods are present.

### Input vs Calculated Variables

**Input Variables** (stored in .h5):
- Pure demographic data: `age`, `sex`, `state_fips`
- Income sources: `employment_income_before_lsr`, `qualified_dividend_income`
- Entity IDs: `person_id`, `household_id`, `tax_unit_id`
- Weights: `household_weight`, `person_weight`

**Calculated Variables** (computed at runtime):
- Derived incomes: `employment_income` (from `employment_income_before_lsr` + labor supply response)
- Tax calculations: `income_tax`, `taxable_income`, `adjusted_gross_income`
- Credits/deductions with complex logic: credits that phase out, bracket adjustments

### The Uprating Mechanism

PolicyEngine has built-in uprating that automatically adjusts values based on the `period` parameter:
- **Wage growth**: Employment income grows ~5% annually
- **CPI inflation**: Benefits adjust for inflation
- **Tax bracket indexing**: Standard deduction, brackets adjust with inflation
- **Credit phase-outs**: Income thresholds for credits adjust

When you call `sim.calculate("employment_income", period=2026)`, PolicyEngine:
1. Reads the stored input value (e.g., `employment_income_before_lsr` at period 2024)
2. Applies uprating factors to project to 2026
3. Applies any formulas (e.g., labor supply response)
4. Returns the 2026 value

---

## Problems Encountered

### Problem 1: Default Period Mismatch

**Initial Issue**: Created 2026.h5 files but they had `default_calculation_period = '2024'`

**Cause**: Data was stored at period 2024, not 2026. PolicyEngine sets default period based on what periods exist in the file.

**Solution**: Rename all DataFrame columns from `variable__2024` → `variable__2026` before creating Dataset.

**Status**: ✅ RESOLVED

### Problem 2: Calibrated Weights Not Applied

**Issue**: Income tax calculations didn't reflect demographic calibration (missing $60B effect)

**Cause**: Weights were set via `sim.set_input()` but then overwritten when we called `sim.calculate("household_weight", period=year)` in the uprating loop.

**Solution**: Explicitly set calibrated weights in DataFrame before processing other variables, and skip weight variables in uprating loop.

**Status**: ✅ RESOLVED - Population now correctly shows 348.8M

### Problem 3: Formula Variables Stored Incorrectly

**Issue**: Variables like `american_opportunity_credit`, `cdcc_relevant_expenses` were being stored with 2024 calculations, then used as-is in 2026.

**Cause**: These variables have formulas (time-dependent logic), but were being pre-calculated at period 2024 and stored. When PolicyEngine loaded them, it used the stored values instead of recalculating with 2026 parameters.

**Attempted Solution**: Drop all formula variables before storing.

**Result**: ❌ FAILED - Dropping them caused bigger problems because some formula variables are actually needed as inputs (they're pre-calculated in base file).

**Current Status**: Not dropping formula variables, but this causes issues.

### Problem 4: Entity-Level Mismatches

**Issue**: Income tax calculations are $40B too low even with baseline weights ($2132.9B vs $2174.7B expected).

**Root Cause**: PolicyEngine has multiple entity levels:
- **Person level**: 53,430 individuals
- **Household level**: 21,108 households
- **Tax unit level**: 29,271 tax units (filing units)
- **SPM unit level**: ~30,000 SPM units
- **Marital unit level**: ~25,000 marital units

The DataFrame from `to_input_dataframe()` is at **person level** (53,430 rows). All values must be person-level.

**What We Handle**:
- ✅ Person-level variables (53,430 values) → use directly
- ✅ Household-level variables (21,108 values) → map via `person_household_id`
- ❌ Tax-unit level variables (29,271 values) → **CURRENTLY FAILING**
- ❌ SPM-unit level variables → **CURRENTLY FAILING**
- ❌ Marital-unit level variables → **CURRENTLY FAILING**

**Variables Affected** (20+ tax-unit level variables not being uprated):
```
unreported_payroll_tax
self_employed_health_insurance_ald
self_employed_pension_contribution_ald
domestic_production_ald
health_savings_account_ald
interest_deduction
recapture_of_investment_credit
savers_credit
cdcc_relevant_expenses
energy_efficient_home_improvement_credit
... (10 more)
```

These are tax-unit level deductions and credits. When we try to uprate them:
1. `sim.calculate(var, period=2026)` returns 29,271 values (tax-unit level)
2. Our DataFrame has 53,430 rows (person level)
3. Dimension mismatch → we skip uprating → they keep base 2024 values
4. In 2026 dollars, these deductions/credits should be ~10% higher
5. Lower deductions/credits → higher taxable income → higher tax by ~$40B ✓

**Status**: 🔴 UNRESOLVED - This is likely the main cause of the $40B discrepancy

---

## Technical Deep Dive

### The Uprating Loop Logic

Current logic in `create_year_h5()`:

```python
for col in df.columns:
    if f"__{base_period}" in col:
        var_name = col.replace(f"__{base_period}", "")

        # Skip weights (already set)
        if var_name in ['household_weight', 'person_weight', 'tax_unit_weight']:
            continue

        # Skip invalid variables
        if var_name not in valid_variables:
            df.rename(columns={col: col_name_new}, inplace=True)
            continue

        # Calculate uprated value
        try:
            uprated_values = sim.calculate(var_name, period=year).values
        except:
            df.rename(columns={col: col_name_new}, inplace=True)
            continue

        # Check entity level
        if len(uprated_values) == n_persons:
            # Person-level: use directly
            df[col_name_new] = uprated_values
        elif len(uprated_values) == len(household_ids):
            # Household-level: expand to persons
            hh_to_value = dict(zip(household_ids, uprated_values))
            df[col_name_new] = person_household_id_series.map(hh_to_value)
        else:
            # PROBLEM: Other entity levels fall through here!
            # Just rename without uprating
            df.rename(columns={col: col_name_new}, inplace=True)
            continue

        df.drop(columns=[col], inplace=True)
```

**The Gap**: The `else` clause catches tax-unit, SPM-unit, and marital-unit level variables and doesn't uprate them.

### Why Entity Mapping is Hard

For household-level variables, we can map via `person_household_id`:
- Each person belongs to exactly one household
- Mapping is straightforward: `person → household_id → household_value`

For tax-unit level variables, we need `person_tax_unit_id`:
- Each person belongs to exactly one tax unit
- Mapping: `person → tax_unit_id → tax_unit_value`

**BUT**: We need the actual tax_unit IDs to create the mapping dictionary.

Current code only has:
- `household_ids` (from `sim.calculate("household_id", map_to="household")`)
- We DON'T have `tax_unit_ids`

---

## Hypotheses for Solutions

### Hypothesis 1: Add Multi-Entity Level Mapping

**Approach**: Calculate IDs for all entity types and handle each appropriately.

```python
# Get all entity IDs
household_ids = sim.calculate("household_id", map_to="household").values
tax_unit_ids = sim.calculate("tax_unit_id", map_to="tax_unit").values
spm_unit_ids = sim.calculate("spm_unit_id", map_to="spm_unit").values
marital_unit_ids = sim.calculate("marital_unit_id", map_to="marital_unit").values

# Get person-to-entity mappings
person_household_id = df[f"person_household_id__{base_period}"]
person_tax_unit_id = df[f"person_tax_unit_id__{base_period}"]
person_spm_unit_id = df[f"person_spm_unit_id__{base_period}"]
person_marital_unit_id = df[f"person_marital_unit_id__{base_period}"]

# In the uprating loop:
if len(uprated_values) == n_persons:
    df[col_name_new] = uprated_values
elif len(uprated_values) == len(household_ids):
    hh_to_value = dict(zip(household_ids, uprated_values))
    df[col_name_new] = person_household_id.map(hh_to_value)
elif len(uprated_values) == len(tax_unit_ids):
    tu_to_value = dict(zip(tax_unit_ids, uprated_values))
    df[col_name_new] = person_tax_unit_id.map(tu_to_value)
elif len(uprated_values) == len(spm_unit_ids):
    spm_to_value = dict(zip(spm_unit_ids, uprated_values))
    df[col_name_new] = person_spm_unit_id.map(spm_to_value)
elif len(uprated_values) == len(marital_unit_ids):
    mu_to_value = dict(zip(marital_unit_ids, uprated_values))
    df[col_name_new] = person_marital_unit_id.map(mu_to_value)
else:
    # Unknown entity type - just rename without uprating
    df.rename(columns={col: col_name_new}, inplace=True)
    continue
```

**Pros**:
- Handles all entity types systematically
- Should capture the missing $40B in tax-unit level adjustments

**Cons**:
- Adds complexity
- Still doesn't handle truly non-entity variables

**Likelihood of Success**: 🟢 HIGH - This should resolve the $40B baseline discrepancy

### Hypothesis 2: Use PolicyEngine's Native Uprating

**Approach**: Don't try to manually uprate variables. Instead:
1. Store all input data at base period (2024)
2. Store calibrated weights at target period (2026)
3. Let PolicyEngine's native uprating handle everything when calculating

**Implementation**:
```python
# Don't uprate anything - just copy input data and relabel period
df = sim.to_input_dataframe()

# Only update weights to target year
df[f"household_weight__{year}"] = person_household_weight_uprated
df[f"person_weight__{year}"] = person_weights

# For all other variables, just rename period
for col in df.columns:
    if f"__{base_period}" in col and "weight" not in col:
        new_col = col.replace(f"__{base_period}", f"__{year}")
        df.rename(columns={col: new_col}, inplace=True)

# Store everything at target year period
dataset = Dataset.from_dataframe(df, year)
```

**Pros**:
- Simpler - no entity-level logic needed
- Relies on PolicyEngine's tested uprating logic
- Avoids manually calculating uprated values

**Cons**:
- **Won't work** - PolicyEngine's uprating is RELATIVE to stored period
- If you store 2024 data labeled as 2026, PolicyEngine thinks it IS 2026 data
- Asking for period 2028 would uprate FROM that "2026" value
- **This is fundamentally incompatible with PolicyEngine's design**

**Likelihood of Success**: 🔴 ZERO - Architecturally impossible

### Hypothesis 3: Hybrid Approach - Minimal Uprating

**Approach**: Only uprate the core income/benefit variables that drive most tax calculations. Leave minor variables at base values.

**Key variables to uprate**:
- Employment income (`employment_income_before_lsr`)
- Self-employment income (`self_employment_income_before_lsr`)
- Investment income (dividends, interest, capital gains)
- Social Security benefits
- Other transfer income (SSI, unemployment, etc.)

**Leave at base values**:
- Deductions/credits (these are formula variables anyway)
- Demographic variables (age, state, etc.)
- Entity structure (IDs, relationships)

**Pros**:
- Simpler than full entity-level mapping
- Captures 90%+ of the dollar impact

**Cons**:
- Still has the entity-level problem for major deductions
- Partial solution - $40B error might reduce to $10B but won't eliminate
- Creates inconsistency (some 2026, some 2024)

**Likelihood of Success**: 🟡 MEDIUM - Would reduce but not eliminate discrepancy

### Hypothesis 4: Don't Store - Dynamically Generate

**Approach**: Don't create .h5 files at all. Instead:
1. Store only the calibrated weights (76 weight vectors, one per year)
2. When user needs year 2075, dynamically:
   - Load base 2024 .h5
   - Apply year 2075 calibrated weights
   - PolicyEngine handles uprating at calculation time

**Implementation**:
```python
# Save weights matrix
np.save('calibrated_weights_2025_2100.npy', weights_matrix)

# When user needs 2075:
sim = Microsimulation(dataset="enhanced_cps_2024.h5")
weights_2075 = weights_matrix[:, year_idx_2075]
sim.set_input("household_weight", 2075, weights_2075)
# ... calculate income_tax at period 2075 ...
```

**Pros**:
- Avoids all .h5 creation problems
- Leverages PolicyEngine's uprating correctly
- Much smaller storage (76 weight vectors vs 76 full datasets)

**Cons**:
- User can't directly load "2075.h5" in PolicyEngine
- Requires wrapper code to apply weights
- Doesn't provide standalone datasets

**Likelihood of Success**: 🟢 HIGH - Would work correctly

**Note**: This may not meet user requirements if they need standalone .h5 files for distribution or external use.

---

## Recommended Path Forward

### Short-term Fix (for testing):

Implement **Hypothesis 1** - Add multi-entity level mapping:

```python
# Add at beginning of create_year_h5():
tax_unit_ids = sim.calculate("tax_unit_id", map_to="tax_unit").values
spm_unit_ids = sim.calculate("spm_unit_id", map_to="spm_unit").values
marital_unit_ids = sim.calculate("marital_unit_id", map_to="marital_unit").values

# Update the entity-level checking in uprating loop (see code above)
```

**Expected result**:
- Baseline income tax: $2174.7B (currently $2132.9B) - **$40B improvement** ✓
- Calibrated income tax: ~$2233B (currently $2190.5B) - **$43B improvement** ✓

### Long-term Consideration:

Evaluate **Hypothesis 4** - Dynamic weight application:
- If users can work with a weight matrix + base file instead of 76 separate .h5 files
- This would be more maintainable and guaranteed to work correctly
- Could provide a wrapper function that makes it feel like loading separate datasets

---

## Open Questions

1. **Formula Variables**: Should we store pre-calculated formula variables or let PolicyEngine recalculate them?
   - Current: Storing them (keeping whatever's in input DataFrame)
   - Risk: May have time-dependent logic that breaks
   - Unknown impact: Small? Large?

2. **Tax Law Changes**: Our approach assumes tax law is constant (2024 law applied to all years)
   - Is this the intended behavior?
   - Or should we account for scheduled tax changes (TCJA expiration, etc.)?

3. **Why are some formula variables in the input DataFrame?**
   - `american_opportunity_credit`, `savers_credit`, etc. have formulas
   - But they're stored as inputs in enhanced_cps_2024.h5
   - Are these override values? Pre-computed for performance?

4. **Validation**: How do we validate that 2075.h5 is "correct"?
   - Compare to: base .h5 calculated at period 2075 with same weights?
   - Check: income tax, population, income distributions?

---

## Diagnostic Summary

### Current State (as of last test):

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Default period | 2026 | 2026 | ✅ |
| Population (persons) | 348.8M | 348.8M | ✅ |
| Households | 149.8M | 149.8M | ✅ |
| Baseline income tax | $2174.7B | $2132.9B | ❌ (-$41.8B) |
| Calibrated income tax | $2233.8B | $2190.5B | ❌ (-$43.3B) |
| DataFrame columns | 179 vars | 159 vars | ❌ (-20 vars) |

**Missing 20 variables**: Tax-unit level deductions/credits not being uprated due to entity-level mismatch.

**Impact**: ~$40B underestimate of income tax (both baseline and calibrated).

---

## Conclusion

Creating year-specific .h5 files for PolicyEngine is complex because:

1. **Entity Levels**: Variables exist at multiple entity levels (person, household, tax unit, etc.) and must be properly mapped to person-level for DataFrame storage

2. **Uprating Complexity**: Income and benefit variables need period-specific uprating, but this must be done carefully to avoid breaking PolicyEngine's internal uprating logic

3. **Input vs Formula Variables**: Some variables with formulas are stored as inputs (pre-calculated), others should be calculated at runtime. Distinguishing these is non-trivial.

4. **Time-Dependent Logic**: Tax law parameters (brackets, phase-outs, standard deductions) change over time and must be properly applied

The most promising immediate fix is implementing multi-entity level mapping to ensure all 179 input variables are properly uprated to the target year. This should resolve the $40B discrepancy.

A longer-term architectural question is whether creating 76 separate .h5 files is the right approach, or if a more lightweight solution (storing only calibrated weights + using base file) would be more maintainable and reliable.

---

## Design Decision: Household-Only Projection Pathway (2025)

### Context

While `run_full_projection.py` handles all entity levels (person, household, tax_unit, spm_unit, marital_unit), we created `run_household_projection.py` as a simpler alternative that uses `map_to="household"` exclusively in the projection calculations.

### Key Design Choices

#### 1. Projection Calculations: Household-Level Only

In the main projection loop, we calculate aggregates at household level:
```python
# Simplified approach
income_tax_hh = sim.calculate('income_tax', period=year, map_to='household')
income_tax_values = income_tax_hh.values  # Already in household order
```

**Rationale**:
- Eliminates person→household aggregation steps
- Values come directly in household order for weight calibration
- ~60% less code than multi-entity approach
- Suitable for aggregate revenue projections

**Limitation**: Cannot analyze by person-level characteristics in the projection loop

#### 2. H5 File Creation: Person-Level Storage Required

Despite using household-level calculations for projections, when creating .h5 files we still need person-level detail:

```python
def create_household_year_h5(year, household_weights, base_dataset_path, output_dir):
    # Broadcast calibrated household weights to persons
    person_weights = person_household_id.map(hh_to_weight)  # ✓ Correct

    # Uprate variables at PERSON level, not household
    uprated_values = sim.calculate(var_name, period=year).values  # ✓ Person-level
```

**Critical Distinction**:
- **Household weights → Persons**: Broadcast (all persons in household share same weight) ✓
- **Data variables**: Calculate at person level to preserve individual-specific values ✓

**Why Not Broadcast Data Variables?**
If we did `sim.calculate('social_security', map_to='household')` and broadcast to persons:
- Each person would get the TOTAL household SS benefits
- Person with $2000/month SS would show as $3500 (household total)
- Age, income, benefits are individual attributes, not shared across household

**Why Broadcast Weights?**
- Weights represent the sampling/expansion factor
- All persons in a household represent the same number of population members
- This is statistically correct for survey weighting

#### 3. The "Household-Only" Name

The pathway is called "household-only" because:
- **Projection loop**: Uses `map_to='household'` for simplicity
- **Weight calibration**: Works with household-level weights
- **Aggregation**: Household-level totals for revenue projections

But NOT because:
- ❌ It broadcasts household data to persons (that would be wrong)
- ❌ It ignores person-level characteristics (those are preserved in .h5)
- ❌ It can't create proper datasets (it can, using person-level uprating)

### Summary of Approach

| Component | Level | Rationale |
|-----------|-------|-----------|
| Projection calculations | Household | Simpler, faster, suitable for aggregates |
| Weight calibration | Household | IPF/GREG works on household weights |
| Weight storage | Broadcast to person | All household members share weight |
| Variable uprating | Person | Preserves individual characteristics |
| H5 file structure | Person-level rows | Required by PolicyEngine architecture |

This design provides a simpler pathway for aggregate projections while still producing valid person-level datasets when needed.
