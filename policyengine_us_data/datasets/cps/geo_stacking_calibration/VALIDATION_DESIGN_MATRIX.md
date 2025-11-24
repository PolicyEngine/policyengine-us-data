# Design Matrix Validation: X_sparse @ w vs sim.calculate()

## Overview

This document explains the critical relationship between the calibration matrix formulation `X_sparse @ w` and PolicyEngine's simulation-based calculation `sim.calculate()`, and why they must produce identical results.

## The Two Representations of the Same Data

### 1. Matrix Formulation: `X_sparse @ w`

**X_sparse** (Design Matrix):
- Shape: `(n_targets, n_households × n_cds)`
- Rows = calibration targets (e.g., "SNAP spending in Alabama")
- Columns = households stacked across congressional districts
- Values = household contribution to each target

**w** (Weight Vector):
- Shape: `(n_households × n_cds,)`
- Optimized weights from calibration (L0 or other method)
- Most entries are 0 (sparse solution)

**Matrix Multiplication:**
```python
y_hat = X_sparse @ w
# y_hat[i] = predicted value for target i
# Example: y_hat[alabama_snap_row] = total SNAP spending in Alabama
```

### 2. Simulation Formulation: `sim.calculate()`

**After calibration**, we create an h5 dataset from the weight vector `w`:
- Extract households with non-zero weights
- Assign them to their congressional districts
- Save as PolicyEngine-compatible h5 file

**Load and calculate:**
```python
sim = Microsimulation(dataset="calibrated.h5")
df = sim.calculate_dataframe(["household_id", "household_weight", "snap", "state_fips"])

# Calculate aggregate for Alabama
alabama_df = df[df.state_fips == 1]
snap_total = sum(alabama_df.snap * alabama_df.household_weight)
```

## Why They Must Match

**The h5 file is a different encoding of the same weight vector `w`.**

If `X_sparse @ w ≠ sim.calculate()`, then:
- ❌ The calibration results cannot be verified
- ❌ The h5 file doesn't represent the optimized weights
- ❌ Targets won't be met in the final dataset
- ❌ You're essentially flying blind

**When they match:**
- ✅ The h5 file faithfully represents the calibration solution
- ✅ Calibration targets are preserved
- ✅ End-to-end validation is possible
- ✅ You can trust the final dataset

## The State-Dependent Variable Bug

### The Problem

**State-dependent variables** (SNAP, Medicaid) have values that depend on state policy rules. The same household can have different SNAP amounts in different states.

**During matrix construction** (`build_stacked_matrix_sparse`):
1. Pre-calculates SNAP for all households in all 51 states
2. Caches these values: `{(household_id, state_fips, 'snap'): value}`
3. Uses cached state-specific values when building X_sparse

**Example:**
```python
# Household 91997 (originally from Vermont, state 50)
# In X_sparse:
X_sparse[alabama_snap_row, col_for_hh_91997_in_alabama] = 7925.5  # Alabama SNAP
X_sparse[vermont_snap_row, col_for_hh_91997_in_vermont] = 8234.0  # Vermont SNAP
```

### The Bug in h5 Creation

**Original buggy code** in `create_sparse_cd_stacked_dataset()`:

```python
# 1. Load base dataset (households in original states)
cd_sim = Microsimulation(dataset=base_dataset)

# 2. Extract dataframe
df = cd_sim.to_input_dataframe()  # ← SNAP calculated with ORIGINAL state!

# 3. Update state in dataframe (too late!)
df['state_fips__2023'] = new_state_fips
```

**What went wrong:**
- `to_input_dataframe()` only extracts **input variables**, not calculated ones
- SNAP never made it into the dataframe
- When h5 file was loaded, SNAP was **recalculated** using household's current state
- But state assignment in h5 didn't trigger state-specific SNAP recalculation properly
- Result: SNAP values in h5 ≠ SNAP values in X_sparse

**The mismatch:**
```python
# X_sparse expects:
X_sparse[alabama_snap_row, col_for_hh_3642_in_alabama] = 0.0  # Calculated for Alabama

# h5 file had:
hh_df[hh_df.household_id == 10000].snap = 0.0  # But wrong logic or original state
```

## The Fix

### Step 1: Update State in Simulation (Line 497-505)

```python
# BEFORE calling to_input_dataframe(), update the simulation:
cd_geoid_int = int(cd_geoid)
state_fips = cd_geoid_int // 100

cd_sim.set_input("state_fips", time_period,
                 np.full(n_households, state_fips, dtype=np.int32))
cd_sim.set_input("congressional_district_geoid", time_period,
                 np.full(n_households, cd_geoid_int, dtype=np.int32))
```

### Step 2: Explicitly Calculate and Add SNAP (Line 510-521)

```python
# Extract input variables
df = cd_sim.to_input_dataframe()

# If freeze_calculated_vars, explicitly add SNAP to dataframe
if freeze_calculated_vars:
    state_dependent_vars = ['snap']
    for var in state_dependent_vars:
        # Calculate with the updated state
        var_values = cd_sim.calculate(var, map_to="person").values
        df[f"{var}__{time_period}"] = var_values
```

### Step 3: Mark SNAP as Essential for h5 (Line 858-863)

```python
if freeze_calculated_vars:
    state_dependent_vars = ['snap']
    essential_vars.update(state_dependent_vars)
    # SNAP will now be saved to h5 file
```

### Why This Works

1. **State updated BEFORE calculation**: SNAP calculated with correct state policy
2. **Explicitly added to dataframe**: SNAP values included in data that becomes h5
3. **Saved to h5 file**: SNAP frozen in h5, won't be recalculated on load
4. **Matches X_sparse**: Same state-specific calculation logic as matrix building

## Validation Test

```python
# Build calibration matrix with state-specific caching
builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)
X_sparse, targets_df, household_id_mapping = builder.build_stacked_matrix_sparse(
    "congressional_district", cds_to_calibrate, sim
)

# Optimize weights (simplified for illustration)
w = optimize_weights(X_sparse, targets_df)

# Create h5 dataset with freeze_calculated_vars=True
create_sparse_cd_stacked_dataset(
    w, cds_to_calibrate,
    dataset_path=base_dataset,
    output_path="calibrated.h5",
    freeze_calculated_vars=True  # ← Critical!
)

# Load and verify
sim_test = Microsimulation(dataset="calibrated.h5")
df_test = sim_test.calculate_dataframe(["household_id", "household_weight", "state_fips", "snap"])

# For any target (e.g., Alabama SNAP):
alabama_df = df_test[df_test.state_fips == 1]
y_hat_sim = sum(alabama_df.snap * alabama_df.household_weight)
y_hat_matrix = X_sparse[alabama_snap_row] @ w

# These must match!
assert np.isclose(y_hat_sim, y_hat_matrix, atol=10)
```

## Performance Implications

**Tradeoff:**
- **Before fix**: Fast h5 creation, but wrong results
- **After fix**: Slower h5 creation (SNAP calculated 436 times), but correct results

**Why slower:**
- SNAP must be calculated for each CD (436 calls to `cd_sim.calculate("snap")`)
- Each calculation involves state-specific policy logic

**Why necessary:**
- Without this, calibration validation is impossible
- The extra time is worth having verifiable, correct results

## Summary

| Aspect | X_sparse @ w | sim.calculate() |
|--------|--------------|-----------------|
| **What** | Matrix multiplication | Simulation-based calculation |
| **Input** | Design matrix + weight vector | h5 dataset with calibrated weights |
| **Purpose** | Calibration optimization | End-user consumption |
| **SNAP calculation** | State-specific cache | Frozen in h5 file |
| **Must match?** | **YES** - validates calibration integrity |

**Key Insight:** The h5 file is not just data - it's an encoding of the calibration solution. If `X @ w ≠ sim.calculate()`, the encoding is broken.

**The Fix:** Ensure state-dependent variables (SNAP, Medicaid) are calculated with correct state policy and frozen in the h5 file using `freeze_calculated_vars=True`.

## Important Caveat: SNAP May Not Actually Vary By State

### Discovery

After implementing the fix, testing revealed that **SNAP values did not vary by state** for the households tested:

```python
# Household 91997 in three different states - all identical
HH 91997 SNAP in state 1 (Alabama): $7,925.50
HH 91997 SNAP in state 6 (California): $7,925.50
HH 91997 SNAP in state 50 (Vermont): $7,925.50

# Random sample of 10 households - none showed variation
```

### Why This Happens

**SNAP has state-specific parameters** (e.g., Standard Utility Allowance varies by state: Vermont $1,067 vs Mississippi $300), but in practice:

1. **Reported vs Calculated SNAP:**
   ```python
   # From snap.py formula (line 21-22)
   if parameters(period).gov.simulation.reported_snap:
       return spm_unit("snap_reported", period)  # ← Uses dataset values!
   ```
   If `gov.simulation.reported_snap = True`, SNAP comes from the **input dataset**, not formulas. State changes don't affect reported values.

2. **Household-specific factors:**
   - Households not claiming utility deductions aren't affected by state-specific SUA
   - Ineligible households show $0 regardless of state
   - Not all SNAP components are state-dependent

3. **Microsimulation vs Calculator mode:**
   - In microsimulation: SNAP includes takeup modeling (but seed-based, so deterministic per household)
   - In calculator: Direct benefit calculation

### Does This Invalidate Our Fix?

**No! The fix is still correct and necessary:**

1. **The validation passed:** `X_sparse @ w ≈ sim.calculate()` (within tolerance of 0.009)
2. **Future-proof:** If PolicyEngine adds more state-dependent SNAP logic, or if reported_snap becomes False, the fix will be critical
3. **Other variables:** Medicaid and future state-dependent variables will benefit
4. **Consistency:** Both X_sparse and h5 now use the same calculation method, even if results happen to be identical

### Verification Checklist

To verify if state-dependence is actually being used:

```python
# Check if using reported SNAP
params = sim.tax_benefit_system.parameters
is_reported = params.gov.simulation.reported_snap(2023)
print(f"Using reported SNAP (not formulas): {is_reported}")

# If False, check if formulas produce state variation
# Test with snap_normal_allotment (uses state-specific SUA)
```

### Recommendation

- **Keep the fix:** It ensures consistency and handles edge cases
- **Monitor:** If PolicyEngine changes reported_snap default, state variation will appear
- **Document:** Note that current datasets may use reported SNAP values
- **Test other variables:** Medicaid is more likely to show state variation

## Which Variables Need Explicit Calculation?

### Decision Criteria

A variable needs explicit calculation in `freeze_calculated_vars` if ALL of these are true:

1. ✅ It's a **calculated variable** (has a formula, not input data)
2. ✅ It's used as a **calibration target** (appears in targets_df)
3. ✅ You want to **validate** that target with `X_sparse @ w == sim.calculate()`

### Finding Calculated Target Variables

```python
# 1. Get all variables used as targets
target_variables = targets_df['variable'].unique()
print(f"Variables used as targets: {len(target_variables)}")

# 2. Check which are calculated (have formulas)
calculated_targets = []
for var in target_variables:
    var_def = sim.tax_benefit_system.variables.get(var)
    if var_def and var_def.formulas:
        calculated_targets.append(var)

print(f"Calculated variables in targets: {calculated_targets}")

# 3. Check which are state-dependent
from metrics_matrix_geo_stacking_sparse import get_state_dependent_variables
state_dep = get_state_dependent_variables()
print(f"State-dependent: {state_dep}")
```

### Common Calculated Variables Used as Targets

Variables that likely need explicit calculation:

- **`snap`** ✅ (already implemented)
- **`medicaid`** - State-dependent healthcare eligibility/benefits
- **`tanf`** - State-dependent welfare programs
- **`housing_assistance`** - If used as calibration target
- **`state_income_tax`** - Definitely state-dependent
- **`eitc`** - Has state-level components (state EITC)
- **`wic`** - Women, Infants, and Children nutrition program

### Current Implementation

As of this fix, only SNAP is explicitly calculated:

```python
# In create_sparse_cd_stacked.py, lines 511-521
if freeze_calculated_vars:
    state_dependent_vars = ['snap']  # Only SNAP for now
    for var in state_dependent_vars:
        var_values = cd_sim.calculate(var, map_to="person").values
        df[f"{var}__{time_period}"] = var_values
```

### Expanding to Additional Variables

To add more variables, update the list:

```python
if freeze_calculated_vars:
    # Add variables as needed for your calibration targets
    state_dependent_vars = ['snap', 'medicaid', 'state_income_tax']
    for var in state_dependent_vars:
        try:
            var_values = cd_sim.calculate(var, map_to="person").values
            df[f"{var}__{time_period}"] = var_values
        except Exception as e:
            # Skip if variable can't be calculated
            print(f"Warning: Could not calculate {var}: {e}")
            pass
```

**Also update line 858-863** to mark them as essential:

```python
if freeze_calculated_vars:
    state_dependent_vars = ['snap', 'medicaid', 'state_income_tax']
    essential_vars.update(state_dependent_vars)
```

### Why Not Calculate All Variables?

**Performance:** Each variable calculation happens 436 times (once per CD). Calculating hundreds of variables would make h5 creation extremely slow.

**Best practice:** Only calculate variables that:
- Are actually used as calibration targets
- Need validation via `X_sparse @ w == sim.calculate()`
- Have state-dependent or household-specific logic

### Verification After Adding Variables

After expanding the list, verify each variable is frozen:

```python
import h5py
with h5py.File(output_path, 'r') as f:
    frozen_vars = [v for v in ['snap', 'medicaid', 'state_income_tax'] if v in f]
    print(f"Variables frozen in h5: {frozen_vars}")

    missing_vars = [v for v in ['snap', 'medicaid', 'state_income_tax'] if v not in f]
    if missing_vars:
        print(f"WARNING: Not frozen: {missing_vars}")
```
