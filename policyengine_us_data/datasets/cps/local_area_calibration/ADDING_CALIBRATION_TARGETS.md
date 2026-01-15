# Adding Calibration Targets to Local Area Geo-Stacking

This document summarizes key learnings from adding `health_insurance_premiums_without_medicare_part_b` as a calibration target. Use this as a reference when adding future targets.

## Key Discovery: No Code Changes Needed for Most Targets

The `sparse_matrix_builder.py` is **already entity-agnostic**. PolicyEngine's `map_to="household"` parameter automatically handles aggregation from any entity level (person, tax_unit, spm_unit) to household level.

```python
# This line in sparse_matrix_builder.py (line 220-222) handles ALL entity types:
target_values = state_sim.calculate(
    target["variable"], map_to="household"
).values
```

**Verified behavior:**
- Person-level variables (like health_insurance_premiums): automatically summed to household
- SPM-unit variables (like snap): automatically aggregated to household
- Household variables: returned as-is

## Architecture Overview

### File Locations

```
policyengine_us_data/
├── storage/
│   ├── calibration/
│   │   └── policy_data.db          # Target database (SQLite)
│   └── stratified_extended_cps_2023.h5  # Base dataset for calibration
└── datasets/cps/local_area_calibration/
    ├── sparse_matrix_builder.py    # Builds X_sparse matrix (GENERIC)
    ├── matrix_tracer.py            # Diagnostics for debugging matrices
    ├── calibration_utils.py        # Helper functions
    └── build_calibration_matrix.py # Runner script
```

### Database Schema (policy_data.db)

```sql
-- Core tables
targets(target_id, variable, period, stratum_id, value, active, ...)
strata(stratum_id, definition_hash, stratum_group_id, ...)
stratum_constraints(stratum_id, constraint_variable, operation, value, ...)
```

**Key stratum_group_ids:**
- Group 1: National hardcoded targets (20 variables including health insurance, medicaid, snap national, etc.)
- Group 4: SNAP state/CD targets (538 targets: 51 state snap + 487 household_count)

### Target Filter Logic

The `build_matrix()` method uses **OR logic** for filters:

```python
# Gets SNAP targets OR health insurance target
target_filter={
    "stratum_group_ids": [4],  # All SNAP targets
    "variables": ["health_insurance_premiums_without_medicare_part_b"],  # Specific variable
}
```

## How to Add a New Target

### Step 1: Check if Target Exists in Database

```python
import sqlite3
from policyengine_us_data.storage import STORAGE_FOLDER

conn = sqlite3.connect(STORAGE_FOLDER / "calibration" / "policy_data.db")
cursor = conn.cursor()

# Find your target
cursor.execute("""
    SELECT t.target_id, t.variable, t.value, t.period, t.stratum_id,
           s.stratum_group_id
    FROM targets t
    JOIN strata s ON t.stratum_id = s.stratum_id
    WHERE t.variable = 'your_variable_name'
""")
print(cursor.fetchall())

# Check constraints for that stratum
cursor.execute("""
    SELECT * FROM stratum_constraints WHERE stratum_id = <stratum_id>
""")
print(cursor.fetchall())
```

### Step 2: Determine Entity Type

```python
from policyengine_us import Microsimulation

sim = Microsimulation()
var = sim.tax_benefit_system.variables['your_variable_name']
print(f"Entity: {var.entity.key}")  # person, household, tax_unit, spm_unit, etc.
```

### Step 3: Verify Aggregation Works

```python
# For non-household variables, verify totals are preserved
person_total = sim.calculate('your_variable', 2023, map_to='person').values.sum()
household_total = sim.calculate('your_variable', 2023, map_to='household').values.sum()
print(f"Match: {np.isclose(person_total, household_total, rtol=1e-6)}")
```

### Step 4: Update the Runner Script

Edit `build_calibration_matrix.py` to include your new target:

```python
targets_df, X_sparse, household_id_mapping = builder.build_matrix(
    sim,
    target_filter={
        "stratum_group_ids": [4],  # SNAP
        "variables": [
            "health_insurance_premiums_without_medicare_part_b",
            "your_new_variable",  # Add here
        ],
    },
)
```

### Step 5: Run and Verify

```bash
cd policyengine_us_data/datasets/cps/local_area_calibration
python build_calibration_matrix.py
```

## When Code Changes ARE Needed

The current implementation may need modification for:

1. **Count variables with special semantics**: Variables ending in `_count` might need `.nunique()` instead of `.sum()` for aggregation. The junkyard implementation handles this but our current builder doesn't.

2. **Variables with state-specific calculations**: SNAP and Medicaid are already handled (state_fips is set before calculation). Other state-dependent variables should work the same way.

3. **Constraint evaluation at non-household level**: Currently all constraints are evaluated at household level after aggregation. If you need person-level constraint evaluation (e.g., "only count persons with income > X"), the junkyard has this pattern but our builder doesn't.

## The Junkyard Reference

Location: `~/devl/policyengine-us-data/policyengine_us_data/datasets/cps/local_area_calibration/metrics_matrix_geo_stacking_sparse.py`

This 2,400+ line file has extensive logic we intentionally avoided:
- Hard-coded variable names and stratum_group_ids
- Complex entity relationship tracking
- Person-level constraint evaluation with `.any()` aggregation

**Key pattern from junkyard (if ever needed):**
```python
# Dynamic entity detection
target_entity = sim.tax_benefit_system.variables[target_variable].entity.key

# Entity relationship DataFrame
entity_rel = pd.DataFrame({
    "person_id": sim.calculate("person_id", map_to="person").values,
    "household_id": sim.calculate("household_id", map_to="person").values,
    "tax_unit_id": sim.calculate("tax_unit_id", map_to="person").values,
    # ... other entities
})

# For counts: use .nunique() on entity IDs
# For amounts: use .sum() on values
```

## Matrix Structure

The sparse matrix X has shape `(n_targets, n_households × n_cds)`:

```
Columns: [CD1_hh0, CD1_hh1, ..., CD1_hhN, CD2_hh0, ..., CDM_hhN]
Rows: One per target (geographic_id + variable combination)

Column index formula: col_idx = cd_idx * n_households + hh_idx
```

Use `MatrixTracer` for debugging:
```python
from matrix_tracer import MatrixTracer

tracer = MatrixTracer(targets_df, X_sparse, household_id_mapping, cds_to_calibrate, sim)
tracer.print_matrix_structure()
tracer.get_column_info(100)  # Info about column 100
tracer.get_row_info(0)       # Info about row 0 (first target)
```

## Environment Setup

```bash
# Use the sep environment for this repo
source ~/envs/sep/bin/activate

# Run from the local_area_calibration directory
cd ~/devl/sep/policyengine-us-data/policyengine_us_data/datasets/cps/local_area_calibration

# Run tests
pytest ../../tests/test_sparse_matrix_builder.py -v
```

## Common Queries

### List all target variables
```sql
SELECT DISTINCT variable FROM targets;
```

### List all constraint variables
```sql
SELECT DISTINCT constraint_variable FROM stratum_constraints;
```

### Find targets by geographic level
```sql
-- National targets (no geographic constraints)
SELECT t.* FROM targets t
JOIN strata s ON t.stratum_id = s.stratum_id
WHERE t.stratum_id NOT IN (
    SELECT stratum_id FROM stratum_constraints
    WHERE constraint_variable IN ('state_fips', 'congressional_district_geoid')
);

-- State-level targets
SELECT t.* FROM targets t
WHERE t.stratum_id IN (
    SELECT stratum_id FROM stratum_constraints
    WHERE constraint_variable = 'state_fips'
);
```

## Summary

For most new targets:
1. Verify target exists in `policy_data.db`
2. Add variable name to the target filter in `build_calibration_matrix.py`
3. Run and verify with `MatrixTracer`

No code changes to `sparse_matrix_builder.py` needed unless you have special aggregation or constraint requirements.
