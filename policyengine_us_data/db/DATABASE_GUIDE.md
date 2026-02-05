# PolicyEngine US Data - Database Guide

## Overview

This database uses a hierarchical stratum-based model to organize US demographic and economic data for PolicyEngine calibration. Data is organized into "strata" - population subgroups defined by constraints - with calibration targets attached to each stratum.

The database is a **compiled artifact**: built locally from government data sources, validated, and promoted to HuggingFace for consumption by downstream pipelines.

## Building the Database

### Quick Start
```bash
source ~/envs/sep/bin/activate
cd ~/devl/sep/policyengine-us-data

make database           # Build (uses cached downloads if available)
make database-refresh   # Force re-download all sources and rebuild
make promote-database   # Copy DB + raw inputs to HuggingFace clone
```

### Pipeline Stages

`make database` runs these scripts sequentially:

| # | Script | Network? | What it does |
|---|--------|----------|--------------|
| 1 | `create_database_tables.py` | No | Creates SQLite schema (8 tables) + validation triggers |
| 2 | `create_field_valid_values.py` | No | Populates field_valid_values with allowed values |
| 3 | `create_initial_strata.py` | Census ACS 5-year | Builds geographic hierarchy: US > 51 states > 436 CDs |
| 4 | `etl_national_targets.py` | No | Loads ~40 hardcoded national targets (CBO, Treasury, CMS) |
| 5 | `etl_age.py` | Census ACS 1-year | Age distribution: 18 bins x 488 geographies |
| 6 | `etl_medicaid.py` | Census ACS + CMS | Medicaid enrollment (admin state-level, survey district-level) |
| 7 | `etl_snap.py` | USDA FNS + Census ACS | SNAP participation (admin state-level, survey district-level) |
| 8 | `etl_state_income_tax.py` | No | State income tax collections (Census STC FY2023, hardcoded) |
| 9 | `etl_irs_soi.py` | IRS | Tax variables, EITC by child count, AGI brackets, conditional strata |
| 10 | `validate_database.py` | No | Checks all target variables exist in policyengine-us |

### Raw Input Caching

All network downloads are cached in `storage/calibration/raw_inputs/`. On subsequent runs, cached files are used instead of hitting external APIs. This decouples extraction from transformation so you can iterate on ETL logic without network access.

Set `PE_REFRESH_RAW=1` to force re-download:
```bash
PE_REFRESH_RAW=1 make database
```

### Promotion to HuggingFace

After building and validating:
```bash
make promote-database
cd ~/devl/huggingface/policyengine-us-data
git add calibration/policy_data.db calibration/raw_inputs/
git commit -m "Update policy_data.db - <description>"
git push
```

This copies both the database and the raw inputs that built it, preserving provenance in the HF repo's git history.

### Recovery

If a step fails mid-pipeline, delete the database and re-run. With cached downloads this takes ~10-15 minutes:
```bash
rm -f policyengine_us_data/storage/calibration/policy_data.db
make database
```

## Database Schema

### Core Tables

**strata** - Population subgroups
- `stratum_id`: Auto-generated primary key
- `parent_stratum_id`: Links to parent in hierarchy
- `stratum_group_id`: Conceptual category (see below)
- `definition_hash`: SHA-256 of constraints for deduplication

**stratum_constraints** - Rules defining each stratum
- `constraint_variable`: Variable name (e.g., `age`, `state_fips`)
- `operation`: Comparison operator (`==`, `!=`, `>`, `>=`, `<`, `<=`)
- `value`: String-encoded value

**targets** - Calibration data values
- `variable`: PolicyEngine US variable name (e.g., `eitc`, `income_tax`)
- `period`: Year
- `value`: Numerical value
- `source_id`: Foreign key to sources table
- `active`: Boolean flag

### Metadata Tables

**sources** - Data provenance (e.g., "Census ACS", "IRS SOI", "CBO")

**variable_groups** - Logical groupings (e.g., "age_distribution", "snap_recipients")

**variable_metadata** - Display info for variables (display name, units, ordering)

### Validation Table

**field_valid_values** - Centralized registry of valid values for semantic fields

This table is the source of truth for what values are allowed in specific fields throughout
the database. Expecifically those that deal with semantic external information rather than designing relationships inherent to teh database itself. SQL triggers enforce validation on INSERT and UPDATE operations.

| Field Validated | Table | Valid Values |
|-----------------|-------|--------------|
| `operation` | stratum_constraints | `==`, `!=`, `>`, `>=`, `<`, `<=` |
| `constraint_variable` | stratum_constraints | All policyengine-us variables |
| `active` | targets | `0`, `1` |
| `period` | targets | `2022`, `2023`, `2024`, `2025` |
| `variable` | targets | All policyengine-us variables |
| `type` | sources | `administrative`, `survey`, `synthetic`, `derived`, `hardcoded` |

**Triggers**: `validate_stratum_constraints_insert`, `validate_stratum_constraints_update`,
`validate_targets_insert`, `validate_targets_update`, `validate_sources_insert`, `validate_sources_update`

To add a new valid value (e.g., a new year):
```sql
INSERT INTO field_valid_values (field_name, valid_value, description)
VALUES ('period', '2026', NULL);
```

To check what values are valid for a field:
```sql
SELECT valid_value, description FROM field_valid_values WHERE field_name = 'operation';
```

## Key Concepts

### Stratum Groups

The `stratum_group_id` field categorizes strata:

| ID | Category | Description |
|----|----------|-------------|
| 0 | Uncategorized | Legacy strata not yet assigned a group |
| 1 | Geographic | US, states, congressional districts |
| 2 | Age/Filer population | Age brackets, tax filer intermediate strata |
| 3 | Income/AGI | 9 income brackets per geography |
| 4 | SNAP | SNAP recipient strata |
| 5 | Medicaid | Medicaid enrollment strata |
| 6 | EITC | EITC recipients by qualifying children |
| 7 | State Income Tax | State-level income tax collections (Census STC) |
| 100-118 | IRS Conditional | Each IRS variable paired with conditional count constraints |

### Conditional Strata (IRS SOI)

IRS variables use a "filer population" intermediate layer and conditional strata:

```
Geographic stratum (group_id=1)
  └── Tax Filer stratum (group_id=2, constraint: tax_unit_is_filer==1)
       ├── AGI bracket strata (group_id=3, constraint: AGI range)
       ├── EITC by child count (group_id=6, constraint: eitc_child_count)
       └── IRS variable strata (group_id=100+, constraint: variable > 0)
            - Targets: tax_unit_count + variable amount
```

Each IRS variable (e.g., `rental_income`, `self_employment_income`) gets its own stratum_group_id (100+) with a constraint requiring that variable > 0. This captures both the count of filers with that income type and the total amount.

### Geographic Hierarchy

```
United States (no constraints)
  ├── Alabama (state_fips == 1)
  │   ├── AL-01 (congressional_district_geoid == 101)
  │   ├── AL-02 (congressional_district_geoid == 102)
  │   └── ...
  ├── Alaska (state_fips == 2)
  │   └── AK-01 (congressional_district_geoid == 201)
  └── ...
```

Geographic strata use `state_fips` and `congressional_district_geoid` constraints (not UCGIDs). The `parse_ucgid()` and `get_geographic_strata()` functions in `utils/db.py` bridge between Census UCGID strings and these internal identifiers.

### UCGID Translation

Census Bureau API responses use UCGIDs (Universal Census Geographic IDs):
- `0100000US` = National
- `0400000USXX` = State (XX = state FIPS)
- `5001800USXXDD` = Congressional district (XX = state FIPS, DD = district number)

ETL scripts that pull Census data receive UCGIDs and create their own domain-specific strata with `ucgid_str` constraints. The geographic hierarchy strata (stratum_group_id=1) use `state_fips`/`congressional_district_geoid` instead.

### Constraint Operations

All constraints use standardized operators validated by the `field_valid_values` table:
`==`, `!=`, `>`, `>=`, `<`, `<=`

### Constraint Validation

ETL scripts validate constraint sets before inserting them into the database using `ensure_consistent_constraint_set()` from `policyengine_us_data.utils.constraint_validation`. This prevents logically inconsistent constraints from being stored.

**Validation Rules:**

1. **Operation Compatibility** (per constraint_variable):

| Operation | Can combine with | Rationale |
|-----------|-----------------|-----------|
| `==` | Nothing (must be alone) | Equality is absolute |
| `!=` | Nothing (must be alone) | Exclusion is absolute |
| `>` | `<` or `<=` only | Forms valid range |
| `>=` | `<` or `<=` only | Forms valid range |
| `<` | `>` or `>=` only | Forms valid range |
| `<=` | `>` or `>=` only | Forms valid range |

**Invalid combinations:**
- `>` with `>=` (redundant/conflicting lower bounds)
- `<` with `<=` (redundant/conflicting upper bounds)
- `==` with anything else
- `!=` with anything else

2. **Value Checks** (if operations are compatible):
- No empty ranges: lower bound must be < upper bound
- For equal bounds, both must be inclusive (`>=` and `<=`) to be valid

**Usage in ETL:**
```python
from policyengine_us_data.utils.constraint_validation import (
    Constraint,
    ensure_consistent_constraint_set,
)

# Build constraint list
constraint_list = [
    Constraint(variable="age", operation=">=", value="25"),
    Constraint(variable="age", operation="<", value="30"),
]

# Validate before creating StratumConstraint objects
ensure_consistent_constraint_set(constraint_list)

# Now safe to add to stratum
stratum.constraints_rel = [
    StratumConstraint(
        constraint_variable=c.variable,
        operation=c.operation,
        value=c.value,
    )
    for c in constraint_list
]
```

### Constraint Value Types

The `value` column stores all values as strings. Downstream code deserializes:
- Numeric strings -> int/float (age, income)
- `"True"`/`"False"` -> booleans (medicaid_enrolled)
- Other strings stay as strings (state_fips with leading zeros)

## Important Warnings

### stratum_id != FIPS Code

The `stratum_id` is auto-generated and has **no relationship** to FIPS codes:
- California: stratum_id=6, state_fips="06" (coincidental!)
- North Carolina: stratum_id=35, state_fips="37" (no match)

Always look up strata by constraint values:
```sql
SELECT s.stratum_id, s.notes
FROM strata s
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE sc.constraint_variable = 'state_fips'
  AND sc.value = '37';
```

### IRS SOI A59664 Data Issue

The IRS SOI column A59664 (EITC with 3+ children amount) is reported in dollars, not thousands like all other monetary columns. The ETL code detects and compensates for this. See `IRS_SOI_DATA_ISSUE.md` for details.

## Utility Functions

**`policyengine_us_data/utils/db.py`**:
- `get_stratum_by_id(session, id)` - Retrieve stratum by primary key
- `get_simple_stratum_by_ucgid(session, ucgid)` - Find stratum with single ucgid_str constraint
- `get_root_strata(session)` - Get strata with no parent
- `get_stratum_children(session, id)` / `get_stratum_parent(session, id)` - Navigate hierarchy
- `parse_ucgid(ucgid_str)` - Parse UCGID to type/state_fips/district info
- `get_geographic_strata(session)` - Map of all geographic strata by type

**`policyengine_us_data/utils/db_metadata.py`**:
- `get_or_create_source(session, ...)` - Upsert data source metadata
- `get_or_create_variable_group(session, ...)` - Upsert variable group
- `get_or_create_variable_metadata(session, ...)` - Upsert variable display info

**`policyengine_us_data/utils/raw_cache.py`**:
- `is_cached(filename)` - Check if a raw input is cached
- `save_json(filename, data)` / `load_json(filename)` - Cache JSON data
- `save_bytes(filename, data)` / `load_bytes(filename)` - Cache binary data

## Example Queries

### Count strata by group
```sql
SELECT
    stratum_group_id,
    CASE stratum_group_id
        WHEN 0 THEN 'Uncategorized'
        WHEN 1 THEN 'Geographic'
        WHEN 2 THEN 'Age/Filer'
        WHEN 3 THEN 'Income/AGI'
        WHEN 4 THEN 'SNAP'
        WHEN 5 THEN 'Medicaid'
        WHEN 6 THEN 'EITC'
        WHEN 7 THEN 'State Income Tax'
    END AS group_name,
    COUNT(*) AS stratum_count
FROM strata
GROUP BY stratum_group_id
ORDER BY stratum_group_id;
```

### Get targets for a specific state
```sql
SELECT t.variable, t.value, t.period, s.notes
FROM targets t
JOIN strata s ON t.stratum_id = s.stratum_id
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE sc.constraint_variable = 'state_fips'
  AND sc.value = '37'
ORDER BY t.variable;
```

### Compare admin vs survey data sources
```sql
SELECT
    src.type AS source_type,
    src.name AS source_name,
    st.notes AS location,
    t.value
FROM targets t
JOIN sources src ON t.source_id = src.source_id
JOIN strata st ON t.stratum_id = st.stratum_id
WHERE t.variable = 'household_count'
    AND st.notes LIKE '%SNAP%'
ORDER BY src.type, st.notes;
```

## Database Location

`policyengine_us_data/storage/calibration/policy_data.db`

Downloaded from HuggingFace by `download_private_prerequisites.py` and `download_calibration_inputs()` in `utils/huggingface.py`.
