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
| 1 | `create_database_tables.py` | No | Creates empty SQLite schema (3 tables + 2 views) |
| 2 | `create_initial_strata.py` | Census ACS 5-year | Builds geographic hierarchy: US > 51 states > 436 CDs |
| 3 | `etl_national_targets.py` | No | Loads ~40 hardcoded national targets (CBO, Treasury, CMS) |
| 4 | `etl_age.py` | Census ACS 1-year | Age distribution: 18 bins x 488 geographies |
| 5 | `etl_medicaid.py` | Census ACS + CMS | Medicaid enrollment (admin state-level, survey district-level) |
| 6 | `etl_snap.py` | USDA FNS + Census ACS | SNAP participation (admin state-level, survey district-level) |
| 7 | `etl_state_income_tax.py` | No | State income tax collections (Census STC FY2023, hardcoded) |
| 8 | `etl_irs_soi.py` | IRS | Tax variables, EITC by child count, AGI brackets, conditional strata |
| 9 | `validate_database.py` | No | Checks all target variables exist in policyengine-us |

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
- `definition_hash`: SHA-256 of constraints for deduplication

**stratum_constraints** - Rules defining each stratum
- `constraint_variable`: Variable name (e.g., `age`, `state_fips`)
- `operation`: Comparison operator (`==`, `!=`, `>`, `>=`, `<`, `<=`)
- `value`: String-encoded value

**targets** - Calibration data values
- `variable`: PolicyEngine US variable name (e.g., `eitc`, `income_tax`)
- `period`: Year
- `value`: Numerical value
- `active`: Boolean flag

### SQL Views

**stratum_domain** - Derives the "domain variable" for each stratum by looking at its non-geographic constraints. Geographic constraints (`state_fips`, `congressional_district_geoid`, `tax_unit_is_filer`, `ucgid_str`) are excluded.

```sql
SELECT * FROM stratum_domain WHERE stratum_id = 123;
-- Returns: stratum_id=123, domain_variable='age'
```

**target_overview** - Flattened view of all targets with geographic level, geographic ID, and domain variable derived from constraints. This is the primary query path for calibration code.

```sql
SELECT * FROM target_overview
WHERE domain_variable = 'snap' AND geo_level = 'state';
-- Returns: target_id, stratum_id, variable, value, period,
--          active, geo_level, geographic_id, domain_variable
```

## Key Concepts

### Stratum Domains (replacing stratum_group_id)

Strata are categorized by their **constraints**, not by a separate group ID field. The `stratum_domain` view derives each stratum's domain from its non-geographic constraints:

| Domain Variable | Description |
|----------------|-------------|
| *(none)* | Geographic strata (US, states, congressional districts) |
| `age` | Age distribution brackets |
| `adjusted_gross_income` | Income/AGI brackets |
| `snap` | SNAP recipient strata |
| `medicaid_enrolled` | Medicaid enrollment strata |
| `eitc_child_count` | EITC recipients by qualifying children |
| `state_income_tax` | State-level income tax collections |
| `aca_ptc` | ACA Premium Tax Credit strata |
| Various IRS variables | Each IRS variable with conditional count constraints |

### Conditional Strata (IRS SOI)

IRS variables use a "filer population" intermediate layer and conditional strata:

```
Geographic stratum (no domain constraints)
  └── Tax Filer stratum (constraint: tax_unit_is_filer==1)
       ├── AGI bracket strata (constraint: adjusted_gross_income range)
       ├── EITC by child count (constraint: eitc_child_count)
       └── IRS variable strata (constraint: variable > 0)
            - Targets: tax_unit_count + variable amount
```

Each IRS variable (e.g., `rental_income`, `self_employment_income`) gets its own stratum with a constraint requiring that variable > 0. This captures both the count of filers with that income type and the total amount.

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

ETL scripts that pull Census data receive UCGIDs and create their own domain-specific strata with `ucgid_str` constraints. The geographic hierarchy strata use `state_fips`/`congressional_district_geoid` instead.

### Constraint Operations

All constraints use standardized operators validated by the `ConstraintOperation` enum:
`==`, `!=`, `>`, `>=`, `<`, `<=`

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

**`policyengine_us_data/utils/raw_cache.py`**:
- `is_cached(filename)` - Check if a raw input is cached
- `save_json(filename, data)` / `load_json(filename)` - Cache JSON data
- `save_bytes(filename, data)` / `load_bytes(filename)` - Cache binary data

## Example Queries

### Count strata by domain
```sql
SELECT
    COALESCE(sd.domain_variable, '(geographic)') AS domain,
    COUNT(DISTINCT s.stratum_id) AS stratum_count
FROM strata s
LEFT JOIN stratum_domain sd ON s.stratum_id = sd.stratum_id
GROUP BY domain
ORDER BY domain;
```

### Get targets for a specific state via target_overview
```sql
SELECT variable, value, period, geo_level, geographic_id, domain_variable
FROM target_overview
WHERE geographic_id = '37'
  AND geo_level = 'state'
ORDER BY variable;
```

### Get targets for a specific state (join-based)
```sql
SELECT t.variable, t.value, t.period, s.notes
FROM targets t
JOIN strata s ON t.stratum_id = s.stratum_id
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE sc.constraint_variable = 'state_fips'
  AND sc.value = '37'
ORDER BY t.variable;
```

### Get all SNAP targets at district level
```sql
SELECT variable, value, geographic_id, period
FROM target_overview
WHERE domain_variable = 'snap'
  AND geo_level = 'district'
ORDER BY geographic_id;
```

## Database Location

`policyengine_us_data/storage/calibration/policy_data.db`

Downloaded from HuggingFace by `download_private_prerequisites.py` and `download_calibration_inputs()` in `utils/huggingface.py`.
