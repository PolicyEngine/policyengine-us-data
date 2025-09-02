# PolicyEngine US Data - Database Getting Started Guide

## Current Task: Matrix Generation for Calibration Targets

### Objective
Create a comprehensive matrix of calibration targets with the following requirements:
1. **Rows grouped by target type** - All age targets together, all income targets together, etc.
2. **Known counts per group** - Each group has a predictable number of entries (e.g., 18 age groups, 9 income brackets)
3. **Source selection** - Ability to specify which data source to use when multiple exist
4. **Geographic filtering** - Ability to select specific geographic levels (national, state, or congressional district)

### Implementation Strategy
The `stratum_group_id` field now categorizes strata by conceptual type, making matrix generation straightforward:
- Query by `stratum_group_id` to get all related targets together
- Each demographic group appears consistently across all 488 geographic areas
- Join with `sources` table to filter/identify data provenance
- Use parent-child relationships to navigate geographic hierarchy

### Example Matrix Query
```sql
-- Generate matrix for a specific geography (e.g., national level)
SELECT 
    CASE s.stratum_group_id
        WHEN 2 THEN 'Age'
        WHEN 3 THEN 'Income'
        WHEN 4 THEN 'SNAP'
        WHEN 5 THEN 'Medicaid'
        WHEN 6 THEN 'EITC'
    END AS group_name,
    s.notes AS stratum_description,
    t.variable,
    t.value,
    src.name AS source
FROM strata s
JOIN targets t ON s.stratum_id = t.stratum_id
JOIN sources src ON t.source_id = src.source_id
WHERE s.parent_stratum_id = 1  -- National level (or any specific geography)
    AND s.stratum_group_id > 1  -- Exclude geographic strata
ORDER BY s.stratum_group_id, s.stratum_id;
```

## Overview
This database uses a hierarchical stratum-based model to organize US demographic and economic data for PolicyEngine calibration. The core concept is that data is organized into "strata" - population subgroups defined by constraints.

## Key Concepts

### Strata Hierarchy
The database uses a parent-child hierarchy:
```
United States (national)
├── States (51 including DC)
│   ├── Congressional Districts (436 total)
│   │   ├── Age groups (18 brackets per geographic area)
│   │   ├── Income groups (AGI stubs)
│   │   └── Other demographic strata (EITC recipients, SNAP, Medicaid, etc.)
```

### Stratum Groups
The `stratum_group_id` field categorizes strata by their conceptual type:
- `1`: Geographic boundaries (US, states, congressional districts)
- `2`: Age-based strata (18 age groups per geography)
- `3`: Income/AGI-based strata (9 income brackets per geography)
- `4`: SNAP recipient strata (1 per geography)
- `5`: Medicaid enrollment strata (1 per geography)
- `6`: EITC recipient strata (4 groups by qualifying children per geography)

### UCGID Translation
The Census Bureau uses UCGIDs (Universal Census Geographic IDs) in their API responses:
- `0100000US`: National level
- `0400000USXX`: State (XX = state FIPS code)
- `5001800USXXDD`: Congressional district (XX = state FIPS, DD = district number)

We parse these into our internal model using `state_fips` and `congressional_district_geoid`.

### Constraint Operations
All constraints use standardized operators:
- `==`: Equals
- `!=`: Not equals
- `>`: Greater than
- `>=`: Greater than or equal
- `<`: Less than
- `<=`: Less than or equal

## Database Structure

### Core Tables
1. **strata**: Main table for population subgroups
   - `stratum_id`: Primary key
   - `parent_stratum_id`: Links to parent in hierarchy
   - `stratum_group_id`: Conceptual category (1=Geographic, 2=Age, 3=Income, 4=SNAP, 5=Medicaid, 6=EITC)
   - `definition_hash`: Unique hash of constraints for deduplication

2. **stratum_constraints**: Defines rules for each stratum
   - `constraint_variable`: Variable name (e.g., "age", "state_fips")
   - `operation`: Comparison operator (==, >, <, etc.)
   - `value`: Constraint value

3. **targets**: Stores actual data values
   - `variable`: PolicyEngine US variable name
   - `period`: Year
   - `value`: Numerical value
   - `source_id`: Foreign key to sources table
   - `active`: Boolean flag for active/inactive targets
   - `tolerance`: Allowed relative error percentage

### Metadata Tables (New)
4. **sources**: Data source metadata
   - `source_id`: Primary key (auto-generated)
   - `name`: Source name (e.g., "IRS Statistics of Income")
   - `type`: SourceType enum (administrative, survey, hardcoded)
   - `vintage`: Year or version of data
   - `description`: Detailed description
   - `url`: Reference URL
   - `notes`: Additional notes

5. **variable_groups**: Logical groupings of related variables
   - `group_id`: Primary key (auto-generated)
   - `name`: Unique group name (e.g., "age_distribution", "snap_recipients")
   - `category`: High-level category (demographic, benefit, tax, income, expense)
   - `is_histogram`: Whether this represents a distribution
   - `is_exclusive`: Whether variables are mutually exclusive
   - `aggregation_method`: How to aggregate (sum, weighted_avg, etc.)
   - `display_order`: Order for display in matrices/reports
   - `description`: What this group represents

6. **variable_metadata**: Display information for variables
   - `metadata_id`: Primary key
   - `variable`: PolicyEngine variable name
   - `group_id`: Foreign key to variable_groups
   - `display_name`: Human-readable name
   - `display_order`: Order within group
   - `units`: Units of measurement (dollars, count, percent)
   - `is_primary`: Whether this is a primary vs derived variable
   - `notes`: Additional notes

## Building the Database

### Step 1: Create Tables
```bash
source ~/envs/pe/bin/activate
cd policyengine_us_data/db
python create_database_tables.py
```

### Step 2: Create Geographic Hierarchy
```bash
python create_initial_strata.py
```
Creates: 1 national + 51 state + 436 congressional district strata

### Step 3: Load Data (in order)
```bash
# Age demographics (Census ACS)
python etl_age.py

# Economic data (IRS SOI)
python etl_irs_soi.py

# Benefits data
python etl_medicaid.py
python etl_snap.py

# National hardcoded targets
python etl_national_targets.py
```

### Step 4: Validate
```bash
python validate_hierarchy.py
```

Expected output:
- 488 geographic strata
- 8,784 age strata (18 age groups × 488 areas)
- All strata have unique definition hashes

## Common Utility Functions

Located in `policyengine_us_data/utils/db.py`:

- `parse_ucgid(ucgid_str)`: Convert Census UCGID to geographic info
- `get_geographic_strata(session)`: Get mapping of geographic strata IDs
- `get_stratum_by_id(session, id)`: Retrieve stratum by ID
- `get_stratum_children(session, id)`: Get child strata
- `get_stratum_parent(session, id)`: Get parent stratum

## ETL Script Pattern

Each ETL script follows this pattern:

1. **Extract**: Pull data from source (Census API, IRS files, etc.)
2. **Transform**: 
   - Parse UCGIDs to get geographic info
   - Map to existing geographic strata
   - Create demographic strata as children
3. **Load**:
   - Check for existing strata to avoid duplicates
   - Add constraints and targets
   - Commit to database

## Important Notes

### Avoiding Duplicates
Always check if a stratum exists before creating:
```python
existing_stratum = session.exec(
    select(Stratum).where(
        Stratum.parent_stratum_id == parent_id,
        Stratum.stratum_group_id == group_id,  # Use appropriate group_id (2 for age, 3 for income, etc.)
        Stratum.notes == note
    )
).first()
```

### Geographic Constraints
- National strata: No geographic constraints needed
- State strata: `state_fips` constraint
- District strata: `congressional_district_geoid` constraint

### Congressional District Normalization
- District 00 → 01 (at-large districts)
- DC district 98 → 01 (delegate district)

### IRS AGI Ranges
AGI stubs use >= for lower bound, < for upper bound:
- Stub 3: $10,000 <= AGI < $25,000
- Stub 4: $25,000 <= AGI < $50,000
- etc.

## Troubleshooting

### "WARNING: Expected 8784 age strata, found 16104"
**Status: RESOLVED**

The validation script was incorrectly counting all demographic strata (stratum_group_id = 0) as age strata. After implementing the new stratum_group_id scheme (1=Geographic, 2=Age, 3=Income, etc.), the validation correctly identifies 8,784 age strata.

Expected: 8,784 age strata (18 age groups × 488 geographic areas)
Actual: 8,784 age strata

**RESOLVED**: Fixed validation script to only count strata with "Age" in notes, not all demographic strata

### Fixed: Synthetic Variable Names
Previously, the IRS SOI ETL was creating invalid variable names like `eitc_tax_unit_count` that don't exist in PolicyEngine. Now correctly uses `tax_unit_count` with appropriate stratum constraints to indicate what's being counted.

### UCGID strings in notes
Legacy UCGID references have been replaced with human-readable identifiers:
- "US" for national
- "State FIPS X" for states
- "CD XXXX" for congressional districts

### Mixed operation types
All operations now use standardized symbols (==, >, <, etc.) validated by ConstraintOperation enum.

## Database Location
`/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db`

## Example SQLite Queries with New Metadata Features

### Compare Administrative vs Survey Data for SNAP
```sql
-- Compare SNAP household counts from different source types
SELECT 
    s.type AS source_type,
    s.name AS source_name,
    st.notes AS location,
    t.value AS household_count
FROM targets t
JOIN sources s ON t.source_id = s.source_id
JOIN strata st ON t.stratum_id = st.stratum_id
WHERE t.variable = 'household_count'
    AND st.notes LIKE '%SNAP%'
ORDER BY s.type, st.notes;
```

### Get All Variables in a Group with Their Metadata
```sql
-- List all EITC-related variables with their display information
SELECT 
    vm.display_name,
    vm.variable,
    vm.units,
    vm.display_order,
    vg.description AS group_description
FROM variable_metadata vm
JOIN variable_groups vg ON vm.group_id = vg.group_id
WHERE vg.name = 'eitc_recipients'
ORDER BY vm.display_order;
```

### Create a Matrix of Benefit Programs by Source Type
```sql
-- Show all benefit programs with admin vs survey values at national level
SELECT 
    vg.name AS benefit_program,
    vm.variable,
    vm.display_name,
    SUM(CASE WHEN s.type = 'administrative' THEN t.value END) AS admin_value,
    SUM(CASE WHEN s.type = 'survey' THEN t.value END) AS survey_value
FROM variable_groups vg
JOIN variable_metadata vm ON vg.group_id = vm.group_id
LEFT JOIN targets t ON vm.variable = t.variable AND t.stratum_id = 1
LEFT JOIN sources s ON t.source_id = s.source_id
WHERE vg.category = 'benefit'
GROUP BY vg.name, vm.variable, vm.display_name
ORDER BY vg.display_order, vm.display_order;
```

### Find All Data from IRS SOI Source
```sql
-- List all variables and values from IRS Statistics of Income
SELECT 
    t.variable,
    vm.display_name,
    t.value / 1e9 AS value_billions,
    vm.units
FROM targets t
JOIN sources s ON t.source_id = s.source_id
LEFT JOIN variable_metadata vm ON t.variable = vm.variable
WHERE s.name = 'IRS Statistics of Income'
    AND t.stratum_id = 1  -- National totals
ORDER BY t.value DESC;
```

### Analyze Data Coverage by Source Type
```sql
-- Show data point counts and geographic coverage by source type
SELECT 
    s.type AS source_type,
    COUNT(DISTINCT t.target_id) AS total_targets,
    COUNT(DISTINCT t.variable) AS unique_variables,
    COUNT(DISTINCT st.stratum_id) AS geographic_coverage,
    s.name AS source_name,
    s.vintage
FROM sources s
LEFT JOIN targets t ON s.source_id = t.source_id
LEFT JOIN strata st ON t.stratum_id = st.stratum_id
GROUP BY s.source_id, s.type, s.name, s.vintage
ORDER BY s.type, total_targets DESC;
```

### Find Variables That Appear in Multiple Sources
```sql
-- Identify variables with both administrative and survey data
SELECT 
    t.variable,
    vm.display_name,
    GROUP_CONCAT(DISTINCT s.type) AS source_types,
    COUNT(DISTINCT s.source_id) AS source_count
FROM targets t
JOIN sources s ON t.source_id = s.source_id
LEFT JOIN variable_metadata vm ON t.variable = vm.variable
GROUP BY t.variable, vm.display_name
HAVING COUNT(DISTINCT s.type) > 1
ORDER BY source_count DESC;
```

### Show Variable Group Hierarchy
```sql
-- Display all variable groups with their categories and metadata
SELECT 
    vg.display_order,
    vg.category,
    vg.name,
    vg.description,
    CASE WHEN vg.is_histogram THEN 'Yes' ELSE 'No' END AS is_histogram,
    vg.aggregation_method,
    COUNT(vm.variable) AS variable_count
FROM variable_groups vg
LEFT JOIN variable_metadata vm ON vg.group_id = vm.group_id
GROUP BY vg.group_id
ORDER BY vg.display_order;
```

### Audit Query: Find Variables Without Metadata
```sql
-- Identify variables in targets that lack metadata entries
SELECT DISTINCT 
    t.variable,
    COUNT(*) AS usage_count,
    GROUP_CONCAT(DISTINCT s.name) AS sources_using
FROM targets t
LEFT JOIN variable_metadata vm ON t.variable = vm.variable
LEFT JOIN sources s ON t.source_id = s.source_id
WHERE vm.metadata_id IS NULL
GROUP BY t.variable
ORDER BY usage_count DESC;
```

### Query by Stratum Group
```sql
-- Get all age-related strata and their targets
SELECT 
    s.stratum_id,
    s.notes,
    t.variable,
    t.value,
    src.name AS source
FROM strata s
JOIN targets t ON s.stratum_id = t.stratum_id
JOIN sources src ON t.source_id = src.source_id
WHERE s.stratum_group_id = 2  -- Age strata
LIMIT 20;

-- Count strata by group
SELECT 
    stratum_group_id,
    CASE stratum_group_id
        WHEN 1 THEN 'Geographic'
        WHEN 2 THEN 'Age'
        WHEN 3 THEN 'Income/AGI'
        WHEN 4 THEN 'SNAP'
        WHEN 5 THEN 'Medicaid'
        WHEN 6 THEN 'EITC'
    END AS group_name,
    COUNT(*) AS stratum_count
FROM strata
GROUP BY stratum_group_id
ORDER BY stratum_group_id;
```

## Key Improvements Made
1. Removed UCGID as a constraint variable (legacy Census concept)
2. Standardized constraint operations with validation
3. Consolidated duplicate code (parse_ucgid, get_geographic_strata)
4. Fixed epsilon hack in IRS AGI ranges
5. ~~Added proper duplicate checking in age ETL (still has known bug causing duplicates)~~ **RESOLVED**
6. Improved human-readable notes without UCGID strings
7. **NEW: Added metadata tables for sources, variable groups, and variable metadata**
8. **NEW: Fixed synthetic variable name bug (e.g., eitc_tax_unit_count → tax_unit_count)**
9. **NEW: Auto-generated source IDs instead of hardcoding**
10. **NEW: Proper categorization of admin vs survey data for same concepts**
11. **NEW: Implemented conceptual stratum_group_id scheme for better organization and querying**