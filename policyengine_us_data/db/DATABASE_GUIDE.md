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

### Metadata Tables
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
source ~/envs/sep/bin/activate
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
# National hardcoded targets
python etl_national_targets.py

# Age demographics (Census ACS)
python etl_age.py

# Economic data (IRS SOI)
python etl_irs_soi.py

# Benefits data
python etl_medicaid.py
python etl_snap.py
```

### Step 4: Validate
```bash
python validate_database.py
```

Expected output:
- 488 geographic strata
- 8,784 age strata (18 age groups × 488 areas)
- All strata have unique definition hashes

## Common Utility Functions

Located in `policyengine_us_data/utils/db.py`:

- `get_stratum_by_id(session, id)`: Retrieve stratum by ID
- `get_simple_stratum_by_ucgid(session, ucgid)`: Get stratum by UCGID
- `get_root_strata(session)`: Get root strata
- `get_stratum_children(session, id)`: Get child strata
- `get_stratum_parent(session, id)`: Get parent stratum

Located in `policyengine_us_data/utils/db_metadata.py`:

- `get_or_create_source(session, ...)`: Get or create a data source
- `get_or_create_variable_group(session, ...)`: Get or create a variable group
- `get_or_create_variable_metadata(session, ...)`: Get or create variable metadata

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
        Stratum.stratum_group_id == group_id,
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
`policyengine_us_data/storage/calibration/policy_data.db`

## Example SQLite Queries with Metadata Features

### Compare Administrative vs Survey Data for SNAP
```sql
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

## Key Improvements
1. Removed UCGID as a constraint variable (legacy Census concept)
2. Standardized constraint operations with validation
3. Consolidated duplicate code (parse_ucgid, get_geographic_strata)
4. Fixed epsilon hack in IRS AGI ranges
5. Added proper duplicate checking in age ETL
6. Improved human-readable notes without UCGID strings
7. Added metadata tables for sources, variable groups, and variable metadata
8. Fixed synthetic variable name bug (e.g., eitc_tax_unit_count → tax_unit_count)
9. Auto-generated source IDs instead of hardcoding
10. Proper categorization of admin vs survey data for same concepts
11. Implemented conceptual stratum_group_id scheme for better organization and querying

## Known Issues / TODOs

### IMPORTANT: stratum_id vs state_fips Codes
**WARNING**: The `stratum_id` is an auto-generated sequential ID and has NO relationship to FIPS codes, despite some confusing coincidences:
- California: stratum_id = 6, state_fips = "06" (coincidental match!)
- North Carolina: stratum_id = 35, state_fips = "37" (no match)
- Ohio: stratum_id = 37, state_fips = "39" (no match)

When querying for states, ALWAYS use the `state_fips` constraint value, never assume stratum_id matches FIPS.

Example of correct lookup:
```sql
-- Find North Carolina's stratum_id by FIPS code
SELECT s.stratum_id, s.notes
FROM strata s
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE sc.constraint_variable = 'state_fips'
  AND sc.value = '37';  -- Returns stratum_id = 35
```

### Type Conversion for Constraint Values
**DESIGN DECISION**: The `value` column in `stratum_constraints` must store heterogeneous data types as strings. The calibration code deserializes these:
- Numeric strings → int/float (for age, income constraints)
- "True"/"False" → Python booleans (for medicaid_enrolled, snap_enrolled)
- Other strings remain strings (for state_fips, which may have leading zeros)

### Medicaid Data Structure
- Medicaid uses `person_count` variable (not `medicaid`) because it's structured as a histogram with constraints
- State-level targets use administrative data (T-MSIS source)
- Congressional district level uses survey data (ACS source)
- No national Medicaid target exists (intentionally, to avoid double-counting when using state-level data)
