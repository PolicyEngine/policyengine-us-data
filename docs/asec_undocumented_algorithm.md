# ASEC Undocumented Algorithm

This document describes the implementation of the ASEC Undocumented Algorithm used to assign SSN card types in the Enhanced CPS dataset, based on the research paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4662801

## Algorithm Overview

The ASEC Undocumented Algorithm implements a process of elimination approach to identify individuals who are likely undocumented immigrants by systematically removing people who have clear indicators of legal immigration status.

The algorithm assigns SSN card type codes:
- **Code 0**: `"NONE"` - Likely undocumented immigrants
- **Code 1**: `"CITIZEN"` - US citizens (born or naturalized)  
- **Code 2**: `"NON_CITIZEN_VALID_EAD"` - Non-citizens with work/study authorization
- **Code 3**: `"OTHER_NON_CITIZEN"` - Non-citizens with indicators of legal status

## Implementation Details

The algorithm is implemented in the `add_ssn_card_type()` function in `policyengine_us_data/datasets/cps/cps.py:848-1305`.

### Function Parameters
- `undocumented_target`: 13 million (default target for total undocumented population)
- `undocumented_workers_target`: 8.3 million (target for undocumented workers)
- `undocumented_students_target`: ~399k (21% of 1.9M total undocumented students)

## Algorithm Steps

### Step 0: Initialization
- All persons start with SSN card type code 0 (`ssn_card_type = np.full(len(person), 0)`)
- Prints initial Code 0 population count

### Step 1: Citizen Classification
Citizens are identified and moved to code 1 based on citizenship status:
- **Condition**: `PRCITSHP` values 1, 2, 3, or 4 (all citizen types)
- **Result**: Citizens assigned to code 1, non-citizens (`PRCITSHP == 5`) remain for further processing

### Step 2: ASEC Undocumented Algorithm Conditions
The algorithm applies 14 specific conditions to identify non-citizens with indicators of legal status. Only individuals not already classified as citizens (code 1) or authorized workers/students (code 2) are evaluated (`potentially_undocumented` mask).

#### Condition 1: Pre-1982 Arrivals (IRCA Amnesty Eligible)
- **Variable**: `PEINUSYR` codes 1-7
- **Logic**: Immigrants who arrived before 1982 were eligible for IRCA amnesty
- **Codes**: 01=Before 1950, 02=1950-1959, 03=1960-1964, 04=1965-1969, 05=1970-1974, 06=1975-1979, 07=1980-1981

#### Condition 2: Eligible Naturalized Citizens
- **Variables**: `PRCITSHP == 4` (naturalized citizen), `A_AGE >= 18`, `PEINUSYR` for residency
- **Logic**: Naturalized citizens meeting residency requirements:
  - 5+ years in US (codes 8-26: 1982-2019), OR
  - 3+ years in US + married to citizen (codes 8-27: 1982-2021)
- **Marriage check**: `A_MARITL` in [1,2] AND `A_SPOUSE > 0`

#### Condition 3: Medicare Recipients
- **Variable**: `MCARE == 1`
- **Logic**: Medicare eligibility indicates legal status

#### Condition 4: Federal Retirement Benefits
- **Variables**: `PEN_SC1 == 3` OR `PEN_SC2 == 3`
- **Logic**: Federal government pension recipients have legal work history

#### Condition 5: Social Security Disability
- **Variables**: `RESNSS1 == 2` OR `RESNSS2 == 2`
- **Logic**: Social Security disability benefits indicate legal status (code 2 = disabled adult or child)

#### Condition 6: Indian Health Service Coverage
- **Variable**: `IHSFLG == 1`
- **Logic**: IHS coverage indicates legal status or tribal membership

#### Condition 7: Medicaid Recipients
- **Variable**: `CAID == 1`
- **Logic**: Medicaid eligibility generally requires legal status (simplified implementation without state-specific adjustments)

#### Condition 8: CHAMPVA Recipients
- **Variable**: `CHAMPVA == 1`
- **Logic**: CHAMPVA (Veterans Affairs health coverage) indicates military family connection

#### Condition 9: Military Health Insurance
- **Variable**: `MIL == 1`
- **Logic**: TRICARE/military health insurance indicates military service or family connection

#### Condition 10: Government Employees
- **Variables**: `PEIO1COW` codes 1-3 OR `A_MJOCC == 11`
- **Logic**: Government employment requires legal work authorization
- **Codes**: 1=Federal government, 2=State government, 3=Local government, A_MJOCC 11=Military occupation

#### Condition 11: Social Security Recipients
- **Variable**: `SS_YN == 1`
- **Logic**: Social Security benefits indicate legal status and work history

#### Condition 12: Housing Assistance
- **Variable**: `SPM_CAPHOUSESUB > 0` (mapped from SPM unit data)
- **Logic**: Housing assistance programs generally require legal status
- **Implementation**: Uses SPM unit mapping: `spm_housing_map = dict(zip(smp_unit.SPM_ID, spm_unit.SPM_CAPHOUSESUB))`

#### Condition 13: Veterans/Military Personnel
- **Variables**: `PEAFEVER == 1` OR `A_MJOCC == 11`
- **Logic**: Military service or veteran status indicates legal status

#### Condition 14: SSI Recipients
- **Variable**: `SSI_YN == 1`
- **Logic**: SSI eligibility generally requires legal status (simplified implementation)

### Step 3: Target-Driven EAD Assignment for Workers
- **Target**: 8.3 million undocumented workers (from Pew Research)
- **Eligibility**: Non-citizens not in code 3 with earnings (`WSAL_VAL > 0` OR `SEMP_VAL > 0`)
- **Process**: Uses `select_random_subset_to_target()` function with random seed 0
- **Logic**: Calculates how many workers need EAD status to hit the target, then randomly selects from eligible pool
- **Result**: Selected workers moved from code 0 to code 2

### Step 4: Target-Driven EAD Assignment for Students  
- **Target**: ~399k undocumented students (21% of 1.9M total, from Higher Ed Immigration Portal)
- **Eligibility**: Non-citizens not in code 3 currently in college (`A_HSCOL == 2`)
- **Process**: Uses `select_random_subset_to_target()` function with random seed 1
- **Result**: Selected students moved from code 0 to code 2

### Step 5: Family Correlation Adjustment
- **Logic**: If any household member has code 0 (undocumented), all other eligible household members are also assigned code 0
- **Scope**: Only affects people with codes 0 or 3 (not citizens or EAD holders)
- **Process**: 
  1. Iterate through each unique household ID
  2. Check if household has any code 0 members
  3. If yes, change all code 3 members in that household to code 0
- **Implementation**: 100% correlation within households (not probabilistic)

### Step 6: Final Population Targeting
- **Target**: 13 million total undocumented immigrants
- **Process**: Uses `select_random_subset_to_target()` function with random seed 42
- **Logic**: If current undocumented population exceeds target, randomly move some code 0 individuals to code 3
- **Eligibility**: Remaining code 0 non-citizens only

## Helper Functions

### `select_random_subset_to_target()`
A sophisticated targeting function that handles both population reduction and increase scenarios:
- **Population reduction**: Uses new random number generator (`np.random.default_rng`) for refinement steps
- **Population increase**: Uses legacy `np.random` for EAD assignment compatibility
- **Weighting**: Accounts for household weights in selection probability
- **Bounds checking**: Caps selection percentage at 100%
