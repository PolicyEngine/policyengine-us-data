# Geo-Stacking Calibration: Project Status

### Congressional District Calibration - FIX APPLIED, AWAITING VALIDATION ⏳

**Matrix Dimensions Verified**: 34,089 × 4,612,880
- 30 national targets
- 7,848 age targets (18 bins × 436 CDs)  
- 436 CD SNAP household counts
- 487 total SNAP targets (436 CD + 51 state costs)
- 25,288 IRS SOI targets (58 × 436 CDs)
- **Total: 34,089 targets** ✓

**Critical Fix Applied (2024-12-24)**: Fixed IRS target deduplication by including constraint operations in concept IDs. AGI bins with boundaries like `< 10000` and `>= 10000` are now properly distinguished.

**Fix Reverted (2024-12-25)**: Reverted tax_unit_count changes after investigation showed the original implementation was correct. Testing demonstrated that summing tax unit counts to household level produces virtually perfect results (0.0% error). The perceived issue was a misunderstanding of how tax unit weights work in PolicyEngine.

**Key Design Decision for CD Calibration**: State SNAP cost targets (51 total) apply to households within each state but remain state-level constraints. Households in CDs within a state have non-zero values in the design matrix for their state's SNAP cost target.

**Note**: This target accounting is specific to congressional district calibration. State-level calibration will have a different target structure and count.

#### What Should Happen (Hierarchical Target Selection)
For each target concept (e.g., "age 25-30 population in Texas"):
1. **If CD-level target exists** → use it for that CD only
2. **If no CD target but state target exists** → use state target for all CDs in that state  
3. **If neither CD nor state target exists** → use national target

For administrative data (e.g., SNAP):
- **Always prefer administrative over survey data**, even if admin is less granular
- State-level SNAP admin data should override CD-level survey estimates

## Next Steps

### Immediate (After Matrix Rebuild)
1. **Run calibration with new matrix** - Test if EITC and other tax_unit_count targets now converge properly
2. **Validate fix effectiveness** - Check if tax_unit_count predictions are within reasonable error bounds (<50% instead of 200-300%)
3. **Monitor convergence** - Ensure the fix doesn't negatively impact other target types

### If Fix Validated
1. **Full CD calibration run** - Run complete calibration with appropriate epochs and sparsity settings
2. **Document final performance** - Update with actual error rates for all target groups
3. **Create sparse CD-stacked dataset** - Use calibrated weights to create final dataset

### Known Issues to Watch
- **Sparsity constraints**: Current L0 settings may be too aggressive (99.17% sparsity is extreme)
- **Rental income targets**: Some showing very high errors (check if this persists)
- **Multi-tax-unit household weighting**: Our scaling assumption may need refinement

## Analysis

#### State Activation Patterns

#### Population Target Achievement

## L0 Package (~/devl/L0)
- `l0/calibration.py` - Core calibration class
- `tests/test_calibration.py` - Test coverage

## Hierarchical Target Reconciliation

### Implementation Status
A reconciliation system has been implemented to adjust lower-level survey targets to match higher-level administrative totals when available.

#### ETL Files and Reconciliation Needs

1. **etl_age.py** ✅ No reconciliation needed
   - Source: Census ACS Table S0101 (survey data for both state and CD)
   - Status: Age targets already sum correctly (state = sum of CDs)
   - Example: California age < 5: State = 2,086,820, Sum of 52 CDs = 2,086,820

2. **etl_medicaid.py** ✅ Reconciliation ACTIVE
   - State: Medicaid T-MSIS (administrative)
   - CD: Census ACS Table S2704 (survey)
   - Adjustment factor: 1.1962 (16.4% undercount)
   - Example: California adjusted from 10,474,055 → 12,529,315

3. **etl_snap.py** ✅ Reconciliation ACTIVE
   - State: USDA FNS SNAP Data (administrative)
   - CD: Census ACS Table S2201 (survey)
   - Adjustment factor: 1.6306 (38.7% undercount)
   - Example: California households adjusted from 1,833,346 → 2,989,406

4. **etl_irs_soi.py** ✅ No reconciliation needed
   - Source: IRS Statistics of Income (administrative at both levels)
   - Both state and CD use same administrative source

5. **etl_national_targets.py** ✅ No reconciliation needed
   - National-level hardcoded targets only

### Reconciliation System Features
- Calculates adjustment factors by comparing administrative totals to survey sums
- Applies proportional adjustments to maintain relative distributions
- Tracks diagnostic information (original values, factors, undercount percentages)
- Currently active for:
  - Medicaid enrollment (stratum_group_id = 5)
  - SNAP household counts (stratum_group_id = 4)

## Calibration Performance Analysis (2024-09-24)

### Critical Finding: Extreme Sparsity Constraints Preventing Convergence

**Dataset**: 644MB calibration log with 3.4M records tracking 10,979 targets over 10,000 epochs

#### Sparsity Progression
- **Initial (epoch 100)**: 0.01% sparsity, 4,612,380 active weights
- **Final (epoch 10,000)**: 99.17% sparsity, only 38,168 active weights (0.83% of original!)
- **Critical failure**: Catastrophic pruning event at epochs 2500-2600 dropped from 1.3M to 328K weights

#### Performance Impact
1. **Loss vs Error Mismatch**: Loss reduced 99.92% but error only reduced 86.62%
2. **Plateau after epoch 1000**: No meaningful improvement despite 9000 more epochs
3. **Insufficient capacity**: Only 3.5 weights per target on average (38K weights for 11K targets)

#### Problem Areas
- **Rental Income**: 43 targets with >100% error, worst case 1,987x target value
- **Tax Unit Counts**: 976 CD-level counts still >100% error at final epoch
- **Congressional Districts**: 1,460 targets never converged below 100% error

#### Root Cause
The aggressive L0 sparsity regularization is starving the model of parameters needed to fit complex geographic patterns. Previous runs without these constraints performed much better. The model cannot represent the relationships between household features and geographic targets with such extreme sparsity.

## Calibration Variable Exclusions (2025-01-01)

### Variables Excluded from Calibration
Based on analysis of calibration errors, the following variables are excluded:

#### CD/State-Level Exclusions (applied across all geographic levels)
**Tax/Income Variables with Consistent High Errors:**
- `rental_income_rental_income>0`
- `salt_salt>0`
- `tax_unit_count_salt>0`
- `net_capital_gains`
- `net_capital_gain`
- `self_employment`
- `medical_deduction`
- `QBI_deduction`
- `rental_income`
- `qualified_dividends`
- `dividends`
- `partnership_S_corp`
- `taxable_IRA_distributions`
- `taxable_interest`
- `tax_exempt_interest`
- `income_tax_paid`
- `income_tax_before_credits`
- `SALT_deduction`
- `real_estate_taxes`
- `taxable_pension`
- `all_filers`
- `unemployment_comp`
- `refundable_CTC`

**Variables with "_national" suffix:**
- `alimony_expense_national`
- `charitable_deduction_national`
- `health_insurance_premiums_without_medicare_part_b_national`
- `medicare_part_b_premiums_national`
- `other_medical_expenses_national`
- `real_estate_taxes_national`
- `salt_deduction_national`

#### National-Level Only Exclusions (only removed for geographic_id == 'US')
**Specific problematic national targets with >50% error:**
- `medical_expense_deduction_tax_unit_is_filer==1` (440% error)
- `interest_deduction_tax_unit_is_filer==1` (325% error)
- `qualified_business_income_deduction_tax_unit_is_filer==1` (146% error)
- `charitable_deduction_tax_unit_is_filer==1` (122% error)
- `alimony_expense_tax_unit_is_filer==1` (96% error)
- `person_count_aca_ptc>0` (114% error)
- `person_count_ssn_card_type=NONE` (62% error)
- `child_support_expense` (51% error)
- `health_insurance_premiums_without_medicare_part_b` (51% error)

**IMPORTANT**: AGI, EITC, and age demographics are NOT excluded at CD level as they are critical for calibration.

## Documentation
- `GEO_STACKING_TECHNICAL.md` - Technical documentation and architecture
- `PROJECT_STATUS.md` - This file (active project management)
