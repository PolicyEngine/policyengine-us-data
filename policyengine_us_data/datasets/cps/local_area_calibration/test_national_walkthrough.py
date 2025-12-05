# National Target Walkthrough:
# This validates the sparse matrix for NATIONAL targets where:
# - There is 1 target row (not 51 like state SNAP)
# - Matrix values are non-zero for ALL 436 CD columns (no geographic filtering)

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.local_area_calibration.metrics_matrix_geo_stacking_sparse import (
    SparseGeoStackingMatrixBuilder,
)
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    create_target_groups,
)
from policyengine_us_data.datasets.cps.local_area_calibration.household_tracer import HouseholdTracer
from policyengine_us_data.datasets.cps.local_area_calibration.stacked_dataset_builder import create_sparse_cd_stacked_dataset

rng_ben = np.random.default_rng(seed=42)


# Step 1: Setup - same as SNAP walkthrough
db_path = STORAGE_FOLDER / "policy_data.db"
db_uri = f"sqlite:///{db_path}"
builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)

engine = create_engine(db_uri)

query = """
SELECT DISTINCT sc.value as cd_geoid
FROM strata s
JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
WHERE s.stratum_group_id = 1
  AND sc.constraint_variable = 'congressional_district_geoid'
ORDER BY sc.value
"""

with engine.connect() as conn:
    result = conn.execute(text(query)).fetchall()
    all_cd_geoids = [row[0] for row in result]

cds_to_calibrate = all_cd_geoids
dataset_uri = STORAGE_FOLDER / "stratified_10k.h5"
sim = Microsimulation(dataset=str(dataset_uri))

targets_df, X_sparse, household_id_mapping = (
    builder.build_stacked_matrix_sparse(
        "congressional_district", cds_to_calibrate, sim
    )
)

target_groups, group_info = create_target_groups(targets_df)
tracer = HouseholdTracer(targets_df, X_sparse, household_id_mapping, cds_to_calibrate, sim)

tracer.print_matrix_structure()

hh_agi_df = sim.calculate_dataframe(['household_id', 'adjusted_gross_income'])

# Alimony Expense -------------------------------------------------------------------

# Group 0 is national alimony_expense - a single target
group_0 = tracer.get_group_rows(0)
print(f"\nGroup 0 info:\n{group_0}")

assert group_0.shape[0] == 1, f"Expected 1 national target, got {group_0.shape[0]}"

row_loc = group_0.iloc[0]['row_index']
row_info = tracer.get_row_info(row_loc)
var = row_info['variable']

# Is var calculated?
calculated = [v for v in sim.tax_benefit_system.variables
              if v not in sim.input_variables]

print(f"{var} is calculated by the engine: {var in calculated}")
print(f"{var} is an input: {var in sim.input_variables}")

print(f"\nRow info for national alimony_expense target:")
print(row_info)

assert var == 'alimony_expense', f"Expected alimony_expense, got {var}"
assert row_loc == 0, f"Expected row 0, got {row_loc}"

# Step 3: Find a household with positive alimony_expense
# alimony_expense is a tax_unit level variable

entity_rel = pd.DataFrame(
    {
        "person_id": sim.calculate("person_id", map_to="person").values,
        "household_id": sim.calculate("household_id", map_to="person").values,
        "tax_unit_id": sim.calculate("tax_unit_id", map_to="person").values,
    }
)

# Get alimony_expense at tax_unit level
tu_df = sim.calculate_dataframe(['tax_unit_id', 'alimony_expense'])
print(f"\nTax units with alimony_expense > 0: {(tu_df.alimony_expense > 0).sum()}")
print(tu_df.loc[tu_df.alimony_expense > 0].head(10))

# Find households with positive alimony expense
tu_with_alimony = tu_df.loc[tu_df.alimony_expense > 0]

# Map tax_units to households
tu_to_hh = entity_rel[['tax_unit_id', 'household_id']].drop_duplicates()
tu_with_alimony_hh = tu_with_alimony.merge(tu_to_hh, on='tax_unit_id')

# Aggregate alimony_expense at household level (sum across tax units)
hh_alimony = tu_with_alimony_hh.groupby('household_id')['alimony_expense'].sum().reset_index()
hh_alimony.columns = ['household_id', 'alimony_expense']
print(f"\nHouseholds with alimony_expense > 0: {hh_alimony.shape[0]}")
print(hh_alimony.head(10))

# Pick a test household
hh_id = hh_alimony.iloc[0]['household_id']
hh_alimony_goal = hh_alimony.iloc[0]['alimony_expense']

print(f"\nTest household: {hh_id}")
print(f"Household alimony_expense: {hh_alimony_goal}")

# Step 4: Validate Matrix Values - KEY DIFFERENCE FROM SNAP
# For national targets, the matrix value should be the SAME in ALL 436 CD columns
# (unlike state SNAP where it's only non-zero in home state CDs)

hh_col_lku = tracer.get_household_column_positions(hh_id)

values_found = []
for cd in hh_col_lku.keys():
    col_loc = hh_col_lku[cd]
    col_info = tracer.get_column_info(col_loc)

    assert col_info['household_id'] == hh_id

    metric = X_sparse[row_loc, col_loc]
    values_found.append(metric)

    # For national target: value should be hh_alimony_goal in ALL CDs
    assert metric == hh_alimony_goal, f"Expected {hh_alimony_goal} for CD {cd}, got {metric}"

print(f"\nAll {len(hh_col_lku)} CD column values validated for household {hh_id}")
print(f"All values equal to {hh_alimony_goal}: {all(v == hh_alimony_goal for v in values_found)}")

# Step 5: Verify a household with zero alimony also has zeros everywhere
hh_df = sim.calculate_dataframe(['household_id'])
all_hh_ids = set(hh_df.household_id.values)
hh_with_alimony_ids = set(hh_alimony.household_id.values)
hh_without_alimony = all_hh_ids - hh_with_alimony_ids

# Pick one household without alimony
hh_zero_id = list(hh_without_alimony)[0]
hh_zero_col_lku = tracer.get_household_column_positions(hh_zero_id)

for cd in list(hh_zero_col_lku.keys())[:10]:  # Check first 10 CDs
    col_loc = hh_zero_col_lku[cd]
    metric = X_sparse[row_loc, col_loc]
    assert metric == 0, f"Expected 0 for zero-alimony household {hh_zero_id} in CD {cd}, got {metric}"

print(f"\nVerified household {hh_zero_id} (no alimony) has zeros in matrix")

# Step 6: End-to-End Validation
# Create a sparse weight vector and verify X @ w matches simulation

n_nonzero = 50000
total_size = X_sparse.shape[1]

w = np.zeros(total_size)
nonzero_indices = rng_ben.choice(total_size, n_nonzero, replace=False)
w[nonzero_indices] = 7
w[hh_col_lku['101']] = 11  # Give our test household a specific weight in CD 101

output_dir = './temp'
output_path = f"{output_dir}/national_alimony_test.h5"

output_file = create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    dataset_path=str(dataset_uri),
    output_path=output_path,
)

# Load and calculate
sim_test = Microsimulation(dataset=output_path)
hh_alimony_df = pd.DataFrame(sim_test.calculate_dataframe([
    "household_id", "household_weight", "alimony_expense"])
)

print(f"\nOutput dataset has {hh_alimony_df.shape[0]} households")

# Matrix multiplication prediction
y_hat = X_sparse @ w
alimony_hat_matrix = y_hat[row_loc]

# Simulation-based calculation (national sum)
alimony_hat_sim = np.sum(hh_alimony_df.alimony_expense.values * hh_alimony_df.household_weight.values)

print(f"\nMatrix multiplication (X @ w)[{row_loc}] = {alimony_hat_matrix:,.2f}")
print(f"Simulation sum(alimony_expense * weight) = {alimony_hat_sim:,.2f}")

assert np.isclose(alimony_hat_sim, alimony_hat_matrix, atol=10), f"Mismatch: {alimony_hat_sim} vs {alimony_hat_matrix}"
print("\nEnd-to-end validation PASSED")

# ============================================================================
# Part 2: income_tax - FEDERAL income tax (NOT state-dependent)
# ============================================================================
# NOTE: income_tax in PolicyEngine is FEDERAL income tax only!
# It does NOT include state_income_tax. The formula is:
#   income_tax = income_tax_before_refundable_credits - income_tax_refundable_credits
# Therefore, income_tax should be the SAME across all CDs for a given household.

print("\n" + "="*80)
print("PART 2: income_tax (Federal Only) - Should NOT vary by state")
print("="*80)

print(f"\nincome_tax is calculated: {'income_tax' not in sim.input_variables}")

# Find the income_tax target row in X_sparse (Group 7)
group_7 = tracer.get_group_rows(7)
income_tax_row = group_7.iloc[0]['row_index']
income_tax_row_info = tracer.get_row_info(income_tax_row)
print(f"\nincome_tax row info: {income_tax_row_info}")

# Find a high-income household for federal income_tax test
hh_agi_df = sim.calculate_dataframe(['household_id', 'adjusted_gross_income'])
high_income_hh = hh_agi_df[
    (hh_agi_df.adjusted_gross_income > 400000) &
    (hh_agi_df.adjusted_gross_income < 600000)
].sort_values('adjusted_gross_income')

if len(high_income_hh) > 0:
    test_hh_id = high_income_hh.iloc[0]['household_id']
    test_hh_agi = high_income_hh.iloc[0]['adjusted_gross_income']
else:
    test_hh_id = hh_agi_df.sort_values('adjusted_gross_income', ascending=False).iloc[0]['household_id']
    test_hh_agi = hh_agi_df[hh_agi_df.household_id == test_hh_id].adjusted_gross_income.values[0]

print(f"\nTest household for income_tax: {test_hh_id}, AGI: ${test_hh_agi:,.0f}")

# Get matrix values for TX vs CA CDs
test_hh_col_lku = tracer.get_household_column_positions(test_hh_id)
tx_cds = [cd for cd in test_hh_col_lku.keys() if cd.startswith('48')]
ca_cds = [cd for cd in test_hh_col_lku.keys() if cd.startswith('6') and len(cd) == 3]

if tx_cds and ca_cds:
    tx_cd, ca_cd = tx_cds[0], ca_cds[0]
    tx_col, ca_col = test_hh_col_lku[tx_cd], test_hh_col_lku[ca_cd]

    income_tax_tx_matrix = X_sparse[income_tax_row, tx_col]
    income_tax_ca_matrix = X_sparse[income_tax_row, ca_col]

    print(f"\nincome_tax in TX CD {tx_cd}: ${income_tax_tx_matrix:,.2f}")
    print(f"income_tax in CA CD {ca_cd}: ${income_tax_ca_matrix:,.2f}")

    assert income_tax_tx_matrix == income_tax_ca_matrix, \
        f"Federal income_tax should be identical across CDs! TX={income_tax_tx_matrix}, CA={income_tax_ca_matrix}"
    print("\n✓ PASSED: Federal income_tax is identical across all CDs (as expected)")


# ============================================================================
# Part 3: salt_deduction - NOT state-dependent (based on INPUTS)
# ============================================================================
# IMPORTANT: salt_deduction does NOT vary by state in geo-stacking!
#
# Why? The SALT deduction formula is:
#   salt_deduction = min(salt_cap, reported_salt)
#   reported_salt = salt (possibly limited to AGI)
#   salt = state_and_local_sales_or_income_tax + real_estate_taxes
#   state_and_local_sales_or_income_tax = max(income_tax_component, sales_tax_component)
#   income_tax_component = state_withheld_income_tax + local_income_tax
#
# The key variables are INPUTS from the CPS/tax data:
#   - state_withheld_income_tax: INPUT (actual withholding reported)
#   - local_income_tax: INPUT
#   - real_estate_taxes: INPUT
#
# These represent what the household ACTUALLY PAID in their original state.
# When we change state_fips for geo-stacking, these input values don't change
# because they're historical data from tax returns, not calculated liabilities.
#
# Truly state-dependent variables must be CALCULATED based on state policy,
# like: snap, medicaid (benefit programs with state-specific rules)

print("\n" + "="*80)
print("PART 3: salt_deduction - Should NOT vary by state (input-based)")
print("="*80)

#from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import get_state_dependent_variables
#state_dep_vars = get_state_dependent_variables()
#print(f"\nState-dependent variables: {state_dep_vars}")

# Find salt_deduction target (Group 21)
group_21 = tracer.get_group_rows(21)
print(f"\nGroup 21 info:\n{group_21}")

salt_row = group_21.iloc[0]['row_index']
salt_row_info = tracer.get_row_info(salt_row)
print(f"\nsalt_deduction row info: {salt_row_info}")

# Use a moderate-income household for testing
moderate_income_hh = hh_agi_df[
    (hh_agi_df.adjusted_gross_income > 75000) &
    (hh_agi_df.adjusted_gross_income < 150000)
].sort_values('adjusted_gross_income')

if len(moderate_income_hh) > 0:
    salt_test_hh_id = moderate_income_hh.iloc[0]['household_id']
    salt_test_hh_agi = moderate_income_hh.iloc[0]['adjusted_gross_income']
else:
    salt_test_hh_id = test_hh_id
    salt_test_hh_agi = test_hh_agi

print(f"\nTest household for salt_deduction: {salt_test_hh_id}, AGI: ${salt_test_hh_agi:,.0f}")

# Get column positions for this household
salt_hh_col_lku = tracer.get_household_column_positions(salt_test_hh_id)
salt_tx_cds = [cd for cd in salt_hh_col_lku.keys() if cd.startswith('48')]
salt_ca_cds = [cd for cd in salt_hh_col_lku.keys() if cd.startswith('6') and len(cd) == 3]

# Check matrix values for TX vs CA - they SHOULD be identical (input-based)
if salt_tx_cds and salt_ca_cds:
    salt_tx_cd, salt_ca_cd = salt_tx_cds[0], salt_ca_cds[0]
    salt_tx_col = salt_hh_col_lku[salt_tx_cd]
    salt_ca_col = salt_hh_col_lku[salt_ca_cd]

    salt_tx_matrix = X_sparse[salt_row, salt_tx_col]
    salt_ca_matrix = X_sparse[salt_row, salt_ca_col]

    print(f"\nsalt_deduction for household {salt_test_hh_id}:")
    print(f"  TX CD {salt_tx_cd}: ${salt_tx_matrix:,.2f}")
    print(f"  CA CD {salt_ca_cd}: ${salt_ca_matrix:,.2f}")




# Bringing in the snap parts of the test:

p_df = sim.calculate_dataframe(['person_household_id', 'person_id', 'snap'], map_to="person")

hh_stats = p_df.groupby('person_household_id').agg(
    person_count=('person_id', 'nunique'),
    snap_min=('snap', 'min'),
    snap_unique=('snap', 'nunique')
).reset_index()

candidates = hh_stats[(hh_stats.person_count > 1) & (hh_stats.snap_min > 0) & (hh_stats.snap_unique > 1)]
candidates.head(10)

hh_id = candidates.iloc[2]['person_household_id']
p_df.loc[p_df.person_household_id == hh_id]

hh_snap_goal = 7925.5

entity_rel = pd.DataFrame(
    {
        "person_id": sim.calculate("person_id", map_to="person").values,
        "household_id": sim.calculate("household_id", map_to="person").values,
        "tax_unit_id": sim.calculate("tax_unit_id", map_to="person").values,
        "spm_unit_id": sim.calculate("spm_unit_id", map_to="person").values,
        "family_id": sim.calculate("family_id", map_to="person").values,
        "marital_unit_id": sim.calculate("marital_unit_id", map_to="person").values,
    }
)

snap_df = sim.calculate_dataframe(['spm_unit_id', 'snap'])
snap_subset = entity_rel.loc[entity_rel.household_id == hh_id]
snap_df.loc[snap_df.spm_unit_id.isin(list(snap_subset.spm_unit_id))]


hh_df = sim.calculate_dataframe(['household_id', 'state_fips'])
hh_loc = np.where(hh_df.household_id == hh_id)[0][0]
hh_one = hh_df.iloc[hh_loc]
hh_home_state = hh_one.state_fips
hh_col_lku = tracer.get_household_column_positions(hh_id)

print(f"Household {hh_id} is from state FIPS {hh_home_state}")
hh_one

n_nonzero = 1000000
total_size = X_sparse.shape[1]

w = np.zeros(total_size)
nonzero_indices = rng_ben.choice(total_size, n_nonzero, replace=False)
w[nonzero_indices] = 2

cd1 = '601'
cd2 = '2001'
output_dir = './temp'
w[hh_col_lku[cd1]] = 1.5
w[hh_col_lku[cd2]] = 1.7

output_path = f"{output_dir}/mapping1.h5"
output_file = create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    cd_subset=[cd1, cd2],
    dataset_path=str(dataset_uri),
    output_path=output_path,
)

sim_test = Microsimulation(dataset=output_path)
df_test = sim_test.calculate_dataframe([
    'congressional_district_geoid',
    'household_id', 'household_weight', 'snap'])

print(f"Output dataset shape: {df_test.shape}")
assert np.isclose(df_test.shape[0] / 2 * 436, n_nonzero, rtol=0.10)

mapping = pd.read_csv(f"{output_dir}/mapping1_household_mapping.csv")
match = mapping.loc[mapping.original_household_id == hh_id].shape[0]
assert match == 2, f"Household should appear twice (once per CD), got {match}"

hh_mapping = mapping.loc[mapping.original_household_id == hh_id]
hh_mapping

df_test_cd1 = df_test.loc[df_test.congressional_district_geoid == int(cd1)]
df_test_cd2 = df_test.loc[df_test.congressional_district_geoid == int(cd2)]

hh_mapping_cd1 = hh_mapping.loc[hh_mapping.congressional_district == int(cd1)]
new_hh_id_cd1 = hh_mapping_cd1['new_household_id'].values[0]

assert hh_mapping_cd1.shape[0] == 1
assert hh_mapping_cd1.original_household_id.values[0] == hh_id

w_hh_cd1 = w[hh_col_lku[cd1]]
assert_cd1_df = df_test_cd1.loc[df_test_cd1.household_id == new_hh_id_cd1]

assert np.isclose(assert_cd1_df.household_weight.values[0], w_hh_cd1, atol=0.001)
assert np.isclose(assert_cd1_df.snap.values[0], hh_snap_goal, atol=0.001)

print(f"CD {cd1}: weight={w_hh_cd1}, snap={assert_cd1_df.snap.values[0]}")
assert_cd1_df


hh_mapping_cd2 = hh_mapping.loc[hh_mapping.congressional_district == int(cd2)]
new_hh_id_cd2 = hh_mapping_cd2['new_household_id'].values[0]

assert hh_mapping_cd2.shape[0] == 1
assert hh_mapping_cd2.original_household_id.values[0] == hh_id

w_hh_cd2 = w[hh_col_lku[cd2]]
assert_cd2_df = df_test_cd2.loc[df_test_cd2.household_id == new_hh_id_cd2]

assert np.isclose(assert_cd2_df.household_weight.values[0], w_hh_cd2, atol=0.001)
assert np.isclose(assert_cd2_df.snap.values[0], hh_snap_goal, atol=0.001)

print(f"CD {cd2}: weight={w_hh_cd2}, snap={assert_cd2_df.snap.values[0]}")

## Another household that requires BBCE to get in

# Calculate household-level variables
hh_df = sim.calculate_dataframe([
    'household_id',
    'state_fips',
    'snap_gross_income_fpg_ratio',
     'gross_income',
    'snap',
    'spm_unit_size',
    'is_snap_eligible',
    'is_tanf_non_cash_eligible'
], map_to="household")

# Filter for BBCE-relevant households
# Between 130% and 200% FPL (where CA qualifies via BBCE, KS doesn't)
candidates = hh_df[
    (hh_df['snap_gross_income_fpg_ratio'] >= 1.50) &
    (hh_df['snap_gross_income_fpg_ratio'] <= 1.80) &
    (hh_df['is_tanf_non_cash_eligible'] > 1)
].copy()

# Sort by FPG ratio to find households near 165%
candidates['distance_from_165'] = abs(candidates['snap_gross_income_fpg_ratio'] - 1.65)
candidates_sorted = candidates.sort_values('distance_from_165')

# Show top 10 candidates
candidates_sorted[['household_id', 'state_fips', 'snap_gross_income_fpg_ratio', 'snap', 'is_snap_eligible', 'spm_unit_size']].head(10)


# There was always a reason why I couldn't get the BBCE pathway to work!
from policyengine_us import Microsimulation

# Load CPS 2023
sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/cps_2023.h5")


# Find PURE BBCE cases - no elderly/disabled exemption
ca_bbce_pure = candidates[
    #(candidates['state_fips'] == 6) &
    (candidates['snap_gross_income_fpg_ratio'] >= 1.30) &
    (candidates['snap_gross_income_fpg_ratio'] <= 2.0) &
    (candidates['is_tanf_non_cash_eligible'] > 0) &
    (candidates['meets_snap_categorical_eligibility'] > 0) &
    (candidates['is_snap_eligible'] > 0) &
    (candidates['snap'] > 0)
].copy()

# Now check which ones FAIL the normal gross test
for idx, row in ca_bbce_pure.head(20).iterrows():
    hh_id = row['household_id']
    check = sim.calculate_dataframe(
        ['household_id', 'meets_snap_gross_income_test', 'has_usda_elderly_disabled'],
        map_to='household'
    )
    hh_check = check[check['household_id'] == hh_id].iloc[0]
    if hh_check['meets_snap_gross_income_test'] == 0:
        print(f"HH {hh_id}: Pure BBCE case! (no elderly/disabled exemption)")
        print(f"  Gross FPL: {row['snap_gross_income_fpg_ratio']:.1%}")
        print(f"  SNAP: ${row['snap']:.2f}")
        break


# Cleanup
import shutil
import os
if os.path.exists('./temp'):
    shutil.rmtree('./temp')
    print("\nCleaned up ./temp directory")
