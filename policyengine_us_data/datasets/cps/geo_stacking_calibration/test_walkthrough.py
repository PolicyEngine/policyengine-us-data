# Step 1: Setup: get the design matrix, X_sparse, in place!

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.geo_stacking_calibration.metrics_matrix_geo_stacking_sparse import (
    SparseGeoStackingMatrixBuilder,
)
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
    create_target_groups,
)
from policyengine_us_data.datasets.cps.geo_stacking_calibration.household_tracer import HouseholdTracer

from policyengine_us_data.datasets.cps.geo_stacking_calibration.create_sparse_cd_stacked import create_sparse_cd_stacked_dataset


rng_ben = np.random.default_rng(seed=42)



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


# Step 2: Pick a group to validate:

tracer.print_matrix_structure()

# Let's go with Group 71, SNAP state targets
# Group 71: SNAP Cost (State) (51 targets across 51 geographies) - rows [33166, 33167, 33168, '...', 33215, 33216]

group_71 = tracer.get_group_rows(71)
# I pick the first one of those rows to get some information 

# I had one row_loc, but really I need many!
row_loc = group_71.iloc[0]['row_index']  # one target, for this particular case, it's of a  
row_info = tracer.get_row_info(row_loc)
var = row_info['variable']
var_desc = row_info['variable_desc']
target_geo_id = int(row_info['geographic_id'])  # For SNAP, these will be state ids. Other targets will be different!

# I'm a little annoyed that I have to exploit a broadcast rather than just get this from the group, but I'll take it
print(row_info)
#Out[28]:
#{'row_index': 33166,
# 'variable': 'snap',
# 'variable_desc': 'snap_cost_state',
# 'geographic_id': '1',
# 'geographic_level': 'unknown',
# 'target_value': 2048985036.0,
# 'stratum_id': 9766,
# 'stratum_group_id': 'state_snap_cost'}

# So this is a state level variable, 
state_snap = tracer.row_catalog[
    (tracer.row_catalog['variable'] == row_info['variable']) &
    (tracer.row_catalog['variable_desc'] == row_info['variable_desc'])
].sort_values('geographic_id')
print(state_snap)

assert state_snap.shape[0] == 51

# The first thing to take away is that the policyengine-us variable is 'snap'
# Let's find an interesting household
# So I think an interesting household is one that
# - Has more than one person per SPM unit
# - Has more than one SPM units
# - each SPM unit has positive snap
# For other variables that are not snap, you'd want to replace spm_unit with whatever that variable's unit is 

entity_rel = pd.DataFrame(
    {
        "person_id": sim.calculate(
            "person_id", map_to="person"
        ).values,
        "household_id": sim.calculate(
            "household_id", map_to="person"
        ).values,
        "tax_unit_id": sim.calculate(
            "tax_unit_id", map_to="person"
        ).values,
        "spm_unit_id": sim.calculate(
            "spm_unit_id", map_to="person"
        ).values,
        "family_id": sim.calculate(
            "family_id", map_to="person"
        ).values,
        "marital_unit_id": sim.calculate(
            "marital_unit_id", map_to="person"
        ).values,
    }
)

# Side Note: understand that these are fundamentally different!
sim.calculate_dataframe(['spm_unit_id', 'snap'])  # Rows are spm_units
sim.calculate_dataframe(['household_id', 'spm_unit_id', 'snap_take_up_seed', 'snap'])  # Rows are households
p_df = sim.calculate_dataframe(['person_household_id', 'person_id', 'snap'], map_to="person")  # Rows are people

# Important information about randomenss in snap, and the snap takeup seed,
# The snap takeup seed comes from the microdata! It's not random in the calculation!
# The key point: For the same household computed twice, SNAP will always be the same because the seed is fixed. But across different households, the
# different seeds create variation in takeup behavior, which models the real-world fact that not all eligible households actually claim SNAP benefits.

# Let's find an example where more than one person from more than one household has 
hh_stats = p_df.groupby('person_household_id').agg(
    person_count=('person_id', 'nunique'),
    snap_min=('snap', 'min'), snap_unique=('snap', 'nunique')).reset_index()
candidates = hh_stats[(hh_stats.person_count > 1) & (hh_stats.snap_min > 0) & (hh_stats.snap_unique > 1)]
candidates.head(10)

hh_id = candidates.iloc[2]['person_household_id']

p_df.loc[p_df.person_household_id == hh_id]

# So I looped through until I found an interesting example
# Two people obviously have snap from a broadcast of the same spm unit, and 
# On person has a snap value of a different SPM unit. So I believe the correct answer for the
# household is 3592 + 4333.5 = 7925.5
# NOT, 3592 + 4333.5 + 4333.5
#Out[76]:
#       person_household_id  person_id    snap  __tmp_weights
#15319                91997    9199706  3592.0            0.0
#15320                91997    9199707  4333.5            0.0
#15321                91997    9199708  4333.5            0.0
hh_snap_goal = 7925.5

# Let's just learn a bit more about this household
hh_df = sim.calculate_dataframe(['household_id', 'snap', 'state_fips'])
hh_df.loc[hh_df.household_id == 91997]

snap_df = sim.calculate_dataframe(['spm_unit_id', 'snap'])
snap_df

# See the  
snap_subset = entity_rel.loc[entity_rel.household_id == hh_id]
snap_df.loc[snap_df.spm_unit_id.isin(list(snap_subset.spm_unit_id))]

# Ok, let's get some baseline info on our test household_id. Remember that Everything needs to go to the household level!
hh_df = sim.calculate_dataframe(['household_id', 'state_fips'])

hh_loc = np.where(hh_df.household_id == hh_id)[0][0]

# Remember that in the matrix, the households are the columns:
hh_one = hh_df.iloc[hh_loc]
#Out[94]:
#household_id    91997
#state_fips         50
#Name: 5109, dtype: int32

hh_home_state = hh_one.state_fips

hh_col_lku = tracer.get_household_column_positions(hh_id)

# loop through congressional districts
for cd in hh_col_lku.keys():

    # Remember, this household from hh_home_state is a donor to all districts covering all 51 states
    hh_away_state = int(cd) // 100
    
    col_loc = hh_col_lku[cd]
    
    col_info = tracer.get_column_info(col_loc)
    assert col_info['household_id'] == hh_id
    value_lku = tracer.lookup_matrix_cell(row_idx=row_loc, col_idx=col_loc)
   
    assert value_lku['household']['household_id'] == hh_id
    
    metric = value_lku['matrix_value']
    assert X_sparse[row_loc, col_loc] == metric
    
    # This code below ONLY Works because this is a state-level attribute!
    # For national and congressional district level targets, then the metric
    # IF it was a cd target, then the equality is not strict enough! 
    if hh_away_state != target_geo_id:
        assert metric == 0
    else:
        assert metric == hh_snap_goal


# Now I think it's time to create a random weight vector, create the .h5 file, and see if I can find this household again
# Make sure it's got the same structure, and same sub units, and that the household map_to gets to the right number, 1906.5

n_nonzero = 500000
total_size = X_sparse.shape[1]

# Create the h5 file from the weight, and test that the household is in the mappings ---
# 3 examples: 2 cds that the target state contains, and 1 that it doesn't

w = np.zeros(total_size)
nonzero_indices = rng_ben.choice(total_size, n_nonzero, replace=False)
w[nonzero_indices] = 2 

# cd 103, from the same state state, weight is 1.5 -----
target_geo_id
cd1 = '103'
cd2 = '3703'
output_dir = './temp'
w[hh_col_lku[cd1]] = 1.5
w[hh_col_lku[cd2]] = 1.7

output_path = f"{output_dir}/mapping1.h5"   # The mapping file and the h5 file will contain 2 cds 
output_file = create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    cd_subset=[cd1, cd2],
    dataset_path=str(dataset_uri),
    output_path=output_path,
)

sim_test = Microsimulation(dataset = output_path)

df_test = sim_test.calculate_dataframe([
    'congressional_district_geoid',
    'household_id', 'household_weight', 'snap'])
df_test.shape 
assert np.isclose(df_test.shape[0] / 2 * 436, n_nonzero, .10)

df_test_cd1 = df_test.loc[df_test.congressional_district_geoid == int(cd1)]
df_test_cd2 = df_test.loc[df_test.congressional_district_geoid == int(cd2)]

# Let's read in the mapping file for cd1, which is in the target geography of interest
mapping = pd.read_csv(f"{output_dir}/mapping1_household_mapping.csv")
match = mapping.loc[mapping.original_household_id == hh_id].shape[0]
assert match == 2  # houshold should be in there twice, for each district

hh_mapping = mapping.loc[mapping.original_household_id == hh_id]

# cd1 checks
hh_mapping_cd1 = hh_mapping.loc[hh_mapping.congressional_district == int(cd1)]
new_hh_id_cd1 = hh_mapping_cd1['new_household_id'].values[0]

assert hh_mapping_cd1.shape[0] == 1
assert hh_mapping_cd1.original_household_id.values[0] == hh_id

w_hh_cd1 = w[hh_col_lku[cd1]]

assert_cd1_df = df_test_cd1.loc[df_test_cd1.household_id == new_hh_id_cd1]
assert np.isclose(assert_cd1_df.household_weight.values[0], w_hh_cd1, atol=0.001)
assert np.isclose(assert_cd1_df.snap.values[0], hh_snap_goal, atol=0.001)

# cd2 checks
# Note: at first I thought that the snap should be zero since it's a different
# state, but I really neglected to see how this household is legitamitely part
# of cd 103 and cd 3701, and its snap value doesn't change. I would have to get
# a household from another state to show that it is zero
hh_mapping_cd2 = hh_mapping.loc[hh_mapping.congressional_district == int(cd2)]
new_hh_id_cd2 = hh_mapping_cd2['new_household_id'].values[0]

assert hh_mapping_cd2.shape[0] == 1
assert hh_mapping_cd2.original_household_id.values[0] == hh_id

w_hh_cd2 = w[hh_col_lku[cd2]]

assert_cd2_df = df_test_cd2.loc[df_test_cd2.household_id == new_hh_id_cd2]
assert np.isclose(assert_cd2_df.household_weight.values[0], w_hh_cd2, atol=0.001)
assert np.isclose(assert_cd2_df.snap.values[0], hh_snap_goal, atol=0.001)

# How can I check to see that households from different states all have snap of 0?
# Eh, you can see it with your eyes because the indicies are contiguous. How could
# formalize this? They're zero if they're not in df_test.

# I don't know, the mapping file has the district and those are the households you're working
# with. You're only dealing with these donor households given to each congressional
# district separately, so I think the zero is there, though we could look at X_sparse
# in those positions. Ah, you're already doing that!

# Now let's get the mapping file for the 

# cd 3703, weight is 0 -----
target_geo_id
cd2 = '3703'
output_dir = './temp'
w[hh_col_lku[cd2]] = 0 

output_path = f"{output_dir}/{cd2}.h5" 
output_file = create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    cd_subset=[cd2],
    dataset_path=str(dataset_uri),
    output_path=output_path,
)

sim_test = Microsimulation(dataset = output_path)

df_test = sim_test.calculate_dataframe(['household_id', 'household_weight', 'snap'])
df_test.shape 
assert np.isclose(df_test.shape[0] * 436, n_nonzero, .10)

# Let's read in the mapping file!
cd2_mapping = pd.read_csv(f"{output_dir}/{cd2}_household_mapping.csv")
match = cd2_mapping.loc[cd2_mapping.original_household_id == hh_id].shape[0]
assert match == 0

hh_mapping = cd2_mapping.loc[cd2_mapping.original_household_id == hh_id]

assert hh_mapping.shape[0] == 0
# Full end-to-end test to ensure sim.calculate matches y_hat = X_sparse @ w
#  To do this, we'll need to freeze the calculated variables upon writing
#  When you set freeze_calculated_vars=True, the function will:
#
#  1. Save calculated variables (like SNAP, Medicaid) to the h5 file (lines 836-840 in create_sparse_cd_stacked.py)
#  2. Prevent recalculation when the h5 file is loaded later

# Let's do a full test of the whole file and see if we can match sim.calculate
total_size = X_sparse.shape[1]
w = np.zeros(total_size)
# Smaller number of non-zero weights because we want to hold the file in memory
n_nonzero = 50000
nonzero_indices = rng_ben.choice(total_size, n_nonzero, replace=False)
w[nonzero_indices] = 7 
w[hh_col_lku[cd1]] = 11 
w[hh_col_lku[cd2]] = 12 
assert np.sum(w > 0) <= n_nonzero + 2

output_path = f"{output_dir}/national.h5" 
output_file = create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    dataset_path=str(dataset_uri),
    output_path=output_path,
    freeze_calculated_vars=True,
)

mapping = pd.read_csv(f"{output_dir}/national_household_mapping.csv")
mapping.loc[mapping.new_household_id == 10000]
mapping.loc[mapping.original_household_id == 3642]

hh_loc_101 = hh_col_lku['101']
X_sparse[row_info['row_index'], hh_loc_101]

sim_test = Microsimulation(dataset = output_path)
hh_snap_df = pd.DataFrame(sim_test.calculate_dataframe([
    "household_id", "household_weight", "congressional_district_geoid", "state_fips", "snap"])
)
hh_snap_df.loc[hh_snap_df.household_id == 10000]

assert np.sum(w > 0) == hh_snap_df.shape[0]

# Reminder:
print(row_info)

y_hat = X_sparse @ w
snap_hat_geo1 = y_hat[row_loc]

geo_1_df = hh_snap_df.loc[hh_snap_df.state_fips == 1]

y_hat_sim = np.sum(geo_1_df.snap.values * geo_1_df.household_weight.values)

assert np.isclose(y_hat_sim, snap_hat_geo1, atol=10)
