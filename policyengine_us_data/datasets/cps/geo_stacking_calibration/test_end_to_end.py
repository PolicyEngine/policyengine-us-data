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

# ------

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
dataset_uri = STORAGE_FOLDER / "stratified_extended_cps_2023.h5"
sim = Microsimulation(dataset=str(dataset_uri))

# ------
targets_df, X_sparse, household_id_mapping = (
    builder.build_stacked_matrix_sparse(
        "congressional_district", cds_to_calibrate, sim
    )
)

target_groups, group_info = create_target_groups(targets_df)
tracer = HouseholdTracer(targets_df, X_sparse, household_id_mapping, cds_to_calibrate, sim)

# Get NC's state SNAP info:
group_71 = tracer.get_group_rows(71)
row_loc = group_71.iloc[28]['row_index']  # The row of X_sparse
row_info = tracer.get_row_info(row_loc)
var = row_info['variable']
var_desc = row_info['variable_desc']
target_geo_id = int(row_info['geographic_id'])

print("Row info for first SNAP state target:")
row_info


# Create a weight vector
total_size = X_sparse.shape[1]

w = np.zeros(total_size)
n_nonzero = 50000
nonzero_indices = rng_ben.choice(total_size, n_nonzero, replace=False)
w[nonzero_indices] = 7

output_dir = "./temp"
h5_name = "national"
output_path = f"{output_dir}/{h5_name}.h5"
output_file = create_sparse_cd_stacked_dataset(
    w,
    cds_to_calibrate,
    dataset_path=str(dataset_uri),
    output_path=output_path,
)

sim_test = Microsimulation(dataset=output_path)
hh_snap_df = pd.DataFrame(sim_test.calculate_dataframe([
    "household_id", "household_weight", "congressional_district_geoid", "state_fips", "snap"])
)
mapping_df = pd.read_csv(f"{output_dir}/mappings/{h5_name}_household_mapping.csv")

merged_df = mapping_df.merge(
    hh_snap_df,
    how='inner',
    left_on='new_household_id',
    right_on='household_id'
)
fips_equal = (merged_df['state_fips_x'] == merged_df['state_fips_y']).all()
assert fips_equal

# These are the households corresponding to the non-zero weight values
merged_df = merged_df.rename(columns={'state_fips_x': 'state_fips'}).drop(columns=['state_fips_y'])

y_hat = X_sparse @ w
snap_hat_state = y_hat[row_loc]

state_df = hh_snap_df.loc[hh_snap_df.state_fips == target_geo_id]
y_hat_sim = np.sum(state_df.snap.values * state_df.household_weight.values)
print(state_df.shape)

assert np.isclose(y_hat_sim, snap_hat_state, atol=10), f"Mismatch: {y_hat_sim} vs {snap_hat_state}"

merged_df['col_pos'] = merged_df.apply(lambda row: tracer.get_household_column_positions(int(row.original_household_id))[str(int(row.congressional_district))], axis=1)
merged_df['sparse_value'] = X_sparse[row_loc, merged_df['col_pos'].values].toarray().ravel()


# Check 1. All w not in the 50k dataframe of households are zero:
w_check = w.copy()
w_check[merged_df['col_pos']] = 0
total_remainder = np.abs(w_check).sum()

if total_remainder == 0:
    print("Success: All indices outside the DataFrame have zero weight.")
else:
    offending_indices = np.nonzero(w_check)[0]
    print(f"First 5 offending indices: {offending_indices[:5]}")

# Check 2. All sparse_value values are 0 unless state_fips = 37
violations = merged_df[
    (merged_df['state_fips'] != 37) & 
    (merged_df['sparse_value'] != 0)
]

if violations.empty:
    print("Check 2 Passed: All non-37 locations have 0 sparse_value.")
else:
    print(f"Check 2 Failed: Found {len(violations)} violations.")
    print(violations[['state_fips', 'sparse_value']].head())

# Check 3. snap values are what is in the row of X_sparse for all rows where state_fips = 37
merged_state_df = merged_df.loc[merged_df.state_fips == 37]
merged_state_df.loc[merged_state_df.snap > 0.0]

# -------------------------------------------
# Debugging ---------------------------------
# -------------------------------------------
# Problem! Original household id of 178010 (new household id 5250083)
# Why does it have 2232 for snap but zero in the X_sparse matrix!?
merged_state_df.loc[merged_state_df.original_household_id == 178010]
# Let me just check the column position
tracer.get_household_column_positions(178010)['3705']

X_sparse[row_loc, 2850099]

tracer.get_household_column_positions(178010)['3701']
X_sparse[row_loc, 2796067]

# Let's check the original home state
tracer.get_household_column_positions(178010)['1501']
X_sparse[row_loc, 702327]

# Are any not zero?
for cd in cds_to_calibrate:
    col_loc = tracer.get_household_column_positions(178010)[cd]
    val = X_sparse[row_loc, col_loc]
    if val > 0:
        print(f"cd {cd} has val {val}")
# Nothing!

# Let's take a look at this household in the original simulation
debug_df = sim.calculate_dataframe(['household_id', 'state_fips', 'snap'])
debug_df.loc[debug_df.household_id == 178010]

# Interesting. It's not either one!
#Out[93]: 
#       weight  household_id  state_fips    snap
#13419     0.0        178010          15  4262.0

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

entity_rel.loc[entity_rel.household_id == 178010]

# I'm really suprised to see only one spm_unit_id
spm_df = sim.calculate_dataframe(['spm_unit_id', 'snap'], map_to="spm_unit")
spm_df.loc[spm_df.spm_unit_id == 178010002]
#Out[102]: 
#       weight  spm_unit_id    snap
#14028     0.0    178010002  4262.0

# Debugging problem
# There's just some tough questions here. Why does the base simulation show the snap as $4262 while
# the simulation that comes out of the output show $2232 while the sparse matrix has all zeros!
