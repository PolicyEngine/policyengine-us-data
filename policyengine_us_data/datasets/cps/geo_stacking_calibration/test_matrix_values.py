"""Test matrix values with our own weights."""

import numpy as np
from policyengine_us import Microsimulation
from metrics_matrix_geo_stacking import GeoStackingMatrixBuilder

# Database path
db_uri = "sqlite:////home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"

# Initialize builder
builder = GeoStackingMatrixBuilder(db_uri, time_period=2023)

# Create microsimulation
print("Loading microsimulation...")
sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim.default_calculation_period = 2023
sim.build_from_dataset()

# Build matrix for California
print("\nBuilding matrix for California (FIPS 6)...")
targets_df, matrix_df = builder.build_matrix_for_geography('state', '6', sim)

print("\nTarget Summary:")
print(f"Total targets: {len(targets_df)}")
print(f"Matrix shape: {matrix_df.shape} (targets x households)")

# Create our own weights - start with uniform
n_households = matrix_df.shape[1]
uniform_weights = np.ones(n_households) / n_households

# Calculate estimates with uniform weights
estimates = matrix_df.values @ uniform_weights

print("\nMatrix check:")
print(f"Non-zero entries in matrix: {(matrix_df.values != 0).sum()}")
print(f"Max value in matrix: {matrix_df.values.max()}")

print("\nFirst 5 rows (targets) sum across households:")
for i in range(min(5, len(targets_df))):
    row_sum = matrix_df.iloc[i].sum()
    target = targets_df.iloc[i]
    print(f"  {target['description']}: row sum={row_sum:.0f} (count of people in this age group)")

print("\nEstimates with uniform weights (1/n for each household):")
for i in range(min(5, len(targets_df))):
    target = targets_df.iloc[i]
    estimate = estimates[i]
    print(f"  {target['description']}: target={target['value']:,.0f}, estimate={estimate:.2f}")

# Try with equal total weight = US population
us_population = 330_000_000  # Approximate
scaled_weights = np.ones(n_households) * (us_population / n_households)

scaled_estimates = matrix_df.values @ scaled_weights

print(f"\nEstimates with scaled weights (total weight = {us_population:,}):")
for i in range(min(5, len(targets_df))):
    target = targets_df.iloc[i]
    estimate = scaled_estimates[i]
    ratio = estimate / target['value'] if target['value'] > 0 else 0
    print(f"  {target['description']}: target={target['value']:,.0f}, estimate={estimate:,.0f}, ratio={ratio:.2f}")

print("\nKey insights:")
print("1. The matrix values are counts of people in each age group per household")
print("2. Row sums show total people in that age group across all households (unweighted)")
print("3. With uniform weights, we get the average per household")
print("4. With scaled weights, we see the estimates are ~7-8x the CA targets")
print("5. This makes sense: US population / CA population ≈ 8")
print("6. The calibration will find weights that match CA targets exactly")