import os
import numpy as np
import pandas as pd
from scipy import sparse as sp
from policyengine_us import Microsimulation

print("=" * 70)
print("CONGRESSIONAL DISTRICT CALIBRATION DIAGNOSTICS")
print("=" * 70)

# Load the microsimulation that was used for CD calibration
# CRITICAL: Must use stratified CPS for CDs
print("\nLoading stratified CPS microsimulation...")
dataset_path = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/stratified_extended_cps_2023.h5"
sim = Microsimulation(dataset=dataset_path)
sim.build_from_dataset()

household_ids = sim.calculate("household_id", map_to="household").values
n_households_total = len(household_ids)
print(f"Total households in stratified simulation: {n_households_total:,}")

# Set up paths
export_dir = os.path.expanduser("~/Downloads/cd_calibration_data")
os.makedirs(export_dir, exist_ok=True)

# Load CD calibration matrix and weights
print("\nLoading calibration matrix and weights...")
X_sparse = sp.load_npz(os.path.join(export_dir, "cd_matrix_sparse.npz"))
print(f"Matrix shape: {X_sparse.shape}")

w = np.load('w_cd_20250911_102023.npy')
n_active = sum(w != 0)
print(f"Sparsity: {n_active:,} active weights out of {len(w):,} ({100*n_active/len(w):.2f}%)")

targets_df = pd.read_pickle(os.path.join(export_dir, "cd_targets_df.pkl"))
print(f"Number of targets: {len(targets_df):,}")

# Calculate predictions
print("\nCalculating predictions...")
y_pred = X_sparse @ w
y_actual = targets_df['value'].values

correlation = np.corrcoef(y_pred, y_actual)[0, 1]
print(f"Correlation between predicted and actual: {correlation:.4f}")

# Calculate errors
abs_errors = np.abs(y_actual - y_pred)
rel_errors = np.abs((y_actual - y_pred) / (y_actual + 1))

targets_df['y_pred'] = y_pred
targets_df['abs_error'] = abs_errors
targets_df['rel_error'] = rel_errors

# Overall statistics
print("\n" + "=" * 70)
print("OVERALL ERROR STATISTICS")
print("=" * 70)
print(f"Mean relative error: {np.mean(rel_errors):.2%}")
print(f"Median relative error: {np.median(rel_errors):.2%}")
print(f"Max relative error: {np.max(rel_errors):.2%}")
print(f"95th percentile error: {np.percentile(rel_errors, 95):.2%}")
print(f"99th percentile error: {np.percentile(rel_errors, 99):.2%}")

# Worst performing targets
print("\n" + "=" * 70)
print("WORST PERFORMING TARGETS (Top 10)")
print("=" * 70)

worst_targets = targets_df.nlargest(10, 'rel_error')
for idx, row in worst_targets.iterrows():
    cd_label = f"CD {row['geographic_id']}" if row['geographic_id'] != 'US' else "National"
    print(f"\n{cd_label} - {row['variable']} (Group {row['stratum_group_id']})")
    print(f"  Description: {row['description']}")
    print(f"  Target: {row['value']:,.0f}, Predicted: {row['y_pred']:,.0f}")
    print(f"  Relative Error: {row['rel_error']:.1%}")

# Error by congressional district
print("\n" + "=" * 70)
print("ERROR ANALYSIS BY CONGRESSIONAL DISTRICT")
print("=" * 70)

cd_errors = targets_df[targets_df['geographic_id'] != 'US'].groupby('geographic_id').agg({
    'rel_error': ['mean', 'median', 'max', 'count']
}).round(4)

cd_errors = cd_errors.sort_values(('rel_error', 'mean'), ascending=False)

print("\nTop 10 CDs with highest mean relative error:")
for cd_id in cd_errors.head(10).index:
    cd_data = cd_errors.loc[cd_id]
    n_targets = cd_data[('rel_error', 'count')]
    mean_err = cd_data[('rel_error', 'mean')]
    max_err = cd_data[('rel_error', 'max')]
    median_err = cd_data[('rel_error', 'median')]
    
    # Parse CD GEOID (e.g., '3601' = Alabama 1st)
    state_fips = cd_id[:-2] if len(cd_id) > 2 else cd_id
    district = cd_id[-2:]
    print(f"CD {cd_id} (State {state_fips}, District {district}): Mean={mean_err:.1%}, Median={median_err:.1%}, Max={max_err:.1%} ({n_targets:.0f} targets)")

print("\nTop 10 CDs with lowest mean relative error:")
for cd_id in cd_errors.tail(10).index:
    cd_data = cd_errors.loc[cd_id]
    n_targets = cd_data[('rel_error', 'count')]
    mean_err = cd_data[('rel_error', 'mean')]
    median_err = cd_data[('rel_error', 'median')]
    
    state_fips = cd_id[:-2] if len(cd_id) > 2 else cd_id
    district = cd_id[-2:]
    print(f"CD {cd_id} (State {state_fips}, District {district}): Mean={mean_err:.1%}, Median={median_err:.1%} ({n_targets:.0f} targets)")

# Error by target type
print("\n" + "=" * 70)
print("ERROR ANALYSIS BY TARGET TYPE")
print("=" * 70)

type_errors = targets_df.groupby('stratum_group_id').agg({
    'rel_error': ['mean', 'median', 'max', 'count']
}).round(4)

type_errors = type_errors.sort_values(('rel_error', 'mean'), ascending=False)

group_name_map = {
    2: 'Age histogram',
    3: 'AGI distribution', 
    4: 'SNAP',
    5: 'Medicaid',
    6: 'EITC'
}

print("\nError by target type (sorted by mean error):")
for type_id in type_errors.index:
    type_data = type_errors.loc[type_id]
    n_targets = type_data[('rel_error', 'count')]
    mean_err = type_data[('rel_error', 'mean')]
    max_err = type_data[('rel_error', 'max')]
    median_err = type_data[('rel_error', 'median')]
    
    if type_id in group_name_map:
        type_label = group_name_map[type_id]
    else:
        type_label = str(type_id)[:30]
    
    print(f"{type_label:30}: Mean={mean_err:.1%}, Median={median_err:.1%}, Max={max_err:.1%} ({n_targets:.0f} targets)")

# Group-wise performance
from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import create_target_groups
target_groups, group_info = create_target_groups(targets_df)

print("\n" + "=" * 70)
print("GROUP-WISE PERFORMANCE")
print("=" * 70)

group_means = []
for group_id in np.unique(target_groups):
    group_mask = target_groups == group_id
    group_errors = rel_errors[group_mask]
    group_means.append(np.mean(group_errors))

print(f"Mean of group means: {np.mean(group_means):.2%}")
print(f"Max group mean: {np.max(group_means):.2%}")

# Active weights analysis by CD
print("\n" + "=" * 70)
print("ACTIVE WEIGHTS ANALYSIS")
print("=" * 70)

print(f"\nTotal weights: {len(w):,}")
print(f"Active weights (non-zero): {n_active:,}")

# Load CD list from calibration
print("\nLoading CD list...")
# Get unique CD GEOIDs from targets_df
cds_to_calibrate = sorted([cd for cd in targets_df['geographic_id'].unique() if cd != 'US'])
n_cds = len(cds_to_calibrate)
print(f"Found {n_cds} congressional districts in targets")
n_households_per_cd = n_households_total

print(f"\nWeight vector structure:")
print(f"  Congressional Districts: {n_cds}")
print(f"  Households per CD: {n_households_per_cd:,}")
print(f"  Expected weight length: {n_cds * n_households_per_cd:,}")
print(f"  Actual weight length: {len(w):,}")

# Map weights to CDs and households
weight_to_cd = {}
weight_to_household = {}
for cd_idx, cd_geoid in enumerate(cds_to_calibrate):
    start_idx = cd_idx * n_households_per_cd
    for hh_idx, hh_id in enumerate(household_ids):
        weight_idx = start_idx + hh_idx
        weight_to_cd[weight_idx] = cd_geoid
        weight_to_household[weight_idx] = (hh_id, cd_geoid)

# Count active weights per CD
active_weights_by_cd = {}
for idx, weight_val in enumerate(w):
    if weight_val != 0:
        cd = weight_to_cd.get(idx, 'unknown')
        if cd not in active_weights_by_cd:
            active_weights_by_cd[cd] = 0
        active_weights_by_cd[cd] += 1

# Activation rates
activation_rates = [(cd, active_weights_by_cd.get(cd, 0) / n_households_per_cd) 
                   for cd in cds_to_calibrate]
activation_rates.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 CDs by activation rate:")
for cd, rate in activation_rates[:10]:
    active = active_weights_by_cd.get(cd, 0)
    cd_targets = targets_df[targets_df['geographic_id'] == cd]
    if not cd_targets.empty:
        mean_error = cd_targets['rel_error'].mean()
        print(f"  CD {cd}: {100*rate:.1f}% active ({active}/{n_households_per_cd}), Mean error: {mean_error:.1%}")
    else:
        print(f"  CD {cd}: {100*rate:.1f}% active ({active}/{n_households_per_cd})")

print("\nBottom 10 CDs by activation rate:")
for cd, rate in activation_rates[-10:]:
    active = active_weights_by_cd.get(cd, 0)
    cd_targets = targets_df[targets_df['geographic_id'] == cd]
    if not cd_targets.empty:
        mean_error = cd_targets['rel_error'].mean()
        print(f"  CD {cd}: {100*rate:.1f}% active ({active}/{n_households_per_cd}), Mean error: {mean_error:.1%}")
    else:
        print(f"  CD {cd}: {100*rate:.1f}% active ({active}/{n_households_per_cd})")

# Universal donor analysis
print("\n" + "=" * 70)
print("UNIVERSAL DONOR HOUSEHOLDS")
print("=" * 70)

household_cd_counts = {}
for idx, weight_val in enumerate(w):
    if weight_val != 0:
        hh_id, cd = weight_to_household.get(idx, (None, None))
        if hh_id is not None:
            if hh_id not in household_cd_counts:
                household_cd_counts[hh_id] = []
            household_cd_counts[hh_id].append(cd)

unique_households = len(household_cd_counts)
total_appearances = sum(len(cds) for cds in household_cd_counts.values())
avg_cds_per_household = total_appearances / unique_households if unique_households > 0 else 0

print(f"\nUnique active households: {unique_households:,}")
print(f"Total household-CD pairs: {total_appearances:,}")
print(f"Average CDs per active household: {avg_cds_per_household:.2f}")

# Distribution
cd_count_distribution = {}
for hh_id, cds in household_cd_counts.items():
    count = len(cds)
    if count not in cd_count_distribution:
        cd_count_distribution[count] = 0
    cd_count_distribution[count] += 1

print("\nDistribution of households by number of CDs they appear in:")
for count in sorted(cd_count_distribution.keys())[:10]:
    n_households = cd_count_distribution[count]
    pct = 100 * n_households / unique_households
    print(f"  {count} CD(s): {n_households:,} households ({pct:.1f}%)")

if max(cd_count_distribution.keys()) > 10:
    print(f"  ...")
    print(f"  Maximum: {max(cd_count_distribution.keys())} CDs")

# Weight distribution by CD
print("\n" + "=" * 70)
print("WEIGHT DISTRIBUTION BY CD")
print("=" * 70)

weights_by_cd = {}
for idx, weight_val in enumerate(w):
    if weight_val != 0:
        cd = weight_to_cd.get(idx, 'unknown')
        if cd not in weights_by_cd:
            weights_by_cd[cd] = []
        weights_by_cd[cd].append(weight_val)

# Get CD populations
cd_populations = {}
for cd_geoid in cds_to_calibrate:
    cd_age_targets = targets_df[(targets_df['geographic_id'] == cd_geoid) & 
                                (targets_df['variable'] == 'person_count') &
                                (targets_df['description'].str.contains('age', na=False))]
    if not cd_age_targets.empty:
        unique_ages = cd_age_targets.drop_duplicates(subset=['description'])
        cd_populations[cd_geoid] = unique_ages['value'].sum()

print("\nPopulation Target Achievement for Sample CDs:")
print("-" * 70)
print(f"{'CD':<10} {'State':<8} {'Population':<12} {'Active':<8} {'Sum Weights':<12} {'Achievement':<12}")
print("-" * 70)

# Sample some interesting CDs
sample_cds = ['3601', '601', '1201', '2701', '3611', '4801', '5301']  # AL-01, CA-01, FL-01, MN-01, NY-11, TX-01, WA-01
for cd_geoid in sample_cds:
    if cd_geoid in weights_by_cd and cd_geoid in cd_populations:
        population_target = cd_populations[cd_geoid]
        active_weights = np.array(weights_by_cd[cd_geoid])
        total_weight = np.sum(active_weights)
        achievement_ratio = total_weight / population_target if population_target > 0 else 0
        n_active = len(active_weights)
        
        state_fips = cd_geoid[:-2] if len(cd_geoid) > 2 else cd_geoid
        district = cd_geoid[-2:]
        
        print(f"{cd_geoid:<10} {state_fips:<8} {population_target:>11,.0f} {n_active:>7} {total_weight:>11,.0f} {achievement_ratio:>11.1%}")

print("\n" + "=" * 70)
print("CALIBRATION DIAGNOSTICS COMPLETE")
print("=" * 70)
print("\nFor sparse CD-stacked dataset creation, use:")
print("  python create_sparse_cd_stacked.py")
print("\nTo use the dataset:")
print('  sim = Microsimulation(dataset="/path/to/sparse_cd_stacked_2023.h5")')

# Export to calibration log CSV format
print("\n" + "=" * 70)
print("EXPORTING TO CALIBRATION LOG CSV FORMAT")
print("=" * 70)

# Create calibration log rows
log_rows = []
for idx, row in targets_df.iterrows():
    # Create target name in hierarchical format
    if row['geographic_id'] == 'US':
        target_name = f"nation/{row['variable']}/{row['description']}"
    else:
        # Congressional district format - use CD GEOID
        target_name = f"CD{row['geographic_id']}/{row['variable']}/{row['description']}"
    
    # Calculate metrics
    estimate = row['y_pred']
    target = row['value']
    error = estimate - target
    rel_error = error / target if target != 0 else 0
    abs_error = abs(error)
    rel_abs_error = abs(rel_error)
    loss = rel_error ** 2
    
    log_rows.append({
        'target_name': target_name,
        'estimate': estimate,
        'target': target,
        'epoch': 0,  # Single evaluation, not training epochs
        'error': error,
        'rel_error': rel_error,
        'abs_error': abs_error,
        'rel_abs_error': rel_abs_error,
        'loss': loss
    })

# Create DataFrame and save
calibration_log_df = pd.DataFrame(log_rows)
csv_path = 'cd_calibration_log.csv'
calibration_log_df.to_csv(csv_path, index=False)
print(f"\nSaved calibration log to: {csv_path}")
print(f"Total rows: {len(calibration_log_df):,}")

# Show sample of the CSV
print("\nSample rows from calibration log:")
print(calibration_log_df.head(10).to_string(index=False, max_colwidth=50))