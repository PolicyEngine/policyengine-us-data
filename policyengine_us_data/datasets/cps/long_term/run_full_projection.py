"""
Full pipeline for projecting income tax revenue 2025-2100
based on demographic changes using IPF or GREG reweighting.

Usage:
    python run_full_projection.py [END_YEAR] [--greg] [--use-ss] [--save-h5]

    END_YEAR: Optional ending year (default: 2035)
    --greg: Use GREG calibration instead of IPF (optional)
    --use-ss: Include Social Security benefit totals as calibration target (requires --greg)
    --save-h5: Save year-specific .h5 files to ./projected_datasets/ (optional, time-consuming)

Examples:
    python run_full_projection.py 2030        # Quick test with IPF (6 years)
    python run_full_projection.py 2050 --greg # Medium run with GREG (26 years)
    python run_full_projection.py 2100        # Full projection with IPF (76 years)
    python run_full_projection.py 2100 --greg # Full projection with GREG (76 years)
    python run_full_projection.py 2050 --greg --use-ss # GREG with SS constraint
    python run_full_projection.py 2100 --greg --save-h5 # Save individual year datasets
"""

import sys
import gc
import os
import psutil
import numpy as np
import pandas as pd
import h5py
from policyengine_us import Microsimulation
from policyengine_core.data.dataset import Dataset
from policyengine_core.enums import Enum
from create_reweighting_matrix import iterative_proportional_fitting
from age_projection import load_ssa_projections


# =========================================================================
# DATASET CONFIGURATION - Switch between datasets here
# =========================================================================

DATASET_OPTIONS = {
    "enhanced_cps_2024": {
        "path": "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        "base_year": 2024,
    },
    "national_2023": {
        "path": "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/datasets/cps/geo_stacking_calibration/national/national.h5",
        "base_year": 2023,
    },
}

# SELECT DATASET HERE (change this to switch datasets)
SELECTED_DATASET = "national_2023"  # or "enhanced_cps_2024"

# Projection always starts at 2025 regardless of base year
START_YEAR = 2025

# =========================================================================


def load_ssa_benefit_projections(year):
    """
    Load SSA Trustee Report projections for Social Security benefits.

    Data from 2025 Trustees Report Table VI.G9 (OASDI Cost).
    Values are in nominal billions of dollars (converted from real 2025 dollars using CPI-W).
    """
    csv_file = os.path.join(os.path.dirname(__file__), "social_security_aux.csv")
    df = pd.read_csv(csv_file, thousands=',')

    row = df[df['year'] == year]
    nominal_billions = row['oasdi_cost_in_billion_nominal_usd'].values[0]
    return nominal_billions * 1e9


def create_year_h5(year, household_weights, base_dataset_path, output_dir):
    """
    Create a year-specific .h5 file with calibrated weights and uprated values.

    Args:
        year: The year for this dataset
        household_weights: Calibrated household weights for this year
        base_dataset_path: Path to base dataset
        output_dir: Directory to save the .h5 file

    Returns:
        Path to the created .h5 file
    """
    output_path = os.path.join(output_dir, f"{year}.h5")

    sim = Microsimulation(dataset=base_dataset_path)
    base_period = int(sim.default_calculation_period)

    # Define essential variables that must be kept even if they have formulas
    # This single definition is used consistently throughout the function
    essential_vars = {
        'person_id', 'household_id', 'tax_unit_id', 'spm_unit_id',
        'marital_unit_id', 'person_weight', 'household_weight', 'tax_unit_weight',
        'person_household_id', 'person_tax_unit_id', 'person_spm_unit_id',
        'person_marital_unit_id'
    }

    # Get all entity IDs for proper mapping
    household_ids = sim.calculate("household_id", map_to="household").values
    person_household_id = sim.calculate("person_household_id").values

    # Get other entity IDs for complete entity-level mapping
    tax_unit_ids = sim.calculate("tax_unit_id", map_to="tax_unit").values
    spm_unit_ids = sim.calculate("spm_unit_id", map_to="spm_unit").values
    marital_unit_ids = sim.calculate("marital_unit_id", map_to="marital_unit").values

    hh_id_to_idx = {int(hh_id): idx for idx, hh_id in enumerate(household_ids)}

    person_weights = np.array([
        household_weights[hh_id_to_idx[int(hh_id)]]
        for hh_id in person_household_id
    ])

    df = sim.to_input_dataframe()
    n_persons = len(df)

    person_household_id_series = df[f"person_household_id__{base_period}"]

    hh_to_weight = dict(zip(household_ids, household_weights))
    person_household_weight_uprated = person_household_id_series.map(hh_to_weight)

    df[f"household_weight__{year}"] = person_household_weight_uprated
    df[f"person_weight__{year}"] = person_weights
    df.drop(columns=[f"household_weight__{base_period}", f"person_weight__{base_period}"], inplace=True, errors='ignore')

    person_tax_unit_id_series = df[f"person_tax_unit_id__{base_period}"]
    person_spm_unit_id_series = df[f"person_spm_unit_id__{base_period}"]
    person_marital_unit_id_series = df[f"person_marital_unit_id__{base_period}"]

    valid_variables = set(sim.tax_benefit_system.variables.keys())

    for col in df.columns:
        if f"__{base_period}" in col:
            var_name = col.replace(f"__{base_period}", "")
            col_name_new = f"{var_name}__{year}"

            if var_name in ['household_weight', 'person_weight', 'tax_unit_weight']:
                continue

            if var_name not in valid_variables:
                df.rename(columns={col: col_name_new}, inplace=True)
                continue

            # Skip variables that have formulas (calculated variables)
            # Also skip variables with 'adds' attributes (they sum components)
            # Only uprate simple input variables
            # Exception: keep essential ID variables and weights even if they have formulas
            if var_name in sim.tax_benefit_system.variables and var_name not in essential_vars:
                var = sim.tax_benefit_system.variables[var_name]
                if var.formulas:  # Has formulas = calculated variable
                    # Don't uprate calculated variables, let them be recalculated
                    df.drop(columns=[col], inplace=True)
                    continue
                if (hasattr(var, 'adds') and var.adds) or (hasattr(var, 'subtracts') and var.subtracts):
                    # Don't uprate - let it calculate from its components
                    df.drop(columns=[col], inplace=True)
                    continue

            try:
                uprated_values = sim.calculate(var_name, period=year).values
            except:
                df.rename(columns={col: col_name_new}, inplace=True)
                continue

            if len(uprated_values) == n_persons:
                df[col_name_new] = uprated_values
            elif len(uprated_values) == len(household_ids):
                hh_to_value = dict(zip(household_ids, uprated_values))
                df[col_name_new] = person_household_id_series.map(hh_to_value)
            elif len(uprated_values) == len(tax_unit_ids):
                tu_to_value = dict(zip(tax_unit_ids, uprated_values))
                df[col_name_new] = person_tax_unit_id_series.map(tu_to_value)
            elif len(uprated_values) == len(spm_unit_ids):
                spm_to_value = dict(zip(spm_unit_ids, uprated_values))
                df[col_name_new] = person_spm_unit_id_series.map(spm_to_value)
            elif len(uprated_values) == len(marital_unit_ids):
                mu_to_value = dict(zip(marital_unit_ids, uprated_values))
                df[col_name_new] = person_marital_unit_id_series.map(mu_to_value)
            else:
                # Unknown entity type - just rename without uprating
                df.rename(columns={col: col_name_new}, inplace=True)
                continue

            df.drop(columns=[col], inplace=True)

    # Note: We've already dropped formula-based variables above to ensure they
    # are recalculated rather than stored with potentially incorrect uprated values

    dataset = Dataset.from_dataframe(df, year)

    new_sim = Microsimulation()
    new_sim.dataset = dataset
    new_sim.build_from_dataset()

    data = {}

    # Use the essential_vars already defined at the top of the function
    for variable in new_sim.tax_benefit_system.variables:
        var_def = new_sim.tax_benefit_system.variables[variable]

        # Skip calculated variables (those with formulas) unless they're essential IDs/weights
        if variable not in essential_vars:
            if var_def.formulas:
                continue

            # Skip aggregate variables (those with adds/subtracts)
            if (hasattr(var_def, 'adds') and var_def.adds) or (hasattr(var_def, 'subtracts') and var_def.subtracts):
                continue

        data[variable] = {}
        for period in new_sim.get_holder(variable).get_known_periods():
            values = new_sim.get_holder(variable).get_array(period)

            value_type = new_sim.tax_benefit_system.variables.get(variable).value_type
            if value_type in (Enum, str) and variable != "county_fips":
                if hasattr(values, "decode_to_str"):
                    values = values.decode_to_str().astype("S")
                else:
                    values = values.astype("S")
            elif variable == "county_fips":
                values = values.astype("int32")
            else:
                values = np.array(values)

            if values is not None:
                data[variable][period] = values

        if len(data[variable]) == 0:
            del data[variable]

    with h5py.File(output_path, "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    del sim, new_sim
    gc.collect()

    return output_path


# Load selected dataset configuration
BASE_DATASET_PATH = DATASET_OPTIONS[SELECTED_DATASET]["path"]
BASE_YEAR = DATASET_OPTIONS[SELECTED_DATASET]["base_year"]

# Check for --greg flag before parsing END_YEAR
USE_GREG = '--greg' in sys.argv
if USE_GREG:
    sys.argv.remove('--greg')  # Remove so it doesn't interfere with END_YEAR

# Check for --use-ss flag (only works with GREG)
# Use social security target?
USE_SS = '--use-ss' in sys.argv
if USE_SS:
    sys.argv.remove('--use-ss')
    if not USE_GREG:
        print("Warning: --use-ss requires --greg, enabling GREG automatically")
        USE_GREG = True

# Check for --save-h5 flag to create year-specific .h5 files
SAVE_H5 = '--save-h5' in sys.argv
if SAVE_H5:
    sys.argv.remove('--save-h5')

END_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2035

# Import samplics if using GREG
if USE_GREG:
    from samplics.weighting import SampleWeight
    calibrator = SampleWeight()

OUTPUT_DIR = "./projected_datasets"

print("="*70)
print(f"INCOME TAX PROJECTION: {START_YEAR}-{END_YEAR}")
print("="*70)
print(f"\nConfiguration:")
print(f"  Base year: {BASE_YEAR} (CPS microdata)")
print(f"  Projection: {START_YEAR}-{END_YEAR}")
print(f"  Calibration method: {'GREG' if USE_GREG else 'IPF'}")
if USE_SS:
    print(f"  Including Social Security benefits constraint: Yes")
if SAVE_H5:
    print(f"  Saving year-specific .h5 files: Yes (to {OUTPUT_DIR}/)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
else:
    print(f"  Saving year-specific .h5 files: No (use --save-h5 to enable)")
print(f"  Years to process: {END_YEAR - START_YEAR + 1}")
print(f"  Note: Each year requires PolicyEngine sim.calculate() calls")
est_time_low = (END_YEAR - START_YEAR + 1) * (3 if SAVE_H5 else 2)
est_time_high = (END_YEAR - START_YEAR + 1) * (7 if SAVE_H5 else 5)
print(f"        Estimated time: ~{est_time_low:.0f}-{est_time_high:.0f} minutes")

# =========================================================================
# STEP 1: LOAD SSA DEMOGRAPHIC PROJECTIONS
# =========================================================================
print("\n" + "="*70)
print("STEP 1: DEMOGRAPHIC PROJECTIONS")
print("="*70)

target_matrix = load_ssa_projections(end_year=END_YEAR)
n_years = target_matrix.shape[1]
n_ages = target_matrix.shape[0]

print(f"\nLoaded SSA projections: {n_ages} ages x {n_years} years")
print(f"\nPopulation projections:")

display_years = [y for y in [START_YEAR, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
                 if START_YEAR <= y <= END_YEAR]
if END_YEAR not in display_years:
    display_years.append(END_YEAR)

for y in display_years:
    idx = y - START_YEAR
    if idx < n_years:
        pop = target_matrix[:, idx].sum()
        print(f"  {y}: {pop/1e6:6.1f}M")

# =========================================================================
# STEP 2: LOAD CPS DATA
# =========================================================================
print("\n" + "="*70)
print("STEP 2: LOADING CPS MICROSIMULATION DATA")
print("="*70)

sim = Microsimulation(dataset=BASE_DATASET_PATH)

age_person = sim.calculate("age")
person_household_id = sim.calculate("person_household_id")

household_ids_unique = np.unique(person_household_id.values)
n_households = len(household_ids_unique)

print(f"\nLoaded {n_households:,} households")

X = np.zeros((n_households, n_ages))
hh_id_to_idx = {hh_id: idx for idx, hh_id in enumerate(household_ids_unique)}

for person_idx in range(len(age_person)):
    age = int(age_person.values[person_idx])
    hh_id = person_household_id.values[person_idx]
    hh_idx = hh_id_to_idx[hh_id]
    age_idx = min(age, 85)
    X[hh_idx, age_idx] += 1

household_microseries = sim.calculate("household_id", map_to="household")
initial_weights = household_microseries.weights.values

print(f"Design matrix shape: {X.shape}")
print(f"Initial weights shape: {initial_weights.shape}")

# Clean up initial sim before entering loop
del sim
gc.collect()

# =========================================================================
# STEP 3: REWEIGHT AND PROJECT INCOME TAX
# =========================================================================
print("\n" + "="*70)
print("STEP 3: REWEIGHTING AND PROJECTING INCOME TAX")
print("="*70)
print("\nMethodology:")
print(f"  1. PolicyEngine uprates {BASE_YEAR} microdata to each projection year")
print(f"     (applies wage growth, inflation, tax bracket adjustments)")
print(f"  2. IPF adjusts weights to match SSA age demographics")
print(f"  3. Compare: baseline (economic only) vs adjusted (economic + demographic)")

years = np.arange(START_YEAR, END_YEAR + 1)
total_income_tax = np.zeros(n_years)
total_income_tax_baseline = np.zeros(n_years)
total_population = np.zeros(n_years)
weights_matrix = np.zeros((n_households, n_years))
baseline_weights_matrix = np.zeros((n_households, n_years))
avg_abs_weight_diff = np.zeros(n_years)

process = psutil.Process()
print(f"\nInitial memory usage: {process.memory_info().rss / 1024**3:.2f} GB")

print("\nYear    Population    Income Tax    Baseline Tax    Wt Diff%   IPF Its   Memory")
print("-" * 85)

for year_idx in range(n_years):
    year = START_YEAR + year_idx

    # Reload simulation each year to prevent memory leak in PolicyEngine
    sim = Microsimulation(dataset=BASE_DATASET_PATH)

    # Calculate baseline totals using MicroSeries (proper person-level weighting)
    income_tax_baseline_total = sim.calculate('income_tax', period=year, map_to='person').sum()

    if USE_SS:
        ss_baseline_total = sim.calculate('social_security', period=year, map_to='person').sum()
        ss_target = load_ssa_benefit_projections(year)
        print(f"  SS before calibration: ${ss_baseline_total/1e9:.1f}B")
        print(f"  SS target for {year}: ${ss_target/1e9:.1f}B (need {ss_target/ss_baseline_total:.2f}x adjustment)")

    # Get person-level data for aggregation to household
    df = pd.DataFrame({
        'household_id': sim.calculate('household_id', map_to='person').values,
        'income_tax': sim.calculate('income_tax', period=year, map_to='person').values,
    })

    if USE_SS:
        df['social_security'] = sim.calculate('social_security', period=year, map_to='person').values

    # Aggregate to household level (sum person values within each household)
    hh_agg = df.groupby('household_id').agg({
        'income_tax': 'sum',
    }).reset_index()

    if USE_SS:
        hh_agg['social_security'] = df.groupby('household_id')['social_security'].sum().values

    # Get baseline household weights and household IDs in correct order
    household_microseries = sim.calculate("household_id", map_to="household")
    household_ids_hh = household_microseries.values
    baseline_weights = household_microseries.weights.values

    # Match aggregated household values to the household weight order
    hh_to_income_tax = dict(zip(hh_agg['household_id'], hh_agg['income_tax']))
    income_tax_year = np.array([hh_to_income_tax[hh_id] for hh_id in household_ids_hh])

    if USE_SS:
        hh_to_ss = dict(zip(hh_agg['household_id'], hh_agg['social_security']))
        ss_benefits_hh = np.array([hh_to_ss[hh_id] for hh_id in household_ids_hh])

    # Get SSA demographic targets for this year
    y_target = target_matrix[:, year_idx]

    # Adjust weights using either IPF or GREG to match SSA age distribution
    # baseline_weights -> w_new (demographic adjustment on top of economic uprating)

    if USE_GREG:
        # Build auxiliary matrix and controls
        controls = {}
        for age_idx in range(n_ages):
            controls[f'age_{age_idx}'] = y_target[age_idx]

        # Add Social Security benefits if requested
        if USE_SS:
            # Convert to DataFrame to ensure samplics can match control names to columns
            import pandas as pd
            age_cols = {f'age_{i}': X[:, i] for i in range(n_ages)}
            aux_df = pd.DataFrame(age_cols)
            aux_df['ss_total'] = ss_benefits_hh

            controls['ss_total'] = ss_target

            aux_vars = aux_df
        else:
            # Just use age variables
            aux_vars = X

        # GREG calibration
        try:
            w_new = calibrator.calibrate(
                samp_weight=baseline_weights,
                aux_vars=aux_vars,  # sample data
                control=controls   # population_data
            )
            info = {'iterations': 1}  # samplics doesn't return iteration count
        except Exception as e:
            print(f"  GREG failed for {year}: {e}, falling back to IPF")
            w_new, info = iterative_proportional_fitting(
                X, y_target, baseline_weights,
                max_iters=100, tol=1e-6, verbose=False
            )
    else:
        # Original IPF calibration
        w_new, info = iterative_proportional_fitting(
            X, y_target, baseline_weights,
            max_iters=100, tol=1e-6, verbose=False
        )

    # Diagnostic: Check if SS calibration actually worked
    if USE_SS:
        ss_actual = np.sum(ss_benefits_hh * w_new)
        ss_residual = ss_actual - ss_target
        ss_rel_error = (ss_residual / ss_target) * 100
        print(f"  SS calibration check: Actual=${ss_actual/1e9:.1f}B, Target=${ss_target/1e9:.1f}B, Residual=${ss_residual/1e9:.1f}B ({ss_rel_error:+.2f}%)")

    # Store results
    weights_matrix[:, year_idx] = w_new
    baseline_weights_matrix[:, year_idx] = baseline_weights
    # Two versions: baseline uses MicroSeries sum, calibrated uses our weights
    total_income_tax[year_idx] = np.sum(income_tax_year * w_new)  # Economic + demographic (our calibration)
    total_income_tax_baseline[year_idx] = income_tax_baseline_total  # Economic only (MicroSeries sum)
    total_population[year_idx] = np.sum(y_target)
    avg_abs_weight_diff[year_idx] = np.mean(np.abs((w_new - baseline_weights) / baseline_weights)) * 100

    # Create year-specific .h5 file if requested
    if SAVE_H5:
        h5_path = create_year_h5(year, w_new, BASE_DATASET_PATH, OUTPUT_DIR)
        if year in display_years:
            print(f"  Saved {year}.h5")

    # Clean up simulation object to free memory
    del sim
    gc.collect()

    mem_gb = process.memory_info().rss / 1024**3

    if year in display_years:
        tax_billions = total_income_tax[year_idx] / 1e9
        baseline_billions = total_income_tax_baseline[year_idx] / 1e9
        pop_millions = total_population[year_idx] / 1e6
        wt_diff_pct = avg_abs_weight_diff[year_idx]
        print(f"{year}    {pop_millions:7.1f}M     ${tax_billions:7.1f}B     ${baseline_billions:7.1f}B      {wt_diff_pct:6.1f}%     {info['iterations']:3d}    {mem_gb:.2f}GB")
    elif year_idx % 2 == 0:
        print(f"{year}    Processing... ({year_idx+1}/{n_years})                                              {mem_gb:.2f}GB")

# =========================================================================
# STEP 4: ANALYZE RESULTS
# =========================================================================
print("\n" + "="*70)
print("STEP 4: ANALYSIS")
print("="*70)

# Create results dataframe
results_df = pd.DataFrame({
    'year': years,
    'population': total_population,
    'income_tax': total_income_tax,
    'income_tax_baseline': total_income_tax_baseline,
    'income_tax_billions': total_income_tax / 1e9,
    'income_tax_baseline_billions': total_income_tax_baseline / 1e9,
    'population_millions': total_population / 1e6,
    'income_tax_per_capita': total_income_tax / total_population,
    'avg_abs_weight_diff': avg_abs_weight_diff
})

# Calculate age distribution metrics
elderly_share = np.zeros(n_years)
working_age_share = np.zeros(n_years)
for year_idx in range(n_years):
    year_pop = target_matrix[:, year_idx]
    total = year_pop.sum()
    working_age_share[year_idx] = np.sum(year_pop[20:65]) / total  # 20-64
    elderly_share[year_idx] = np.sum(year_pop[65:]) / total  # 65+

results_df['working_age_share'] = working_age_share
results_df['elderly_share'] = elderly_share

# Key metrics
print("\nKEY FINDINGS:")
print("-" * 40)

pop_change = (total_population[-1] / total_population[0] - 1) * 100
print(f"Population change 2025-2100: {pop_change:+.1f}%")

tax_change = (total_income_tax[-1] / total_income_tax[0] - 1) * 100
tax_baseline_change = (total_income_tax_baseline[-1] / total_income_tax_baseline[0] - 1) * 100
print(f"Income tax change (with demographics): {tax_change:+.1f}%")
print(f"Income tax change (baseline uprating): {tax_baseline_change:+.1f}%")

demographic_effect = total_income_tax[-1] - total_income_tax_baseline[-1]
print(f"Demographic adjustment (2100): ${demographic_effect/1e9:+.1f}B")

working_2025 = working_age_share[0] * 100
working_2100 = working_age_share[-1] * 100
print(f"Working age (20-64) share: {working_2025:.1f}% → {working_2100:.1f}%")

elderly_2025 = elderly_share[0] * 100
elderly_2100 = elderly_share[-1] * 100
print(f"Elderly (65+) share: {elderly_2025:.1f}% → {elderly_2100:.1f}%")

per_capita_2025 = total_income_tax[0] / total_population[0]
per_capita_2100 = total_income_tax[-1] / total_population[-1]
print(f"Per capita income tax: ${per_capita_2025:.0f} → ${per_capita_2100:.0f}")

avg_weight_diff_start = avg_abs_weight_diff[0]
avg_weight_diff_end = avg_abs_weight_diff[-1]
print(f"Avg abs weight adjustment: {avg_weight_diff_start:.1f}% → {avg_weight_diff_end:.1f}%")

# Optional: Save weights matrix if needed for other analyses
save_weights = False
if save_weights:
    np.save('household_weights_full.npy', weights_matrix)
    print(f"\nSaved weights matrix: household_weights_full.npy")

# =========================================================================
# STEP 5: CREATE VISUALIZATION
# =========================================================================
print("\n" + "="*70)
print("STEP 5: CREATING VISUALIZATION")
print("="*70)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Calculate youth share for the plot
youth_share = np.zeros(n_years)
for year_idx in range(n_years):
    year_pop = target_matrix[:, year_idx]
    total = year_pop.sum()
    youth_share[year_idx] = np.sum(year_pop[:20]) / total  # 0-19

# Create 4-panel figure
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Total Income Tax', 'Income Tax Per Capita',
                   'Total Population', 'Age Distribution'),
    specs=[[{'secondary_y': False}, {'secondary_y': False}],
           [{'secondary_y': False}, {'secondary_y': False}]]
)

# Plot 1: Total Income Tax
fig.add_trace(
    go.Scatter(x=years, y=total_income_tax/1e9, name='Income Tax',
               line=dict(color='blue', width=2)),
    row=1, col=1
)

# Plot 2: Income Tax Per Capita
fig.add_trace(
    go.Scatter(x=years, y=total_income_tax/total_population, name='Per Capita',
               line=dict(color='green', width=2)),
    row=1, col=2
)

# Plot 3: Total Population
fig.add_trace(
    go.Scatter(x=years, y=total_population/1e6, name='Population',
               line=dict(color='purple', width=2)),
    row=2, col=1
)

# Plot 4: Age Distribution
fig.add_trace(
    go.Scatter(x=years, y=youth_share*100, name='Youth (0-19)',
               line=dict(color='cyan', width=2)),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=years, y=working_age_share*100, name='Working (20-64)',
               line=dict(color='orange', width=2)),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=years, y=elderly_share*100, name='Elderly (65+)',
               line=dict(color='red', width=2)),
    row=2, col=2
)

# Update axes labels
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=2, col=2)
fig.update_yaxes(title_text="Billions ($)", row=1, col=1)
fig.update_yaxes(title_text="Dollars", row=1, col=2)
fig.update_yaxes(title_text="Millions", row=2, col=1)
fig.update_yaxes(title_text="Percent (%)", row=2, col=2)

# Update layout
fig.update_layout(
    title=f'Income Tax and Demographics Projections (2025-{END_YEAR})',
    height=700,
    width=1000,
    showlegend=True
)

fig.write_html('income_tax_projections.html')
print("Saved visualization: income_tax_projections.html")
