"""
Full pipeline for projecting income tax revenue 2025-2100
based on demographic changes using IPF reweighting.

Usage:
    python run_full_projection.py [END_YEAR]

    END_YEAR: Optional ending year (default: 2035)

Examples:
    python run_full_projection.py 2030  # Quick test (6 years)
    python run_full_projection.py 2050  # Medium run (26 years)
    python run_full_projection.py 2100  # Full projection (76 years)
"""

import sys
import gc
import psutil
import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from create_reweighting_matrix import iterative_proportional_fitting
from age_projection import load_ssa_projections

BASE_YEAR = 2024
START_YEAR = BASE_YEAR + 1
END_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2035

print("="*70)
print(f"INCOME TAX PROJECTION: {START_YEAR}-{END_YEAR}")
print("="*70)
print(f"\nConfiguration:")
print(f"  Base year: {BASE_YEAR} (CPS microdata)")
print(f"  Projection: {START_YEAR}-{END_YEAR}")
print(f"  Years to process: {END_YEAR - START_YEAR + 1}")
print(f"  Note: Each year requires PolicyEngine sim.calculate() calls")
print(f"        Estimated time: ~{(END_YEAR - START_YEAR + 1) * 2:.0f}-{(END_YEAR - START_YEAR + 1) * 5:.0f} minutes")

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

sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

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
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

    # Calculate income tax for this year using PolicyEngine's uprating
    # This applies economic factors (wages, inflation, tax law) to BASE_YEAR data
    income_tax_microseries = sim.calculate("income_tax", period=year, map_to="household")
    baseline_weights = income_tax_microseries.weights.values  # PolicyEngine's standard weights
    income_tax_year = income_tax_microseries.values  # Uprated income tax values

    # Get SSA demographic targets for this year
    y_target = target_matrix[:, year_idx]

    # Adjust weights using IPF to match SSA age distribution
    # baseline_weights -> w_new (demographic adjustment on top of economic uprating)
    w_new, info = iterative_proportional_fitting(
        X, y_target, baseline_weights,
        max_iters=100, tol=1e-6, verbose=False
    )

    # Store results
    weights_matrix[:, year_idx] = w_new
    baseline_weights_matrix[:, year_idx] = baseline_weights
    # Same uprated income_tax_year, two different weightings:
    total_income_tax[year_idx] = np.sum(income_tax_year * w_new)  # Economic + demographic
    total_income_tax_baseline[year_idx] = np.sum(income_tax_year * baseline_weights)  # Economic only
    total_population[year_idx] = np.sum(y_target)
    avg_abs_weight_diff[year_idx] = np.mean(np.abs((w_new - baseline_weights) / baseline_weights)) * 100

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
