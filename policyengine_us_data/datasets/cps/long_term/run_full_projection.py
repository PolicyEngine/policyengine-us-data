"""
Full pipeline for projecting income tax revenue 2025-2100
based on demographic changes using IPF reweighting.
"""

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from create_reweighting_matrix import iterative_proportional_fitting, create_age_design_matrix
from age_projection import create_annual_transition_matrix, y_age_2024


# =========================================================================
# STEP 1: CREATE DEMOGRAPHIC PROJECTIONS
# =========================================================================
print("="*70)
print("DEMOGRAPHIC PROJECTIONS (2025-2100)")
print("="*70)

# Setup
n_years = 76  # 2025-2100
n_brackets = 18
annual_births = 3_800_000

# Create transition matrix
T = create_annual_transition_matrix()

# Project population year by year
projections = np.zeros((n_years + 1, n_brackets))
projections[0] = y_age_2024

for year in range(n_years):
    projections[year + 1] = T @ projections[year]
    projections[year + 1, 0] += annual_births  # Add births to 0-4 bracket

# Extract 2025-2100 (skip 2024) and transpose to get 18 x 76
target_matrix = projections[1:, :].T

print(f"\nPopulation projections:")
for y in [2025, 2050, 2075, 2100]:
    idx = y - 2025
    pop = target_matrix[:, idx].sum()
    print(f"  {y}: {pop/1e6:6.1f}M")

# =========================================================================
# STEP 2: LOAD CPS DATA
# =========================================================================
print("\n" + "="*70)
print("LOADING CPS MICROSIMULATION DATA")
print("="*70)

sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

# Create design matrix (households x age brackets)
X, initial_weights = create_age_design_matrix(sim)
n_households = X.shape[0]
print(f"\nLoaded {n_households:,} households")
print(f"Design matrix shape: {X.shape}")
print(f"Initial total weight: {initial_weights.sum()/1e6:.1f}M")

# Calculate income tax for each household (2024 baseline)
income_tax_values = sim.calculate("income_tax", period=2024, map_to="household").values
baseline_total = np.sum(income_tax_values * initial_weights)
print(f"Baseline income tax (2024): ${baseline_total/1e9:.1f}B")

# =========================================================================
# STEP 3: REWEIGHT AND PROJECT INCOME TAX
# =========================================================================
print("\n" + "="*70)
print("REWEIGHTING AND PROJECTING INCOME TAX")
print("="*70)

# Initialize results arrays
years = np.arange(2025, 2101)
total_income_tax = np.zeros(n_years)
total_population = np.zeros(n_years)
weights_matrix = np.zeros((n_households, n_years))

print("\nYear    Population    Income Tax    Per Capita    IPF Iters")
print("-" * 60)

# Process each year
for year_idx in range(n_years):
    year = 2025 + year_idx
    y_target = target_matrix[:, year_idx]

    # Run IPF to get new weights
    w_new, info = iterative_proportional_fitting(
        X, y_target, initial_weights,
        max_iters=100, tol=1e-6, verbose=False
    )

    # Store results
    weights_matrix[:, year_idx] = w_new
    total_income_tax[year_idx] = np.sum(income_tax_values * w_new)
    total_population[year_idx] = np.sum(y_target)

    # Report progress for selected years
    if year in [2025, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]:
        tax_billions = total_income_tax[year_idx] / 1e9
        pop_millions = total_population[year_idx] / 1e6
        per_capita = total_income_tax[year_idx] / total_population[year_idx]
        print(f"{year}    {pop_millions:7.1f}M     ${tax_billions:7.1f}B    ${per_capita:7.0f}     {info['iterations']:3d}")

# =========================================================================
# STEP 4: ANALYZE RESULTS
# =========================================================================
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

# Create results dataframe
results_df = pd.DataFrame({
    'year': years,
    'population': total_population,
    'income_tax': total_income_tax,
    'income_tax_billions': total_income_tax / 1e9,
    'population_millions': total_population / 1e6,
    'income_tax_per_capita': total_income_tax / total_population
})

# Calculate age distribution metrics
elderly_share = np.zeros(n_years)
working_age_share = np.zeros(n_years)
for year_idx in range(n_years):
    year_pop = target_matrix[:, year_idx]
    total = year_pop.sum()
    working_age_share[year_idx] = np.sum(year_pop[4:13]) / total  # 20-64
    elderly_share[year_idx] = np.sum(year_pop[13:]) / total  # 65+

results_df['working_age_share'] = working_age_share
results_df['elderly_share'] = elderly_share

# Key metrics
print("\nKEY FINDINGS:")
print("-" * 40)

# Population change
pop_change = (total_population[-1] / total_population[0] - 1) * 100
print(f"Population change 2025-2100: {pop_change:+.1f}%")

# Tax revenue change
tax_change = (total_income_tax[-1] / total_income_tax[0] - 1) * 100
print(f"Income tax change 2025-2100: {tax_change:+.1f}%")

# Working age population
working_2025 = working_age_share[0] * 100
working_2100 = working_age_share[-1] * 100
print(f"Working age (20-64) share: {working_2025:.1f}% → {working_2100:.1f}%")

# Elderly population
elderly_2025 = elderly_share[0] * 100
elderly_2100 = elderly_share[-1] * 100
print(f"Elderly (65+) share: {elderly_2025:.1f}% → {elderly_2100:.1f}%")

# Per capita tax
per_capita_2025 = total_income_tax[0] / total_population[0]
per_capita_2100 = total_income_tax[-1] / total_population[-1]
print(f"Per capita income tax: ${per_capita_2025:.0f} → ${per_capita_2100:.0f}")

# Optional: Save weights matrix if needed for other analyses
save_weights = False
if save_weights:
    np.save('household_weights_full.npy', weights_matrix)
    print(f"\nSaved weights matrix: household_weights_full.npy")

# =========================================================================
# STEP 5: CREATE VISUALIZATION
# =========================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATION")
print("="*70)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Calculate youth share for the plot
youth_share = np.zeros(n_years)
for year_idx in range(n_years):
    year_pop = target_matrix[:, year_idx]
    total = year_pop.sum()
    youth_share[year_idx] = np.sum(year_pop[:4]) / total  # 0-19

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
    title='Income Tax and Demographics Projections (2025-2100)',
    height=700,
    width=1000,
    showlegend=True
)

fig.write_html('income_tax_projections.html')
print("Saved visualization: income_tax_projections.html")
