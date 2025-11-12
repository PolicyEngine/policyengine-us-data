"""
Household-level projection pathway for income tax revenue 2025-2100.


Usage:
    python run_household_projection.py [END_YEAR] [--greg] [--use-ss] [--save-h5]

    END_YEAR: Optional ending year (default: 2035)
    --greg: Use GREG calibration instead of IPF (optional)
    --use-ss: Include Social Security benefit totals as calibration target (requires --greg)
    --save-h5: Save year-specific .h5 files with calibrated weights to ./projected_datasets/

Examples:
    python run_household_projection.py 2100 --greg --use-ss --save-h5
"""

import sys
import gc
import os
import psutil

import numpy as np

from policyengine_us import Microsimulation

from ssa_data import load_ssa_age_projections, load_ssa_benefit_projections
from calibration import calibrate_weights
from projection_utils import (
    build_household_age_matrix,
    create_household_year_h5,
)


# =========================================================================
# DATASET CONFIGURATION
# =========================================================================

DATASET_OPTIONS = {
    "cps_2024_full": {
        "path": "./policyengine-us-data/policyengine_us_data/storage/cps_2024_full.h5",
        "base_year": 2024,
    },
    "enhanced_cps_2024": {
        "path": "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        "base_year": 2024,
    },
}

SELECTED_DATASET = "enhanced_cps_2024"
START_YEAR = 2025

# Load selected dataset configuration
BASE_DATASET_PATH = DATASET_OPTIONS[SELECTED_DATASET]["path"]
BASE_YEAR = DATASET_OPTIONS[SELECTED_DATASET]["base_year"]


USE_GREG = "--greg" in sys.argv
if USE_GREG:
    sys.argv.remove("--greg")

USE_SS = "--use-ss" in sys.argv
if USE_SS:
    sys.argv.remove("--use-ss")
    if not USE_GREG:
        print("Warning: --use-ss requires --greg, enabling GREG automatically")
        USE_GREG = True

SAVE_H5 = "--save-h5" in sys.argv
if SAVE_H5:
    sys.argv.remove("--save-h5")

END_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2035

if USE_GREG:
    from samplics.weighting import SampleWeight

    calibrator = SampleWeight()
else:
    calibrator = None

OUTPUT_DIR = "./projected_datasets"

print("=" * 70)
print(f"HOUSEHOLD-LEVEL INCOME TAX PROJECTION: {START_YEAR}-{END_YEAR}")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Base year: {BASE_YEAR} (CPS microdata)")
print(f"  Projection: {START_YEAR}-{END_YEAR}")
print(f"  Calculation level: HOUSEHOLD ONLY (simplified)")
print(f"  Calibration method: {'GREG' if USE_GREG else 'IPF'}")
if USE_SS:
    print(f"  Including Social Security benefits constraint: Yes")
if SAVE_H5:
    print(f"  Saving year-specific .h5 files: Yes (to {OUTPUT_DIR}/)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
else:
    print(f"  Saving year-specific .h5 files: No (use --save-h5 to enable)")
print(f"  Years to process: {END_YEAR - START_YEAR + 1}")
est_time = (END_YEAR - START_YEAR + 1) * (3 if SAVE_H5 else 2)
print(f"  Estimated time: ~{est_time:.0f} minutes")

# =========================================================================
# STEP 1: LOAD SSA DEMOGRAPHIC PROJECTIONS
# =========================================================================
print("\n" + "=" * 70)
print("STEP 1: DEMOGRAPHIC PROJECTIONS")
print("=" * 70)

target_matrix = load_ssa_age_projections(end_year=END_YEAR)
n_years = target_matrix.shape[1]
n_ages = target_matrix.shape[0]

print(f"\nLoaded SSA projections: {n_ages} ages x {n_years} years")
print(f"\nPopulation projections:")

display_years = [
    y
    for y in [START_YEAR, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
    if START_YEAR <= y <= END_YEAR
]
if END_YEAR not in display_years:
    display_years.append(END_YEAR)

for y in display_years:
    idx = y - START_YEAR
    if idx < n_years:
        pop = target_matrix[:, idx].sum()
        print(f"  {y}: {pop/1e6:6.1f}M")

# =========================================================================
# STEP 2: BUILD HOUSEHOLD AGE MATRIX
# =========================================================================
print("\n" + "=" * 70)
print("STEP 2: BUILDING HOUSEHOLD AGE COMPOSITION")
print("=" * 70)

sim = Microsimulation(dataset=BASE_DATASET_PATH)
X, household_ids_unique, hh_id_to_idx = build_household_age_matrix(sim, n_ages)
n_households = len(household_ids_unique)

print(f"\nLoaded {n_households:,} households")
print(f"Household age matrix shape: {X.shape}")

del sim
gc.collect()

# =========================================================================
# STEP 3: PROJECT INCOME TAX WITH HOUSEHOLD-LEVEL CALCULATIONS
# =========================================================================
print("\n" + "=" * 70)
print("STEP 3: HOUSEHOLD-LEVEL PROJECTION")
print("=" * 70)
print("\nMethodology (SIMPLIFIED):")
print("  1. PolicyEngine uprates to each projection year")
print("  2. Calculate all values at household level (map_to='household')")
print("  3. IPF/GREG adjusts weights to match SSA demographics")
print("  4. Apply calibrated weights directly (no aggregation needed)")

years = np.arange(START_YEAR, END_YEAR + 1)
total_income_tax = np.zeros(n_years)
total_income_tax_baseline = np.zeros(n_years)
total_population = np.zeros(n_years)
weights_matrix = np.zeros((n_households, n_years))
baseline_weights_matrix = np.zeros((n_households, n_years))

process = psutil.Process()
print(f"\nInitial memory usage: {process.memory_info().rss / 1024**3:.2f} GB")

print("\nYear    Population    Income Tax    Baseline Tax    Memory")
print("-" * 65)

for year_idx in range(n_years):
    year = START_YEAR + year_idx

    sim = Microsimulation(dataset=BASE_DATASET_PATH)

    income_tax_hh = sim.calculate(
        "income_tax", period=year, map_to="household"
    )
    income_tax_baseline_total = income_tax_hh.sum()
    income_tax_values = income_tax_hh.values

    household_microseries = sim.calculate("household_id", map_to="household")
    baseline_weights = household_microseries.weights.values
    household_ids_hh = household_microseries.values

    assert len(household_ids_hh) == n_households

    ss_values = None
    ss_target = None
    if USE_SS:
        ss_hh = sim.calculate(
            "social_security", period=year, map_to="household"
        )
        ss_values = ss_hh.values
        ss_target = load_ssa_benefit_projections(year)

    y_target = target_matrix[:, year_idx]

    w_new, iterations = calibrate_weights(
        X=X,
        y_target=y_target,
        baseline_weights=baseline_weights,
        method="greg" if USE_GREG else "ipf",
        calibrator=calibrator,
        ss_values=ss_values,
        ss_target=ss_target,
        n_ages=n_ages,
        max_iters=100,
        tol=1e-6,
        verbose=False,
    )

    weights_matrix[:, year_idx] = w_new
    baseline_weights_matrix[:, year_idx] = baseline_weights
    total_income_tax[year_idx] = np.sum(income_tax_values * w_new)
    total_income_tax_baseline[year_idx] = income_tax_baseline_total
    total_population[year_idx] = np.sum(y_target)

    if SAVE_H5:
        h5_path = create_household_year_h5(
            year, w_new, BASE_DATASET_PATH, OUTPUT_DIR
        )
        if year in display_years:
            print(f"  Saved {year}.h5")

    del sim
    gc.collect()

    mem_gb = process.memory_info().rss / 1024**3

    if year in display_years:
        tax_billions = total_income_tax[year_idx] / 1e9
        baseline_billions = total_income_tax_baseline[year_idx] / 1e9
        pop_millions = total_population[year_idx] / 1e6
        print(
            f"{year}    {pop_millions:7.1f}M     ${tax_billions:7.1f}B     ${baseline_billions:7.1f}B     {mem_gb:.2f}GB"
        )
    elif year_idx % 5 == 0:
        print(
            f"{year}    Processing... ({year_idx+1}/{n_years})                        {mem_gb:.2f}GB"
        )
