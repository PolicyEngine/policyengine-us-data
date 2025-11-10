"""
Household-level projection pathway for income tax revenue 2025-2100.

SIMPLER ALTERNATIVE to run_full_projection.py
- Uses map_to="household" exclusively
- No multi-entity complexity
- Suitable for aggregate revenue projections
- NOT suitable for person-level policy analysis

Use this when you need:
- Fast aggregate projections
- Simple, maintainable code
- Memory-efficient processing

Use run_full_projection.py instead when you need:
- Person-level detail for policy analysis
- Tax-unit-level outputs
- Full microsimulation datasets

Usage:
    python run_household_projection.py [END_YEAR] [--greg] [--use-ss] [--save-h5]

    END_YEAR: Optional ending year (default: 2035)
    --greg: Use GREG calibration instead of IPF (optional)
    --use-ss: Include Social Security benefit totals as calibration target (requires --greg)
    --save-h5: Save year-specific .h5 files with calibrated weights to ./projected_datasets/

Examples:
    python run_household_projection.py 2030        # Quick test with IPF (6 years)
    python run_household_projection.py 2050 --greg # Medium run with GREG (26 years)
    python run_household_projection.py 2100        # Full projection with IPF (76 years)
    python run_household_projection.py 2100 --greg --use-ss # GREG with SS constraint
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
from create_reweighting_matrix import iterative_proportional_fitting
from age_projection import load_ssa_projections


# =========================================================================
# DATASET CONFIGURATION
# =========================================================================

DATASET_OPTIONS = {
    "cps_2024_full": {
        "path": "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/cps_2024_full.h5",
        "base_year": 2024,
    },
    "cps_2023": {
        "path": "hf://policyengine/policyengine-us-data/cps_2023.h5",
        "base_year": 2023,
    },
    "enhanced_cps_2024": {
        "path": "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5",
        "base_year": 2024,
    },
    "national_2023": {
        "path": "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/datasets/cps/geo_stacking_calibration/national/national.h5",
        "base_year": 2023,
    },
}

# SELECT DATASET HERE
# SELECTED_DATASET = "cps_2024_full"
SELECTED_DATASET = "enhanced_cps_2024"

# Projection always starts at 2025
START_YEAR = 2025

# Load selected dataset configuration
BASE_DATASET_PATH = DATASET_OPTIONS[SELECTED_DATASET]["path"]
BASE_YEAR = DATASET_OPTIONS[SELECTED_DATASET]["base_year"]


def load_ssa_benefit_projections(year):
    """
    Load SSA Trustee Report projections for Social Security benefits.
    Values are in nominal billions of dollars.
    """
    csv_file = "social_security_aux.csv"
    df = pd.read_csv(csv_file, thousands=",")

    row = df[df["year"] == year]
    nominal_billions = row["oasdi_cost_in_billion_nominal_usd"].values[0]
    return nominal_billions * 1e9


def create_household_year_h5(
    year, household_weights, base_dataset_path, output_dir
):
    """
    Create a year-specific .h5 file with calibrated household weights only.

    This simplified version only saves weights and essential IDs, letting PolicyEngine
    uprate and calculate all other variables on-the-fly when the dataset is loaded.

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

    # Get person-level data
    df = sim.to_input_dataframe()

    # Get household IDs and weights
    household_ids = sim.calculate("household_id", map_to="household").values
    person_household_id = df[f"person_household_id__{base_period}"]

    # Map household weights to persons
    hh_to_weight = dict(zip(household_ids, household_weights))
    person_weights = person_household_id.map(hh_to_weight)

    # Update household weights only (person weights handled by PolicyEngine)
    df[f"household_weight__{year}"] = person_weights
    df.drop(
        columns=[
            f"household_weight__{base_period}",
            f"person_weight__{base_period}",
        ],
        inplace=True,
        errors="ignore",
    )

    # For all other variables: uprate them at person level
    # (person-specific variables need person-level calculation, not household)
    for col in df.columns:
        if f"__{base_period}" in col:
            var_name = col.replace(f"__{base_period}", "")
            col_name_new = f"{var_name}__{year}"

            # Skip weights (already handled)
            if var_name in ["household_weight", "person_weight"]:
                continue

            # Uprate at person level to preserve individual-specific values
            try:
                uprated_values = sim.calculate(var_name, period=year).values

                # Should be person-level array
                if len(uprated_values) == len(df):
                    df[col_name_new] = uprated_values
                    df.drop(columns=[col], inplace=True)
                else:
                    # Unexpected length, just rename
                    df.rename(columns={col: col_name_new}, inplace=True)

            except:
                # If calculation fails, just rename
                df.rename(columns={col: col_name_new}, inplace=True)

    # Create dataset and save
    dataset = Dataset.from_dataframe(df, year)

    # Build new sim and save to h5
    new_sim = Microsimulation()
    new_sim.dataset = dataset
    new_sim.build_from_dataset()

    data = {}
    for variable in new_sim.tax_benefit_system.variables:
        holder = new_sim.get_holder(variable)
        known_periods = holder.get_known_periods()

        if len(known_periods) > 0:
            data[variable] = {}
            for period in known_periods:
                values = holder.get_array(period)
                values = np.array(values)

                # Handle object dtypes that HDF5 can't save
                if values.dtype == np.object_:
                    try:
                        values = values.astype("S")
                    except:
                        continue

                data[variable][period] = values

    with h5py.File(output_path, "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)

    del sim, new_sim, dataset
    gc.collect()

    return output_path


# Parse command line arguments
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

## Manual Override
# USE_GREG=True
# USE_SS=True
# SAVE_H5=True
# END_YEAR = 2100


# Import samplics if using GREG
if USE_GREG:
    from samplics.weighting import SampleWeight

    calibrator = SampleWeight()

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

target_matrix = load_ssa_projections(end_year=END_YEAR)
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

# Load initial simulation to build age matrix
sim = Microsimulation(dataset=BASE_DATASET_PATH)

# Get person-level age and household mapping
age_person = sim.calculate("age")
person_household_id = sim.calculate("person_household_id")

# Get unique household IDs
household_ids_unique = np.unique(person_household_id.values)
n_households = len(household_ids_unique)

print(f"\nLoaded {n_households:,} households")

# Build household age composition matrix (X)
X = np.zeros((n_households, n_ages))
hh_id_to_idx = {hh_id: idx for idx, hh_id in enumerate(household_ids_unique)}

for person_idx in range(len(age_person)):
    age = int(age_person.values[person_idx])
    hh_id = person_household_id.values[person_idx]
    hh_idx = hh_id_to_idx[hh_id]
    age_idx = min(age, 85)  # Cap at 85+
    X[hh_idx, age_idx] += 1

print(f"Household age matrix shape: {X.shape}")

# Clean up initial sim
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

    # Reload simulation each year
    sim = Microsimulation(dataset=BASE_DATASET_PATH)

    # SIMPLIFIED: All calculations at household level
    income_tax_hh = sim.calculate(
        "income_tax", period=year, map_to="household"
    )
    income_tax_baseline_total = income_tax_hh.sum()  # Uses household weights
    income_tax_values = income_tax_hh.values  # Already in household order

    # Get baseline household weights
    household_microseries = sim.calculate("household_id", map_to="household")
    baseline_weights = household_microseries.weights.values
    household_ids_hh = household_microseries.values

    # Verify ordering matches our X matrix
    assert len(household_ids_hh) == n_households

    # Optional: Social Security for GREG
    if USE_SS:
        ss_hh = sim.calculate(
            "social_security", period=year, map_to="household"
        )
        ss_baseline_total = ss_hh.sum()
        ss_values = ss_hh.values
        ss_target = load_ssa_benefit_projections(year)

    # Get SSA demographic targets for this year
    y_target = target_matrix[:, year_idx]

    # Calibrate weights
    if USE_GREG:
        # Build controls
        controls = {}
        for age_idx in range(n_ages):
            controls[f"age_{age_idx}"] = y_target[age_idx]

        if USE_SS:
            # Add SS as auxiliary variable
            age_cols = {f"age_{i}": X[:, i] for i in range(n_ages)}
            aux_df = pd.DataFrame(age_cols)
            aux_df["ss_total"] = ss_values
            controls["ss_total"] = ss_target
            aux_vars = aux_df
        else:
            aux_vars = X

        try:
            w_new = calibrator.calibrate(
                samp_weight=baseline_weights,
                aux_vars=aux_vars,
                control=controls,
            )
            iterations = 1  # GREG doesn't report iterations
        except Exception as e:
            print(f"  GREG failed for {year}: {e}, falling back to IPF")
            w_new, info = iterative_proportional_fitting(
                X,
                y_target,
                baseline_weights,
                max_iters=100,
                tol=1e-6,
                verbose=False,
            )
            iterations = info["iterations"]
    else:
        # IPF calibration
        w_new, info = iterative_proportional_fitting(
            X,
            y_target,
            baseline_weights,
            max_iters=100,
            tol=1e-6,
            verbose=False,
        )
        iterations = info["iterations"]

    # Store results - SIMPLIFIED: direct calculation
    weights_matrix[:, year_idx] = w_new
    baseline_weights_matrix[:, year_idx] = baseline_weights
    total_income_tax[year_idx] = np.sum(income_tax_values * w_new)
    total_income_tax_baseline[year_idx] = income_tax_baseline_total
    total_population[year_idx] = np.sum(y_target)

    # Create year-specific .h5 file if requested
    if SAVE_H5:
        h5_path = create_household_year_h5(
            year, w_new, BASE_DATASET_PATH, OUTPUT_DIR
        )
        if year in display_years:
            print(f"  Saved {year}.h5")

    # Clean up
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
