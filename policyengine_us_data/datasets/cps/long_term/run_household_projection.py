"""
Household-level projection pathway for income tax revenue 2025-2100.


Usage:
    python run_household_projection.py [END_YEAR] [--greg] [--use-ss] [--use-payroll] [--use-h6-reform] [--save-h5]

    END_YEAR: Optional ending year (default: 2035)
    --greg: Use GREG calibration instead of IPF (optional)
    --use-ss: Include Social Security benefit totals as calibration target (requires --greg)
    --use-payroll: Include taxable payroll totals as calibration target (requires --greg)
    --use-h6-reform: Include H6 reform income impact ratio as calibration target (requires --greg)
    --save-h5: Save year-specific .h5 files with calibrated weights to ./projected_datasets/

Examples:
    python run_household_projection.py 2100 --greg --use-ss --use-payroll --use-h6-reform --save-h5
"""

import sys
import gc
import os
import psutil

import numpy as np

from policyengine_us import Microsimulation

from ssa_data import (
    load_ssa_age_projections,
    load_ssa_benefit_projections,
    load_taxable_payroll_projections,
)
from calibration import calibrate_weights
from projection_utils import (
    build_household_age_matrix,
    create_household_year_h5,
)


def create_h6_reform():
    """
    Create H6 Social Security reform that phases out benefit taxation.

    The reform has two phases:
    1. Phase-in (2045-2053): Gradually increase thresholds
    2. Elimination (2054-2100): Set thresholds to infinity
    """
    reform_payload = {
        "gov.irs.social_security.taxability.threshold.base.main.SINGLE": {},
        "gov.irs.social_security.taxability.threshold.base.main.JOINT": {},
        "gov.irs.social_security.taxability.threshold.base.main.SEPARATE": {},
        "gov.irs.social_security.taxability.threshold.base.main.HEAD_OF_HOUSEHOLD": {},
        "gov.irs.social_security.taxability.threshold.base.main.SURVIVING_SPOUSE": {},
    }

    # Phase-in period: 2045 to 2053
    for year in range(2045, 2054):
        # Calculate the index (0 for 2045, 1 for 2046, etc.)
        i = year - 2045

        # H6 Formulas
        single_val = 32_500 + (7_500 * i)
        joint_val = 65_000 + (15_000 * i)

        # Create the time key for this specific year
        time_key = f"{year}-01-01.{year}-12-31"

        # Assign values
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.SINGLE"][time_key] = single_val
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.SEPARATE"][time_key] = single_val
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.HEAD_OF_HOUSEHOLD"][time_key] = single_val
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.SURVIVING_SPOUSE"][time_key] = single_val
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.JOINT"][time_key] = joint_val

    # Elimination period: 2054 to 2100
    # To "Eliminate" taxation, we set the threshold to Infinity (or an arbitrarily high number)
    final_period_key = "2054-01-01.2100-12-31"
    inf_value = 9e99  # Effectively infinity

    reform_payload["gov.irs.social_security.taxability.threshold.base.main.SINGLE"][final_period_key] = inf_value
    reform_payload["gov.irs.social_security.taxability.threshold.base.main.SEPARATE"][final_period_key] = inf_value
    reform_payload["gov.irs.social_security.taxability.threshold.base.main.HEAD_OF_HOUSEHOLD"][final_period_key] = inf_value
    reform_payload["gov.irs.social_security.taxability.threshold.base.main.SURVIVING_SPOUSE"][final_period_key] = inf_value
    reform_payload["gov.irs.social_security.taxability.threshold.base.main.JOINT"][final_period_key] = inf_value

    # Create the Reform Object
    from policyengine_core.reforms import Reform
    return Reform.from_dict(reform_payload, country_id="us")


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

USE_PAYROLL = "--use-payroll" in sys.argv
if USE_PAYROLL:
    sys.argv.remove("--use-payroll")
    if not USE_GREG:
        print(
            "Warning: --use-payroll requires --greg, enabling GREG automatically"
        )
        USE_GREG = True

USE_H6_REFORM = "--use-h6-reform" in sys.argv
if USE_H6_REFORM:
    sys.argv.remove("--use-h6-reform")
    if not USE_GREG:
        print("Warning: --use-h6-reform requires --greg, enabling GREG automatically")
        USE_GREG = True
    from ssa_data import load_h6_income_rate_change

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
if USE_PAYROLL:
    print(f"  Including taxable payroll constraint: Yes")
if USE_H6_REFORM:
    print(f"  Including H6 reform income impact constraint: Yes")
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
        if year in display_years:
            ss_baseline = np.sum(ss_values * baseline_weights)
            print(
                f"  [DEBUG {year}] SS baseline: ${ss_baseline/1e9:.1f}B, target: ${ss_target/1e9:.1f}B"
            )

    payroll_values = None
    payroll_target = None
    if USE_PAYROLL:
        # SSA taxable payroll = W-2 wages capped at wage base + SE income within remaining cap room
        taxable_wages_hh = sim.calculate(
            "taxable_earnings_for_social_security",
            period=year,
            map_to="household",
        )
        taxable_self_emp_hh = sim.calculate(
            "social_security_taxable_self_employment_income",
            period=year,
            map_to="household",
        )
        payroll_values = taxable_wages_hh.values + taxable_self_emp_hh.values
        payroll_target = load_taxable_payroll_projections(year)
        if year in display_years:
            payroll_baseline = np.sum(payroll_values * baseline_weights)
            print(
                f"  [DEBUG {year}] Payroll baseline: ${payroll_baseline/1e9:.1f}B, target: ${payroll_target/1e9:.1f}B"
            )

    h6_income_values = None
    h6_revenue_target = None
    if USE_H6_REFORM:
        # Load target ratio from CSV
        h6_target_ratio = load_h6_income_rate_change(year)

        # Only calculate H6 reform impacts if the target ratio is non-zero
        # (Reform has no effect before 2045, so skip computation for efficiency)
        if h6_target_ratio != 0:
            # Create and apply H6 reform
            h6_reform = create_h6_reform()
            reform_sim = Microsimulation(dataset=BASE_DATASET_PATH, reform=h6_reform)

            # Calculate reform income tax
            income_tax_reform_hh = reform_sim.calculate(
                "income_tax", period=year, map_to="household"
            )
            income_tax_reform = income_tax_reform_hh.values

            # Revenue impact per household
            h6_income_values = income_tax_reform - income_tax_values

            # Calculate H6 revenue target: ratio × payroll target
            # This converts the ratio constraint to an absolute revenue constraint
            payroll_target_year = load_taxable_payroll_projections(year)
            h6_revenue_target = h6_target_ratio * payroll_target_year

            # Debug output for key years
            if year in display_years:
                h6_impact_baseline = np.sum(h6_income_values * baseline_weights)
                print(
                    f"  [DEBUG {year}] H6 baseline revenue: ${h6_impact_baseline/1e9:.3f}B, target: ${h6_revenue_target/1e9:.3f}B"
                )
                print(
                    f"  [DEBUG {year}] H6 target ratio: {h6_target_ratio:.4f} × payroll ${payroll_target_year/1e9:.1f}B"
                )

            del reform_sim
            gc.collect()

    y_target = target_matrix[:, year_idx]

    w_new, iterations = calibrate_weights(
        X=X,
        y_target=y_target,
        baseline_weights=baseline_weights,
        method="greg" if USE_GREG else "ipf",
        calibrator=calibrator,
        ss_values=ss_values,
        ss_target=ss_target,
        payroll_values=payroll_values,
        payroll_target=payroll_target,
        h6_income_values=h6_income_values,
        h6_revenue_target=h6_revenue_target,
        n_ages=n_ages,
        max_iters=100,
        tol=1e-6,
        verbose=False,
    )

    if year in display_years and (USE_SS or USE_PAYROLL or USE_H6_REFORM):
        if USE_SS:
            ss_achieved = np.sum(ss_values * w_new)
            print(
                f"  [DEBUG {year}] SS achieved: ${ss_achieved/1e9:.1f}B (error: {(ss_achieved - ss_target)/ss_target*100:.1f}%)"
            )
        if USE_PAYROLL:
            payroll_achieved = np.sum(payroll_values * w_new)
            print(
                f"  [DEBUG {year}] Payroll achieved: ${payroll_achieved/1e9:.1f}B (error: {(payroll_achieved - payroll_target)/payroll_target*100:.1f}%)"
            )
        if USE_H6_REFORM and h6_revenue_target is not None:
            h6_revenue_achieved = np.sum(h6_income_values * w_new)
            error_pct = (h6_revenue_achieved - h6_revenue_target) / abs(h6_revenue_target) * 100 if h6_revenue_target != 0 else 0
            print(
                f"  [DEBUG {year}] H6 achieved revenue: ${h6_revenue_achieved/1e9:.3f}B (error: {error_pct:.1f}%)"
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
