"""
Household-level projection pathway for income tax revenue 2025-2100.


Usage:
    python run_household_projection.py [START_YEAR] [END_YEAR] [--greg] [--use-ss] [--use-payroll] [--use-h6-reform] [--use-tob] [--save-h5]

    START_YEAR: Optional starting year (default: 2025)
    END_YEAR: Optional ending year (default: 2035)
    --greg: Use GREG calibration instead of IPF (optional)
    --use-ss: Include Social Security benefit totals as calibration target (requires --greg)
    --use-payroll: Include taxable payroll totals as calibration target (requires --greg)
    --use-h6-reform: Include H6 reform income impact ratio as calibration target (requires --greg)
    --use-tob: Include TOB (Taxation of Benefits) revenue as calibration target (requires --greg)
    --save-h5: Save year-specific .h5 files with calibrated weights to ./projected_datasets/

Examples:
    python run_household_projection.py 2045 2045 --greg --use-ss  # single year
    python run_household_projection.py 2025 2100 --greg --use-ss --use-payroll --use-tob --save-h5
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
    Implements Proposal H6:
    1. Phase out OASDI taxation (Tier 1) from 2045-2053 by raising thresholds.
    2. Eliminate OASDI taxation fully in 2054+ (set Tier 1 rate to 0%).
    3. HOLD HARMLESS: Maintain HI taxation (Tier 2) revenue at current law levels throughout.

    CRITICAL: Handles the "Threshold Crossover" problem.
    As OASDI thresholds rise above HI thresholds ($34k/$44k), we must
    swap the parameter definitions to prevent the engine from breaking.
    """

    reform_payload = {
        # Thresholds
        "gov.irs.social_security.taxability.threshold.base.main.SINGLE": {},
        "gov.irs.social_security.taxability.threshold.base.main.JOINT": {},
        "gov.irs.social_security.taxability.threshold.base.main.HEAD_OF_HOUSEHOLD": {},
        "gov.irs.social_security.taxability.threshold.base.main.SURVIVING_SPOUSE": {},
        "gov.irs.social_security.taxability.threshold.base.main.SEPARATE": {},
        "gov.irs.social_security.taxability.threshold.adjusted_base.main.SINGLE": {},
        "gov.irs.social_security.taxability.threshold.adjusted_base.main.JOINT": {},
        "gov.irs.social_security.taxability.threshold.adjusted_base.main.HEAD_OF_HOUSEHOLD": {},
        "gov.irs.social_security.taxability.threshold.adjusted_base.main.SURVIVING_SPOUSE": {},
        "gov.irs.social_security.taxability.threshold.adjusted_base.main.SEPARATE": {},
        # Rates - Base (Tier 1)
        "gov.irs.social_security.taxability.rate.base.benefit_cap": {},
        "gov.irs.social_security.taxability.rate.base.excess": {},
        # Rates - Additional (Tier 2 - HI)
        "gov.irs.social_security.taxability.rate.additional.benefit_cap": {},
        "gov.irs.social_security.taxability.rate.additional.excess": {},
    }

    # --- CONSTANTS: CURRENT LAW HI THRESHOLDS (FROZEN) ---
    # We must preserve these specific triggers to protect the HI Trust Fund
    HI_SINGLE = 34_000
    HI_JOINT = 44_000

    # --- PHASE 1: THE TRANSITION (2045-2053) ---
    for year in range(2045, 2054):
        period = f"{year}-01-01"
        i = year - 2045

        # 1. Calculate the Target OASDI Thresholds (Rising)
        #    (a) 2045 = $32,500 ... (i) 2053 = $92,500
        oasdi_target_single = 32_500 + (7_500 * i)
        oasdi_target_joint = 65_000 + (15_000 * i)

        # 2. Handle Threshold Crossover
        #    OASDI thresholds rise above HI thresholds during phase-out.
        #    We must swap parameters: put lower threshold in 'base' slot.

        # --- SET RATES FOR TRANSITION (2045-2053) ---
        # Joint filers cross immediately in 2045 ($65k OASDI > $44k HI).
        # Single filers cross in 2046 ($40k OASDI > $34k HI).
        #
        # PolicyEngine forces one global rate structure per year.
        # We choose swapped rates (0.35/0.85) for ALL years to minimize error:
        #
        # Trade-off in 2045:
        #   - Single filers: $225 undertax (15% on $1.5k range) ✓ acceptable
        #   - Joint filers: Would be $3,150 overtax with default rates ✗ unacceptable
        #
        # The swapped rate error is 14x smaller and aligns with tax-cutting intent.

        # Tier 1 (Base): HI ONLY (35%)
        reform_payload[
            "gov.irs.social_security.taxability.rate.base.benefit_cap"
        ][period] = 0.35
        reform_payload["gov.irs.social_security.taxability.rate.base.excess"][
            period
        ] = 0.35

        # Tier 2 (Additional): HI + OASDI Combined (85%)
        reform_payload[
            "gov.irs.social_security.taxability.rate.additional.benefit_cap"
        ][period] = 0.85
        reform_payload[
            "gov.irs.social_security.taxability.rate.additional.excess"
        ][period] = 0.85

        # --- SET THRESHOLDS (MIN/MAX SWAP) ---
        # Always put the smaller number in 'base' and larger in 'adjusted_base'

        # Single
        reform_payload[
            "gov.irs.social_security.taxability.threshold.base.main.SINGLE"
        ][period] = min(oasdi_target_single, HI_SINGLE)
        reform_payload[
            "gov.irs.social_security.taxability.threshold.adjusted_base.main.SINGLE"
        ][period] = max(oasdi_target_single, HI_SINGLE)

        # Joint
        reform_payload[
            "gov.irs.social_security.taxability.threshold.base.main.JOINT"
        ][period] = min(oasdi_target_joint, HI_JOINT)
        reform_payload[
            "gov.irs.social_security.taxability.threshold.adjusted_base.main.JOINT"
        ][period] = max(oasdi_target_joint, HI_JOINT)

        # Map other statuses (Head/Surviving Spouse -> Single logic, Separate -> Single logic usually)
        # Note: Separate is usually 0, but for H6 strictness we map to Single logic here
        for status in ["HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE", "SEPARATE"]:
            reform_payload[
                f"gov.irs.social_security.taxability.threshold.base.main.{status}"
            ][period] = min(oasdi_target_single, HI_SINGLE)
            reform_payload[
                f"gov.irs.social_security.taxability.threshold.adjusted_base.main.{status}"
            ][period] = max(oasdi_target_single, HI_SINGLE)

    # --- PHASE 2: ELIMINATION (2054+) ---
    # OASDI is gone. We only collect HI.
    # Logic: "Base" becomes the HI tier ($34k). Rate is 0.35.
    # "Adjusted" becomes irrelevant (set high or rate to same).

    elim_period = "2054-01-01.2100-12-31"

    # 1. Set Thresholds to "HI Only" mode
    # Base = $34k / $44k
    reform_payload[
        "gov.irs.social_security.taxability.threshold.base.main.SINGLE"
    ][elim_period] = HI_SINGLE
    reform_payload[
        "gov.irs.social_security.taxability.threshold.base.main.JOINT"
    ][elim_period] = HI_JOINT

    # Adjusted = Infinity (Disable the second tier effectively)
    reform_payload[
        "gov.irs.social_security.taxability.threshold.adjusted_base.main.SINGLE"
    ][elim_period] = 9_999_999
    reform_payload[
        "gov.irs.social_security.taxability.threshold.adjusted_base.main.JOINT"
    ][elim_period] = 9_999_999

    # Map others
    for status in ["HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE", "SEPARATE"]:
        reform_payload[
            f"gov.irs.social_security.taxability.threshold.base.main.{status}"
        ][elim_period] = HI_SINGLE
        reform_payload[
            f"gov.irs.social_security.taxability.threshold.adjusted_base.main.{status}"
        ][elim_period] = 9_999_999

    # 2. Set Rates for HI Only Revenue
    # Tier 1 (Now the ONLY tier) = 35% (HI Share)
    reform_payload["gov.irs.social_security.taxability.rate.base.benefit_cap"][
        elim_period
    ] = 0.35
    reform_payload["gov.irs.social_security.taxability.rate.base.excess"][
        elim_period
    ] = 0.35

    # Tier 2 (Disabled via threshold, but zero out for safety)
    reform_payload[
        "gov.irs.social_security.taxability.rate.additional.benefit_cap"
    ][elim_period] = 0.35
    reform_payload[
        "gov.irs.social_security.taxability.rate.additional.excess"
    ][elim_period] = 0.35

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
        print(
            "Warning: --use-h6-reform requires --greg, enabling GREG automatically"
        )
        USE_GREG = True
    from ssa_data import load_h6_income_rate_change

USE_TOB = "--use-tob" in sys.argv
if USE_TOB:
    sys.argv.remove("--use-tob")
    if not USE_GREG:
        print(
            "Warning: --use-tob requires --greg, enabling GREG automatically"
        )
        USE_GREG = True
    from ssa_data import load_oasdi_tob_projections, load_hi_tob_projections

SAVE_H5 = "--save-h5" in sys.argv
if SAVE_H5:
    sys.argv.remove("--save-h5")

START_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
END_YEAR = int(sys.argv[2]) if len(sys.argv) > 2 else 2035

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
if USE_TOB:
    print(f"  Including TOB revenue constraint: Yes")
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

target_matrix = load_ssa_age_projections(
    start_year=START_YEAR, end_year=END_YEAR
)
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
        if h6_target_ratio == 0:
            if year in display_years:
                print(f"  [DEBUG {year}] H6 reform not active until 2045")
        else:
            # Create and apply H6 reform
            h6_reform = create_h6_reform()
            reform_sim = Microsimulation(
                dataset=BASE_DATASET_PATH, reform=h6_reform
            )

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
                h6_impact_baseline = np.sum(
                    h6_income_values * baseline_weights
                )
                print(
                    f"  [DEBUG {year}] H6 baseline revenue: ${h6_impact_baseline/1e9:.3f}B, target: ${h6_revenue_target/1e9:.3f}B"
                )
                print(
                    f"  [DEBUG {year}] H6 target ratio: {h6_target_ratio:.4f} × payroll ${payroll_target_year/1e9:.1f}B"
                )

            del reform_sim
            gc.collect()

    oasdi_tob_values = None
    oasdi_tob_target = None
    hi_tob_values = None
    hi_tob_target = None
    if USE_TOB:
        oasdi_tob_hh = sim.calculate(
            "tob_revenue_oasdi", period=year, map_to="household"
        )
        oasdi_tob_values = oasdi_tob_hh.values
        oasdi_tob_target = load_oasdi_tob_projections(year)

        hi_tob_hh = sim.calculate(
            "tob_revenue_medicare_hi", period=year, map_to="household"
        )
        hi_tob_values = hi_tob_hh.values
        hi_tob_target = load_hi_tob_projections(year)

        if year in display_years:
            oasdi_baseline = np.sum(oasdi_tob_values * baseline_weights)
            hi_baseline = np.sum(hi_tob_values * baseline_weights)
            print(
                f"  [DEBUG {year}] OASDI TOB baseline: ${oasdi_baseline/1e9:.1f}B, target: ${oasdi_tob_target/1e9:.1f}B"
            )
            print(
                f"  [DEBUG {year}] HI TOB baseline: ${hi_baseline/1e9:.1f}B, target: ${hi_tob_target/1e9:.1f}B"
            )

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
        oasdi_tob_values=oasdi_tob_values,
        oasdi_tob_target=oasdi_tob_target,
        hi_tob_values=hi_tob_values,
        hi_tob_target=hi_tob_target,
        n_ages=n_ages,
        max_iters=100,
        tol=1e-6,
        verbose=False,
    )

    if year in display_years and (
        USE_SS or USE_PAYROLL or USE_H6_REFORM or USE_TOB
    ):
        if USE_SS:
            ss_achieved = np.sum(ss_values * w_new)
            print(
                f"  [DEBUG {year}] SS achieved: ${ss_achieved/1e9:.1f}B (error: ${abs(ss_achieved - ss_target)/1e6:.1f}M, {(ss_achieved - ss_target)/ss_target*100:.3f}%)"
            )
        if USE_PAYROLL:
            payroll_achieved = np.sum(payroll_values * w_new)
            print(
                f"  [DEBUG {year}] Payroll achieved: ${payroll_achieved/1e9:.1f}B (error: ${abs(payroll_achieved - payroll_target)/1e6:.1f}M, {(payroll_achieved - payroll_target)/payroll_target*100:.3f}%)"
            )
        if USE_H6_REFORM and h6_revenue_target is not None:
            h6_revenue_achieved = np.sum(h6_income_values * w_new)
            error_pct = (
                (h6_revenue_achieved - h6_revenue_target)
                / abs(h6_revenue_target)
                * 100
                if h6_revenue_target != 0
                else 0
            )
            print(
                f"  [DEBUG {year}] H6 achieved revenue: ${h6_revenue_achieved/1e9:.3f}B (error: ${abs(h6_revenue_achieved - h6_revenue_target)/1e6:.1f}M, {error_pct:.3f}%)"
            )
        if USE_TOB:
            oasdi_achieved = np.sum(oasdi_tob_values * w_new)
            hi_achieved = np.sum(hi_tob_values * w_new)
            print(
                f"  [DEBUG {year}] OASDI TOB achieved: ${oasdi_achieved/1e9:.1f}B (error: ${abs(oasdi_achieved - oasdi_tob_target)/1e6:.1f}M, {(oasdi_achieved - oasdi_tob_target)/oasdi_tob_target*100:.3f}%)"
            )
            print(
                f"  [DEBUG {year}] HI TOB achieved: ${hi_achieved/1e9:.1f}B (error: ${abs(hi_achieved - hi_tob_target)/1e6:.1f}M, {(hi_achieved - hi_tob_target)/hi_tob_target*100:.3f}%)"
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
