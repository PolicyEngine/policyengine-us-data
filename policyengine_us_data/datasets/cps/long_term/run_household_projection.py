"""
Household-level projection pathway for income tax revenue 2025-2100.


Usage:
    python run_household_projection.py [START_YEAR] [END_YEAR] [--profile PROFILE] [--target-source SOURCE] [--tax-assumption ASSUMPTION] [--output-dir DIR] [--save-h5] [--allow-validation-failures]
    python run_household_projection.py [START_YEAR] [END_YEAR] [--profile PROFILE] [--target-source SOURCE] [--support-augmentation-profile donor-backed-synthetic-v1] [--support-augmentation-target-year YEAR]
    python run_household_projection.py [START_YEAR] [END_YEAR] [--profile PROFILE] [--target-source SOURCE] [--support-augmentation-profile donor-backed-composite-v1] [--support-augmentation-target-year YEAR] [--support-augmentation-align-to-run-year] [--support-augmentation-blueprint-base-weight-scale SCALE]
    python run_household_projection.py [START_YEAR] [END_YEAR] [--greg] [--use-ss] [--use-payroll] [--use-h6-reform] [--use-tob] [--save-h5]

    START_YEAR: Optional starting year (default: 2025)
    END_YEAR: Optional ending year (default: 2035)
    --profile: Named calibration contract (recommended)
    --target-source: Named long-term target source package
    --tax-assumption: Long-run federal tax assumption (`trustees-core-thresholds-v1` by default)
    --output-dir: Output directory for generated H5 files and metadata
    --allow-validation-failures: Record validation issues in metadata and continue instead of aborting the run
    --support-augmentation-profile: Experimental late-year support expansion profile
    --support-augmentation-target-year: Year whose extreme support is used to build the supplement
    --support-augmentation-align-to-run-year: Rebuild the augmentation support for each run year instead of reusing one target-year support
    --support-augmentation-blueprint-base-weight-scale: Prior scaling applied to original households when target-year donor-composite blueprint calibration is active
    --greg: Use GREG calibration instead of IPF (optional)
    --use-ss: Include Social Security benefit totals as calibration target (requires --greg)
    --use-payroll: Include taxable payroll totals as calibration target (requires --greg)
    --use-h6-reform: Include H6 reform income impact ratio as calibration target (requires --greg)
    --use-tob: Include TOB (Taxation of Benefits) revenue as a hard calibration target (requires --greg)
    --save-h5: Save year-specific .h5 files with calibrated weights to ./projected_datasets/

Examples:
    python run_household_projection.py 2045 2045 --profile ss --target-source trustees_2025_current_law --save-h5
    python run_household_projection.py 2025 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --save-h5
    python run_household_projection.py 2075 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --support-augmentation-profile donor-backed-synthetic-v1 --support-augmentation-target-year 2100 --allow-validation-failures
    python run_household_projection.py 2100 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --support-augmentation-profile donor-backed-composite-v1 --support-augmentation-target-year 2100 --support-augmentation-blueprint-base-weight-scale 0.5 --allow-validation-failures --save-h5
    python run_household_projection.py 2075 2100 --profile ss-payroll-tob --target-source trustees_2025_current_law --support-augmentation-profile donor-backed-composite-v1 --support-augmentation-align-to-run-year --support-augmentation-blueprint-base-weight-scale 0.5 --allow-validation-failures
"""

import sys
import gc
import os
import psutil

import numpy as np

from policyengine_us import Microsimulation

from ssa_data import (
    describe_long_term_target_source,
    get_long_term_target_source,
    load_ssa_age_projections,
    load_ssa_benefit_projections,
    load_taxable_payroll_projections,
    set_long_term_target_source,
)
from calibration import build_calibration_audit, calibrate_weights
from calibration_artifacts import (
    update_dataset_manifest,
    write_support_augmentation_report,
    write_year_metadata,
)
from calibration_profiles import (
    approximate_window_for_year,
    build_profile_from_flags,
    classify_calibration_quality,
    get_profile,
    validate_calibration_audit,
)
from projection_utils import (
    aggregate_age_targets,
    aggregate_household_age_matrix,
    build_age_bins,
    build_household_age_matrix,
    create_household_year_h5,
    validate_projected_social_security_cap,
)
from tax_assumptions import (
    TRUSTEES_CORE_THRESHOLD_ASSUMPTION,
    create_wage_indexed_core_thresholds_reform,
    get_long_run_tax_assumption_metadata,
)
from prototype_synthetic_2100_support import (
    build_role_composite_calibration_blueprint,
    build_donor_backed_augmented_dataset,
    build_role_composite_augmented_dataset,
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
        reform_payload["gov.irs.social_security.taxability.rate.base.benefit_cap"][
            period
        ] = 0.35
        reform_payload["gov.irs.social_security.taxability.rate.base.excess"][
            period
        ] = 0.35

        # Tier 2 (Additional): HI + OASDI Combined (85%)
        reform_payload[
            "gov.irs.social_security.taxability.rate.additional.benefit_cap"
        ][period] = 0.85
        reform_payload["gov.irs.social_security.taxability.rate.additional.excess"][
            period
        ] = 0.85

        # --- SET THRESHOLDS (MIN/MAX SWAP) ---
        # Always put the smaller number in 'base' and larger in 'adjusted_base'

        # Single
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.SINGLE"][
            period
        ] = min(oasdi_target_single, HI_SINGLE)
        reform_payload[
            "gov.irs.social_security.taxability.threshold.adjusted_base.main.SINGLE"
        ][period] = max(oasdi_target_single, HI_SINGLE)

        # Joint
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.JOINT"][
            period
        ] = min(oasdi_target_joint, HI_JOINT)
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
    reform_payload["gov.irs.social_security.taxability.threshold.base.main.SINGLE"][
        elim_period
    ] = HI_SINGLE
    reform_payload["gov.irs.social_security.taxability.threshold.base.main.JOINT"][
        elim_period
    ] = HI_JOINT

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
    reform_payload["gov.irs.social_security.taxability.rate.additional.benefit_cap"][
        elim_period
    ] = 0.35
    reform_payload["gov.irs.social_security.taxability.rate.additional.excess"][
        elim_period
    ] = 0.35

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

SUPPORTED_AUGMENTATION_PROFILES = {
    "donor-backed-synthetic-v1",
    "donor-backed-composite-v1",
}
SUPPORTED_TAX_ASSUMPTIONS = {
    "current-law-literal",
    TRUSTEES_CORE_THRESHOLD_ASSUMPTION["name"],
}


PROFILE_NAME = None
if "--profile" in sys.argv:
    profile_index = sys.argv.index("--profile")
    if profile_index + 1 >= len(sys.argv):
        raise ValueError("--profile requires a profile name")
    PROFILE_NAME = sys.argv[profile_index + 1]
    del sys.argv[profile_index : profile_index + 2]

TARGET_SOURCE = None
if "--target-source" in sys.argv:
    source_index = sys.argv.index("--target-source")
    if source_index + 1 >= len(sys.argv):
        raise ValueError("--target-source requires a source name")
    TARGET_SOURCE = sys.argv[source_index + 1]
    del sys.argv[source_index : source_index + 2]

OUTPUT_DIR = "./projected_datasets"
if "--output-dir" in sys.argv:
    output_dir_index = sys.argv.index("--output-dir")
    if output_dir_index + 1 >= len(sys.argv):
        raise ValueError("--output-dir requires a directory path")
    OUTPUT_DIR = sys.argv[output_dir_index + 1]
    del sys.argv[output_dir_index : output_dir_index + 2]

SUPPORT_AUGMENTATION_PROFILE = None
if "--support-augmentation-profile" in sys.argv:
    augmentation_index = sys.argv.index("--support-augmentation-profile")
    if augmentation_index + 1 >= len(sys.argv):
        raise ValueError(
            "--support-augmentation-profile requires a profile name"
        )
    SUPPORT_AUGMENTATION_PROFILE = sys.argv[augmentation_index + 1]
    del sys.argv[augmentation_index : augmentation_index + 2]

SUPPORT_AUGMENTATION_TARGET_YEAR = None
if "--support-augmentation-target-year" in sys.argv:
    target_year_index = sys.argv.index("--support-augmentation-target-year")
    if target_year_index + 1 >= len(sys.argv):
        raise ValueError(
            "--support-augmentation-target-year requires a year"
        )
    SUPPORT_AUGMENTATION_TARGET_YEAR = int(sys.argv[target_year_index + 1])
    del sys.argv[target_year_index : target_year_index + 2]

SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR = (
    "--support-augmentation-align-to-run-year" in sys.argv
)
if SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR:
    sys.argv.remove("--support-augmentation-align-to-run-year")

SUPPORT_AUGMENTATION_START_YEAR = 2075
if "--support-augmentation-start-year" in sys.argv:
    start_year_index = sys.argv.index("--support-augmentation-start-year")
    if start_year_index + 1 >= len(sys.argv):
        raise ValueError(
            "--support-augmentation-start-year requires a year"
        )
    SUPPORT_AUGMENTATION_START_YEAR = int(sys.argv[start_year_index + 1])
    del sys.argv[start_year_index : start_year_index + 2]

SUPPORT_AUGMENTATION_TOP_N_TARGETS = 20
if "--support-augmentation-top-n-targets" in sys.argv:
    top_n_index = sys.argv.index("--support-augmentation-top-n-targets")
    if top_n_index + 1 >= len(sys.argv):
        raise ValueError(
            "--support-augmentation-top-n-targets requires an integer"
        )
    SUPPORT_AUGMENTATION_TOP_N_TARGETS = int(sys.argv[top_n_index + 1])
    del sys.argv[top_n_index : top_n_index + 2]

SUPPORT_AUGMENTATION_DONORS_PER_TARGET = 5
if "--support-augmentation-donors-per-target" in sys.argv:
    donor_index = sys.argv.index("--support-augmentation-donors-per-target")
    if donor_index + 1 >= len(sys.argv):
        raise ValueError(
            "--support-augmentation-donors-per-target requires an integer"
        )
    SUPPORT_AUGMENTATION_DONORS_PER_TARGET = int(sys.argv[donor_index + 1])
    del sys.argv[donor_index : donor_index + 2]

SUPPORT_AUGMENTATION_MAX_DISTANCE = 3.0
if "--support-augmentation-max-distance" in sys.argv:
    distance_index = sys.argv.index("--support-augmentation-max-distance")
    if distance_index + 1 >= len(sys.argv):
        raise ValueError(
            "--support-augmentation-max-distance requires a float"
        )
    SUPPORT_AUGMENTATION_MAX_DISTANCE = float(sys.argv[distance_index + 1])
    del sys.argv[distance_index : distance_index + 2]

SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE = 0.1
if "--support-augmentation-clone-weight-scale" in sys.argv:
    weight_scale_index = sys.argv.index(
        "--support-augmentation-clone-weight-scale"
    )
    if weight_scale_index + 1 >= len(sys.argv):
        raise ValueError(
            "--support-augmentation-clone-weight-scale requires a float"
        )
    SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE = float(
        sys.argv[weight_scale_index + 1]
    )
    del sys.argv[weight_scale_index : weight_scale_index + 2]

SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE = 0.5
if "--support-augmentation-blueprint-base-weight-scale" in sys.argv:
    blueprint_scale_index = sys.argv.index(
        "--support-augmentation-blueprint-base-weight-scale"
    )
    if blueprint_scale_index + 1 >= len(sys.argv):
        raise ValueError(
            "--support-augmentation-blueprint-base-weight-scale requires a float"
        )
    SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE = float(
        sys.argv[blueprint_scale_index + 1]
    )
    del sys.argv[blueprint_scale_index : blueprint_scale_index + 2]

TAX_ASSUMPTION = TRUSTEES_CORE_THRESHOLD_ASSUMPTION["name"]
if "--tax-assumption" in sys.argv:
    tax_assumption_index = sys.argv.index("--tax-assumption")
    if tax_assumption_index + 1 >= len(sys.argv):
        raise ValueError("--tax-assumption requires a value")
    TAX_ASSUMPTION = sys.argv[tax_assumption_index + 1]
    del sys.argv[tax_assumption_index : tax_assumption_index + 2]
if TAX_ASSUMPTION not in SUPPORTED_TAX_ASSUMPTIONS:
    raise ValueError(
        "Unsupported --tax-assumption: "
        f"{TAX_ASSUMPTION}. Valid values: {sorted(SUPPORTED_TAX_ASSUMPTIONS)}"
    )

ALLOW_VALIDATION_FAILURES = "--allow-validation-failures" in sys.argv
if ALLOW_VALIDATION_FAILURES:
    sys.argv.remove("--allow-validation-failures")
ALLOW_VALIDATION_FAILURES = ALLOW_VALIDATION_FAILURES or (
    os.environ.get("PEUD_ALLOW_INVALID_ARTIFACTS", "").lower() in {"1", "true", "yes"}
)


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
        print("Warning: --use-payroll requires --greg, enabling GREG automatically")
        USE_GREG = True

USE_H6_REFORM = "--use-h6-reform" in sys.argv
if USE_H6_REFORM:
    sys.argv.remove("--use-h6-reform")
    if not USE_GREG:
        print("Warning: --use-h6-reform requires --greg, enabling GREG automatically")
        USE_GREG = True

USE_TOB = "--use-tob" in sys.argv
if USE_TOB:
    sys.argv.remove("--use-tob")
    if not USE_GREG:
        print("Warning: --use-tob requires --greg, enabling GREG automatically")
        USE_GREG = True

SAVE_H5 = "--save-h5" in sys.argv
if SAVE_H5:
    sys.argv.remove("--save-h5")

START_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
END_YEAR = int(sys.argv[2]) if len(sys.argv) > 2 else 2035

if SUPPORT_AUGMENTATION_TARGET_YEAR is None:
    SUPPORT_AUGMENTATION_TARGET_YEAR = END_YEAR

if SUPPORT_AUGMENTATION_PROFILE is not None:
    if SUPPORT_AUGMENTATION_PROFILE not in SUPPORTED_AUGMENTATION_PROFILES:
        raise ValueError(
            "Unsupported support augmentation profile: "
            f"{SUPPORT_AUGMENTATION_PROFILE}"
        )
    if START_YEAR < SUPPORT_AUGMENTATION_START_YEAR:
        raise ValueError(
            "Support augmentation is only supported for late-year runs. "
            f"Received START_YEAR={START_YEAR}, requires >= "
            f"{SUPPORT_AUGMENTATION_START_YEAR}."
        )

legacy_flags_used = any([USE_GREG, USE_SS, USE_PAYROLL, USE_H6_REFORM, USE_TOB])
if PROFILE_NAME and legacy_flags_used:
    raise ValueError("Use either --profile or legacy calibration flags, not both.")

if PROFILE_NAME:
    PROFILE = get_profile(PROFILE_NAME)
else:
    PROFILE = build_profile_from_flags(
        use_greg=USE_GREG,
        use_ss=USE_SS,
        use_payroll=USE_PAYROLL,
        use_h6_reform=USE_H6_REFORM,
        use_tob=USE_TOB,
    )

if TARGET_SOURCE:
    set_long_term_target_source(TARGET_SOURCE)
TARGET_SOURCE = get_long_term_target_source()
TARGET_SOURCE_METADATA = describe_long_term_target_source(TARGET_SOURCE)


def _compose_reforms(*reforms):
    reforms = tuple(reform for reform in reforms if reform is not None)
    if not reforms:
        return None
    if len(reforms) == 1:
        return reforms[0]
    return reforms


if TAX_ASSUMPTION == "current-law-literal":
    ACTIVE_LONG_RUN_TAX_REFORM = None
    LONG_RUN_TAX_ASSUMPTION_METADATA = {
        "name": "current-law-literal",
        "description": (
            "Use the baseline PolicyEngine federal tax parameter uprating "
            "without long-run Trustees-style wage-index overrides."
        ),
        "source": "PolicyEngine baseline",
        "start_year": None,
        "end_year": int(END_YEAR),
    }
else:
    ACTIVE_LONG_RUN_TAX_REFORM = create_wage_indexed_core_thresholds_reform(
        start_year=TRUSTEES_CORE_THRESHOLD_ASSUMPTION["start_year"],
        end_year=END_YEAR,
    )
    LONG_RUN_TAX_ASSUMPTION_METADATA = get_long_run_tax_assumption_metadata(
        TAX_ASSUMPTION,
        end_year=END_YEAR,
    )

BASE_DATASET = BASE_DATASET_PATH
SUPPORT_AUGMENTATION_METADATA = None
SUPPORT_AUGMENTATION_REPORT = None
MANIFEST_SUPPORT_AUGMENTATION_METADATA = None

CALIBRATION_METHOD = PROFILE.calibration_method
USE_GREG = CALIBRATION_METHOD == "greg"
USE_SS = PROFILE.use_ss
USE_PAYROLL = PROFILE.use_payroll
USE_H6_REFORM = PROFILE.use_h6_reform
USE_TOB = PROFILE.use_tob
BENCHMARK_TOB = PROFILE.benchmark_tob

if USE_H6_REFORM:
    from ssa_data import load_h6_income_rate_change

if USE_TOB or BENCHMARK_TOB:
    from ssa_data import load_hi_tob_projections, load_oasdi_tob_projections

if USE_GREG:
    try:
        from samplics.weighting import SampleWeight
    except ImportError:
        raise ImportError(
            "samplics is required for GREG calibration. "
            "Install with: pip install policyengine-us-data[calibration]"
        )
    calibrator = SampleWeight()
else:
    calibrator = None

print("=" * 70)
print(f"HOUSEHOLD-LEVEL INCOME TAX PROJECTION: {START_YEAR}-{END_YEAR}")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Base year: {BASE_YEAR} (CPS microdata)")
print(f"  Projection: {START_YEAR}-{END_YEAR}")
print(f"  Calculation level: HOUSEHOLD ONLY (simplified)")
print(f"  Calibration profile: {PROFILE.name}")
print(f"  Profile description: {PROFILE.description}")
print(f"  Target source: {TARGET_SOURCE}")
print(f"  Long-run tax assumption: {LONG_RUN_TAX_ASSUMPTION_METADATA['name']}")
print(f"  Calibration method: {CALIBRATION_METHOD.upper()}")
if SUPPORT_AUGMENTATION_PROFILE:
    print(f"  Support augmentation: {SUPPORT_AUGMENTATION_PROFILE}")
    if SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR:
        print("  Support augmentation target year: each run year")
    else:
        print(
            "  Support augmentation target year: "
            f"{SUPPORT_AUGMENTATION_TARGET_YEAR}"
        )
    print(
        "  Support augmentation blueprint base-weight scale: "
        f"{SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE}"
    )
if USE_SS:
    print(f"  Including Social Security benefits constraint: Yes")
if USE_PAYROLL:
    print(f"  Including taxable payroll constraint: Yes")
if USE_H6_REFORM:
    print(f"  Including H6 reform income impact constraint: Yes")
if USE_TOB:
    print(f"  Including TOB revenue constraint: Yes")
elif BENCHMARK_TOB:
    print(f"  Benchmarking TOB after calibration: Yes")
if SAVE_H5:
    print(f"  Saving year-specific .h5 files: Yes (to {OUTPUT_DIR}/)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
else:
    print(f"  Saving year-specific .h5 files: No (use --save-h5 to enable)")
print(f"  Years to process: {END_YEAR - START_YEAR + 1}")
est_time = (END_YEAR - START_YEAR + 1) * (3 if SAVE_H5 else 2)
print(f"  Estimated time: ~{est_time:.0f} minutes")


def _build_support_augmentation(
    target_year: int,
    *,
    report_filename: str | None = None,
):
    if SUPPORT_AUGMENTATION_PROFILE is None:
        return BASE_DATASET_PATH, None, None, None

    if SUPPORT_AUGMENTATION_PROFILE == "donor-backed-synthetic-v1":
        augmented_dataset, augmentation_report = build_donor_backed_augmented_dataset(
            base_dataset=BASE_DATASET_PATH,
            base_year=BASE_YEAR,
            target_year=target_year,
            top_n_targets=SUPPORT_AUGMENTATION_TOP_N_TARGETS,
            donors_per_target=SUPPORT_AUGMENTATION_DONORS_PER_TARGET,
            max_distance_for_clone=SUPPORT_AUGMENTATION_MAX_DISTANCE,
            clone_weight_scale=SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE,
        )
    else:
        augmented_dataset, augmentation_report = build_role_composite_augmented_dataset(
            base_dataset=BASE_DATASET_PATH,
            base_year=BASE_YEAR,
            target_year=target_year,
            top_n_targets=SUPPORT_AUGMENTATION_TOP_N_TARGETS,
            donors_per_target=SUPPORT_AUGMENTATION_DONORS_PER_TARGET,
            max_older_distance=SUPPORT_AUGMENTATION_MAX_DISTANCE,
            max_worker_distance=SUPPORT_AUGMENTATION_MAX_DISTANCE,
            clone_weight_scale=SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE,
        )

    report_path = write_support_augmentation_report(
        OUTPUT_DIR,
        augmentation_report,
        filename=(
            report_filename
            if report_filename is not None
            else "support_augmentation_report.json"
        ),
    )
    year_metadata = {
        "name": SUPPORT_AUGMENTATION_PROFILE,
        "activation_start_year": SUPPORT_AUGMENTATION_START_YEAR,
        "target_year": int(target_year),
        "target_year_strategy": (
            "run_year" if SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR else "fixed"
        ),
        "top_n_targets": SUPPORT_AUGMENTATION_TOP_N_TARGETS,
        "donors_per_target": SUPPORT_AUGMENTATION_DONORS_PER_TARGET,
        "max_distance_for_clone": SUPPORT_AUGMENTATION_MAX_DISTANCE,
        "clone_weight_scale": SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE,
        "blueprint_base_weight_scale": (
            SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE
        ),
        "report_file": report_path.name,
        "report_summary": {
            "base_household_count": augmentation_report["base_household_count"],
            "augmented_household_count": augmentation_report[
                "augmented_household_count"
            ],
            "base_person_count": augmentation_report["base_person_count"],
            "augmented_person_count": augmentation_report[
                "augmented_person_count"
            ],
            "clone_household_count": augmentation_report.get(
                "clone_household_count", 0
            ),
            "successful_target_count": sum(
                report["successful_clone_count"] > 0
                for report in augmentation_report["target_reports"]
            ),
            "skipped_target_count": len(augmentation_report["skipped_targets"]),
        },
    }
    manifest_metadata = {
        "name": SUPPORT_AUGMENTATION_PROFILE,
        "activation_start_year": SUPPORT_AUGMENTATION_START_YEAR,
        "target_year": (
            int(target_year) if not SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR else None
        ),
        "target_year_strategy": (
            "run_year" if SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR else "fixed"
        ),
        "top_n_targets": SUPPORT_AUGMENTATION_TOP_N_TARGETS,
        "donors_per_target": SUPPORT_AUGMENTATION_DONORS_PER_TARGET,
        "max_distance_for_clone": SUPPORT_AUGMENTATION_MAX_DISTANCE,
        "clone_weight_scale": SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE,
        "blueprint_base_weight_scale": (
            SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE
        ),
        "report_file": (
            None
            if SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR
            else report_path.name
        ),
    }
    return augmented_dataset, augmentation_report, year_metadata, manifest_metadata


def _print_support_augmentation_summary(augmentation_report: dict) -> None:
    print(
        "  Base households -> augmented households: "
        f"{augmentation_report['base_household_count']:,} -> "
        f"{augmentation_report['augmented_household_count']:,}"
    )
    print(
        "  Base people -> augmented people: "
        f"{augmentation_report['base_person_count']:,} -> "
        f"{augmentation_report['augmented_person_count']:,}"
    )
    print(
        "  Successful target clones: "
        f"{sum(report['successful_clone_count'] > 0 for report in augmentation_report['target_reports'])}"
    )
    print(
        "  Skipped synthetic targets: "
        f"{len(augmentation_report['skipped_targets'])}"
    )

# =========================================================================
# STEP 1: LOAD SSA DEMOGRAPHIC PROJECTIONS
# =========================================================================
print("\n" + "=" * 70)
print("STEP 1: DEMOGRAPHIC PROJECTIONS")
print("=" * 70)

target_matrix = load_ssa_age_projections(start_year=START_YEAR, end_year=END_YEAR)
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
        print(f"  {y}: {pop / 1e6:6.1f}M")

augmentation_cache: dict[int, tuple[object, dict, dict, dict]] = {}
X = None
hh_id_to_idx = None
n_households = None
household_ids_unique = None
aggregated_age_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

if SUPPORT_AUGMENTATION_PROFILE in {
    "donor-backed-synthetic-v1",
    "donor-backed-composite-v1",
}:
    print("\n" + "=" * 70)
    print("STEP 1B: BUILD DONOR-BACKED LATE-YEAR SUPPORT")
    print("=" * 70)
    if SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR:
        print(
            "  Dynamic mode: donor-backed support will be rebuilt separately for "
            "each run year."
        )
    else:
        (
            BASE_DATASET,
            SUPPORT_AUGMENTATION_REPORT,
            SUPPORT_AUGMENTATION_METADATA,
            MANIFEST_SUPPORT_AUGMENTATION_METADATA,
        ) = _build_support_augmentation(
            SUPPORT_AUGMENTATION_TARGET_YEAR,
        )
        augmentation_cache[SUPPORT_AUGMENTATION_TARGET_YEAR] = (
            BASE_DATASET,
            SUPPORT_AUGMENTATION_REPORT,
            SUPPORT_AUGMENTATION_METADATA,
            MANIFEST_SUPPORT_AUGMENTATION_METADATA,
        )
        _print_support_augmentation_summary(SUPPORT_AUGMENTATION_REPORT)

if SUPPORT_AUGMENTATION_PROFILE is not None and SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR:
    MANIFEST_SUPPORT_AUGMENTATION_METADATA = {
        "name": SUPPORT_AUGMENTATION_PROFILE,
        "activation_start_year": SUPPORT_AUGMENTATION_START_YEAR,
        "target_year": None,
        "target_year_strategy": "run_year",
        "top_n_targets": SUPPORT_AUGMENTATION_TOP_N_TARGETS,
        "donors_per_target": SUPPORT_AUGMENTATION_DONORS_PER_TARGET,
        "max_distance_for_clone": SUPPORT_AUGMENTATION_MAX_DISTANCE,
        "clone_weight_scale": SUPPORT_AUGMENTATION_CLONE_WEIGHT_SCALE,
        "blueprint_base_weight_scale": (
            SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE
        ),
        "report_file": None,
    }

# =========================================================================
# STEP 2: BUILD HOUSEHOLD AGE MATRIX
# =========================================================================
print("\n" + "=" * 70)
print("STEP 2: BUILDING HOUSEHOLD AGE COMPOSITION")
print("=" * 70)

if SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR:
    print("\nDynamic augmentation enabled; base support will be used before the activation year and rebuilt per-year after that.")

sim = Microsimulation(dataset=BASE_DATASET)
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

process = psutil.Process()
print(f"\nInitial memory usage: {process.memory_info().rss / 1024**3:.2f} GB")

print("\nYear    Population    Income Tax    Baseline Tax    Memory")
print("-" * 65)

for year_idx in range(n_years):
    year = START_YEAR + year_idx
    current_dataset = BASE_DATASET
    current_support_augmentation_report = SUPPORT_AUGMENTATION_REPORT
    current_support_augmentation_metadata = SUPPORT_AUGMENTATION_METADATA
    current_X = X
    current_hh_id_to_idx = hh_id_to_idx

    if (
        SUPPORT_AUGMENTATION_PROFILE is not None
        and SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR
        and year >= SUPPORT_AUGMENTATION_START_YEAR
    ):
        cached = augmentation_cache.get(year)
        if cached is None:
            cached = _build_support_augmentation(
                year,
                report_filename=f"support_augmentation_report_{year}.json",
            )
            augmentation_cache[year] = cached
        (
            current_dataset,
            current_support_augmentation_report,
            current_support_augmentation_metadata,
            _,
        ) = cached
        if year in display_years:
            print(
                f"  [DEBUG {year}] Rebuilt support augmentation for run year {year}"
            )
            _print_support_augmentation_summary(current_support_augmentation_report)
        sim = Microsimulation(
            dataset=current_dataset,
            reform=ACTIVE_LONG_RUN_TAX_REFORM,
        )
        current_X, _, current_hh_id_to_idx = build_household_age_matrix(sim, n_ages)
    else:
        sim = Microsimulation(
            dataset=current_dataset,
            reform=ACTIVE_LONG_RUN_TAX_REFORM,
        )

    income_tax_hh = sim.calculate("income_tax", period=year, map_to="household")
    income_tax_baseline_total = income_tax_hh.sum()
    income_tax_values = income_tax_hh.values

    household_microseries = sim.calculate("household_id", map_to="household")
    baseline_weights = household_microseries.weights.values
    household_ids_hh = household_microseries.values

    if not (
        SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR
        and current_support_augmentation_report is not None
    ):
        assert len(household_ids_hh) == n_households

    ss_values = None
    ss_target = None
    if USE_SS:
        ss_hh = sim.calculate("social_security", period=year, map_to="household")
        ss_values = ss_hh.values
        ss_target = load_ssa_benefit_projections(year)
        if year in display_years:
            ss_baseline = np.sum(ss_values * baseline_weights)
            print(
                f"  [DEBUG {year}] SS baseline: ${ss_baseline / 1e9:.1f}B, target: ${ss_target / 1e9:.1f}B"
            )

    payroll_values = None
    payroll_target = None
    if USE_PAYROLL:
        payroll_cap = validate_projected_social_security_cap(
            sim.tax_benefit_system.parameters,
            year,
        )
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
                f"  [DEBUG {year}] Payroll cap: ${payroll_cap:,.0f}"
            )
            print(
                f"  [DEBUG {year}] Payroll baseline: ${payroll_baseline / 1e9:.1f}B, target: ${payroll_target / 1e9:.1f}B"
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
                dataset=current_dataset,
                reform=_compose_reforms(ACTIVE_LONG_RUN_TAX_REFORM, h6_reform),
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
                h6_impact_baseline = np.sum(h6_income_values * baseline_weights)
                print(
                    f"  [DEBUG {year}] H6 baseline revenue: ${h6_impact_baseline / 1e9:.3f}B, target: ${h6_revenue_target / 1e9:.3f}B"
                )
                print(
                    f"  [DEBUG {year}] H6 target ratio: {h6_target_ratio:.4f} × payroll ${payroll_target_year / 1e9:.1f}B"
                )

            del reform_sim
            gc.collect()

    oasdi_tob_values = None
    oasdi_tob_target = None
    hi_tob_values = None
    hi_tob_target = None
    if USE_TOB or BENCHMARK_TOB:
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
                f"  [DEBUG {year}] OASDI TOB baseline: ${oasdi_baseline / 1e9:.1f}B, target: ${oasdi_tob_target / 1e9:.1f}B"
            )
            print(
                f"  [DEBUG {year}] HI TOB baseline: ${hi_baseline / 1e9:.1f}B, target: ${hi_tob_target / 1e9:.1f}B"
            )

    approximate_window = approximate_window_for_year(PROFILE, year)
    age_bucket_size = (
        approximate_window.age_bucket_size
        if approximate_window is not None
        else None
    )
    if age_bucket_size and age_bucket_size > 1:
        if SUPPORT_AUGMENTATION_ALIGN_TO_RUN_YEAR and current_support_augmentation_report is not None:
            age_bins = build_age_bins(n_ages=n_ages, bucket_size=age_bucket_size)
            X_current = aggregate_household_age_matrix(current_X, age_bins)
            aggregated_target_matrix = aggregate_age_targets(target_matrix, age_bins)
        else:
            if age_bucket_size not in aggregated_age_cache:
                age_bins = build_age_bins(n_ages=n_ages, bucket_size=age_bucket_size)
                aggregated_age_cache[age_bucket_size] = (
                    aggregate_household_age_matrix(current_X, age_bins),
                    aggregate_age_targets(target_matrix, age_bins),
                )
            X_current, aggregated_target_matrix = aggregated_age_cache[age_bucket_size]
        y_target = aggregated_target_matrix[:, year_idx]
    else:
        X_current = current_X
        y_target = target_matrix[:, year_idx]
        age_bucket_size = 1

    X_actual_current = X_current
    ss_values_actual = None if ss_values is None else np.asarray(ss_values, dtype=float)
    payroll_values_actual = (
        None if payroll_values is None else np.asarray(payroll_values, dtype=float)
    )
    calibration_baseline_weights = baseline_weights
    X_calibration = X_current.copy()
    ss_values_calibration = (
        None if ss_values_actual is None else ss_values_actual.copy()
    )
    payroll_values_calibration = (
        None if payroll_values_actual is None else payroll_values_actual.copy()
    )
    blueprint_summary = None
    if (
            SUPPORT_AUGMENTATION_PROFILE == "donor-backed-composite-v1"
        and current_support_augmentation_report is not None
        and year >= SUPPORT_AUGMENTATION_START_YEAR
    ):
        calibration_blueprint = build_role_composite_calibration_blueprint(
            current_support_augmentation_report,
            year=year,
            age_bins=build_age_bins(n_ages=n_ages, bucket_size=age_bucket_size),
            hh_id_to_idx=current_hh_id_to_idx,
            baseline_weights=baseline_weights,
            base_weight_scale=SUPPORT_AUGMENTATION_BLUEPRINT_BASE_WEIGHT_SCALE,
        )
        if calibration_blueprint is not None:
            calibration_baseline_weights = calibration_blueprint["baseline_weights"]
            for idx, age_vector in calibration_blueprint["age_overrides"].items():
                X_calibration[idx] = age_vector
            if ss_values_calibration is not None:
                for idx, target_value in calibration_blueprint["ss_overrides"].items():
                    ss_values_calibration[idx] = target_value
            if payroll_values_calibration is not None:
                for idx, target_value in calibration_blueprint[
                    "payroll_overrides"
                ].items():
                    payroll_values_calibration[idx] = target_value
            blueprint_summary = calibration_blueprint["summary"]
            if year in display_years:
                print(
                    f"  [DEBUG {year}] Using support blueprint for "
                    f"{blueprint_summary['clone_household_count']} clone households "
                    f"(base-weight scale {blueprint_summary['base_weight_scale']:.3f})"
                )

    w_new, iterations, calibration_event = calibrate_weights(
        X=X_calibration,
        y_target=y_target,
        baseline_weights=calibration_baseline_weights,
        method=CALIBRATION_METHOD,
        calibrator=calibrator,
        ss_values=ss_values_calibration,
        ss_target=ss_target,
        payroll_values=payroll_values_calibration,
        payroll_target=payroll_target,
        h6_income_values=h6_income_values,
        h6_revenue_target=h6_revenue_target,
        oasdi_tob_values=oasdi_tob_values if USE_TOB else None,
        oasdi_tob_target=oasdi_tob_target if USE_TOB else None,
        hi_tob_values=hi_tob_values if USE_TOB else None,
        hi_tob_target=hi_tob_target if USE_TOB else None,
        n_ages=X_current.shape[1],
        max_iters=100,
        tol=1e-6,
        verbose=False,
        allow_fallback_to_ipf=PROFILE.allow_greg_fallback,
        allow_approximate_entropy=approximate_window is not None,
        approximate_max_error_pct=(
            approximate_window.max_constraint_error_pct
            if approximate_window is not None
            else None
        ),
    )

    calibration_audit = build_calibration_audit(
        X=X_actual_current,
        y_target=y_target,
        weights=w_new,
        baseline_weights=calibration_baseline_weights,
        calibration_event=calibration_event,
        ss_values=ss_values_actual,
        ss_target=ss_target,
        payroll_values=payroll_values_actual,
        payroll_target=payroll_target,
        h6_income_values=h6_income_values,
        h6_revenue_target=h6_revenue_target,
        oasdi_tob_values=oasdi_tob_values if USE_TOB else None,
        oasdi_tob_target=oasdi_tob_target if USE_TOB else None,
        hi_tob_values=hi_tob_values if USE_TOB else None,
        hi_tob_target=hi_tob_target if USE_TOB else None,
    )
    if blueprint_summary is not None:
        calibration_audit["support_blueprint"] = blueprint_summary
    if BENCHMARK_TOB and oasdi_tob_values is not None and hi_tob_values is not None:
        calibration_audit["benchmarks"] = {
            "oasdi_tob": {
                "target": float(oasdi_tob_target),
                "achieved": float(np.sum(oasdi_tob_values * w_new)),
            },
            "hi_tob": {
                "target": float(hi_tob_target),
                "achieved": float(np.sum(hi_tob_values * w_new)),
            },
        }
        for benchmark in calibration_audit["benchmarks"].values():
            benchmark["error"] = benchmark["achieved"] - benchmark["target"]
            benchmark["pct_error"] = (
                0.0
                if benchmark["target"] == 0
                else (benchmark["error"] / benchmark["target"] * 100)
            )
            benchmark["source"] = TARGET_SOURCE
    calibration_audit["calibration_quality"] = classify_calibration_quality(
        calibration_audit,
        PROFILE,
        year=year,
    )
    calibration_audit["age_bucket_size"] = age_bucket_size
    calibration_audit["age_bucket_count"] = int(X_current.shape[1])

    validation_issues = validate_calibration_audit(
        calibration_audit,
        PROFILE,
        year=year,
    )
    calibration_audit["validation_issues"] = validation_issues
    calibration_audit["validation_passed"] = not bool(validation_issues)
    if validation_issues:
        issue_text = "; ".join(validation_issues)
        if not ALLOW_VALIDATION_FAILURES:
            raise RuntimeError(f"Calibration validation failed for {year}: {issue_text}")
        print(
            f"  [WARN {year}] Validation issues recorded but not fatal: {issue_text}",
            file=sys.stderr,
        )

    if year in display_years and CALIBRATION_METHOD in {"greg", "entropy"}:
        n_neg = calibration_audit["negative_weight_count"]
        if n_neg > 0:
            pct_neg = calibration_audit["negative_weight_pct"]
            hh_pct_neg = calibration_audit.get("negative_weight_household_pct", 0.0)
            max_neg = calibration_audit["largest_negative_weight"]
            print(
                f"  [DEBUG {year}] Negative weights: {n_neg} households "
                f"({hh_pct_neg:.2f}% of households, {pct_neg:.2f}% of weight mass), "
                f"largest: {max_neg:,.0f}"
            )
        else:
            print(f"  [DEBUG {year}] Negative weights: 0 (all weights non-negative)")

    if year in display_years and (USE_SS or USE_PAYROLL or USE_H6_REFORM or USE_TOB):
        if USE_SS:
            ss_stats = calibration_audit["constraints"]["ss_total"]
            print(
                f"  [DEBUG {year}] SS achieved: ${ss_stats['achieved'] / 1e9:.1f}B "
                f"(error: ${abs(ss_stats['error']) / 1e6:.1f}M, "
                f"{ss_stats['pct_error']:.3f}%)"
            )
        if USE_PAYROLL:
            payroll_stats = calibration_audit["constraints"]["payroll_total"]
            print(
                f"  [DEBUG {year}] Payroll achieved: ${payroll_stats['achieved'] / 1e9:.1f}B "
                f"(error: ${abs(payroll_stats['error']) / 1e6:.1f}M, "
                f"{payroll_stats['pct_error']:.3f}%)"
            )
        if USE_H6_REFORM and h6_revenue_target is not None:
            h6_stats = calibration_audit["constraints"]["h6_revenue"]
            print(
                f"  [DEBUG {year}] H6 achieved revenue: ${h6_stats['achieved'] / 1e9:.3f}B "
                f"(error: ${abs(h6_stats['error']) / 1e6:.1f}M, "
                f"{h6_stats['pct_error']:.3f}%)"
            )
        if USE_TOB:
            oasdi_stats = calibration_audit["constraints"]["oasdi_tob"]
            hi_stats = calibration_audit["constraints"]["hi_tob"]
            print(
                f"  [DEBUG {year}] OASDI TOB achieved: ${oasdi_stats['achieved'] / 1e9:.1f}B "
                f"(error: ${abs(oasdi_stats['error']) / 1e6:.1f}M, "
                f"{oasdi_stats['pct_error']:.3f}%)"
            )
            print(
                f"  [DEBUG {year}] HI TOB achieved: ${hi_stats['achieved'] / 1e9:.1f}B "
                f"(error: ${abs(hi_stats['error']) / 1e6:.1f}M, "
                f"{hi_stats['pct_error']:.3f}%)"
            )
    if year in display_years and BENCHMARK_TOB:
        oasdi_stats = calibration_audit["benchmarks"]["oasdi_tob"]
        hi_stats = calibration_audit["benchmarks"]["hi_tob"]
        print(
            f"  [DEBUG {year}] OASDI TOB benchmark: ${oasdi_stats['achieved'] / 1e9:.1f}B "
            f"(gap: ${abs(oasdi_stats['error']) / 1e6:.1f}M, "
            f"{oasdi_stats['pct_error']:.3f}%)"
        )
        print(
            f"  [DEBUG {year}] HI TOB benchmark: ${hi_stats['achieved'] / 1e9:.1f}B "
            f"(gap: ${abs(hi_stats['error']) / 1e6:.1f}M, "
            f"{hi_stats['pct_error']:.3f}%)"
        )

    total_income_tax[year_idx] = np.sum(income_tax_values * w_new)
    total_income_tax_baseline[year_idx] = income_tax_baseline_total
    total_population[year_idx] = np.sum(y_target)

    if SAVE_H5:
        h5_path = create_household_year_h5(
            year,
            w_new,
            current_dataset,
            OUTPUT_DIR,
            reform=ACTIVE_LONG_RUN_TAX_REFORM,
        )
        metadata_path = write_year_metadata(
            h5_path,
            year=year,
            base_dataset_path=BASE_DATASET_PATH,
            profile=PROFILE.to_dict(),
            calibration_audit=calibration_audit,
            target_source=TARGET_SOURCE_METADATA,
            tax_assumption=LONG_RUN_TAX_ASSUMPTION_METADATA,
            support_augmentation=current_support_augmentation_metadata,
        )
        update_dataset_manifest(
            OUTPUT_DIR,
            year=year,
            h5_path=h5_path,
            metadata_path=metadata_path,
            base_dataset_path=BASE_DATASET_PATH,
            profile=PROFILE.to_dict(),
            calibration_audit=calibration_audit,
            target_source=TARGET_SOURCE_METADATA,
            tax_assumption=LONG_RUN_TAX_ASSUMPTION_METADATA,
            support_augmentation=MANIFEST_SUPPORT_AUGMENTATION_METADATA,
        )
        if year in display_years:
            print(f"  Saved {year}.h5 and metadata")

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
            f"{year}    Processing... ({year_idx + 1}/{n_years})                        {mem_gb:.2f}GB"
        )
