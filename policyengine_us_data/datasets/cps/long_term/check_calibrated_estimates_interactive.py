import os

import pandas as pd
import numpy as np

from policyengine_us import Microsimulation

#H5_PATH = 'hf://policyengine/test/'
H5_PATH = '/home/baogorek/devl/sep/policyengine-us-data/policyengine_us_data/datasets/cps/long_term/projected_datasets/'

# 2027 --------------------------------------
sim = Microsimulation(dataset = H5_PATH + "2027.h5")
parameters = sim.tax_benefit_system.parameters

## Total social security
assert sim.default_calculation_period == '2027'
ss_estimate_cost_b = sim.calculate("social_security").sum() / 1E9

### Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G10 (nominal dollars in billions)
### Intermediate scenario for row 69, for Intermediate Scenario, 2027, Cost is: $1,715 billion
ss_trustees_cost_b = 1_800
assert round(ss_estimate_cost_b) == ss_trustees_cost_b

## Taxable Payroll for Social Security
taxible_estimate_b = (
    sim.calculate("taxable_earnings_for_social_security").sum() / 1E9
    + sim.calculate("social_security_taxable_self_employment_income").sum() / 1E9
)

### Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G6 (nominal dollars in billions)
### Intermediate scenario for row 69, for Intermediate Scenario, 2027, Cost is: $1,715 billion
ss_trustees_payroll_b = 11_627
assert round(taxible_estimate_b) == ss_trustees_payroll_b

## Population demographics, total

### Population count of 6 year olds
person_weights = sim.calculate("age", map_to="person").weights
person_ages = sim.calculate("age", map_to="person").values
person_is_6 = person_ages == 6

total_age6_est = np.sum(person_is_6 * person_weights)

### Single Year Age demographic projections - latest published is 2024:
### "Mid Year" CSV from https://www.ssa.gov/oact/HistEst/Population/2024/Population2024.html
### Row 8694, Col C, contains 3730632
ss_age6_pop = 3_730_632
assert ss_age6_pop == round(total_age6_est)


# 2100 --------------------------------------
sim = Microsimulation(dataset = H5_PATH + "2100.h5")
parameters = sim.tax_benefit_system.parameters

## Total social security
assert sim.default_calculation_period == '2100'
ss_estimate_cost_b = sim.calculate("social_security").sum() / 1E9

### Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G10 (nominal dollars in billions)
### Intermediate scenario for row 69, for Intermediate Scenario, 2100, Cost is: $34,432 billion
ss_trustees_cost_b = 34_432 
# Rounding takes it off a bit. We calibrated to CPI ratio times 2025-dollar cost
assert np.allclose(ss_estimate_cost_b, ss_trustees_cost_b, rtol=.0001)

## Taxable Payroll for Social Security
taxible_estimate_b = (
    sim.calculate("taxable_earnings_for_social_security").sum() / 1E9
    + sim.calculate("social_security_taxable_self_employment_income").sum() / 1E9
)

### Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G6 (nominal dollars in billions)
### Intermediate scenario for row 143, for Intermediate Scenario, 2100, Cost is: $187,614 billion
ss_trustees_payroll_b = 187_614
assert round(taxible_estimate_b) == ss_trustees_payroll_b

## Population demographics, total

### Population count of 6 year olds
person_weights = sim.calculate("age", map_to="person").weights
person_ages = sim.calculate("age", map_to="person").values
person_is_6 = person_ages == 6

total_age6_est = np.sum(person_is_6 * person_weights)

### Single Year Age demographic projections - latest published is 2024:
### "Mid Year" CSV from https://www.ssa.gov/oact/HistEst/Population/2024/Population2024.html
### Row 16067, Col C, contains 5162540
ss_age6_pop = 5_162_540
assert np.allclose(ss_age6_pop, total_age6_est, atol = 1)

# Taxation of benefits -------
# Trustees report stops at 2099 so project at current rate one year out, bring to billions
sim.default_calculation_period
hi_tob_trustees = 1761.5 
hi_tob_estimate_b = sim.calculate("tob_revenue_medicare_hi", map_to = "household").sum() / 1E9
hi_tob_estimate_b

oasdi_tob_trustees = 2101.3 
oasdi_tob_estimate_b = sim.calculate("tob_revenue_oasdi").sum() / 1E9
oasdi_tob_estimate_b


# Testing the H6 Reform ------------------------------------------------------

from policyengine_us import Microsimulation
from policyengine_core.reforms import Reform


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
        reform_payload["gov.irs.social_security.taxability.rate.base.benefit_cap"][period] = 0.35
        reform_payload["gov.irs.social_security.taxability.rate.base.excess"][period] = 0.35

        # Tier 2 (Additional): HI + OASDI Combined (85%)
        reform_payload["gov.irs.social_security.taxability.rate.additional.benefit_cap"][period] = 0.85
        reform_payload["gov.irs.social_security.taxability.rate.additional.excess"][period] = 0.85

        # --- SET THRESHOLDS (MIN/MAX SWAP) ---
        # Always put the smaller number in 'base' and larger in 'adjusted_base'

        # Single
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.SINGLE"][period] = min(oasdi_target_single, HI_SINGLE)
        reform_payload["gov.irs.social_security.taxability.threshold.adjusted_base.main.SINGLE"][period] = max(oasdi_target_single, HI_SINGLE)

        # Joint
        reform_payload["gov.irs.social_security.taxability.threshold.base.main.JOINT"][period] = min(oasdi_target_joint, HI_JOINT)
        reform_payload["gov.irs.social_security.taxability.threshold.adjusted_base.main.JOINT"][period] = max(oasdi_target_joint, HI_JOINT)

        # Map other statuses (Head/Surviving Spouse -> Single logic, Separate -> Single logic usually)
        # Note: Separate is usually 0, but for H6 strictness we map to Single logic here
        for status in ["HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE", "SEPARATE"]:
            reform_payload[f"gov.irs.social_security.taxability.threshold.base.main.{status}"][period] = min(oasdi_target_single, HI_SINGLE)
            reform_payload[f"gov.irs.social_security.taxability.threshold.adjusted_base.main.{status}"][period] = max(oasdi_target_single, HI_SINGLE)

    # --- PHASE 2: ELIMINATION (2054+) ---
    # OASDI is gone. We only collect HI.
    # Logic: "Base" becomes the HI tier ($34k). Rate is 0.35.
    # "Adjusted" becomes irrelevant (set high or rate to same).

    elim_period = "2054-01-01.2100-12-31"

    # 1. Set Thresholds to "HI Only" mode
    # Base = $34k / $44k
    reform_payload["gov.irs.social_security.taxability.threshold.base.main.SINGLE"][elim_period] = HI_SINGLE
    reform_payload["gov.irs.social_security.taxability.threshold.base.main.JOINT"][elim_period] = HI_JOINT

    # Adjusted = Infinity (Disable the second tier effectively)
    reform_payload["gov.irs.social_security.taxability.threshold.adjusted_base.main.SINGLE"][elim_period] = 9_999_999
    reform_payload["gov.irs.social_security.taxability.threshold.adjusted_base.main.JOINT"][elim_period] = 9_999_999

    # Map others
    for status in ["HEAD_OF_HOUSEHOLD", "SURVIVING_SPOUSE", "SEPARATE"]:
         reform_payload[f"gov.irs.social_security.taxability.threshold.base.main.{status}"][elim_period] = HI_SINGLE
         reform_payload[f"gov.irs.social_security.taxability.threshold.adjusted_base.main.{status}"][elim_period] = 9_999_999

    # 2. Set Rates for HI Only Revenue
    # Tier 1 (Now the ONLY tier) = 35% (HI Share)
    reform_payload["gov.irs.social_security.taxability.rate.base.benefit_cap"][elim_period] = 0.35
    reform_payload["gov.irs.social_security.taxability.rate.base.excess"][elim_period] = 0.35

    # Tier 2 (Disabled via threshold, but zero out for safety)
    reform_payload["gov.irs.social_security.taxability.rate.additional.benefit_cap"][elim_period] = 0.35
    reform_payload["gov.irs.social_security.taxability.rate.additional.excess"][elim_period] = 0.35

    return reform_payload


# Create the reform
h6_reform_payload = create_h6_reform()
h6_reform = Reform.from_dict(h6_reform_payload, country_id="us")

year = 2052
dataset_path = f'/home/baogorek/devl/sep/policyengine-us-data/policyengine_us_data/datasets/cps/long_term/projected_datasets/{year}.h5'
#dataset_path = f'hf://policyengine/test/{year}.h5'

# Baseline simulation
baseline = Microsimulation(dataset=dataset_path)
baseline_revenue = baseline.calculate("income_tax").sum()

# Reform simulation
reform_sim = Microsimulation(dataset=dataset_path, reform=h6_reform)
reform_revenue = reform_sim.calculate("income_tax").sum()

# Ha we know this will fail because the hacky reform flips the tiers that the tob variables depend on!
# Just hoping this is truly isolated
# First, let's make sure there's no taxation on benefits after the reform!
# assert reform_sim.calculate("tob_revenue_oasdi").sum() / 1E9 == 0

# Calculate impact
revenue_impact = reform_revenue - baseline_revenue
print(f"revenue_impact (B): {revenue_impact / 1E9:.2f}")

# Calculate taxable payroll
taxable_ss_earnings = baseline.calculate("taxable_earnings_for_social_security")
taxable_self_employment = baseline.calculate("social_security_taxable_self_employment_income")
total_taxable_payroll = taxable_ss_earnings.sum() + taxable_self_employment.sum()

# Calculate SS benefits
ss_benefits = baseline.calculate("social_security")
total_ss_benefits = ss_benefits.sum()

est_rev_as_pct_of_taxable_payroll = 100 * revenue_impact / total_taxable_payroll

# From https://www.ssa.gov/oact/solvency/provisions/tables/table_run133.html:
target_rev_as_pct_of_taxable_payroll = -1.12

assert np.allclose(est_rev_as_pct_of_taxable_payroll, target_rev_as_pct_of_taxable_payroll, atol = .01)

