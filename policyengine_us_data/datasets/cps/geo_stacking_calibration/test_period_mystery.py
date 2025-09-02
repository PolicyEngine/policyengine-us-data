"""
Comprehensive test of period handling behavior with PolicyEngine datasets.
Kept for reference - demonstrates the quirk that requires setting 
default_calculation_period before build_from_dataset() and not passing
period explicitly to calculate() calls.
"""

from policyengine_us import Microsimulation
import numpy as np

print("Investigating period handling with 2024 dataset...")

# Test 1: Set default_calculation_period BEFORE build_from_dataset
print("\n1. Setting default_calculation_period=2023 BEFORE build_from_dataset:")
sim1 = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim1.default_calculation_period = 2023
sim1.build_from_dataset()

ages1 = sim1.calculate("age", period=2023).values
print(f"  With period=2023: Ages min={ages1.min()}, max={ages1.max()}, unique={len(np.unique(ages1))}")

ages1_no_period = sim1.calculate("age").values
print(f"  Without period: Ages min={ages1_no_period.min()}, max={ages1_no_period.max()}, unique={len(np.unique(ages1_no_period))}")

# Test 2: Set default_calculation_period AFTER build_from_dataset
print("\n2. Setting default_calculation_period=2023 AFTER build_from_dataset:")
sim2 = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim2.build_from_dataset()
sim2.default_calculation_period = 2023

ages2 = sim2.calculate("age", period=2023).values
print(f"  With period=2023: Ages min={ages2.min()}, max={ages2.max()}, unique={len(np.unique(ages2))}")

ages2_no_period = sim2.calculate("age").values
print(f"  Without period: Ages min={ages2_no_period.min()}, max={ages2_no_period.max()}, unique={len(np.unique(ages2_no_period))}")

# Test 3: Never set default_calculation_period
print("\n3. Never setting default_calculation_period:")
sim3 = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim3.build_from_dataset()

print(f"  Default period is: {sim3.default_calculation_period}")

ages3_2023 = sim3.calculate("age", period=2023).values
print(f"  With period=2023: Ages min={ages3_2023.min()}, max={ages3_2023.max()}, unique={len(np.unique(ages3_2023))}")

ages3_2024 = sim3.calculate("age", period=2024).values
print(f"  With period=2024: Ages min={ages3_2024.min()}, max={ages3_2024.max()}, unique={len(np.unique(ages3_2024))}")

ages3_no_period = sim3.calculate("age").values
print(f"  Without period: Ages min={ages3_no_period.min()}, max={ages3_no_period.max()}, unique={len(np.unique(ages3_no_period))}")

# Test 4: Check what the original code pattern does
print("\n4. Original code pattern (set period before build):")
sim4 = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim4.default_calculation_period = 2023  # This is what the original does
sim4.build_from_dataset()

# Original doesn't pass period to calculate
ages4 = sim4.calculate("age").values  # No period passed
weights4 = sim4.calculate("person_weight").values
print(f"  Ages without period: min={ages4.min()}, max={ages4.max()}, unique={len(np.unique(ages4))}")
print(f"  Weights sum: {weights4.sum():,.0f}")

# Let's also check household_weight
hh_weights4 = sim4.calculate("household_weight").values
print(f"  Household weights sum: {hh_weights4.sum():,.0f}")