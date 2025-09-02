"""
Test script demonstrating the period handling quirk with PolicyEngine datasets.

IMPORTANT: The 2024 enhanced CPS dataset only contains 2024 data. 
When requesting 2023 data explicitly, it returns defaults (age=40, weight=0).

Solution: Set default_calculation_period=2023 BEFORE build_from_dataset(),
then DON'T pass period to calculate(). This uses the 2024 data for 2023 calculations.
"""

from policyengine_us import Microsimulation
import numpy as np

print("Demonstrating period handling with 2024 dataset for 2023 calculations...")

# WRONG WAY - Returns default values
print("\n1. WRONG: Explicitly passing period=2023")
sim_wrong = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
ages_wrong = sim_wrong.calculate("age", period=2023).values
print(f"   Ages: min={ages_wrong.min()}, max={ages_wrong.max()}, unique={len(np.unique(ages_wrong))}")
print(f"   Result: All ages are 40 (default value)")

# RIGHT WAY - Uses 2024 data for 2023 calculations
print("\n2. RIGHT: Set default period before build, don't pass period to calculate")
sim_right = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
sim_right.default_calculation_period = 2023
sim_right.build_from_dataset()
ages_right = sim_right.calculate("age").values  # No period passed!
print(f"   Ages: min={ages_right.min()}, max={ages_right.max()}, unique={len(np.unique(ages_right))}")
print(f"   Result: Actual age distribution from dataset")

print("\nThis quirk is critical for using 2024 data with 2023 calibration targets!")