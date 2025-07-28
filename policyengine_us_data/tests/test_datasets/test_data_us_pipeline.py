"""
Tests to ensure Microsimulations can run with the us-data direct dependencies only.
"""


def test_microsimulation_runs():
    from policyengine_us import Microsimulation
    from policyengine_us_data.datasets.cps import EnhancedCPS_2024

    sim = Microsimulation(dataset=EnhancedCPS_2024)
    sim.calculate("employment_income", map_to="household", period=2025)
