"""Test to verify enhanced CPS has the target number of active households (20k-25k)."""


def test_enhanced_cps_household_count():
    """Test that EnhancedCPS_2024 has between 20,000 and 25,000 non-zero weights."""
    from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS_2024
    from policyengine_us import Microsimulation
    import numpy as np

    # Load the enhanced dataset
    sim = Microsimulation(dataset=EnhancedCPS_2024)
    weights = sim.calculate("household_weight").values

    # Count non-zero weights (threshold for "active" households)
    threshold = 0.01
    nonzero_weights = np.sum(weights > threshold)

    print(f"\nHousehold count check:")
    print(f"Non-zero weights (> {threshold}): {nonzero_weights:,}")
    print(f"Target range: 20,000 - 25,000")

    # Assert the count is in our target range
    assert 20000 <= nonzero_weights <= 25000, (
        f"Expected 20k-25k active households, got {nonzero_weights:,}. "
        f"Need to adjust L0 penalty: too high if < 20k, too low if > 25k"
    )

    print(f"✅ SUCCESS: {nonzero_weights:,} households in target range!")


if __name__ == "__main__":
    test_enhanced_cps_household_count()
