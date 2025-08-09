"""Test to compare sparsity between old downsampling and new L0 penalty approach."""
import numpy as np
import torch
from policyengine_us_data.utils import HardConcrete, set_seeds, build_loss_matrix
from policyengine_us_data.datasets.cps.enhanced_cps import reweight
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.extended_cps import ExtendedCPS_2024
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

def test_l0_sparsity():
    """Test that the new L0 penalty achieves similar sparsity to 50% downsampling."""
    
    print("Loading ExtendedCPS_2024 dataset...")
    sim = Microsimulation(dataset=ExtendedCPS_2024)
    
    # Get original weights
    original_weights = sim.calculate("household_weight").values
    n_households = len(original_weights)
    print(f"Total households: {n_households}")
    
    # Add small noise to weights (as done in enhanced_cps.py)
    original_weights = original_weights + np.random.normal(1, 0.1, len(original_weights))
    
    # Create a simple loss matrix for testing (just a few targets)
    print("\nBuilding loss matrix for a few test targets...")
    
    # Create dummy targets for testing
    targets_df = pd.DataFrame({
        'nation/total_households': [1.3e8],  # ~130M households
        'nation/total_income': [2e13],  # ~$20T total income
    })
    
    # Build a simple loss matrix
    loss_matrix = pd.DataFrame(
        np.random.randn(n_households, 2) * 1000,
        columns=targets_df.columns
    )
    
    # Test with OLD L0 penalty (2.6445e-07)
    print("\n" + "="*60)
    print("Testing with OLD L0 penalty (2.6445e-07)...")
    weights_old_l0 = reweight(
        original_weights=original_weights,
        loss_matrix=loss_matrix,
        targets_array=targets_df.values[0],
        epochs=100,  # Fewer epochs for testing
        l0_lambda=2.6445e-07,  # OLD value
        dropout_rate=0.05,
        init_mean=0.999,
        temperature=0.25,
        log_path=None,
    )
    
    # Test with NEW L0 penalty (1.0e-06)
    print("\nTesting with NEW L0 penalty (1.0e-06)...")
    weights_new_l0 = reweight(
        original_weights=original_weights,
        loss_matrix=loss_matrix,
        targets_array=targets_df.values[0],
        epochs=100,  # Fewer epochs for testing
        l0_lambda=1.0e-06,  # NEW value (4x larger)
        dropout_rate=0.05,
        init_mean=0.999,
        temperature=0.25,
        log_path=None,
    )
    
    # Calculate sparsity metrics
    threshold = 0.01  # Consider weights below this as effectively zero
    
    # Old L0 sparsity
    n_zero_old = np.sum(weights_old_l0 < threshold)
    n_nonzero_old = n_households - n_zero_old
    sparsity_old = n_zero_old / n_households
    
    # New L0 sparsity
    n_zero_new = np.sum(weights_new_l0 < threshold)
    n_nonzero_new = n_households - n_zero_new
    sparsity_new = n_zero_new / n_households
    
    # Compare with 50% downsampling
    expected_after_downsampling = n_households * 0.5
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("-"*60)
    print(f"Total households: {n_households:,}")
    print(f"\n50% downsampling would keep: {int(expected_after_downsampling):,} households")
    print(f"\nOLD L0 penalty (2.6445e-07):")
    print(f"  - Non-zero weights: {n_nonzero_old:,} ({(1-sparsity_old)*100:.1f}%)")
    print(f"  - Zero weights: {n_zero_old:,} ({sparsity_old*100:.1f}%)")
    print(f"\nNEW L0 penalty (1.0e-06):")
    print(f"  - Non-zero weights: {n_nonzero_new:,} ({(1-sparsity_new)*100:.1f}%)")
    print(f"  - Zero weights: {n_zero_new:,} ({sparsity_new*100:.1f}%)")
    
    print(f"\nSparsity increase: {sparsity_old*100:.1f}% → {sparsity_new*100:.1f}%")
    print(f"Effective household reduction: {n_nonzero_old:,} → {n_nonzero_new:,}")
    
    # Check if new L0 achieves similar sparsity to 50% downsampling
    if n_nonzero_new < expected_after_downsampling * 1.2:  # Within 20% of target
        print("\n✓ NEW L0 penalty achieves similar or better sparsity than 50% downsampling")
    else:
        print(f"\n⚠ NEW L0 penalty keeps more households than 50% downsampling")
        print(f"  Consider increasing L0 penalty further")
    
    # Weight distribution analysis
    print("\n" + "="*60)
    print("WEIGHT DISTRIBUTION ANALYSIS:")
    print("-"*60)
    
    # Old L0
    nonzero_weights_old = weights_old_l0[weights_old_l0 >= threshold]
    print(f"\nOLD L0 non-zero weights:")
    print(f"  Mean: {np.mean(nonzero_weights_old):.2f}")
    print(f"  Median: {np.median(nonzero_weights_old):.2f}")
    print(f"  Std: {np.std(nonzero_weights_old):.2f}")
    
    # New L0
    nonzero_weights_new = weights_new_l0[weights_new_l0 >= threshold]
    print(f"\nNEW L0 non-zero weights:")
    print(f"  Mean: {np.mean(nonzero_weights_new):.2f}")
    print(f"  Median: {np.median(nonzero_weights_new):.2f}")
    print(f"  Std: {np.std(nonzero_weights_new):.2f}")
    
    return {
        'n_households': n_households,
        'n_nonzero_old': n_nonzero_old,
        'n_nonzero_new': n_nonzero_new,
        'sparsity_old': sparsity_old,
        'sparsity_new': sparsity_new,
        'target_50pct': int(expected_after_downsampling)
    }

if __name__ == "__main__":
    print("Testing L0 penalty sparsity comparison")
    print("This will load the ExtendedCPS_2024 dataset and test reweighting...")
    print("="*60)
    
    results = test_l0_sparsity()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Old L0 (2.6445e-07): {results['n_nonzero_old']:,} non-zero weights")
    print(f"New L0 (1.0e-06): {results['n_nonzero_new']:,} non-zero weights")
    print(f"Target (50% of {results['n_households']:,}): {results['target_50pct']:,} weights")
    
    efficiency_ratio = results['n_nonzero_new'] / results['target_50pct']
    if efficiency_ratio < 1.2:
        print(f"\n✅ SUCCESS: New L0 achieves comparable sparsity (ratio: {efficiency_ratio:.2f})")
    else:
        print(f"\n⚠️  WARNING: May need to increase L0 further (ratio: {efficiency_ratio:.2f})")