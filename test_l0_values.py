"""Test to verify L0 penalty values achieve desired sparsity."""
import numpy as np
import torch
from policyengine_us_data.utils import HardConcrete, set_seeds
import torch.nn as nn

def test_l0_sparsity_levels():
    """Test different L0 penalty values to understand sparsity behavior."""
    
    print("Testing L0 penalty values for sparsity...")
    print("="*60)
    
    # Simulate a typical reweighting scenario
    n_households = 100000  # Similar to CPS dataset size
    n_targets = 50  # Number of calibration targets
    
    set_seeds(1456)
    
    # Create synthetic loss matrix and targets
    loss_matrix = torch.randn(n_households, n_targets)
    targets = torch.randn(n_targets) * 1000
    original_weights = torch.ones(n_households)
    
    # Test different L0 values - need much larger values for actual sparsity
    l0_values = [
        2.6445e-07,  # Current value
        1.0e-06,     # ~4x
        1.0e-05,     # ~40x
        5.0e-05,     # ~200x
        1.0e-04,     # ~400x
        5.0e-04,     # ~2000x
        1.0e-03,     # ~4000x
    ]
    
    results = []
    
    for l0_lambda in l0_values:
        print(f"\nTesting L0 = {l0_lambda:.2e}")
        
        # Initialize HardConcrete dropout layer
        dropout_layer = HardConcrete(
            input_dim=n_households,
            init_mean=0.999,  # Start with almost all weights active
            temperature=0.25,
        )
        
        # Create simple weight multiplier network
        weight_multiplier = nn.Parameter(torch.ones(n_households))
        
        # Optimizer
        optimizer = torch.optim.Adam(
            [weight_multiplier] + list(dropout_layer.parameters()),
            lr=1e-2
        )
        
        # Training loop (simplified version of reweighting)
        for epoch in range(200):
            optimizer.zero_grad()
            
            # Get dropout mask
            dropout_mask = dropout_layer()
            
            # Apply dropout to weights
            effective_weights = weight_multiplier * dropout_mask * original_weights
            
            # Calculate loss (simplified)
            predictions = loss_matrix.T @ effective_weights
            mse_loss = ((predictions - targets) ** 2).mean()
            
            # L0 penalty
            l0_loss = dropout_layer.get_penalty()
            
            # Total loss
            total_loss = mse_loss + l0_lambda * l0_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Clamp weights to be positive
            with torch.no_grad():
                weight_multiplier.data = torch.clamp(weight_multiplier.data, min=0)
        
        # Evaluate final sparsity
        with torch.no_grad():
            dropout_layer.eval()
            final_mask = dropout_layer()
            effective_weights = weight_multiplier * final_mask
            
            # Count effectively zero weights (threshold at 0.01)
            n_zeros = (effective_weights < 0.01).sum().item()
            n_nonzero = n_households - n_zeros
            sparsity = n_zeros / n_households
            
            # Get active probability from the layer
            active_prob = dropout_layer.get_active_prob()
            expected_active = active_prob.sum().item()
        
        results.append({
            'l0_lambda': l0_lambda,
            'n_nonzero': n_nonzero,
            'sparsity': sparsity,
            'expected_active': expected_active,
        })
        
        print(f"  Non-zero weights: {n_nonzero:,} / {n_households:,} ({(1-sparsity)*100:.1f}%)")
        print(f"  Sparsity: {sparsity*100:.1f}%")
        print(f"  Expected active (from gate probs): {expected_active:.0f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF L0 PENALTY EFFECTS:")
    print("-"*60)
    print(f"Total households: {n_households:,}")
    print(f"50% downsampling would keep: {n_households//2:,}")
    print("\nL0 Penalty    | Non-zero | Sparsity | vs 50% target")
    print("-"*60)
    
    target_nonzero = n_households // 2
    for r in results:
        ratio = r['n_nonzero'] / target_nonzero
        status = "✓" if ratio < 1.2 else "⚠"
        print(f"{r['l0_lambda']:.2e} | {r['n_nonzero']:8,} | {r['sparsity']*100:6.1f}% | {ratio:5.2f}x {status}")
    
    print("\n" + "="*60)
    
    # Find best L0 value
    best_l0 = None
    best_diff = float('inf')
    for r in results:
        diff = abs(r['n_nonzero'] - target_nonzero)
        if diff < best_diff:
            best_diff = diff
            best_l0 = r
    
    print(f"\nBest L0 value for ~50% sparsity: {best_l0['l0_lambda']:.2e}")
    print(f"  Achieves {best_l0['n_nonzero']:,} non-zero weights ({(1-best_l0['sparsity'])*100:.1f}%)")
    
    # Check our proposed value
    proposed = [r for r in results if abs(r['l0_lambda'] - 1.0e-06) < 1e-9]
    if proposed:
        r = proposed[0]
        print(f"\nProposed L0 (1.0e-06) performance:")
        print(f"  Non-zero: {r['n_nonzero']:,} ({(1-r['sparsity'])*100:.1f}%)")
        if r['n_nonzero'] > target_nonzero * 1.5:
            print("  ⚠ May need higher L0 penalty for better sparsity")
        else:
            print("  ✓ Achieves reasonable sparsity")

if __name__ == "__main__":
    test_l0_sparsity_levels()