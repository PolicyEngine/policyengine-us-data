#!/usr/bin/env python3
"""
Utility functions for testing and debugging geo-stacking calibration.

Consolidated from various debug scripts used during development.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from policyengine_us import Microsimulation
from metrics_matrix_geo_stacking import GeoStackingMatrixBuilder


def debug_national_targets(targets_df):
    """Debug function to check for duplicate national targets."""
    national_targets = targets_df[targets_df['geographic_id'] == 'US']
    print("National targets in stacked matrix:")
    print(national_targets[['stacked_target_id', 'variable', 'value']].head(10))
    
    if len(national_targets) > 5:
        print("\n" + "=" * 60)
        print("WARNING: National targets are being duplicated!")
        print(f"Expected 5, got {len(national_targets)}")


def test_matrix_values_with_weights(matrix_df, targets_df, custom_weights=None):
    """
    Test matrix values with custom weights.
    
    Parameters
    ----------
    matrix_df : pd.DataFrame
        The calibration matrix
    targets_df : pd.DataFrame
        The target values
    custom_weights : np.ndarray, optional
        Custom weights to apply (defaults to uniform)
    """
    if custom_weights is None:
        # Use uniform weights
        n_households = matrix_df.shape[1]
        custom_weights = np.ones(n_households) * 100
    
    # Calculate weighted sums
    weighted_sums = matrix_df @ custom_weights
    
    # Compare to targets
    comparison = pd.DataFrame({
        'target': targets_df['value'].values,
        'weighted_sum': weighted_sums,
        'ratio': weighted_sums / targets_df['value'].values
    })
    
    print("Target vs Weighted Sum Comparison:")
    print(comparison.describe())
    
    return comparison


def verify_sparsity_pattern(matrix_df, targets_df):
    """
    Verify the sparsity pattern of a stacked matrix.
    
    Ensures:
    - National targets apply to all household copies
    - State targets only apply to their respective households
    """
    household_cols = matrix_df.columns.tolist()
    
    # Group households by state
    state_households = {}
    for col in household_cols:
        for state_code in ['6', '37']:  # CA and NC
            if f'_state{state_code}' in col:
                if state_code not in state_households:
                    state_households[state_code] = []
                state_households[state_code].append(col)
                break
    
    results = {}
    
    # Check national targets
    national_targets = targets_df[targets_df['geographic_id'] == 'US']
    if not national_targets.empty:
        nat_target = national_targets.iloc[0]
        nat_id = nat_target['stacked_target_id']
        nat_row = matrix_df.loc[nat_id]
        
        for state_code, households in state_households.items():
            nonzero = (nat_row[households] != 0).sum()
            results[f'national_in_state_{state_code}'] = nonzero
    
    # Check state-specific targets
    for state_code in state_households.keys():
        state_targets = targets_df[targets_df['geographic_id'] == state_code]
        if not state_targets.empty:
            state_target = state_targets.iloc[0]
            state_id = state_target['stacked_target_id']
            state_row = matrix_df.loc[state_id]
            
            # Should be non-zero only for this state
            for check_state, households in state_households.items():
                nonzero = (state_row[households] != 0).sum()
                results[f'state_{state_code}_in_state_{check_state}'] = nonzero
    
    return results


def check_period_handling(sim):
    """
    Debug function to check period handling in the simulation.
    
    The enhanced CPS 2024 dataset only contains 2024 data, but we may
    need to pull targets from different years.
    """
    print(f"Default calculation period: {sim.default_calculation_period}")
    
    # Try to get age for different periods
    test_periods = [2022, 2023, 2024]
    for period in test_periods:
        try:
            age_values = sim.calculate("age", period=period)
            non_zero = (age_values > 0).sum()
            print(f"Period {period}: {non_zero} non-zero age values")
        except Exception as e:
            print(f"Period {period}: Error - {e}")


if __name__ == "__main__":
    # Quick test of utilities
    print("Testing geo-stacking utilities...")
    
    # Setup
    db_uri = f"sqlite:///{Path.home()}/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
    builder = GeoStackingMatrixBuilder(db_uri)
    
    # Create simulation
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
    sim.build_from_dataset()
    
    # Build small test matrix
    print("\nBuilding test matrix for California...")
    targets_df, matrix_df = builder.build_matrix_for_geography('state', '6', sim)
    
    print(f"Matrix shape: {matrix_df.shape}")
    print(f"Number of targets: {len(targets_df)}")
    
    # Test utilities
    print("\nTesting matrix values with uniform weights...")
    comparison = test_matrix_values_with_weights(matrix_df, targets_df)
    
    print("\nUtilities test complete!")