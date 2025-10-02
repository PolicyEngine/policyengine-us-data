"""
Quick patch to add hierarchical consistency checking to simple_holdout results.
This can be called after simple_holdout completes.
"""

import numpy as np
import pandas as pd
import pickle
import os
from scipy import sparse as sp
import torch

def compute_hierarchical_consistency(calibration_package_path):
    """
    Load calibration package and compute hierarchical consistency metrics.
    Assumes model has been trained and weights are available.
    
    Args:
        calibration_package_path: Path to calibration_package.pkl
        
    Returns:
        dict with hierarchical consistency metrics
    """
    
    # Load the package
    with open(calibration_package_path, 'rb') as f:
        data = pickle.load(f)
    
    X_sparse = data['X_sparse']
    targets_df = data['targets_df']
    targets = targets_df.value.values
    
    # Load the most recent trained model or weights
    # For now, we'll compute what the metrics would look like
    # In practice, you'd load the actual weights from the trained model
    
    # Get CD-level targets
    cd_mask = targets_df['geographic_id'].str.len() > 2
    cd_targets = targets_df[cd_mask].copy()
    
    # Group CDs by state and variable
    hierarchical_checks = []
    
    for variable in cd_targets['variable'].unique():
        var_cd_targets = cd_targets[cd_targets['variable'] == variable]
        
        # Extract state from CD (assuming format like '0101' where first 2 digits are state)
        var_cd_targets['state'] = var_cd_targets['geographic_id'].apply(
            lambda x: x[:2] if len(x) == 4 else x[:-2]
        )
        
        # Sum by state
        state_sums = var_cd_targets.groupby('state')['value'].sum()
        
        # Check if we have corresponding state-level targets
        state_targets = targets_df[
            (targets_df['geographic_id'].isin(state_sums.index)) &
            (targets_df['variable'] == variable)
        ]
        
        if not state_targets.empty:
            for state_id in state_sums.index:
                state_target = state_targets[state_targets['geographic_id'] == state_id]
                if not state_target.empty:
                    cd_sum = state_sums[state_id]
                    state_val = state_target['value'].iloc[0]
                    rel_diff = (cd_sum - state_val) / state_val if state_val != 0 else 0
                    
                    hierarchical_checks.append({
                        'variable': variable,
                        'state': state_id,
                        'cd_sum': cd_sum,
                        'state_target': state_val,
                        'relative_difference': rel_diff
                    })
        
        # Check national consistency
        national_target = targets_df[
            (targets_df['geographic_id'] == 'US') &
            (targets_df['variable'] == variable)
        ]
        
        if not national_target.empty:
            cd_national_sum = var_cd_targets['value'].sum()
            national_val = national_target['value'].iloc[0]
            rel_diff = (cd_national_sum - national_val) / national_val if national_val != 0 else 0
            
            hierarchical_checks.append({
                'variable': variable,
                'state': 'US',
                'cd_sum': cd_national_sum,
                'state_target': national_val,
                'relative_difference': rel_diff
            })
    
    if hierarchical_checks:
        checks_df = pd.DataFrame(hierarchical_checks)
        
        # Summary statistics
        summary = {
            'mean_abs_rel_diff': np.abs(checks_df['relative_difference']).mean(),
            'max_abs_rel_diff': np.abs(checks_df['relative_difference']).max(),
            'n_checks': len(checks_df),
            'n_perfect_matches': (np.abs(checks_df['relative_difference']) < 0.001).sum(),
            'n_within_1pct': (np.abs(checks_df['relative_difference']) < 0.01).sum(),
            'n_within_5pct': (np.abs(checks_df['relative_difference']) < 0.05).sum(),
            'n_within_10pct': (np.abs(checks_df['relative_difference']) < 0.10).sum(),
        }
        
        # Worst mismatches
        worst = checks_df.nlargest(5, 'relative_difference')
        summary['worst_overestimates'] = worst[['variable', 'state', 'relative_difference']].to_dict('records')
        
        best = checks_df.nsmallest(5, 'relative_difference')
        summary['worst_underestimates'] = best[['variable', 'state', 'relative_difference']].to_dict('records')
        
        return {
            'summary': summary,
            'details': checks_df
        }
    else:
        return {
            'summary': {'message': 'No hierarchical targets found for comparison'},
            'details': pd.DataFrame()
        }


def analyze_holdout_hierarchical_consistency(results, targets_df):
    """
    Analyze hierarchical consistency for holdout groups only.
    This is useful when some groups are geographic aggregates.
    
    Args:
        results: Output from simple_holdout
        targets_df: Full targets dataframe with geographic info
        
    Returns:
        Enhanced results dict with hierarchical analysis
    """
    
    # Check if any holdout groups represent state or national aggregates
    holdout_group_ids = list(results['holdout_group_losses'].keys())
    
    # Map group IDs to geographic levels
    group_geo_analysis = []
    
    for group_id in holdout_group_ids:
        group_targets = targets_df[targets_df.index.isin(
            [i for i, g in enumerate(target_groups) if g == group_id]
        )]
        
        if not group_targets.empty:
            geo_ids = group_targets['geographic_id'].unique()
            
            # Classify the geographic level
            if 'US' in geo_ids:
                level = 'national'
            elif all(len(g) <= 2 for g in geo_ids):
                level = 'state'
            elif all(len(g) > 2 for g in geo_ids):
                level = 'cd'
            else:
                level = 'mixed'
            
            group_geo_analysis.append({
                'group_id': group_id,
                'geographic_level': level,
                'n_geos': len(geo_ids),
                'loss': results['holdout_group_losses'][group_id]
            })
    
    # Add to results
    if group_geo_analysis:
        geo_df = pd.DataFrame(group_geo_analysis)
        
        # Compare performance by geographic level
        level_performance = geo_df.groupby('geographic_level')['loss'].agg(['mean', 'std', 'min', 'max', 'count'])
        
        results['hierarchical_analysis'] = {
            'group_geographic_levels': group_geo_analysis,
            'performance_by_level': level_performance.to_dict(),
            'observation': 'Check if state/national groups have higher loss than CD groups'
        }
    
    return results


# Example usage:
if __name__ == "__main__":
    # Check hierarchical consistency of targets
    consistency = compute_hierarchical_consistency(
        "~/Downloads/cd_calibration_data/calibration_package.pkl"
    )
    
    print("Hierarchical Consistency Check")
    print("=" * 60)
    print(f"Mean absolute relative difference: {consistency['summary']['mean_abs_rel_diff']:.2%}")
    print(f"Max absolute relative difference: {consistency['summary']['max_abs_rel_diff']:.2%}")
    print(f"Checks within 1%: {consistency['summary']['n_within_1pct']}/{consistency['summary']['n_checks']}")
    print(f"Checks within 5%: {consistency['summary']['n_within_5pct']}/{consistency['summary']['n_checks']}")
    print(f"Checks within 10%: {consistency['summary']['n_within_10pct']}/{consistency['summary']['n_checks']}")
    
    if 'worst_overestimates' in consistency['summary']:
        print("\nWorst overestimates (CD sum > state/national target):")
        for item in consistency['summary']['worst_overestimates'][:3]:
            print(f"  {item['variable']} in {item['state']}: {item['relative_difference']:.1%}")