"""
Shared utilities for calibration scripts.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


def create_target_groups(targets_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Automatically create target groups based on metadata.
    
    Grouping rules:
    1. Each national hardcoded target gets its own group (singleton)
       - These are scalar values like "tip_income" or "medical_expenses" 
       - Each one represents a fundamentally different quantity
       - We want each to contribute equally to the loss
       
    2. All demographic targets grouped by (geographic_id, stratum_group_id)
       - All 18 age bins for California form ONE group
       - All 18 age bins for North Carolina form ONE group
       - This prevents age variables from dominating the loss
       
    The result is that each group contributes equally to the total loss,
    regardless of how many individual targets are in the group.
    
    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame containing target metadata with columns:
        - stratum_group_id: Identifier for the type of target
        - geographic_id: Geographic identifier (US, state FIPS, etc.)
        - variable: Variable name
        - value: Target value
        - description: Human-readable description
    
    Returns
    -------
    target_groups : np.ndarray
        Array of group IDs for each target
    group_info : List[str]
        List of descriptive strings for each group
    """
    target_groups = np.zeros(len(targets_df), dtype=int)
    group_id = 0
    group_info = []
    
    print("\n=== Creating Target Groups ===")
    
    # Process national hardcoded targets first - each gets its own group
    national_mask = targets_df['stratum_group_id'] == 'national_hardcoded'
    national_targets = targets_df[national_mask]
    
    if len(national_targets) > 0:
        print(f"\nNational hardcoded targets (each is a singleton group):")
        for idx in national_targets.index:
            target = targets_df.loc[idx]
            var_name = target['variable']
            value = target['value']
            
            target_groups[idx] = group_id
            group_info.append(f"Group {group_id}: National {var_name} (1 target, value={value:,.0f})")
            print(f"  Group {group_id}: {var_name} = {value:,.0f}")
            group_id += 1
    
    # Process demographic targets - grouped by stratum_group_id ONLY (not geography)
    # This ensures all age targets across all states form ONE group
    demographic_mask = ~national_mask
    demographic_df = targets_df[demographic_mask]
    
    if len(demographic_df) > 0:
        print(f"\nDemographic and IRS targets:")
        
        # Get unique stratum_group_ids (NOT grouped by geography)
        unique_stratum_groups = demographic_df['stratum_group_id'].unique()
        
        for stratum_group in unique_stratum_groups:
            # Handle numeric stratum_group_ids (histograms)
            if isinstance(stratum_group, (int, np.integer)):
                # Find ALL targets with this stratum_group_id across ALL geographies
                mask = (targets_df['stratum_group_id'] == stratum_group)
                
                matching_targets = targets_df[mask]
                target_groups[mask] = group_id
                
                # Create descriptive label
                stratum_labels = {
                    1: 'Geographic',  # This shouldn't appear in demographic targets
                    2: 'Age',
                    3: 'AGI Distribution',
                    4: 'SNAP',
                    5: 'Medicaid', 
                    6: 'EITC'
                }
                stratum_name = stratum_labels.get(stratum_group, f'Unknown({stratum_group})')
                n_targets = mask.sum()
            
            # Handle string stratum_group_ids (IRS scalars and AGI total)
            elif isinstance(stratum_group, str):
                if stratum_group.startswith('irs_scalar_'):
                    # Each IRS scalar variable gets its own group
                    mask = (targets_df['stratum_group_id'] == stratum_group)
                    matching_targets = targets_df[mask]
                    target_groups[mask] = group_id
                    var_name = stratum_group.replace('irs_scalar_', '')
                    stratum_name = f'IRS {var_name}'
                    n_targets = mask.sum()
                elif stratum_group == 'agi_total_amount':
                    # AGI total amount gets its own group
                    mask = (targets_df['stratum_group_id'] == stratum_group)
                    matching_targets = targets_df[mask]
                    target_groups[mask] = group_id
                    stratum_name = 'AGI Total Amount'
                    n_targets = mask.sum()
                else:
                    continue  # Skip unknown string groups
            else:
                continue  # Skip other types
            
            # Count unique geographies in this group
            unique_geos = matching_targets['geographic_id'].unique()
            n_geos = len(unique_geos)
            
            # Get geographic breakdown
            geo_counts = matching_targets.groupby('geographic_id').size()
            
            # Build state name mapping (extend as needed)
            state_names = {
                '6': 'California', 
                '37': 'North Carolina',
                '48': 'Texas',
                '36': 'New York',
                '12': 'Florida',
                '42': 'Pennsylvania',
                '17': 'Illinois',
                '39': 'Ohio',
                '13': 'Georgia',
                '26': 'Michigan',
                # Add more states as needed
            }
            
            geo_breakdown = []
            for geo_id, count in geo_counts.items():
                geo_name = state_names.get(geo_id, f'State {geo_id}')
                geo_breakdown.append(f"{geo_name}: {count}")
            
            group_info.append(f"Group {group_id}: All {stratum_name} targets ({n_targets} total)")
            print(f"  Group {group_id}: {stratum_name} histogram across {n_geos} geographies ({n_targets} total targets)")
            print(f"    Geographic breakdown: {', '.join(geo_breakdown)}")
            
            # Show sample targets from different geographies
            if n_geos > 1 and n_targets > 3:
                for geo_id in unique_geos[:2]:  # Show first two geographies
                    geo_name = state_names.get(geo_id, f'State {geo_id}')
                    geo_targets = matching_targets[matching_targets['geographic_id'] == geo_id]
                    print(f"    {geo_name} samples:")
                    print(f"      - {geo_targets.iloc[0]['description']}")
                    if len(geo_targets) > 1:
                        print(f"      - {geo_targets.iloc[-1]['description']}")
            
            group_id += 1
    
    print(f"\nTotal groups created: {group_id}")
    print("=" * 40)
    
    return target_groups, group_info