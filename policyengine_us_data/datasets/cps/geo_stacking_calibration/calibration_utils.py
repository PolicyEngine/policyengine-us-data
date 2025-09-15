"""
Shared utilities for calibration scripts.
"""
import os
import urllib
import tempfile
from typing import Tuple, List

import numpy as np
import pandas as pd


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
    
    # Process national targets first - each gets its own group
    national_mask = targets_df['stratum_group_id'] == 'national'
    national_targets = targets_df[national_mask]
    
    if len(national_targets) > 0:
        print(f"\nNational targets (each is a singleton group):")
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
            
            # Handle string stratum_group_ids (IRS scalars, AGI total, and state SNAP cost)
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
                elif stratum_group == 'state_snap_cost':
                    # State-level SNAP costs get their own group
                    mask = (targets_df['stratum_group_id'] == stratum_group)
                    matching_targets = targets_df[mask]
                    target_groups[mask] = group_id
                    stratum_name = 'State SNAP Cost (Administrative)'
                    n_targets = mask.sum()
                else:
                    continue  # Skip unknown string groups
            else:
                continue  # Skip other types
            
            # Count unique geographies in this group
            unique_geos = matching_targets['geographic_id'].unique()
            n_geos = len(unique_geos)
            
            group_info.append(f"Group {group_id}: All {stratum_name} targets ({n_targets} total)")
            
            # Only show details for small groups, otherwise just summary
            if n_geos <= 10:
                print(f"  Group {group_id}: {stratum_name} ({n_targets} targets across {n_geos} geographies)")
            else:
                print(f"  Group {group_id}: {stratum_name} ({n_targets} targets)")
            
            group_id += 1
    
    print(f"\nTotal groups created: {group_id}")
    print("=" * 40)
    
    return target_groups, group_info


# NOTE: this is for public files. A TODO is to contrast it with what we already have
def download_from_huggingface(file_name):
    """Download a file from HuggingFace to a temporary location."""
    base_url = "https://huggingface.co/policyengine/test/resolve/main/"
    url = base_url + file_name
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, file_name)
    
    # Check if already downloaded
    if not os.path.exists(local_path):
        print(f"Downloading {file_name} from HuggingFace...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")
    else:
        print(f"Using cached {local_path}")
    
    return local_path
