"""
Sparse geo-stacking calibration matrix creation for PolicyEngine US.

This module creates calibration matrices for the geo-stacking approach where
the same household dataset is treated as existing in multiple geographic areas.
Targets are rows, households are columns (small n, large p formulation).

This version builds sparse matrices directly, avoiding dense intermediate structures.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class SparseGeoStackingMatrixBuilder:
    """Build sparse calibration matrices for geo-stacking approach.
    
    NOTE: Period handling is complex due to mismatched data years:
    - The enhanced CPS 2024 dataset only contains 2024 data
    - Targets in the database exist for different years (2022, 2023, 2024)
    - For now, we pull targets from whatever year they exist and use 2024 data
    - This temporal mismatch will be addressed in future iterations
    """
    
    def __init__(self, db_uri: str, time_period: int = 2024):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.time_period = time_period  # Default to 2024 to match CPS data
        
    def get_national_targets(self) -> pd.DataFrame:
        """
        Get national-level targets from the database.
        These have no state equivalents and apply to all geographies.
        """
        query = """
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.active,
            t.tolerance,
            s.notes as stratum_notes,
            src.name as source_name
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        JOIN sources src ON t.source_id = src.source_id
        WHERE s.parent_stratum_id IS NULL  -- National level
          AND s.stratum_group_id = 1  -- Geographic stratum
          AND UPPER(src.type) = 'HARDCODED'  -- Hardcoded national targets (case-insensitive)
        ORDER BY t.variable
        """
        
        with self.engine.connect() as conn:
            # Don't filter by period for now - get any available hardcoded targets
            df = pd.read_sql(query, conn)
        
        logger.info(f"Found {len(df)} national targets from database")
        return df
    
    def get_irs_scalar_targets(self, geographic_stratum_id: int,
                               geographic_level: str) -> pd.DataFrame:
        """
        Get IRS scalar variables from child strata with constraints.
        These are now in child strata with constraints like "salt > 0"
        """
        query = """
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.active,
            t.tolerance,
            s.notes as stratum_notes,
            s.stratum_group_id,
            src.name as source_name
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        JOIN sources src ON t.source_id = src.source_id
        WHERE s.parent_stratum_id = :stratum_id  -- Look for children of geographic stratum
          AND s.stratum_group_id >= 100  -- IRS strata have group_id >= 100
          AND src.name = 'IRS Statistics of Income'
          AND t.variable NOT IN ('adjusted_gross_income')  -- AGI handled separately
        ORDER BY s.stratum_group_id, t.variable
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'stratum_id': geographic_stratum_id})
        
        if len(df) > 0:
            logger.info(f"Found {len(df)} IRS scalar targets for {geographic_level}")
        return df
    
    def get_agi_total_target(self, geographic_stratum_id: int,
                             geographic_level: str) -> pd.DataFrame:
        """
        Get the total AGI amount for a geography.
        This is a single scalar value, not a distribution.
        """
        query = """
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.active,
            t.tolerance,
            s.notes as stratum_notes,
            src.name as source_name
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        JOIN sources src ON t.source_id = src.source_id
        WHERE s.stratum_id = :stratum_id
          AND t.variable = 'adjusted_gross_income'
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'stratum_id': geographic_stratum_id})
        
        if len(df) > 0:
            logger.info(f"Found AGI total target for {geographic_level}")
        return df
    
    def get_demographic_targets(self, geographic_stratum_id: int, 
                              stratum_group_id: int, 
                              group_name: str) -> pd.DataFrame:
        """
        Generic function to get demographic targets for a geographic area.
        
        Args:
            geographic_stratum_id: The parent geographic stratum
            stratum_group_id: The demographic group (2=Age, 3=Income, 4=SNAP, 5=Medicaid, 6=EITC)
            group_name: Descriptive name for logging
        """
        # First try with the specified period, then fall back to most recent
        query_with_period = """
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.active,
            t.tolerance,
            s.notes as stratum_notes,
            s.stratum_group_id,
            sc.constraint_variable,
            sc.operation,
            sc.value as constraint_value,
            t.period
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        LEFT JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE t.period = :period
          AND s.stratum_group_id = :stratum_group_id
          AND s.parent_stratum_id = :parent_id
        ORDER BY t.variable, sc.constraint_variable
        """
        
        query_any_period = """
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.active,
            t.tolerance,
            s.notes as stratum_notes,
            s.stratum_group_id,
            sc.constraint_variable,
            sc.operation,
            sc.value as constraint_value,
            t.period
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        LEFT JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = :stratum_group_id
          AND s.parent_stratum_id = :parent_id
          AND t.period = (
              SELECT MAX(t2.period)
              FROM targets t2
              JOIN strata s2 ON t2.stratum_id = s2.stratum_id
              WHERE s2.stratum_group_id = :stratum_group_id
                AND s2.parent_stratum_id = :parent_id
          )
        ORDER BY t.variable, sc.constraint_variable
        """
        
        with self.engine.connect() as conn:
            # Try with specified period first
            df = pd.read_sql(query_with_period, conn, params={
                'period': self.time_period,
                'stratum_group_id': stratum_group_id,
                'parent_id': geographic_stratum_id
            })
            
            # If no results, try most recent period
            if len(df) == 0:
                df = pd.read_sql(query_any_period, conn, params={
                    'stratum_group_id': stratum_group_id,
                    'parent_id': geographic_stratum_id
                })
                if len(df) > 0:
                    period_used = df['period'].iloc[0]
                    logger.info(f"No {group_name} targets for {self.time_period}, using {period_used} instead")
        
        logger.info(f"Found {len(df)} {group_name} targets for stratum {geographic_stratum_id}")
        return df
    
    def get_state_stratum_id(self, state_fips: str) -> Optional[int]:
        """Get the stratum_id for a state."""
        query = """
        SELECT s.stratum_id 
        FROM strata s
        JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = 1  -- Geographic
          AND sc.constraint_variable = 'state_fips'
          AND sc.value = :state_fips
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'state_fips': state_fips}).fetchone()
            return result[0] if result else None
    
    def get_cd_stratum_id(self, cd_geoid: str) -> Optional[int]:
        """Get the stratum_id for a congressional district."""
        query = """
        SELECT s.stratum_id 
        FROM strata s
        JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = 1  -- Geographic
          AND sc.constraint_variable = 'congressional_district_geoid'
          AND sc.value = :cd_geoid
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'cd_geoid': cd_geoid}).fetchone()
            return result[0] if result else None
    
    def get_constraints_for_stratum(self, stratum_id: int) -> pd.DataFrame:
        """Get all constraints for a specific stratum."""
        query = """
        SELECT 
            constraint_variable,
            operation,
            value,
            notes
        FROM stratum_constraints
        WHERE stratum_id = :stratum_id
          AND constraint_variable NOT IN ('state_fips', 'congressional_district_geoid')
        ORDER BY constraint_variable
        """
        
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={'stratum_id': stratum_id})
    
    def apply_constraints_to_sim_sparse(self, sim, constraints_df: pd.DataFrame, 
                                       target_variable: str, 
                                       skip_geographic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply constraints and return sparse representation (indices and values).
        
        Args:
            sim: Microsimulation instance
            constraints_df: DataFrame with constraints
            target_variable: Variable to calculate
            skip_geographic: Whether to skip geographic constraints (default True)
        
        Returns:
            Tuple of (nonzero_indices, nonzero_values) at household level
        """
        if sim is None:
            raise ValueError("Microsimulation instance required")
            
        # Get target entity level
        target_entity = sim.tax_benefit_system.variables[target_variable].entity.key
        
        # Start with all ones mask at entity level
        entity_count = len(sim.calculate(f"{target_entity}_id").values)
        entity_mask = np.ones(entity_count, dtype=bool)
        
        # Apply each constraint
        for _, constraint in constraints_df.iterrows():
            var = constraint['constraint_variable']
            op = constraint['operation']
            val = constraint['value']
            
            # Skip geographic constraints only if requested
            if skip_geographic and var in ['state_fips', 'congressional_district_geoid']:
                continue
                
            # Get values for this constraint variable WITHOUT explicit period
            try:
                constraint_values = sim.calculate(var).values
                constraint_entity = sim.tax_benefit_system.variables[var].entity.key
                
                # Parse value based on type
                try:
                    parsed_val = float(val)
                    if parsed_val.is_integer():
                        parsed_val = int(parsed_val)
                except ValueError:
                    if val == "True":
                        parsed_val = True
                    elif val == "False":
                        parsed_val = False
                    else:
                        parsed_val = val
                
                # Apply operation using standardized operators from database
                if op == '==':
                    mask = (constraint_values == parsed_val).astype(bool)
                elif op == '>':
                    mask = (constraint_values > parsed_val).astype(bool)
                elif op == '>=':
                    mask = (constraint_values >= parsed_val).astype(bool)
                elif op == '<':
                    mask = (constraint_values < parsed_val).astype(bool)
                elif op == '<=':
                    mask = (constraint_values <= parsed_val).astype(bool)
                elif op == '!=':
                    mask = (constraint_values != parsed_val).astype(bool)
                else:
                    logger.warning(f"Unknown operation {op}, skipping")
                    continue
                
                # Map to target entity if needed
                if constraint_entity != target_entity:
                    mask = sim.map_result(mask, constraint_entity, target_entity)
                    mask = mask.astype(bool)
                
                # Combine with existing mask
                entity_mask = entity_mask & mask
                
            except Exception as e:
                logger.warning(f"Could not apply constraint {var} {op} {val}: {e}")
                continue
        
        # Calculate target variable values WITHOUT explicit period
        target_values = sim.calculate(target_variable).values
        
        # Apply mask at entity level
        masked_values = target_values * entity_mask
        
        # Map to household level
        if target_entity != "household":
            household_values = sim.map_result(masked_values, target_entity, "household")
        else:
            household_values = masked_values
        
        # Return sparse representation
        nonzero_indices = np.nonzero(household_values)[0]
        nonzero_values = household_values[nonzero_indices]
        
        return nonzero_indices, nonzero_values
    
    def build_matrix_for_geography_sparse(self, geographic_level: str, 
                                         geographic_id: str, 
                                         sim=None) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]:
        """
        Build sparse calibration matrix for any geographic level.
        
        Returns:
            Tuple of (targets_df, sparse_matrix, household_ids)
        """
        # Get the geographic stratum ID
        if geographic_level == 'state':
            geo_stratum_id = self.get_state_stratum_id(geographic_id)
            geo_label = f"state_{geographic_id}"
        elif geographic_level == 'congressional_district':
            geo_stratum_id = self.get_cd_stratum_id(geographic_id)
            geo_label = f"cd_{geographic_id}"
        else:
            raise ValueError(f"Unknown geographic level: {geographic_level}")
        
        if geo_stratum_id is None:
            raise ValueError(f"Could not find {geographic_level} {geographic_id} in database")
        
        # Get national targets from database
        national_targets = self.get_national_targets()
        
        # Get demographic targets for this geography
        age_targets = self.get_demographic_targets(geo_stratum_id, 2, "age")
        
        # For AGI distribution, we want only one count variable (ideally tax_unit_count)
        # Currently the database has person_count, so we'll use that for now
        agi_distribution_targets = self.get_demographic_targets(geo_stratum_id, 3, "AGI_distribution")
        
        snap_targets = self.get_demographic_targets(geo_stratum_id, 4, "SNAP")
        medicaid_targets = self.get_demographic_targets(geo_stratum_id, 5, "Medicaid")
        eitc_targets = self.get_demographic_targets(geo_stratum_id, 6, "EITC")
        
        # Get IRS scalar targets (individual variables, each its own group)
        irs_scalar_targets = self.get_irs_scalar_targets(geo_stratum_id, geographic_level)
        agi_total_target = self.get_agi_total_target(geo_stratum_id, geographic_level)
        
        all_targets = []
        
        # Add national targets
        for _, target in national_targets.iterrows():
            all_targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'active': target['active'],
                'tolerance': target['tolerance'],
                'stratum_id': target['stratum_id'],
                'stratum_group_id': 'national',
                'geographic_level': 'national',
                'geographic_id': 'US',
                'description': f"{target['variable']}_national"
            })
        
        # Process demographic targets (similar to original but simplified)
        processed_strata = set()
        
        # Helper function to process target groups
        def process_target_group(targets_df, group_name):
            for stratum_id in targets_df['stratum_id'].unique():
                if stratum_id in processed_strata:
                    continue
                processed_strata.add(stratum_id)
                
                stratum_targets = targets_df[targets_df['stratum_id'] == stratum_id]
                
                # Build description from constraints once per stratum
                constraints = stratum_targets[['constraint_variable', 'operation', 'constraint_value']].drop_duplicates()
                desc_parts = []
                for _, c in constraints.iterrows():
                    if c['constraint_variable'] in ['age', 'adjusted_gross_income', 'eitc_child_count']:
                        desc_parts.append(f"{c['constraint_variable']}{c['operation']}{c['constraint_value']}")
                
                # Group by variable to handle multiple variables per stratum (e.g., SNAP)
                for variable in stratum_targets['variable'].unique():
                    variable_targets = stratum_targets[stratum_targets['variable'] == variable]
                    # Use the first row for this variable (they should all have same value)
                    target = variable_targets.iloc[0]
                    
                    # Build description with variable name
                    full_desc_parts = [variable] + desc_parts
                    
                    all_targets.append({
                        'target_id': target['target_id'],
                        'variable': target['variable'],
                        'value': target['value'],
                        'active': target['active'],
                        'tolerance': target['tolerance'],
                        'stratum_id': target['stratum_id'],
                        'stratum_group_id': target['stratum_group_id'],
                        'geographic_level': geographic_level,
                        'geographic_id': geographic_id,
                        'description': '_'.join(full_desc_parts)
                    })
        
        process_target_group(age_targets, "age")
        process_target_group(agi_distribution_targets, "agi_distribution")
        process_target_group(snap_targets, "snap")
        process_target_group(medicaid_targets, "medicaid")
        process_target_group(eitc_targets, "eitc")
        
        # Process IRS scalar targets - need to check if they come from constrained strata
        for _, target in irs_scalar_targets.iterrows():
            # Check if this target's stratum has a constraint (indicating it's an IRS child stratum)
            constraints = self.get_constraints_for_stratum(target['stratum_id'])
            
            # If there's a constraint like "salt > 0", use "salt" for the group ID
            if not constraints.empty and len(constraints) > 0:
                # Get the constraint variable (e.g., "salt" from "salt > 0")
                constraint_var = constraints.iloc[0]['constraint_variable']
                # Use the constraint variable for grouping both count and amount
                stratum_group_override = f'irs_scalar_{constraint_var}'
            else:
                # Fall back to using the target variable name
                stratum_group_override = f'irs_scalar_{target["variable"]}'
            
            all_targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'active': target.get('active', True),
                'tolerance': target.get('tolerance', 0.05),
                'stratum_id': target['stratum_id'],
                'stratum_group_id': stratum_group_override,
                'geographic_level': geographic_level,
                'geographic_id': geographic_id,
                'description': f"{target['variable']}_{geographic_level}"
            })
        
        # Process AGI total target
        for _, target in agi_total_target.iterrows():
            all_targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'active': target.get('active', True),
                'tolerance': target.get('tolerance', 0.05),
                'stratum_id': target['stratum_id'],
                'stratum_group_id': 'agi_total_amount',
                'geographic_level': geographic_level,
                'geographic_id': geographic_id,
                'description': f"agi_total_{geographic_level}"
            })
        
        targets_df = pd.DataFrame(all_targets)
        
        # Build sparse matrix if sim provided
        if sim is not None:
            household_ids = sim.calculate("household_id").values
            n_households = len(household_ids)
            n_targets = len(targets_df)
            
            # Use LIL matrix for efficient row-by-row construction
            matrix = sparse.lil_matrix((n_targets, n_households), dtype=np.float32)
            
            for i, (_, target) in enumerate(targets_df.iterrows()):
                # Get constraints for this stratum
                constraints = self.get_constraints_for_stratum(target['stratum_id'])
                
                # Get sparse representation of household values
                nonzero_indices, nonzero_values = self.apply_constraints_to_sim_sparse(
                    sim, constraints, target['variable']
                )
                
                # Set the sparse row
                if len(nonzero_indices) > 0:
                    matrix[i, nonzero_indices] = nonzero_values
            
            # Convert to CSR for efficient operations
            matrix = matrix.tocsr()
            
            logger.info(f"Created sparse matrix for {geographic_level} {geographic_id}: shape {matrix.shape}, nnz={matrix.nnz}")
            return targets_df, matrix, household_ids.tolist()
        
        return targets_df, None, []
    
    def get_state_snap_cost(self, state_fips: str) -> pd.DataFrame:
        """Get state-level SNAP cost target (administrative data)."""
        query = """
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.active,
            t.tolerance
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = 4  -- SNAP
          AND t.variable = 'snap'  -- Cost variable
          AND sc.constraint_variable = 'state_fips'
          AND sc.value = :state_fips
          AND t.period = :period
        """
        
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={
                'state_fips': state_fips,
                'period': self.time_period
            })
    
    def get_state_fips_for_cd(self, cd_geoid: str) -> str:
        """Extract state FIPS from CD GEOID."""
        # CD GEOIDs are formatted as state_fips + district_number
        # e.g., "601" = California (06) district 01
        if len(cd_geoid) == 3:
            return str(int(cd_geoid[:1]))  # Single digit state, return as string of integer
        elif len(cd_geoid) == 4:
            return str(int(cd_geoid[:2]))  # Two digit state, return as string of integer
        else:
            raise ValueError(f"Unexpected CD GEOID format: {cd_geoid}")
    
    def build_stacked_matrix_sparse(self, geographic_level: str, 
                                   geographic_ids: List[str], 
                                   sim=None) -> Tuple[pd.DataFrame, sparse.csr_matrix, Dict[str, List[str]]]:
        """
        Build stacked sparse calibration matrix for multiple geographic areas.
        
        Returns:
            Tuple of (targets_df, sparse_matrix, household_id_mapping)
        """
        all_targets = []
        geo_matrices = []
        household_id_mapping = {}
        
        # First, get national targets once (they apply to all geographic copies)
        national_targets = self.get_national_targets()
        national_targets_list = []
        for _, target in national_targets.iterrows():
            national_targets_list.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'active': target['active'],
                'tolerance': target['tolerance'],
                'stratum_id': target['stratum_id'],
                'stratum_group_id': 'national',
                'geographic_level': 'national',
                'geographic_id': 'US',
                'description': f"{target['variable']}_national",
                'stacked_target_id': f"{target['target_id']}_national"
            })
        
        # Build matrix for each geography
        national_matrix_parts = []
        for i, geo_id in enumerate(geographic_ids):
            logger.info(f"Processing {geographic_level} {geo_id} ({i+1}/{len(geographic_ids)})")
            
            # Build matrix for this geography
            targets_df, matrix, household_ids = self.build_matrix_for_geography_sparse(
                geographic_level, geo_id, sim
            )
            
            if matrix is not None:
                # Separate national and geo-specific targets
                national_mask = targets_df['geographic_id'] == 'US'
                geo_mask = ~national_mask
                
                # Extract submatrices - convert pandas Series to numpy array for indexing
                if national_mask.any():
                    national_part = matrix[national_mask.values, :]
                    national_matrix_parts.append(national_part)
                
                if geo_mask.any():
                    geo_part = matrix[geo_mask.values, :]
                    geo_matrices.append(geo_part)
                
                # Add geo-specific targets
                geo_specific_targets = targets_df[geo_mask].copy()
                prefix = "state" if geographic_level == "state" else "cd"
                geo_specific_targets['stacked_target_id'] = (
                    geo_specific_targets['target_id'].astype(str) + f"_{prefix}{geo_id}"
                )
                all_targets.append(geo_specific_targets)
                
                # Store household ID mapping
                household_id_mapping[f"{prefix}{geo_id}"] = [
                    f"{hh_id}_{prefix}{geo_id}" for hh_id in household_ids
                ]
        
        # If building for congressional districts, add state-level SNAP costs
        state_snap_targets_list = []
        state_snap_matrices = []
        if geographic_level == "congressional_district" and sim is not None:
            # Identify unique states from the CDs
            unique_states = set()
            for cd_id in geographic_ids:
                state_fips = self.get_state_fips_for_cd(cd_id)
                unique_states.add(state_fips)
            
            logger.info(f"Adding state SNAP costs for {len(unique_states)} states")
            
            # Get household info - must match the actual matrix columns
            household_ids = sim.calculate("household_id").values
            n_households = len(household_ids)
            total_cols = n_households * len(geographic_ids)
            
            # Get SNAP cost target for each state
            for state_fips in sorted(unique_states):
                snap_cost_df = self.get_state_snap_cost(state_fips)
                if not snap_cost_df.empty:
                    for _, target in snap_cost_df.iterrows():
                        state_snap_targets_list.append({
                            'target_id': target['target_id'],
                            'variable': target['variable'],
                            'value': target['value'],
                            'active': target.get('active', True),
                            'tolerance': target.get('tolerance', 0.05),
                            'stratum_id': target['stratum_id'],
                            'stratum_group_id': 'state_snap_cost',
                            'geographic_level': 'state',
                            'geographic_id': state_fips,
                            'description': f"snap_cost_state_{state_fips}",
                            'stacked_target_id': f"{target['target_id']}_state_{state_fips}"
                        })
                        
                        # Build matrix row for this state SNAP cost
                        # This row should have SNAP values for households in CDs of this state
                        # Get constraints for this state SNAP stratum to apply to simulation
                        constraints = self.get_constraints_for_stratum(target['stratum_id'])
                        
                        # Create a sparse row with correct dimensions (1 x total_cols)
                        row_data = []
                        row_indices = []
                        
                        # Calculate SNAP values once (only for households with SNAP > 0 in this state)
                        # Apply the state constraint to get SNAP values
                        # Important: skip_geographic=False to apply state_fips constraint
                        nonzero_indices, nonzero_values = self.apply_constraints_to_sim_sparse(
                            sim, constraints, 'snap', skip_geographic=False
                        )
                        
                        # Create a mapping of household indices to SNAP values
                        snap_value_map = dict(zip(nonzero_indices, nonzero_values))
                        
                        # For each CD, check if it's in this state and add SNAP values
                        for cd_idx, cd_id in enumerate(geographic_ids):
                            cd_state_fips = self.get_state_fips_for_cd(cd_id)
                            if cd_state_fips == state_fips:
                                # This CD is in the target state
                                # Add SNAP values at the correct column positions
                                col_offset = cd_idx * n_households
                                for hh_idx, snap_val in snap_value_map.items():
                                    row_indices.append(col_offset + hh_idx)
                                    row_data.append(snap_val)
                        
                        # Create sparse matrix row
                        if row_data:
                            row_matrix = sparse.csr_matrix(
                                (row_data, ([0] * len(row_data), row_indices)),
                                shape=(1, total_cols)
                            )
                            state_snap_matrices.append(row_matrix)
            
            # Add state SNAP targets to all_targets
            if state_snap_targets_list:
                all_targets.append(pd.DataFrame(state_snap_targets_list))
        
        # Add national targets to the list once
        if national_targets_list:
            all_targets.insert(0, pd.DataFrame(national_targets_list))
        
        # Combine all targets
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        # Stack matrices if provided
        if geo_matrices:
            # Stack national targets (horizontally concatenate across all geographies)
            if national_matrix_parts:
                stacked_national = sparse.hstack(national_matrix_parts)
            else:
                stacked_national = None
            
            # Stack geo-specific targets (block diagonal)
            stacked_geo = sparse.block_diag(geo_matrices)
            
            # Combine all matrix parts
            matrix_parts = []
            if stacked_national is not None:
                matrix_parts.append(stacked_national)
            matrix_parts.append(stacked_geo)
            
            # Add state SNAP matrices if we have them (for CD calibration)
            if state_snap_matrices:
                stacked_state_snap = sparse.vstack(state_snap_matrices)
                matrix_parts.append(stacked_state_snap)
            
            # Combine all parts
            combined_matrix = sparse.vstack(matrix_parts)
            
            # Convert to CSR for efficiency
            combined_matrix = combined_matrix.tocsr()
            
            logger.info(f"Created stacked sparse matrix: shape {combined_matrix.shape}, nnz={combined_matrix.nnz}")
            return combined_targets, combined_matrix, household_id_mapping
        
        return combined_targets, None, household_id_mapping


def main():
    """Example usage for California and North Carolina."""
    from policyengine_us import Microsimulation
    
    # Database path
    db_uri = "sqlite:////home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
    
    # Initialize sparse builder
    builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2024)
    
    # Create microsimulation with 2024 data
    print("Loading microsimulation...")
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
    sim.build_from_dataset()
    
    # Test single state
    print("\nBuilding sparse matrix for California (FIPS 6)...")
    targets_df, matrix, household_ids = builder.build_matrix_for_geography_sparse('state', '6', sim)
    
    print("\nTarget Summary:")
    print(f"Total targets: {len(targets_df)}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix sparsity: {matrix.nnz} non-zero elements ({100*matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.4f}%)")
    print(f"Memory usage: {matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes} bytes")
    
    # Test stacking multiple states
    print("\n" + "="*70)
    print("Testing multi-state stacking: California (6) and North Carolina (37)")
    print("="*70)
    
    targets_df, matrix, hh_mapping = builder.build_stacked_matrix_sparse(
        'state', 
        ['6', '37'],
        sim
    )
    
    if matrix is not None:
        print(f"\nStacked matrix shape: {matrix.shape}")
        print(f"Stacked matrix sparsity: {matrix.nnz} non-zero elements ({100*matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.4f}%)")
        print(f"Memory usage: {matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes} bytes")
        
        # Compare to dense matrix memory
        dense_memory = matrix.shape[0] * matrix.shape[1] * 4  # 4 bytes per float32
        print(f"Dense matrix would use: {dense_memory} bytes")
        print(f"Memory savings: {100*(1 - (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)/dense_memory):.2f}%")


if __name__ == "__main__":
    main()