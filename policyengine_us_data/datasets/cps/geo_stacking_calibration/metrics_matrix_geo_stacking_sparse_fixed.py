"""
Fixed version of metrics_matrix_geo_stacking_sparse.py that properly implements:
1. Hierarchical target selection (CD -> State -> National)
2. Correct AGI histogram handling (only tax_unit_count, not all 3 variables)
3. State SNAP cost targets alongside CD SNAP household counts
4. No duplication of national targets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine, text
from scipy import sparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedSparseGeoStackingMatrixBuilder:
    """
    Fixed builder for sparse geo-stacked calibration matrices.
    Implements proper hierarchical target selection for congressional districts.
    """
    
    def __init__(self, db_uri: str, time_period: int = 2023):
        self.engine = create_engine(db_uri)
        self.time_period = time_period
        
    def get_national_hardcoded_targets(self) -> pd.DataFrame:
        """Get the 5 national hardcoded targets."""
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
        WHERE s.stratum_group_id = 1 
          AND s.notes = 'National hardcoded'
          AND t.period = :period
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'period': self.time_period})
        
        logger.info(f"Found {len(df)} national hardcoded targets")
        return df
    
    def get_state_fips_for_cd(self, cd_geoid: str) -> str:
        """Extract state FIPS from CD GEOID."""
        # CD GEOIDs are formatted as state_fips + district_number
        # e.g., "601" = California (06) district 01
        if len(cd_geoid) == 3:
            return cd_geoid[:1].zfill(2)  # Single digit state
        elif len(cd_geoid) == 4:
            return cd_geoid[:2]  # Two digit state
        else:
            raise ValueError(f"Unexpected CD GEOID format: {cd_geoid}")
    
    def get_state_stratum_id(self, state_fips: str) -> Optional[int]:
        """Get the stratum ID for a state."""
        query = """
        SELECT s.stratum_id
        FROM strata s
        JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = 1
          AND sc.constraint_variable = 'state_fips'
          AND sc.value = :state_fips
        LIMIT 1
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'state_fips': state_fips}).fetchone()
            return result[0] if result else None
    
    def get_cd_stratum_id(self, cd_geoid: str) -> Optional[int]:
        """Get the stratum ID for a congressional district."""
        query = """
        SELECT s.stratum_id
        FROM strata s
        JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = 1
          AND sc.constraint_variable = 'congressional_district_geoid'
          AND sc.value = :cd_geoid
        LIMIT 1
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {'cd_geoid': cd_geoid}).fetchone()
            return result[0] if result else None
    
    def get_demographic_targets(self, geographic_stratum_id: int, 
                              stratum_group_id: int, 
                              group_name: str) -> pd.DataFrame:
        """Get demographic targets for a geographic area."""
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
            sc.constraint_variable,
            sc.operation,
            sc.value as constraint_value
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        LEFT JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = :stratum_group_id
          AND s.parent_stratum_id = :parent_id
          AND t.period = :period
        ORDER BY t.variable, sc.constraint_variable
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'period': self.time_period,
                'stratum_group_id': stratum_group_id,
                'parent_id': geographic_stratum_id
            })
        
        if len(df) > 0:
            logger.info(f"Found {len(df)} {group_name} targets for stratum {geographic_stratum_id}")
        return df
    
    def get_irs_scalar_targets(self, geographic_stratum_id: int, geographic_level: str) -> pd.DataFrame:
        """Get IRS scalar targets (20 straightforward targets with count and amount)."""
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
        WHERE s.parent_stratum_id = :stratum_id
          AND t.period = :period
          AND t.variable NOT IN ('person_count', 'adjusted_gross_income')
          AND s.stratum_group_id > 10  -- IRS targets have higher group IDs
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'stratum_id': geographic_stratum_id,
                'period': self.time_period
            })
        
        if len(df) > 0:
            logger.info(f"Found {len(df)} IRS scalar targets for {geographic_level}")
        return df
    
    def get_agi_histogram_targets(self, geographic_stratum_id: int) -> pd.DataFrame:
        """
        Get AGI histogram targets - ONLY tax_unit_count, not all 3 variables.
        This reduces from 27 targets (9 bins × 3 variables) to 9 targets (9 bins × 1 variable).
        """
        query = """
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.active,
            t.tolerance,
            sc.constraint_variable,
            sc.operation,
            sc.value as constraint_value
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        LEFT JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
        WHERE s.stratum_group_id = 3  -- AGI distribution
          AND s.parent_stratum_id = :parent_id
          AND t.period = :period
          AND t.variable = 'tax_unit_count'  -- ONLY tax_unit_count, not person_count or adjusted_gross_income
        ORDER BY sc.constraint_value
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'period': self.time_period,
                'parent_id': geographic_stratum_id
            })
        
        if len(df) > 0:
            logger.info(f"Found {len(df.drop_duplicates('target_id'))} AGI histogram targets (tax_unit_count only)")
        return df
    
    def get_agi_total_target(self, geographic_stratum_id: int) -> pd.DataFrame:
        """Get the single AGI total amount target."""
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
        WHERE s.parent_stratum_id = :stratum_id
          AND t.period = :period
          AND t.variable = 'adjusted_gross_income'
          AND s.stratum_group_id > 10  -- Scalar IRS target
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'stratum_id': geographic_stratum_id,
                'period': self.time_period
            })
        
        return df
    
    def get_state_snap_cost_target(self, state_fips: str) -> pd.DataFrame:
        """Get state-level SNAP cost target (administrative data)."""
        state_stratum_id = self.get_state_stratum_id(state_fips)
        if not state_stratum_id:
            return pd.DataFrame()
        
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
        WHERE s.stratum_group_id = 4  -- SNAP
          AND s.parent_stratum_id = :parent_id
          AND t.period = :period
          AND t.variable = 'snap'  -- The cost variable, not household_count
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'period': self.time_period,
                'parent_id': state_stratum_id
            })
        
        return df
    
    def get_constraints_for_stratum(self, stratum_id: int) -> pd.DataFrame:
        """Get all constraints for a stratum."""
        query = """
        SELECT 
            constraint_variable,
            operation,
            value as constraint_value
        FROM stratum_constraints
        WHERE stratum_id = :stratum_id
        """
        
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={'stratum_id': stratum_id})
    
    def apply_constraints_to_sim_sparse(self, sim, constraints: pd.DataFrame, 
                                       variable: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply constraints and return sparse representation."""
        household_values = sim.calculate(variable).values
        
        # Apply each constraint
        mask = np.ones(len(household_values), dtype=bool)
        for _, constraint in constraints.iterrows():
            constraint_var = constraint['constraint_variable']
            operation = constraint['operation']
            value = constraint['constraint_value']
            
            if constraint_var in ['age', 'adjusted_gross_income', 'eitc_child_count', 
                                 'congressional_district_geoid', 'state_fips']:
                constraint_values = sim.calculate(constraint_var).values
                
                if operation == '<':
                    mask &= constraint_values < value
                elif operation == '>':
                    mask &= constraint_values >= value
                elif operation == '=':
                    mask &= constraint_values == value
        
        # Apply mask
        household_values = household_values * mask
        
        # Return sparse representation
        nonzero_indices = np.nonzero(household_values)[0]
        nonzero_values = household_values[nonzero_indices]
        
        return nonzero_indices, nonzero_values
    
    def build_cd_targets_with_hierarchy(self, cd_geoid: str) -> List[Dict]:
        """
        Build targets for a congressional district with proper hierarchy.
        This is the key function that implements the correct logic.
        """
        targets = []
        
        # Get CD and state stratum IDs
        cd_stratum_id = self.get_cd_stratum_id(cd_geoid)
        state_fips = self.get_state_fips_for_cd(cd_geoid)
        state_stratum_id = self.get_state_stratum_id(state_fips)
        
        if not cd_stratum_id:
            logger.warning(f"No stratum ID found for CD {cd_geoid}")
            return targets
        
        # 1. CD Age targets (7,848 total = 18 bins × 436 CDs)
        age_targets = self.get_demographic_targets(cd_stratum_id, 2, "age")
        for _, target in age_targets.iterrows():
            targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'stratum_id': target['stratum_id'],
                'geographic_level': 'congressional_district',
                'geographic_id': cd_geoid,
                'description': f"age_{cd_geoid}"
            })
        
        # 2. CD Medicaid targets (436 total)
        medicaid_targets = self.get_demographic_targets(cd_stratum_id, 5, "Medicaid")
        for _, target in medicaid_targets.iterrows():
            targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'stratum_id': target['stratum_id'],
                'geographic_level': 'congressional_district',
                'geographic_id': cd_geoid,
                'description': f"medicaid_{cd_geoid}"
            })
        
        # 3. CD SNAP household_count (436 total)
        snap_targets = self.get_demographic_targets(cd_stratum_id, 4, "SNAP")
        # Filter to only household_count
        snap_household = snap_targets[snap_targets['variable'] == 'household_count']
        for _, target in snap_household.iterrows():
            targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'stratum_id': target['stratum_id'],
                'geographic_level': 'congressional_district',
                'geographic_id': cd_geoid,
                'description': f"snap_household_{cd_geoid}"
            })
        
        # 4. State SNAP cost (51 total across all CDs)
        # This is a state-level target that households in this CD contribute to
        state_snap_cost = self.get_state_snap_cost_target(state_fips)
        for _, target in state_snap_cost.iterrows():
            targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'stratum_id': target['stratum_id'],
                'geographic_level': 'state',
                'geographic_id': state_fips,
                'description': f"snap_cost_state_{state_fips}"
            })
        
        # 5. CD IRS targets (21,800 total = 50 × 436)
        # 5a. IRS scalar targets (40 variables: 20 × 2 for count and amount)
        irs_scalar = self.get_irs_scalar_targets(cd_stratum_id, 'congressional_district')
        for _, target in irs_scalar.iterrows():
            targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'stratum_id': target['stratum_id'],
                'geographic_level': 'congressional_district',
                'geographic_id': cd_geoid,
                'description': f"irs_{target['variable']}_{cd_geoid}"
            })
        
        # 5b. AGI histogram (9 bins with ONLY tax_unit_count)
        agi_histogram = self.get_agi_histogram_targets(cd_stratum_id)
        for _, target in agi_histogram.iterrows():
            targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'stratum_id': target['stratum_id'],
                'geographic_level': 'congressional_district',
                'geographic_id': cd_geoid,
                'description': f"agi_bin_{cd_geoid}"
            })
        
        # 5c. AGI total amount (1 scalar)
        agi_total = self.get_agi_total_target(cd_stratum_id)
        for _, target in agi_total.iterrows():
            targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'stratum_id': target['stratum_id'],
                'geographic_level': 'congressional_district',
                'geographic_id': cd_geoid,
                'description': f"agi_total_{cd_geoid}"
            })
        
        return targets
    
    def build_stacked_matrix_sparse(self, congressional_districts: List[str], 
                                   sim=None) -> Tuple[pd.DataFrame, sparse.csr_matrix, Dict[str, List[str]]]:
        """
        Build the complete sparse calibration matrix for congressional districts.
        Should produce exactly 30,576 targets.
        """
        all_targets = []
        household_id_mapping = {}
        
        # 1. Add national targets ONCE (5 targets)
        national_targets = self.get_national_hardcoded_targets()
        for _, target in national_targets.iterrows():
            all_targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'stratum_id': target['stratum_id'],
                'geographic_level': 'national',
                'geographic_id': 'US',
                'description': f"{target['variable']}_national"
            })
        
        # Track unique state SNAP costs to avoid duplication
        state_snap_added = set()
        
        # 2. Process each congressional district
        for i, cd_geoid in enumerate(congressional_districts):
            if i % 50 == 0:
                logger.info(f"Processing CD {cd_geoid} ({i+1}/{len(congressional_districts)})")
            
            # Get all targets for this CD (including its state SNAP cost)
            cd_targets = self.build_cd_targets_with_hierarchy(cd_geoid)
            
            # Add CD-specific targets
            for target in cd_targets:
                if target['geographic_level'] == 'congressional_district':
                    # CD-level target
                    target['stacked_target_id'] = f"{target['target_id']}_cd{cd_geoid}"
                    all_targets.append(target)
                elif target['geographic_level'] == 'state':
                    # State-level target (SNAP cost) - add only once per state
                    state_id = target['geographic_id']
                    if state_id not in state_snap_added:
                        target['stacked_target_id'] = f"{target['target_id']}_state{state_id}"
                        all_targets.append(target)
                        state_snap_added.add(state_id)
            
            # Store household mapping
            if sim is not None:
                household_ids = sim.calculate("household_id").values
                household_id_mapping[f"cd{cd_geoid}"] = [
                    f"{hh_id}_cd{cd_geoid}" for hh_id in household_ids
                ]
        
        # Convert to DataFrame
        targets_df = pd.DataFrame(all_targets)
        
        logger.info(f"Total targets created: {len(targets_df)}")
        logger.info(f"Expected: 30,576 (5 national + 7,848 CD age + 436 CD Medicaid + "
                   f"436 CD SNAP household + 51 state SNAP cost + 21,800 CD IRS)")
        
        # Build sparse matrix if sim provided
        if sim is not None:
            n_households = len(sim.calculate("household_id").values)
            n_targets = len(targets_df)
            n_cds = len(congressional_districts)
            
            # Total columns = n_households × n_CDs
            total_cols = n_households * n_cds
            
            logger.info(f"Building sparse matrix: {n_targets} × {total_cols}")
            
            # Use LIL matrix for efficient construction
            matrix = sparse.lil_matrix((n_targets, total_cols), dtype=np.float32)
            
            # Fill the matrix
            for i, (_, target) in enumerate(targets_df.iterrows()):
                if i % 1000 == 0:
                    logger.info(f"Processing target {i+1}/{n_targets}")
                
                # Get constraints for this target
                constraints = self.get_constraints_for_stratum(target['stratum_id'])
                
                # Determine which CD copies should have non-zero values
                if target['geographic_level'] == 'national':
                    # National targets apply to all CD copies
                    for j, cd in enumerate(congressional_districts):
                        col_start = j * n_households
                        col_end = (j + 1) * n_households
                        
                        nonzero_indices, nonzero_values = self.apply_constraints_to_sim_sparse(
                            sim, constraints, target['variable']
                        )
                        
                        if len(nonzero_indices) > 0:
                            matrix[i, col_start + nonzero_indices] = nonzero_values
                
                elif target['geographic_level'] == 'congressional_district':
                    # CD targets apply only to that CD's copy
                    cd_idx = congressional_districts.index(target['geographic_id'])
                    col_start = cd_idx * n_households
                    
                    nonzero_indices, nonzero_values = self.apply_constraints_to_sim_sparse(
                        sim, constraints, target['variable']
                    )
                    
                    if len(nonzero_indices) > 0:
                        matrix[i, col_start + nonzero_indices] = nonzero_values
                
                elif target['geographic_level'] == 'state':
                    # State targets (SNAP cost) apply to all CDs in that state
                    state_fips = target['geographic_id']
                    for j, cd in enumerate(congressional_districts):
                        cd_state = self.get_state_fips_for_cd(cd)
                        if cd_state == state_fips:
                            col_start = j * n_households
                            
                            nonzero_indices, nonzero_values = self.apply_constraints_to_sim_sparse(
                                sim, constraints, target['variable']
                            )
                            
                            if len(nonzero_indices) > 0:
                                matrix[i, col_start + nonzero_indices] = nonzero_values
            
            # Convert to CSR for efficient operations
            matrix = matrix.tocsr()
            
            logger.info(f"Matrix created: shape {matrix.shape}, nnz={matrix.nnz:,}")
            return targets_df, matrix, household_id_mapping
        
        return targets_df, None, household_id_mapping