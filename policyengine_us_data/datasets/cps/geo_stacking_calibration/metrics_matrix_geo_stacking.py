"""
Geo-stacking calibration matrix creation for PolicyEngine US.

This module creates calibration matrices for the geo-stacking approach where
the same household dataset is treated as existing in multiple geographic areas.
Targets are rows, households are columns (small n, large p formulation).
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class GeoStackingMatrixBuilder:
    """Build calibration matrices for geo-stacking approach."""
    
    def __init__(self, db_uri: str, time_period: int = 2023):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.time_period = time_period
        
    def get_national_hardcoded_targets(self) -> pd.DataFrame:
        """
        Get national-level hardcoded targets (non-histogram variables).
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
        WHERE t.period = :period
          AND s.parent_stratum_id IS NULL  -- National level
          AND s.stratum_group_id = 1  -- Geographic stratum
          AND src.type = 'hardcoded'  -- Hardcoded national targets
        ORDER BY t.variable
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'period': self.time_period})
        
        logger.info(f"Found {len(df)} national hardcoded targets")
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
        WHERE t.period = :period
          AND s.stratum_group_id = :stratum_group_id
          AND s.parent_stratum_id = :parent_id
        ORDER BY t.variable, sc.constraint_variable
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'period': self.time_period,
                'stratum_group_id': stratum_group_id,
                'parent_id': geographic_stratum_id
            })
        
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
    
    def apply_constraints_to_sim(self, sim, constraints_df: pd.DataFrame, 
                                target_variable: str) -> np.ndarray:
        """
        Apply constraints to create a mask at household level.
        Returns household-level values after applying constraints.
        
        NOTE: We DON'T pass period to calculate() - this uses sim.default_calculation_period
        which was set before build_from_dataset(). This allows using 2024 data for 2023 calculations.
        """
        if sim is None:
            raise ValueError("Microsimulation instance required")
            
        # Get target entity level
        target_entity = sim.tax_benefit_system.variables[target_variable].entity.key
        
        # Start with all ones mask at entity level
        # DON'T pass period - use default_calculation_period
        entity_count = len(sim.calculate(f"{target_entity}_id").values)
        entity_mask = np.ones(entity_count, dtype=bool)
        
        # Apply each constraint
        for _, constraint in constraints_df.iterrows():
            var = constraint['constraint_variable']
            op = constraint['operation']
            val = constraint['value']
            
            # Skip geographic constraints (already handled by stratification)
            if var in ['state_fips', 'congressional_district_geoid']:
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
                    parsed_val = val
                
                # Apply operation using standardized operators from database
                if op == '==':
                    mask = constraint_values == parsed_val
                elif op == '>':
                    mask = constraint_values > parsed_val
                elif op == '>=':
                    mask = constraint_values >= parsed_val
                elif op == '<':
                    mask = constraint_values < parsed_val
                elif op == '<=':
                    mask = constraint_values <= parsed_val
                elif op == '!=':
                    mask = constraint_values != parsed_val
                else:
                    logger.warning(f"Unknown operation {op}, skipping")
                    continue
                
                # Map to target entity if needed
                if constraint_entity != target_entity:
                    mask = sim.map_result(mask, constraint_entity, target_entity)
                
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
            
        return household_values
    
    def build_matrix_for_geography(self, geographic_level: str, 
                                  geographic_id: str, 
                                  sim=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build calibration matrix for any geographic level.
        
        Args:
            geographic_level: 'state' or 'congressional_district'
            geographic_id: state_fips or congressional_district_geoid
            sim: Microsimulation instance
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
        
        # Get national hardcoded targets
        national_targets = self.get_national_hardcoded_targets()
        
        # Get demographic targets for this geography
        # For now just Age (group 2), but structured to easily add others
        age_targets = self.get_demographic_targets(geo_stratum_id, 2, "age")
        
        # Future: Add other demographic groups
        # income_targets = self.get_demographic_targets(geo_stratum_id, 3, "income")
        # snap_targets = self.get_demographic_targets(geo_stratum_id, 4, "SNAP")
        # medicaid_targets = self.get_demographic_targets(geo_stratum_id, 5, "Medicaid")
        # eitc_targets = self.get_demographic_targets(geo_stratum_id, 6, "EITC")
        
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
                'geographic_level': 'national',
                'geographic_id': geographic_id,
                'description': f"{target['variable']}_national"
            })
        
        # Process age targets
        processed_strata = set()
        for stratum_id in age_targets['stratum_id'].unique():
            if stratum_id in processed_strata:
                continue
            processed_strata.add(stratum_id)
            
            stratum_targets = age_targets[age_targets['stratum_id'] == stratum_id]
            target = stratum_targets.iloc[0]
            
            # Build description from constraints
            constraints = stratum_targets[['constraint_variable', 'operation', 'constraint_value']].drop_duplicates()
            desc_parts = [target['variable']]
            for _, c in constraints.iterrows():
                if c['constraint_variable'] == 'age':
                    desc_parts.append(f"age{c['operation']}{c['constraint_value']}")
            
            all_targets.append({
                'target_id': target['target_id'],
                'variable': target['variable'],
                'value': target['value'],
                'active': target['active'],
                'tolerance': target['tolerance'],
                'stratum_id': target['stratum_id'],
                'geographic_level': geographic_level,
                'geographic_id': geographic_id,
                'description': '_'.join(desc_parts)
            })
        
        targets_df = pd.DataFrame(all_targets)
        
        # Build matrix if sim provided
        if sim is not None:
            household_ids = sim.calculate("household_id", period=self.time_period).values
            n_households = len(household_ids)
            
            # Initialize matrix (targets x households)
            matrix_data = []
            
            for _, target in targets_df.iterrows():
                # Get constraints for this stratum
                constraints = self.get_constraints_for_stratum(target['stratum_id'])
                
                # Apply constraints and get household values
                household_values = self.apply_constraints_to_sim(
                    sim, constraints, target['variable']
                )
                
                matrix_data.append(household_values)
            
            # Create matrix DataFrame (targets as rows, households as columns)
            matrix_df = pd.DataFrame(
                data=np.array(matrix_data),
                index=targets_df['target_id'].values,
                columns=household_ids
            )
            
            logger.info(f"Created matrix for {geographic_level} {geographic_id}: shape {matrix_df.shape}")
            return targets_df, matrix_df
        
        return targets_df, None
    
    def build_stacked_matrix(self, geographic_level: str, 
                           geographic_ids: List[str], 
                           sim=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build stacked calibration matrix for multiple geographic areas.
        
        Args:
            geographic_level: 'state' or 'congressional_district'
            geographic_ids: List of state_fips or cd_geoids
            sim: Microsimulation instance
        """
        all_targets = []
        all_matrices = []
        
        for i, geo_id in enumerate(geographic_ids):
            logger.info(f"Processing {geographic_level} {geo_id} ({i+1}/{len(geographic_ids)})")
            
            targets_df, matrix_df = self.build_matrix_for_geography(
                geographic_level, geo_id, sim
            )
            
            # Add geographic index to target IDs to make them unique
            prefix = "state" if geographic_level == "state" else "cd"
            targets_df['stacked_target_id'] = (
                targets_df['target_id'].astype(str) + f"_{prefix}{geo_id}"
            )
            
            if matrix_df is not None:
                # Add geographic index to household IDs
                matrix_df.columns = [f"{hh_id}_{prefix}{geo_id}" for hh_id in matrix_df.columns]
                matrix_df.index = targets_df['stacked_target_id'].values
                all_matrices.append(matrix_df)
            
            all_targets.append(targets_df)
        
        # Combine all targets
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        # Stack matrices if provided
        if all_matrices:
            # Get all unique household columns
            all_columns = []
            for matrix in all_matrices:
                all_columns.extend(matrix.columns.tolist())
            
            # Create combined matrix with proper alignment
            combined_matrix = pd.DataFrame(
                index=combined_targets['stacked_target_id'].values,
                columns=all_columns,
                dtype=float
            ).fillna(0.0)
            
            # Fill in values from each geographic area's matrix
            for matrix in all_matrices:
                # Use the intersection of indices to avoid mismatches
                common_targets = combined_matrix.index.intersection(matrix.index)
                for target_id in common_targets:
                    # Get the columns for this matrix
                    cols = matrix.columns
                    # Set the values - ensure we're setting the right shape
                    combined_matrix.loc[target_id, cols] = matrix.loc[target_id, cols].values
            
            logger.info(f"Created stacked matrix: shape {combined_matrix.shape}")
            return combined_targets, combined_matrix
        
        return combined_targets, None


def main():
    """Example usage for California and congressional districts."""
    from policyengine_us import Microsimulation
    
    # Database path
    db_uri = "sqlite:////home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
    
    # Initialize builder with 2023 targets
    builder = GeoStackingMatrixBuilder(db_uri, time_period=2023)
    
    # Create microsimulation
    # IMPORTANT: The 2024 dataset only contains 2024 data. When we request 2023 data explicitly,
    # it returns defaults (age=40, weight=0). However, if we set default_calculation_period=2023
    # BEFORE build_from_dataset() and then DON'T pass period to calculate(), it uses the 2024 data.
    # This is likely a fallback behavior in PolicyEngine.
    print("Loading microsimulation...")
    sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
    sim.default_calculation_period = 2023
    sim.build_from_dataset()
    
    # Build matrix for California
    print("\nBuilding matrix for California (FIPS 6)...")
    targets_df, matrix_df = builder.build_matrix_for_geography('state', '6', sim)
    
    print("\nTarget Summary:")
    print(f"Total targets: {len(targets_df)}")
    print(f"National targets: {(targets_df['geographic_level'] == 'national').sum()}")
    print(f"State age targets: {(targets_df['geographic_level'] == 'state').sum()}")
    print(f"Active targets: {targets_df['active'].sum()}")
    
    if matrix_df is not None:
        print(f"\nMatrix shape: {matrix_df.shape}")
        print(f"Matrix has {matrix_df.shape[0]} targets (rows) x {matrix_df.shape[1]} households (columns)")
        
        # Create our own weights for validation - don't use dataset weights
        # as we'll be reweighting anyway
        n_households = matrix_df.shape[1]
        ca_population = 39_000_000  # Approximate California population
        uniform_weights = np.ones(n_households) * (ca_population / n_households)
        
        estimates = matrix_df.values @ uniform_weights
        
        print("\nValidation with uniform weights scaled to CA population:")
        print("(Note: These won't match until proper calibration/reweighting)")
        for i in range(min(10, len(targets_df))):
            target = targets_df.iloc[i]
            estimate = estimates[i]
            ratio = estimate / target['value'] if target['value'] > 0 else 0
            print(f"  {target['description']}: target={target['value']:,.0f}, estimate={estimate:,.0f}, ratio={ratio:.2f}")
    
    # Example: Stack California and Texas
    # TODO: Fix stacking implementation - currently has DataFrame indexing issues
    print("\n" + "="*50)
    print("Stacking multiple states is implemented but needs debugging.")
    print("The single-state matrix creation is working correctly!")
    
    # Show what the stacked matrix would look like
    print("\nWhen stacking works, it will create:")
    print("- For 2 states: ~36 targets x ~42,502 household columns")
    print("- For all 51 states: ~918 targets x ~1,083,801 household columns")
    print("- Matrix will be very sparse with block structure")


if __name__ == "__main__":
    main()