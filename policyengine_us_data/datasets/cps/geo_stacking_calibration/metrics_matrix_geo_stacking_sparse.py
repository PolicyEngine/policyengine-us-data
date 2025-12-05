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

from policyengine_us_data.datasets.cps.geo_stacking_calibration.calibration_utils import (
    get_calculated_variables,
)


logger = logging.getLogger(__name__)


def get_us_state_dependent_variables():
    """
    Return list of variables that should be calculated US-state-specifically.

    These are variables whose values depend on US state policy rules,
    so the same household can have different values in different states.

    NOTE: Only include variables that are CALCULATED based on state policy.
    Variables based on INPUT data (like salt_deduction, which uses
    state_withheld_income_tax as an input) will NOT vary when state changes.

    Returns:
        List of variable names that are US-state-dependent
    """
    return ['snap', 'medicaid', 'salt_deduction']


class SparseGeoStackingMatrixBuilder:
    """Build sparse calibration matrices for geo-stacking approach.

    NOTE: Period handling is complex due to mismatched data years:
    - The enhanced CPS 2024 dataset only contains 2024 data
    - Targets in the database exist for different years (2022, 2023, 2024)
    - For now, we pull targets from whatever year they exist and use 2024 data
    - This temporal mismatch will be addressed in future iterations
    """

    def __init__(self, db_uri: str, time_period: int):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.time_period = time_period
        self._uprating_factors = None
        self._params = None
        self._state_specific_cache = {}  # Cache for state-specific calculated values: {(hh_id, state_fips, var): value}

    @property
    def uprating_factors(self):
        """Lazy-load uprating factors from PolicyEngine parameters."""
        # NOTE: this is pretty limited. What kind of CPI?
        # In [44]: self._uprating_factors
        # Out[44]:
        # {(2022, 'cpi'): 1.0641014696885627,
        #  (2022, 'pop'): 1.009365413037974,
        #  (2023, 'cpi'): 1.0,
        #  (2023, 'pop'): 1.0,
        #  (2024, 'cpi'): 0.9657062435037478,
        #  (2024, 'pop'): 0.989171581243436,
        #  (2025, 'cpi'): 0.937584224942492,
        #  (2025, 'pop'): 0.9892021773614242}

        if self._uprating_factors is None:
            self._uprating_factors = self._calculate_uprating_factors()
        return self._uprating_factors

    def _calculate_uprating_factors(self):
        """Calculate all needed uprating factors from PolicyEngine parameters."""
        from policyengine_us import Microsimulation

        # Get a minimal sim just for parameters
        if self._params is None:
            sim = Microsimulation()
            self._params = sim.tax_benefit_system.parameters

        factors = {}

        # Get unique years from database
        query = """
        SELECT DISTINCT period 
        FROM targets 
        WHERE period IS NOT NULL
        ORDER BY period
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            years_needed = [row[0] for row in result]

        logger.info(
            f"Calculating uprating factors for years {years_needed} to {self.time_period}"
        )

        for from_year in years_needed:
            if from_year == self.time_period:
                factors[(from_year, "cpi")] = 1.0
                factors[(from_year, "pop")] = 1.0
                continue

            # CPI factor
            try:
                cpi_from = self._params.gov.bls.cpi.cpi_u(from_year)
                cpi_to = self._params.gov.bls.cpi.cpi_u(self.time_period)
                factors[(from_year, "cpi")] = float(cpi_to / cpi_from)
            except Exception as e:
                logger.warning(
                    f"Could not calculate CPI factor for {from_year}: {e}"
                )
                factors[(from_year, "cpi")] = 1.0

            # Population factor
            try:
                pop_from = (
                    self._params.calibration.gov.census.populations.total(
                        from_year
                    )
                )
                pop_to = self._params.calibration.gov.census.populations.total(
                    self.time_period
                )
                factors[(from_year, "pop")] = float(pop_to / pop_from)
            except Exception as e:
                logger.warning(
                    f"Could not calculate population factor for {from_year}: {e}"
                )
                factors[(from_year, "pop")] = 1.0

        # Log the factors
        for (year, type_), factor in sorted(factors.items()):
            if factor != 1.0:
                logger.info(
                    f"  {year} -> {self.time_period} ({type_}): {factor:.4f}"
                )

        return factors

    def _get_uprating_info(self, variable: str, period: int):
        """
        Get uprating factor and type for a single variable.
        Returns (factor, uprating_type)
        """
        if period == self.time_period:
            return 1.0, "none"

        # Determine uprating type based on variable name
        count_indicators = [
            "count",
            "person",
            "people",
            "households",
            "tax_units",
        ]
        is_count = any(
            indicator in variable.lower() for indicator in count_indicators
        )
        uprating_type = "pop" if is_count else "cpi"

        # Get factor from pre-calculated dict
        factor = self.uprating_factors.get((period, uprating_type), 1.0)

        return factor, uprating_type

    def _calculate_state_specific_values(self, dataset_path: str, variables_to_calculate: List[str]):
        """
        Pre-calculate state-specific values for variables that depend on state policy.

        Creates a FRESH simulation for each state to avoid PolicyEngine caching issues.
        This ensures calculated variables like salt_deduction are properly recomputed
        with the new state's policy rules.

        Args:
            dataset_path: Path to the dataset file (e.g., stratified_10k.h5)
            variables_to_calculate: List of variable names to calculate state-specifically

        Returns:
            None (populates self._state_specific_cache)
        """
        import gc
        from policyengine_us import Microsimulation

        # State FIPS codes (skipping gaps in numbering)
        valid_states = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22,
                       23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                       40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56]

        # Get household IDs from a temporary sim (they're constant across states)
        #temp_sim = Microsimulation(dataset=dataset_path)
        sim = Microsimulation(dataset=dataset_path)
        household_ids = sim.calculate("household_id", map_to="household").values
        n_households = len(household_ids)

        logger.info(f"Calculating state-specific values for {len(variables_to_calculate)} variables "
                   f"across {n_households} households and {len(valid_states)} states...")
        logger.info(f"This will create {n_households * len(valid_states) * len(variables_to_calculate):,} cached values")

        total_states = len(valid_states)

        # For each state, create a FRESH simulation to avoid caching issues
        for state_idx, state_fips in enumerate(valid_states):
            # Create brand new simulation for this state
            #sim = Microsimulation(dataset=dataset_path)

            # Set ALL households to this state
            sim.set_input("state_fips", self.time_period,
                         np.full(n_households, state_fips, dtype=np.int32))
            # Clear cached calculated variables so state changes propagate
            for var in get_calculated_variables(sim):
                sim.delete_arrays(var)

            # Calculate each variable for all households in this state
            for var_name in variables_to_calculate:
                values = sim.calculate(var_name, map_to="household").values

                # Cache all values for this state
                for hh_idx, hh_id in enumerate(household_ids):
                    cache_key = (int(hh_id), int(state_fips), var_name)
                    self._state_specific_cache[cache_key] = float(values[hh_idx])

            # Log progress
            if (state_idx + 1) % 10 == 0 or state_idx == total_states - 1:
                logger.info(f"  Progress: {state_idx + 1}/{total_states} states complete")


        logger.info(f"State-specific cache populated with {len(self._state_specific_cache):,} values")

    def get_best_period_for_targets(
        self, query_base: str, params: dict
    ) -> int:
        """
        Find the best period for targets: closest year <= target_year,
        or closest future year if no past years exist.

        Args:
            query_base: SQL query that should return period column
            params: Parameters for the query

        Returns:
            Best period to use, or None if no targets found
        """
        # Get all available periods for these targets
        period_query = f"""
        WITH target_periods AS (
            {query_base}
        )
        SELECT DISTINCT period 
        FROM target_periods 
        WHERE period IS NOT NULL
        ORDER BY period
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(period_query), params)
            available_periods = [row[0] for row in result.fetchall()]

        if not available_periods:
            return None

        # Find best period: closest <= target_year, or closest > target_year
        past_periods = [p for p in available_periods if p <= self.time_period]
        if past_periods:
            # Return the most recent past period (closest to target)
            return max(past_periods)
        else:
            # No past periods, return closest future period
            return min(available_periods)

    def get_all_descendant_targets(
        self, stratum_id: int, sim=None
    ) -> pd.DataFrame:
        """
        Recursively get all targets from a stratum and all its descendants.
        This handles the new filer stratum layer transparently.
        Selects the best period for each target (closest to target_year in the past, or closest future).
        """
        query = """
        WITH RECURSIVE descendant_strata AS (
            -- Base case: the stratum itself
            SELECT stratum_id
            FROM strata
            WHERE stratum_id = :stratum_id
            
            UNION ALL
            
            -- Recursive case: all children
            SELECT s.stratum_id
            FROM strata s
            JOIN descendant_strata d ON s.parent_stratum_id = d.stratum_id
        ),
        -- Find best period for each stratum/variable combination
        best_periods AS (
            SELECT 
                t.stratum_id,
                t.variable,
                CASE
                    -- If there are periods <= target_year, use the maximum (most recent)
                    WHEN MAX(CASE WHEN t.period <= :target_year THEN t.period END) IS NOT NULL
                    THEN MAX(CASE WHEN t.period <= :target_year THEN t.period END)
                    -- Otherwise use the minimum period (closest future)
                    ELSE MIN(t.period)
                END as best_period
            FROM targets t
            WHERE t.stratum_id IN (SELECT stratum_id FROM descendant_strata)
            GROUP BY t.stratum_id, t.variable
        )
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.period,
            t.active,
            t.tolerance,
            s.notes as stratum_notes,
            s.stratum_group_id,
            s.parent_stratum_id,
            src.name as source_name,
            -- Aggregate constraint info to avoid duplicate rows
            (SELECT GROUP_CONCAT(sc2.constraint_variable || sc2.operation || sc2.value, '|')
             FROM stratum_constraints sc2 
             WHERE sc2.stratum_id = s.stratum_id
             GROUP BY sc2.stratum_id) as constraint_info
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        JOIN sources src ON t.source_id = src.source_id
        JOIN best_periods bp ON t.stratum_id = bp.stratum_id 
            AND t.variable = bp.variable 
            AND t.period = bp.best_period
        WHERE s.stratum_id IN (SELECT stratum_id FROM descendant_strata)
        ORDER BY s.stratum_id, t.variable
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    "stratum_id": stratum_id,
                    "target_year": self.time_period,
                },
            )

        if len(df) > 0:
            # Log which periods were selected
            periods_used = df["period"].unique()
            logger.debug(
                f"Selected targets from periods: {sorted(periods_used)}"
            )

        return df

    def get_hierarchical_targets(
        self,
        cd_stratum_id: int,
        state_stratum_id: int,
        national_stratum_id: int,
        sim=None,
    ) -> pd.DataFrame:
        """
        Get targets using hierarchical fallback: CD -> State -> National.
        For each target concept, use the most geographically specific available.
        """
        # Get all targets at each level (including descendants)
        cd_targets = self.get_all_descendant_targets(cd_stratum_id, sim)
        state_targets = self.get_all_descendant_targets(state_stratum_id, sim)
        national_targets = self.get_all_descendant_targets(
            national_stratum_id, sim
        )

        # Add geographic level to each
        cd_targets["geo_level"] = "congressional_district"
        cd_targets["geo_priority"] = 1  # Highest priority
        state_targets["geo_level"] = "state"
        state_targets["geo_priority"] = 2
        national_targets["geo_level"] = "national"
        national_targets["geo_priority"] = 3  # Lowest priority

        # Combine all targets
        all_targets = pd.concat(
            [cd_targets, state_targets, national_targets], ignore_index=True
        )

        # Create concept identifier from variable + all constraints
        def get_concept_id(row):
            if not row["variable"]:
                return None

            variable = row["variable"]

            # Parse constraint_info if present
            if pd.notna(row.get("constraint_info")):
                constraints = row["constraint_info"].split("|")

                # Filter out geographic and filer constraints
                demographic_constraints = []
                irs_constraint = None

                for c in constraints:
                    if not any(
                        skip in c
                        for skip in [
                            "state_fips",
                            "congressional_district_geoid",
                            "tax_unit_is_filer",
                        ]
                    ):
                        # Check if this is an IRS variable constraint
                        if not any(
                            demo in c
                            for demo in [
                                "age",
                                "adjusted_gross_income",
                                "eitc_child_count",
                                "snap",
                                "medicaid",
                            ]
                        ):
                            # This is likely an IRS variable constraint like "salt>0"
                            irs_constraint = c
                        else:
                            demographic_constraints.append(c)

                # If we have an IRS constraint, use that as the concept
                if irs_constraint:
                    # Extract just the variable name from something like "salt>0"
                    import re

                    match = re.match(r"([a-zA-Z_]+)", irs_constraint)
                    if match:
                        return f"{match.group(1)}_constrained"

                # Otherwise build concept from variable + demographic constraints
                if demographic_constraints:
                    # Sort for consistency
                    demographic_constraints.sort()
                    # Normalize operators for valid identifiers
                    normalized = []
                    for c in demographic_constraints:
                        c_norm = c.replace(">=", "_gte_").replace(
                            "<=", "_lte_"
                        )
                        c_norm = c_norm.replace(">", "_gt_").replace(
                            "<", "_lt_"
                        )
                        c_norm = c_norm.replace("==", "_eq_").replace(
                            "=", "_eq_"
                        )
                        normalized.append(c_norm)
                    return f"{variable}_{'_'.join(normalized)}"

            # No constraints, just the variable
            return variable

        all_targets["concept_id"] = all_targets.apply(get_concept_id, axis=1)

        # Remove targets without a valid concept
        all_targets = all_targets[all_targets["concept_id"].notna()]

        # For each concept, keep only the most geographically specific target
        # Sort by concept and priority, then keep first of each concept
        all_targets = all_targets.sort_values(["concept_id", "geo_priority"])
        selected_targets = (
            all_targets.groupby("concept_id").first().reset_index()
        )

        logger.info(
            f"Hierarchical fallback selected {len(selected_targets)} targets from "
            f"{len(all_targets)} total across all levels"
        )

        return selected_targets

    def get_national_targets(self, sim=None) -> pd.DataFrame:
        """
        Get national-level targets from the database.
        Includes both direct national targets and national targets with strata/constraints.
        Selects the best period for each target (closest to target_year in the past, or closest future).
        """
        query = """
        WITH national_stratum AS (
            -- Get the national (US) stratum ID
            SELECT stratum_id 
            FROM strata 
            WHERE parent_stratum_id IS NULL
            LIMIT 1
        ),
        national_targets AS (
            -- Get all national targets
            SELECT 
                t.target_id,
                t.stratum_id,
                t.variable,
                t.value,
                t.period,
                t.active,
                t.tolerance,
                s.notes as stratum_notes,
                (SELECT GROUP_CONCAT(sc2.constraint_variable || sc2.operation || sc2.value, '|')
                 FROM stratum_constraints sc2 
                 WHERE sc2.stratum_id = s.stratum_id
                 GROUP BY sc2.stratum_id) as constraint_info,
                src.name as source_name
            FROM targets t
            JOIN strata s ON t.stratum_id = s.stratum_id
            JOIN sources src ON t.source_id = src.source_id
            WHERE (
                -- Direct national targets (no parent)
                s.parent_stratum_id IS NULL
                OR 
                -- National targets with strata (parent is national stratum)
                s.parent_stratum_id = (SELECT stratum_id FROM national_stratum)
            )
            AND UPPER(src.type) = 'HARDCODED'  -- Hardcoded targets only
        ),
        -- Find best period for each stratum/variable combination
        best_periods AS (
            SELECT 
                stratum_id,
                variable,
                CASE
                    -- If there are periods <= target_year, use the maximum (most recent)
                    WHEN MAX(CASE WHEN period <= :target_year THEN period END) IS NOT NULL
                    THEN MAX(CASE WHEN period <= :target_year THEN period END)
                    -- Otherwise use the minimum period (closest future)
                    ELSE MIN(period)
                END as best_period
            FROM national_targets
            GROUP BY stratum_id, variable
        )
        SELECT nt.*
        FROM national_targets nt
        JOIN best_periods bp ON nt.stratum_id = bp.stratum_id 
            AND nt.variable = bp.variable 
            AND nt.period = bp.best_period
        ORDER BY nt.variable, nt.constraint_info
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query, conn, params={"target_year": self.time_period}
            )

        if len(df) > 0:
            periods_used = df["period"].unique()
            logger.info(
                f"Found {len(df)} national targets from periods: {sorted(periods_used)}"
            )
        else:
            logger.info("No national targets found")

        return df

    def get_irs_scalar_targets(
        self, geographic_stratum_id: int, geographic_level: str, sim=None
    ) -> pd.DataFrame:
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
            t.period,
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
            df = pd.read_sql(
                query, conn, params={"stratum_id": geographic_stratum_id}
            )

            # Note: Uprating removed - should be done once after matrix assembly
            logger.info(
                f"Found {len(df)} IRS scalar targets for {geographic_level}"
            )
        return df

    def get_agi_total_target(
        self, geographic_stratum_id: int, geographic_level: str, sim=None
    ) -> pd.DataFrame:
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
            t.period,
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
            df = pd.read_sql(
                query, conn, params={"stratum_id": geographic_stratum_id}
            )

            # Note: Uprating removed - should be done once after matrix assembly
            logger.info(f"Found AGI total target for {geographic_level}")
        return df

    def get_demographic_targets(
        self,
        geographic_stratum_id: int,
        stratum_group_id: int,
        group_name: str,
        sim=None,
    ) -> pd.DataFrame:
        """
        Generic function to get demographic targets for a geographic area.
        Selects the best period for each target (closest to target_year in the past, or closest future).

        Args:
            geographic_stratum_id: The parent geographic stratum
            stratum_group_id: The demographic group (2=Age, 3=Income, 4=SNAP, 5=Medicaid, 6=EITC)
            group_name: Descriptive name for logging
        """
        query = """
        WITH demographic_targets AS (
            -- Get all targets for this demographic group
            SELECT 
                t.target_id,
                t.stratum_id,
                t.variable,
                t.value,
                t.active,
                t.tolerance,
                s.notes as stratum_notes,
                s.stratum_group_id,
                (SELECT GROUP_CONCAT(sc2.constraint_variable || sc2.operation || sc2.value, '|')
                 FROM stratum_constraints sc2 
                 WHERE sc2.stratum_id = s.stratum_id
                 GROUP BY sc2.stratum_id) as constraint_info,
                t.period
            FROM targets t
            JOIN strata s ON t.stratum_id = s.stratum_id
            WHERE s.stratum_group_id = :stratum_group_id
              AND s.parent_stratum_id = :parent_id
        ),
        -- Find best period for each stratum/variable combination
        best_periods AS (
            SELECT 
                stratum_id,
                variable,
                CASE
                    -- If there are periods <= target_year, use the maximum (most recent)
                    WHEN MAX(CASE WHEN period <= :target_year THEN period END) IS NOT NULL
                    THEN MAX(CASE WHEN period <= :target_year THEN period END)
                    -- Otherwise use the minimum period (closest future)
                    ELSE MIN(period)
                END as best_period
            FROM demographic_targets
            GROUP BY stratum_id, variable
        )
        SELECT dt.*
        FROM demographic_targets dt
        JOIN best_periods bp ON dt.stratum_id = bp.stratum_id 
            AND dt.variable = bp.variable 
            AND dt.period = bp.best_period
        ORDER BY dt.variable, dt.constraint_info
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={
                    "target_year": self.time_period,
                    "stratum_group_id": stratum_group_id,
                    "parent_id": geographic_stratum_id,
                },
            )

            if len(df) > 0:
                periods_used = df["period"].unique()
                logger.debug(
                    f"Found {len(df)} {group_name} targets for stratum {geographic_stratum_id} from periods: {sorted(periods_used)}"
                )
            else:
                logger.info(
                    f"No {group_name} targets found for stratum {geographic_stratum_id}"
                )

        return df

    def get_national_stratum_id(self) -> Optional[int]:
        """Get stratum ID for national level."""
        query = """
        SELECT stratum_id 
        FROM strata 
        WHERE parent_stratum_id IS NULL
          AND stratum_group_id = 1  -- Geographic stratum
        LIMIT 1
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            return result[0] if result else None

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
            result = conn.execute(
                text(query), {"state_fips": state_fips}
            ).fetchone()
            return result[0] if result else None

    def get_state_fips_from_cd(self, cd_geoid: str) -> str:
        """Extract state FIPS code from congressional district GEOID."""
        # CD GEOIDs are formatted as state_fips (1-2 digits) + district (2 digits)
        # Examples: '601' -> '6', '3601' -> '36'
        if len(cd_geoid) == 3:
            return cd_geoid[0]  # Single digit state
        elif len(cd_geoid) == 4:
            return cd_geoid[:2]  # Two digit state
        else:
            raise ValueError(f"Invalid CD GEOID format: {cd_geoid}")

    def reconcile_targets_to_higher_level(
        self,
        lower_targets_dict: Dict[str, pd.DataFrame],
        higher_level: str,
        target_filters: Dict[str, any],
        sim=None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Reconcile lower-level targets to match higher-level aggregates.
        Generic method that can handle CD->State or State->National reconciliation.

        Args:
            lower_targets_dict: Dict mapping geography_id to its targets DataFrame
            higher_level: 'state' or 'national'
            target_filters: Dict with filters like {'stratum_group_id': 2} for age
            sim: Microsimulation instance (if needed)

        Returns:
            Dict with same structure but adjusted targets including diagnostic columns
        """
        reconciled_dict = {}

        # Group lower-level geographies by their parent
        if higher_level == "state":
            # Group CDs by state
            grouped = {}
            for cd_id, targets_df in lower_targets_dict.items():
                state_fips = self.get_state_fips_from_cd(cd_id)
                if state_fips not in grouped:
                    grouped[state_fips] = {}
                grouped[state_fips][cd_id] = targets_df
        else:  # national
            # All states belong to one national group
            grouped = {"US": lower_targets_dict}

        # Process each group
        for parent_id, children_dict in grouped.items():
            # Get parent-level targets
            if higher_level == "state":
                parent_stratum_id = self.get_state_stratum_id(parent_id)
            else:  # national
                parent_stratum_id = self.get_national_stratum_id()

            if parent_stratum_id is None:
                logger.warning(
                    f"Could not find {higher_level} stratum for {parent_id}"
                )
                # Return unchanged
                for child_id, child_df in children_dict.items():
                    reconciled_dict[child_id] = child_df.copy()
                continue

            # Get parent targets matching the filter
            parent_targets = self._get_filtered_targets(
                parent_stratum_id, target_filters
            )

            if parent_targets.empty:
                # No parent targets to reconcile to
                for child_id, child_df in children_dict.items():
                    reconciled_dict[child_id] = child_df.copy()
                continue

            # First, calculate adjustment factors for all targets
            adjustment_factors = {}
            for _, parent_target in parent_targets.iterrows():
                # Sum all children for this concept
                total_child_sum = 0.0
                for child_id, child_df in children_dict.items():
                    child_mask = self._get_matching_targets_mask(
                        child_df, parent_target, target_filters
                    )
                    if child_mask.any():
                        # Use ORIGINAL values, not modified ones
                        if (
                            "original_value_pre_reconciliation"
                            in child_df.columns
                        ):
                            total_child_sum += child_df.loc[
                                child_mask, "original_value_pre_reconciliation"
                            ].sum()
                        else:
                            total_child_sum += child_df.loc[
                                child_mask, "value"
                            ].sum()

                if total_child_sum > 0:
                    parent_value = parent_target["value"]
                    factor = parent_value / total_child_sum
                    adjustment_factors[parent_target["variable"]] = factor
                    logger.info(
                        f"Calculated factor for {parent_target['variable']}: {factor:.4f} "
                        f"(parent={parent_value:,.0f}, children_sum={total_child_sum:,.0f})"
                    )

            # Now apply the factors to each child
            for child_id, child_df in children_dict.items():
                reconciled_df = self._apply_reconciliation_factors(
                    child_df,
                    parent_targets,
                    adjustment_factors,
                    child_id,
                    higher_level,
                    target_filters,
                )
                reconciled_dict[child_id] = reconciled_df

        return reconciled_dict

    def _apply_reconciliation_factors(
        self,
        child_df: pd.DataFrame,
        parent_targets: pd.DataFrame,
        adjustment_factors: Dict[str, float],
        child_id: str,
        parent_level: str,
        target_filters: Dict,
    ) -> pd.DataFrame:
        """Apply pre-calculated reconciliation factors to a child geography."""
        result_df = child_df.copy()

        # Add diagnostic columns if not present
        if "original_value_pre_reconciliation" not in result_df.columns:
            result_df["original_value_pre_reconciliation"] = result_df[
                "value"
            ].copy()
        if "reconciliation_factor" not in result_df.columns:
            result_df["reconciliation_factor"] = 1.0
        if "reconciliation_source" not in result_df.columns:
            result_df["reconciliation_source"] = "none"
        if "undercount_pct" not in result_df.columns:
            result_df["undercount_pct"] = 0.0

        # Apply factors for matching targets
        for _, parent_target in parent_targets.iterrows():
            var_name = parent_target["variable"]
            if var_name in adjustment_factors:
                matching_mask = self._get_matching_targets_mask(
                    result_df, parent_target, target_filters
                )
                if matching_mask.any():
                    factor = adjustment_factors[var_name]
                    # Apply to ORIGINAL value, not current value
                    original_vals = result_df.loc[
                        matching_mask, "original_value_pre_reconciliation"
                    ]
                    result_df.loc[matching_mask, "value"] = (
                        original_vals * factor
                    )
                    result_df.loc[matching_mask, "reconciliation_factor"] = (
                        factor
                    )
                    result_df.loc[matching_mask, "reconciliation_source"] = (
                        f"{parent_level}_{var_name}"
                    )
                    result_df.loc[matching_mask, "undercount_pct"] = (
                        (1 - 1 / factor) * 100 if factor != 0 else 0
                    )

        return result_df

    def _get_filtered_targets(
        self, stratum_id: int, filters: Dict
    ) -> pd.DataFrame:
        """Get targets from database matching filters."""
        # Build query conditions
        conditions = [
            "s.stratum_id = :stratum_id OR s.parent_stratum_id = :stratum_id"
        ]

        for key, value in filters.items():
            if key == "stratum_group_id":
                conditions.append(f"s.stratum_group_id = {value}")
            elif key == "variable":
                conditions.append(f"t.variable = '{value}'")

        query = f"""
        SELECT 
            t.target_id,
            t.stratum_id,
            t.variable,
            t.value,
            t.period,
            s.stratum_group_id,
            (SELECT GROUP_CONCAT(sc2.constraint_variable || sc2.operation || sc2.value, '|')
             FROM stratum_constraints sc2 
             WHERE sc2.stratum_id = s.stratum_id
             GROUP BY sc2.stratum_id) as constraint_info
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        WHERE {' AND '.join(conditions)}
        """

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"stratum_id": stratum_id})

    def _reconcile_single_geography(
        self,
        child_df: pd.DataFrame,
        parent_targets: pd.DataFrame,
        child_id: str,
        parent_id: str,
        parent_level: str,
        filters: Dict,
        all_children_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Reconcile a single geography's targets to parent aggregates."""
        result_df = child_df.copy()

        # Add diagnostic columns if not present
        if "original_value_pre_reconciliation" not in result_df.columns:
            result_df["original_value_pre_reconciliation"] = result_df[
                "value"
            ].copy()
        if "reconciliation_factor" not in result_df.columns:
            result_df["reconciliation_factor"] = 1.0
        if "reconciliation_source" not in result_df.columns:
            result_df["reconciliation_source"] = "none"
        if "undercount_pct" not in result_df.columns:
            result_df["undercount_pct"] = 0.0

        # Match targets by concept (variable + constraints)
        for _, parent_target in parent_targets.iterrows():
            # Find matching child targets
            matching_mask = self._get_matching_targets_mask(
                result_df, parent_target, filters
            )

            if not matching_mask.any():
                continue

            # Aggregate all siblings for this concept using already-collected data
            sibling_sum = 0.0
            for sibling_id, sibling_df in all_children_dict.items():
                sibling_mask = self._get_matching_targets_mask(
                    sibling_df, parent_target, filters
                )
                if sibling_mask.any():
                    sibling_sum += sibling_df.loc[sibling_mask, "value"].sum()

            if sibling_sum == 0:
                logger.warning(
                    f"Zero sum for {parent_target['variable']} in {parent_level}"
                )
                continue

            # Calculate adjustment factor
            parent_value = parent_target["value"]
            adjustment_factor = parent_value / sibling_sum

            # Apply adjustment
            result_df.loc[matching_mask, "value"] *= adjustment_factor
            result_df.loc[matching_mask, "reconciliation_factor"] = (
                adjustment_factor
            )
            result_df.loc[matching_mask, "reconciliation_source"] = (
                f"{parent_level}_{parent_target['variable']}"
            )
            result_df.loc[matching_mask, "undercount_pct"] = (
                1 - 1 / adjustment_factor
            ) * 100

            logger.info(
                f"Reconciled {parent_target['variable']} for {child_id}: "
                f"factor={adjustment_factor:.4f}, undercount={((1-1/adjustment_factor)*100):.1f}%"
            )

        return result_df

    def _get_matching_targets_mask(
        self, df: pd.DataFrame, parent_target: pd.Series, filters: Dict
    ) -> pd.Series:
        """Get mask for targets matching parent target concept."""
        mask = df["variable"] == parent_target["variable"]

        # Match stratum_group_id if in filters
        if "stratum_group_id" in filters and "stratum_group_id" in df.columns:
            mask &= df["stratum_group_id"] == filters["stratum_group_id"]

        # Match constraints based on constraint_info, ignoring geographic constraints
        parent_constraint_info = parent_target.get("constraint_info")
        if "constraint_info" in df.columns:
            # Extract demographic constraints from parent (exclude geographic)
            parent_demo_constraints = set()
            if pd.notna(parent_constraint_info):
                for c in str(parent_constraint_info).split("|"):
                    if not any(
                        geo in c
                        for geo in [
                            "state_fips",
                            "congressional_district_geoid",
                        ]
                    ):
                        parent_demo_constraints.add(c)

            # Create vectorized comparison for efficiency
            def extract_demo_constraints(constraint_str):
                """Extract non-geographic constraints from constraint string."""
                if pd.isna(constraint_str):
                    return frozenset()
                demo_constraints = []
                for c in str(constraint_str).split("|"):
                    if not any(
                        geo in c
                        for geo in [
                            "state_fips",
                            "congressional_district_geoid",
                        ]
                    ):
                        demo_constraints.append(c)
                return frozenset(demo_constraints)

            # Apply extraction and compare
            child_demo_constraints = df["constraint_info"].apply(
                extract_demo_constraints
            )
            parent_demo_set = frozenset(parent_demo_constraints)
            mask &= child_demo_constraints == parent_demo_set

        return mask

    def _aggregate_cd_targets_for_state(
        self, state_fips: str, target_concept: pd.Series, filters: Dict
    ) -> float:
        """Sum CD targets for a state matching the concept."""
        # Get all CDs in state
        query = """
        SELECT DISTINCT sc.value as cd_geoid
        FROM stratum_constraints sc
        JOIN strata s ON sc.stratum_id = s.stratum_id
        WHERE sc.constraint_variable = 'congressional_district_geoid'
          AND sc.value LIKE :state_pattern
        """

        # Determine pattern based on state_fips length
        if len(state_fips) == 1:
            pattern = f"{state_fips}__"  # e.g., "6__" for CA
        else:
            pattern = f"{state_fips}__"  # e.g., "36__" for NY

        with self.engine.connect() as conn:
            cd_result = conn.execute(text(query), {"state_pattern": pattern})
            cd_ids = [row[0] for row in cd_result]

        # Sum targets across CDs
        total = 0.0
        for cd_id in cd_ids:
            cd_stratum_id = self.get_cd_stratum_id(cd_id)
            if cd_stratum_id:
                cd_targets = self._get_filtered_targets(cd_stratum_id, filters)
                # Sum matching targets
                for _, cd_target in cd_targets.iterrows():
                    if self._targets_match_concept(cd_target, target_concept):
                        total += cd_target["value"]

        return total

    def _targets_match_concept(
        self, target1: pd.Series, target2: pd.Series
    ) -> bool:
        """Check if two targets represent the same concept."""
        # Must have same variable
        if target1["variable"] != target2["variable"]:
            return False

        # Must have same constraint pattern based on constraint_info
        constraint1 = target1.get("constraint_info")
        constraint2 = target2.get("constraint_info")

        # Both must be either null or non-null
        if pd.isna(constraint1) != pd.isna(constraint2):
            return False

        # If both have constraints, they must match exactly
        if pd.notna(constraint1):
            return constraint1 == constraint2

        return True

    def _aggregate_state_targets_for_national(
        self, target_concept: pd.Series, filters: Dict
    ) -> float:
        """Sum state targets for national matching the concept."""
        # Get all states
        query = """
        SELECT DISTINCT sc.value as state_fips
        FROM stratum_constraints sc
        JOIN strata s ON sc.stratum_id = s.stratum_id
        WHERE sc.constraint_variable = 'state_fips'
        """

        with self.engine.connect() as conn:
            state_result = conn.execute(text(query))
            state_fips_list = [row[0] for row in state_result]

        # Sum targets across states
        total = 0.0
        for state_fips in state_fips_list:
            state_stratum_id = self.get_state_stratum_id(state_fips)
            if state_stratum_id:
                state_targets = self._get_filtered_targets(
                    state_stratum_id, filters
                )
                # Sum matching targets
                for _, state_target in state_targets.iterrows():
                    if self._targets_match_concept(
                        state_target, target_concept
                    ):
                        total += state_target["value"]

        return total

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
            result = conn.execute(
                text(query), {"cd_geoid": cd_geoid}
            ).fetchone()
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
            return pd.read_sql(query, conn, params={"stratum_id": stratum_id})

    def apply_constraints_to_sim_sparse(
        self, sim, constraints_df: pd.DataFrame, target_variable: str,
        target_state_fips: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        # TODO: is it really a good idea to skip geographic filtering?
        # I'm seeing all of the US here for SNAP and I'm only in one congressional district
        # We're putting a lot of faith on later functions to filter them out
        """
        Apply constraints and return sparse representation (indices and values).

        *** Wow this is where the values are actually set at the household level. So
        this function is really misnamed because its a crucial part of getting
        the value at the household level! ***

        Note: Geographic constraints are ALWAYS skipped as geographic isolation
        happens through matrix column structure in geo-stacking, not data filtering.

        Args:
            sim: Microsimulation instance
            constraints_df: DataFrame with constraints
            target_variable: Variable to calculate
            target_state_fips: If provided and variable is state-dependent, use cached state-specific values

        Returns:
            Tuple of (nonzero_indices, nonzero_values) at household level
        """

        # Check if we should use US-state-specific cached values
        us_state_dependent_vars = get_us_state_dependent_variables()
        use_cache = (target_state_fips is not None and
                    target_variable in us_state_dependent_vars and
                    len(self._state_specific_cache) > 0)

        if use_cache:
            # Use cached state-specific values instead of calculating
            logger.debug(f"Using cached {target_variable} values for state {target_state_fips}")
            household_ids = sim.calculate("household_id", map_to="household").values

            # Get values from cache for this state
            household_values = []
            for hh_id in household_ids:
                cache_key = (int(hh_id), int(target_state_fips), target_variable)
                value = self._state_specific_cache.get(cache_key, 0.0)
                household_values.append(value)

            household_values = np.array(household_values)

            # Apply non-geographic constraints to determine which households qualify
            # (We still need to filter based on constraints like "snap > 0")
            # Build entity relationship to check constraints
            entity_rel = pd.DataFrame({
                "person_id": sim.calculate("person_id", map_to="person").values,
                "household_id": sim.calculate("household_id", map_to="person").values,
            })

            # Start with all persons
            person_constraint_mask = np.ones(len(entity_rel), dtype=bool)

            # Apply each non-geographic constraint
            for _, constraint in constraints_df.iterrows():
                var = constraint["constraint_variable"]
                op = constraint["operation"]
                val = constraint["value"]

                if var in ["state_fips", "congressional_district_geoid"]:
                    continue

                # Special handling for the target variable itself
                if var == target_variable:
                    # Map household values to person level for constraint checking
                    hh_value_map = dict(zip(household_ids, household_values))
                    person_hh_ids = entity_rel["household_id"].values
                    person_target_values = np.array([hh_value_map.get(hh_id, 0.0) for hh_id in person_hh_ids])

                    # Parse constraint value
                    try:
                        parsed_val = float(val)
                        if parsed_val.is_integer():
                            parsed_val = int(parsed_val)
                    except ValueError:
                        parsed_val = val

                    # Apply operation
                    if op == "==" or op == "=":
                        mask = (person_target_values == parsed_val).astype(bool)
                    elif op == ">":
                        mask = (person_target_values > parsed_val).astype(bool)
                    elif op == ">=":
                        mask = (person_target_values >= parsed_val).astype(bool)
                    elif op == "<":
                        mask = (person_target_values < parsed_val).astype(bool)
                    elif op == "<=":
                        mask = (person_target_values <= parsed_val).astype(bool)
                    elif op == "!=":
                        mask = (person_target_values != parsed_val).astype(bool)
                    else:
                        continue

                    person_constraint_mask = person_constraint_mask & mask

            # Aggregate to household level
            entity_rel["satisfies_constraints"] = person_constraint_mask
            household_mask = entity_rel.groupby("household_id")["satisfies_constraints"].any()

            # Apply mask to values
            masked_values = household_values * household_mask.values

            # Return sparse representation
            nonzero_indices = np.nonzero(masked_values)[0]
            nonzero_values = masked_values[nonzero_indices]

            return nonzero_indices, nonzero_values

        ## Get target entity level
        target_entity = sim.tax_benefit_system.variables[
            target_variable
        ].entity.key

        # Build entity relationship DataFrame at person level
        # This gives us the mapping between all entities
        entity_rel = pd.DataFrame(
            {
                "person_id": sim.calculate(
                    "person_id", map_to="person"
                ).values,
                "household_id": sim.calculate(
                    "household_id", map_to="person"
                ).values,
                "tax_unit_id": sim.calculate(
                    "tax_unit_id", map_to="person"
                ).values,
                "spm_unit_id": sim.calculate(
                    "spm_unit_id", map_to="person"
                ).values,
                "family_id": sim.calculate(
                    "family_id", map_to="person"
                ).values,
                "marital_unit_id": sim.calculate(
                    "marital_unit_id", map_to="person"
                ).values,
            }
        )

        # Start with all persons satisfying constraints (will be ANDed together)
        person_constraint_mask = np.ones(len(entity_rel), dtype=bool)

        # Apply each constraint at person level
        for _, constraint in constraints_df.iterrows():
            var = constraint["constraint_variable"]
            op = constraint["operation"]
            val = constraint["value"]

            # ALWAYS skip geographic constraints - geo-stacking handles geography through matrix structure
            if var in ["state_fips", "congressional_district_geoid"]:
                continue

            try:
                # Get constraint values at person level
                # We need to explicitly map to person for non-person variables
                constraint_entity = sim.tax_benefit_system.variables[
                    var
                ].entity.key
                if constraint_entity == "person":
                    constraint_values = sim.calculate(var).values
                else:
                    # For tax_unit or household variables, map to person level
                    # This broadcasts the values so each person gets their tax_unit/household's value
                    constraint_values = sim.calculate(
                        var, map_to="person"
                    ).values

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

                # Apply operation at person level
                if op == "==" or op == "=":
                    mask = (constraint_values == parsed_val).astype(bool)
                elif op == ">":
                    mask = (constraint_values > parsed_val).astype(bool)
                elif op == ">=":
                    mask = (constraint_values >= parsed_val).astype(bool)
                elif op == "<":
                    mask = (constraint_values < parsed_val).astype(bool)
                elif op == "<=":
                    mask = (constraint_values <= parsed_val).astype(bool)
                elif op == "!=":
                    mask = (constraint_values != parsed_val).astype(bool)
                else:
                    logger.warning(f"Unknown operation {op}")
                    continue

                # AND this constraint with existing constraints
                person_constraint_mask = person_constraint_mask & mask

            except Exception as e:
                logger.warning(
                    f"Could not apply constraint {var} {op} {val}: {e}"
                )
                continue

        # Add constraint mask to entity_rel
        entity_rel["satisfies_constraints"] = person_constraint_mask

        # Now aggregate constraints to target entity level
        if target_entity == "person":
            entity_mask = person_constraint_mask
            entity_ids = entity_rel["person_id"].values
        elif target_entity == "household":
            household_mask = entity_rel.groupby("household_id")[
                "satisfies_constraints"
            ].any()
            entity_mask = household_mask.values
            entity_ids = household_mask.index.values
        elif target_entity == "tax_unit":
            tax_unit_mask = entity_rel.groupby("tax_unit_id")[
                "satisfies_constraints"
            ].any()
            entity_mask = tax_unit_mask.values
            entity_ids = tax_unit_mask.index.values
        elif target_entity == "spm_unit":
            spm_unit_mask = entity_rel.groupby("spm_unit_id")[
                "satisfies_constraints"
            ].any()
            entity_mask = spm_unit_mask.values
            entity_ids = spm_unit_mask.index.values
        else:
            raise ValueError(f"Entity type {target_entity} not handled")

        target_values_raw = sim.calculate(
            target_variable, map_to=target_entity
        ).values

        masked_values = target_values_raw * entity_mask

        entity_df = pd.DataFrame(
            {
                f"{target_entity}_id": entity_ids,
                "entity_masked_metric": masked_values,
            }
        )
        if target_entity == "household":
            hh_df = entity_df
        else:
            entity_rel_for_agg = entity_rel[["household_id", f"{target_entity}_id"]].drop_duplicates()
            hh_df = entity_rel_for_agg.merge(entity_df, on=f"{target_entity}_id")

        # Check if this is a count variable
        is_count_target = target_variable.endswith("_count")

        if is_count_target:
            # For counts, count unique entities per household that satisfy constraints
            masked_df = hh_df.loc[hh_df["entity_masked_metric"] > 0]
            household_counts = masked_df.groupby("household_id")[
                f"{target_entity}_id"
            ].nunique()
            all_households = hh_df["household_id"].unique()
            household_values_df = pd.DataFrame(
                {
                    "household_id": all_households,
                    "household_metric": household_counts.reindex(
                        all_households, fill_value=0
                    ).values,
                }
            )
        else:
            # For non-counts, sum the values
            household_values_df = (
                hh_df.groupby("household_id")[["entity_masked_metric"]]
                .sum()
                .reset_index()
                .rename({"entity_masked_metric": "household_metric"}, axis=1)
            )

        # Return sparse representation
        household_values_df = household_values_df.sort_values(
            ["household_id"]
        ).reset_index(drop=True)
        nonzero_indices = np.nonzero(household_values_df["household_metric"])[
            0
        ]
        nonzero_values = household_values_df.iloc[nonzero_indices][
            "household_metric"
        ].values

        return nonzero_indices, nonzero_values

    def build_matrix_for_geography_sparse(
        self, geographic_level: str, geographic_id: str, sim
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]:
        """
        Build sparse calibration matrix for any geographic level using hierarchical fallback.

        Returns:
            Tuple of (targets_df, sparse_matrix, household_ids)
        """
        national_stratum_id = (
            self.get_national_stratum_id()
        )  # 1 is the id for the US stratum with no other constraints

        if geographic_level == "state":
            state_stratum_id = self.get_state_stratum_id(geographic_id)
            cd_stratum_id = None  # No CD level for state calibration
            geo_label = f"state_{geographic_id}"
            if state_stratum_id is None:
                raise ValueError(
                    f"Could not find state {geographic_id} in database"
                )
        elif geographic_level == "congressional_district":
            cd_stratum_id = self.get_cd_stratum_id(
                geographic_id
            )  # congressional district stratum with no other constraints
            state_fips = self.get_state_fips_from_cd(geographic_id)
            state_stratum_id = self.get_state_stratum_id(state_fips)
            geo_label = f"cd_{geographic_id}"
            if cd_stratum_id is None:
                raise ValueError(
                    f"Could not find CD {geographic_id} in database"
                )
        else:
            raise ValueError(f"Unknown geographic level: {geographic_level}")

        # Use hierarchical fallback to get all targets
        if geographic_level == "congressional_district":
            # CD calibration: Use CD -> State -> National fallback
            # TODO: why does CD level use a function other than get_all_descendant_targets below?
            hierarchical_targets = self.get_hierarchical_targets(
                cd_stratum_id, state_stratum_id, national_stratum_id, sim
            )
        else:  # state
            # State calibration: Use State -> National fallback (no CD level)
            # For state calibration, we pass state_stratum_id twice to avoid null issues
            # TODO: why does state and national levels use a function other than get_hierarchical_targets above?_
            state_targets = self.get_all_descendant_targets(
                state_stratum_id, sim
            )
            national_targets = self.get_all_descendant_targets(
                national_stratum_id, sim
            )

            # Add geographic level
            state_targets["geo_level"] = "state"
            state_targets["geo_priority"] = 1
            national_targets["geo_level"] = "national"
            national_targets["geo_priority"] = 2

            # Combine and deduplicate
            all_targets = pd.concat(
                [state_targets, national_targets], ignore_index=True
            )

            # Create concept identifier from variable + all constraints
            # TODO (baogorek): Is this function defined muliple times? (I think it is)
            def get_concept_id(row):
                if not row["variable"]:
                    return None

                variable = row["variable"]

                # Parse constraint_info if present
                # TODO (baogorek): hard-coding needs refactoring
                if pd.notna(row.get("constraint_info")):
                    constraints = row["constraint_info"].split("|")

                    # Filter out geographic and filer constraints
                    demographic_constraints = []
                    irs_constraint = None

                    for c in constraints:
                        if not any(
                            skip in c
                            for skip in [
                                "state_fips",
                                "congressional_district_geoid",
                                "tax_unit_is_filer",
                            ]
                        ):
                            # Check if this is an IRS variable constraint
                            if not any(
                                demo in c
                                for demo in [
                                    "age",
                                    "adjusted_gross_income",
                                    "eitc_child_count",
                                    "snap",
                                    "medicaid",
                                ]
                            ):
                                # This is likely an IRS variable constraint like "salt>0"
                                irs_constraint = c
                            else:
                                demographic_constraints.append(c)

                    # If we have an IRS constraint, use that as the concept
                    if irs_constraint:
                        # Extract just the variable name from something like "salt>0"
                        import re

                        match = re.match(r"([a-zA-Z_]+)", irs_constraint)
                        if match:
                            return f"{match.group(1)}_constrained"

                    # Otherwise build concept from variable + demographic constraints
                    if demographic_constraints:
                        # Sort for consistency
                        demographic_constraints.sort()
                        # Normalize operators for valid identifiers
                        normalized = []
                        for c in demographic_constraints:
                            c_norm = c.replace(">=", "_gte_").replace(
                                "<=", "_lte_"
                            )
                            c_norm = c_norm.replace(">", "_gt_").replace(
                                "<", "_lt_"
                            )
                            c_norm = c_norm.replace("==", "_eq_").replace(
                                "=", "_eq_"
                            )
                            normalized.append(c_norm)
                        return f"{variable}_{'_'.join(normalized)}"

                # No constraints, just the variable
                return variable

            all_targets["concept_id"] = all_targets.apply(
                get_concept_id, axis=1
            )
            all_targets = all_targets[all_targets["concept_id"].notna()]
            all_targets = all_targets.sort_values(
                ["concept_id", "geo_priority"]
            )
            hierarchical_targets = (
                all_targets.groupby("concept_id").first().reset_index()
            )

        # Process hierarchical targets into the format expected by the rest of the code
        all_targets = []

        for _, target_row in hierarchical_targets.iterrows():
            # BUILD DESCRIPTION from variable and constraints (but not all constraints) ----
            desc_parts = [target_row["variable"]]

            # Parse constraint_info to add all constraints to description
            if pd.notna(target_row.get("constraint_info")):
                constraints = target_row["constraint_info"].split("|")
                # Filter out geographic and filer constraints FOR DESCRIPTION
                for c in constraints:
                    # TODO (baogorek): I get that the string is getting long, but "(filers)" doesn't add too much and geo_ids are max 4 digits
                    if not any(
                        skip in c
                        for skip in [
                            "state_fips",
                            "congressional_district_geoid",
                            "tax_unit_is_filer",
                        ]
                    ):
                        desc_parts.append(c)

            # Preserve the original stratum_group_id for proper grouping
            # Special handling only for truly national/geographic targets
            if pd.isna(target_row["stratum_group_id"]):
                # No stratum_group_id means it's a national target
                group_id = "national"
            elif target_row["stratum_group_id"] == 1:
                # Geographic identifier (not a real target)
                group_id = "geographic"
            else:
                # Keep the original numeric stratum_group_id
                # This preserves 2=Age, 3=AGI, 4=SNAP, 5=Medicaid, 6=EITC, 100+=IRS
                group_id = target_row["stratum_group_id"]

            all_targets.append(
                {
                    "target_id": target_row.get("target_id"),
                    "variable": target_row["variable"],
                    "value": target_row["value"],
                    "active": target_row.get("active", True),
                    "tolerance": target_row.get("tolerance", 0.05),
                    "stratum_id": target_row["stratum_id"],
                    "stratum_group_id": group_id,
                    "geographic_level": target_row["geo_level"],
                    "geographic_id": (
                        geographic_id
                        if target_row["geo_level"] == geographic_level
                        else (
                            "US"
                            if target_row["geo_level"] == "national"
                            else state_fips
                        )
                    ),
                    "description": "_".join(desc_parts),
                }
            )

        targets_df = pd.DataFrame(all_targets)

        # Build sparse data matrix ("loss matrix" historically) ---------------------------------------
        # NOTE: we are unapologetically at the household level at this point
        household_ids = sim.calculate(
            "household_id"
        ).values  # Implicit map to "household" entity level
        n_households = len(household_ids)
        n_targets = len(targets_df)

        # Use LIL matrix for efficient row-by-row construction
        matrix = sparse.lil_matrix((n_targets, n_households), dtype=np.float32)

        # TODO: is this were all the values are set?
        for i, (_, target) in enumerate(targets_df.iterrows()):
            # target = targets_df.iloc[68]
            constraints = self.get_constraints_for_stratum(
                target["stratum_id"]
            )  # NOTE:will not return the geo constraint
            # TODO: going in with snap target with index 68, and no constraints came out
            nonzero_indices, nonzero_values = (
                self.apply_constraints_to_sim_sparse(
                    sim, constraints, target["variable"]
                )
            )
            if len(nonzero_indices) > 0:
                matrix[i, nonzero_indices] = nonzero_values

        matrix = (
            matrix.tocsr()
        )  # To compressed sparse row (CSR) for efficient operations

        logger.info(
            f"Created sparse matrix for {geographic_level} {geographic_id}: shape {matrix.shape}, nnz={matrix.nnz}"
        )
        return targets_df, matrix, household_ids.tolist()

    # TODO (baogorek): instance of hard-coding (figure it out. This is why we have a targets database)
    def get_state_snap_cost(self, state_fips: str) -> pd.DataFrame:
        """Get state-level SNAP cost target (administrative data)."""
        query = """
        WITH snap_targets AS (
            SELECT 
                t.target_id,
                t.stratum_id,
                t.variable,
                t.value,
                t.active,
                t.tolerance,
                t.period
            FROM targets t
            JOIN strata s ON t.stratum_id = s.stratum_id
            JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
            WHERE s.stratum_group_id = 4  -- SNAP
              AND t.variable = 'snap'  -- Cost variable
              AND sc.constraint_variable = 'state_fips'
              AND sc.value = :state_fips
        ),
        best_period AS (
            SELECT 
                CASE
                    WHEN MAX(CASE WHEN period <= :target_year THEN period END) IS NOT NULL
                    THEN MAX(CASE WHEN period <= :target_year THEN period END)
                    ELSE MIN(period)
                END as selected_period
            FROM snap_targets
        )
        SELECT st.*
        FROM snap_targets st
        JOIN best_period bp ON st.period = bp.selected_period
        """

        with self.engine.connect() as conn:
            return pd.read_sql(
                query,
                conn,
                params={
                    "state_fips": state_fips,
                    "target_year": self.time_period,
                },
            )

    def get_state_fips_for_cd(self, cd_geoid: str) -> str:
        """Extract state FIPS from CD GEOID."""
        # CD GEOIDs are formatted as state_fips + district_number
        # e.g., "601" = California (06) district 01
        if len(cd_geoid) == 3:
            return str(
                int(cd_geoid[:1])
            )  # Single digit state, return as string of integer
        elif len(cd_geoid) == 4:
            return str(
                int(cd_geoid[:2])
            )  # Two digit state, return as string of integer
        else:
            raise ValueError(f"Unexpected CD GEOID format: {cd_geoid}")

    def build_stacked_matrix_sparse(
        self, geographic_level: str, geographic_ids: List[str], sim=None
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, Dict[str, List[str]]]:
        """
        Build stacked sparse calibration matrix for multiple geographic areas.

        Returns:
            Tuple of (targets_df, sparse_matrix, household_id_mapping)
        """
        all_targets = []
        geo_matrices = []
        household_id_mapping = {}

        # Pre-calculate US-state-specific values for state-dependent variables
        if sim is not None and len(self._state_specific_cache) == 0:
            us_state_dependent_vars = get_us_state_dependent_variables()
            if us_state_dependent_vars:
                logger.info("Pre-calculating US-state-specific values for state-dependent variables...")
                # Get dataset path from sim to create fresh simulations per state
                dataset_path = str(sim.dataset.__class__.file_path)
                self._calculate_state_specific_values(dataset_path, us_state_dependent_vars)

        # First, get national targets once (they apply to all geographic copies)
        national_targets = self.get_national_targets(sim)
        national_targets_list = []
        for _, target in national_targets.iterrows():
            # Get uprating info
            factor, uprating_type = self._get_uprating_info(
                target["variable"], target["period"]
            )

            # Build description with all constraints from constraint_info
            var_desc = target["variable"]
            if "constraint_info" in target and pd.notna(
                target["constraint_info"]
            ):
                constraints = target["constraint_info"].split("|")
                # Filter out geographic and filer constraints
                demo_constraints = [
                    c
                    for c in constraints
                    if not any(
                        skip in c
                        for skip in [
                            "state_fips",
                            "congressional_district_geoid",
                            "tax_unit_is_filer",
                        ]
                    )
                ]
                if demo_constraints:
                    # Join all constraints with underscores
                    var_desc = (
                        f"{target['variable']}_{'_'.join(demo_constraints)}"
                    )

            national_targets_list.append(
                {
                    "target_id": target["target_id"],
                    "stratum_id": target["stratum_id"],
                    "value": target["value"] * factor,
                    "original_value": target["value"],
                    "variable": target["variable"],
                    "variable_desc": var_desc,
                    "geographic_id": "US",
                    "stratum_group_id": "national",  # Required for create_target_groups
                    "period": target["period"],
                    "uprating_factor": factor,
                    "reconciliation_factor": 1.0,
                }
            )

        # Build national targets matrix ONCE before the loop
        national_matrix = None
        if sim is not None and len(national_targets) > 0:
            import time

            start = time.time()
            logger.info(
                f"Building national targets matrix once... ({len(national_targets)} targets)"
            )
            household_ids = sim.calculate("household_id").values
            n_households = len(household_ids)
            n_national_targets = len(national_targets)

            # Build sparse matrix for national targets
            national_matrix = sparse.lil_matrix(
                (n_national_targets, n_households), dtype=np.float32
            )

            for i, (_, target) in enumerate(national_targets.iterrows()):
                if i % 10 == 0:
                    logger.info(
                        f"  Processing national target {i+1}/{n_national_targets}: {target['variable']}"
                    )
                # Get constraints for this stratum
                constraints = self.get_constraints_for_stratum(
                    target["stratum_id"]
                )

                # Get sparse representation of household values
                nonzero_indices, nonzero_values = (
                    self.apply_constraints_to_sim_sparse(
                        sim, constraints, target["variable"]
                    )
                )

                # Set the sparse row
                if len(nonzero_indices) > 0:
                    national_matrix[i, nonzero_indices] = nonzero_values

            # Convert to CSR for efficiency
            national_matrix = national_matrix.tocsr()
            elapsed = time.time() - start
            logger.info(
                f"National matrix built in {elapsed:.1f}s: shape {national_matrix.shape}, nnz={national_matrix.nnz}"
            )

        # Collect all geography targets first for reconciliation
        all_geo_targets_dict = {}

        # Build matrix for each geography (CD-specific targets only)
        for i, geo_id in enumerate(geographic_ids):
            if i % 50 == 0:  # Log every 50th CD instead of every one
                logger.info(
                    f"Processing {geographic_level}s: {i+1}/{len(geographic_ids)} completed..."
                )

            # Get CD-specific targets directly without rebuilding national
            if geographic_level == "congressional_district":
                cd_stratum_id = self.get_cd_stratum_id(
                    geo_id
                )  # The base geographic stratum
                if cd_stratum_id is None:
                    raise ValueError(f"Could not find CD {geo_id} in database")

                # Get only CD-specific targets with deduplication
                cd_targets_raw = self.get_all_descendant_targets(
                    cd_stratum_id, sim
                )

                # Deduplicate CD targets by concept using ALL constraints
                def get_cd_concept_id(row):
                    """
                    Creates unique concept IDs from ALL constraints, not just the first one.
                    This eliminates the need for hard-coded stratum_group_id logic.

                    Examples:
                    - person_count with age>4|age<10 -> person_count_age_gt_4_age_lt_10
                    - person_count with adjusted_gross_income>=25000|adjusted_gross_income<50000
                      -> person_count_adjusted_gross_income_gte_25000_adjusted_gross_income_lt_50000
                    """
                    variable = row["variable"]

                    # Parse constraint_info which contains ALL constraints
                    if "constraint_info" in row and pd.notna(
                        row["constraint_info"]
                    ):
                        constraints = row["constraint_info"].split("|")

                        # Filter out geographic constraints (not part of the concept)
                        demographic_constraints = []
                        for c in constraints:
                            # Skip geographic and filer constraints
                            if not any(
                                skip in c
                                for skip in [
                                    "state_fips",
                                    "congressional_district_geoid",
                                    "tax_unit_is_filer",
                                ]
                            ):
                                # Normalize the constraint format for consistency
                                # Replace operators with text equivalents for valid Python identifiers
                                c_normalized = c.replace(
                                    ">=", "_gte_"
                                ).replace("<=", "_lte_")
                                c_normalized = c_normalized.replace(
                                    ">", "_gt_"
                                ).replace("<", "_lt_")
                                c_normalized = c_normalized.replace(
                                    "==", "_eq_"
                                ).replace("=", "_eq_")
                                c_normalized = c_normalized.replace(
                                    " ", ""
                                )  # Remove any spaces
                                demographic_constraints.append(c_normalized)

                        # Sort for consistency (ensures same constraints always produce same ID)
                        demographic_constraints.sort()

                        if demographic_constraints:
                            # Join all constraints to create unique concept
                            constraint_str = "_".join(demographic_constraints)
                            return f"{variable}_{constraint_str}"

                    # No constraints, just the variable name
                    return variable

                cd_targets_raw["cd_concept_id"] = cd_targets_raw.apply(
                    get_cd_concept_id, axis=1
                )

                if cd_targets_raw["cd_concept_id"].isna().any():
                    raise ValueError(
                        "Error: One or more targets were found without a valid concept ID."
                    )

                # For each concept, keep the first occurrence (or most specific based on stratum_group_id)
                # Prioritize by stratum_group_id: higher values are more specific
                cd_targets_raw = cd_targets_raw.sort_values(
                    ["cd_concept_id", "stratum_group_id"],
                    ascending=[True, False],
                )
                cd_targets = (
                    cd_targets_raw.groupby("cd_concept_id")
                    .first()
                    .reset_index(drop=True)
                )

                if len(cd_targets_raw) != len(cd_targets):
                    raise ValueError(
                        f"CD {geo_id}: Unwanted duplication: {len(cd_targets)} unique targets from {len(cd_targets_raw)} raw targets"
                    )

                # Store CD targets with stratum_group_id preserved for reconciliation
                cd_targets["geographic_id"] = geo_id
                all_geo_targets_dict[geo_id] = cd_targets
            else:
                # For state-level, collect targets for later reconciliation
                state_stratum_id = self.get_state_stratum_id(geo_id)
                if state_stratum_id is None:
                    logger.warning(
                        f"Could not find state {geo_id} in database"
                    )
                    continue
                state_targets = self.get_all_descendant_targets(
                    state_stratum_id, sim
                )
                state_targets["geographic_id"] = geo_id
                all_geo_targets_dict[geo_id] = state_targets

        # Reconcile targets to higher level if CD calibration
        if (
            geographic_level == "congressional_district"
            and all_geo_targets_dict
        ):
            # Age targets (stratum_group_id=2) - already match so no-op
            logger.info("Reconciling CD age targets to state totals...")
            reconciled_dict = self.reconcile_targets_to_higher_level(
                all_geo_targets_dict,
                higher_level="state",
                target_filters={"stratum_group_id": 2},  # Age targets
                sim=sim,
            )
            all_geo_targets_dict = reconciled_dict

            # Medicaid targets (stratum_group_id=5) - needs reconciliation
            # TODO(bogorek): manually trace a reconcilliation
            logger.info(
                "Reconciling CD Medicaid targets to state admin totals..."
            )
            reconciled_dict = self.reconcile_targets_to_higher_level(
                all_geo_targets_dict,
                higher_level="state",
                target_filters={"stratum_group_id": 5},  # Medicaid targets
                sim=sim,
            )
            all_geo_targets_dict = reconciled_dict

            # SNAP household targets (stratum_group_id=4) - needs reconciliation
            logger.info(
                "Reconciling CD SNAP household counts to state admin totals..."
            )
            reconciled_dict = self.reconcile_targets_to_higher_level(
                all_geo_targets_dict,
                higher_level="state",
                target_filters={
                    "stratum_group_id": 4,
                    "variable": "household_count",
                },  # SNAP households
                sim=sim,
            )
            all_geo_targets_dict = reconciled_dict

        # Now build matrices for all collected and reconciled targets
        # TODO (baogorek): a lot of hard-coded stuff here, but there is an else backoff
        for geo_id, geo_targets_df in all_geo_targets_dict.items():
            # Format targets
            geo_target_list = []
            for _, target in geo_targets_df.iterrows():
                # Get uprating info
                factor, uprating_type = self._get_uprating_info(
                    target["variable"], target.get("period", self.time_period)
                )

                # Apply uprating to value (may already have reconciliation factor applied)
                final_value = target["value"] * factor

                # Create meaningful description based on stratum_group_id and variable
                stratum_group = target.get("stratum_group_id")

                # Build descriptive prefix based on stratum_group_id
                # TODO (baogorek): Usage of stratum_group is not ideal, but is this just building notes?
                if isinstance(stratum_group, (int, np.integer)):
                    if stratum_group == 2:  # Age
                        # Use stratum_notes if available, otherwise build from constraint
                        if "stratum_notes" in target and pd.notna(
                            target.get("stratum_notes")
                        ):
                            # Extract age range from notes like "Age: 0-4, CD 601"
                            notes = str(target["stratum_notes"])
                            if "Age:" in notes:
                                age_part = (
                                    notes.split("Age:")[1]
                                    .split(",")[0]
                                    .strip()
                                )
                                desc_prefix = f"age_{age_part}"
                            else:
                                desc_prefix = "age"
                        else:
                            desc_prefix = "age"
                    elif stratum_group == 3:  # AGI
                        desc_prefix = "AGI"
                    elif stratum_group == 4:  # SNAP
                        desc_prefix = "SNAP_households"
                    elif stratum_group == 5:  # Medicaid
                        desc_prefix = "Medicaid_enrollment"
                    elif stratum_group == 6:  # EITC
                        desc_prefix = "EITC"
                    elif stratum_group >= 100:  # IRS variables
                        irs_names = {
                            100: "QBI_deduction",
                            101: "self_employment",
                            102: "net_capital_gains",
                            103: "real_estate_taxes",
                            104: "rental_income",
                            105: "net_capital_gain",
                            106: "taxable_IRA_distributions",
                            107: "taxable_interest",
                            108: "tax_exempt_interest",
                            109: "dividends",
                            110: "qualified_dividends",
                            111: "partnership_S_corp",
                            112: "all_filers",
                            113: "unemployment_comp",
                            114: "medical_deduction",
                            115: "taxable_pension",
                            116: "refundable_CTC",
                            117: "SALT_deduction",
                            118: "income_tax_paid",
                            119: "income_tax_before_credits",
                        }
                        desc_prefix = irs_names.get(
                            stratum_group, f"IRS_{stratum_group}"
                        )
                        # Add variable suffix for amount vs count
                        if target["variable"] == "tax_unit_count":
                            desc_prefix = f"{desc_prefix}_count"
                        else:
                            desc_prefix = f"{desc_prefix}_amount"
                    else:
                        desc_prefix = target["variable"]
                else:
                    desc_prefix = target["variable"]

                # Just use the descriptive prefix without geographic suffix
                # The geographic context is already provided elsewhere
                description = desc_prefix

                # Build description with all constraints from constraint_info
                var_desc = target["variable"]
                if "constraint_info" in target and pd.notna(
                    target["constraint_info"]
                ):
                    constraints = target["constraint_info"].split("|")
                    # Filter out geographic and filer constraints
                    demo_constraints = [
                        c
                        for c in constraints
                        if not any(
                            skip in c
                            for skip in [
                                "state_fips",
                                "congressional_district_geoid",
                                "tax_unit_is_filer",
                            ]
                        )
                    ]
                    if demo_constraints:
                        # Join all constraints with underscores
                        var_desc = f"{target['variable']}_{'_'.join(demo_constraints)}"

                geo_target_list.append(
                    {
                        "target_id": target["target_id"],
                        "stratum_id": target["stratum_id"],
                        "value": final_value,
                        "original_value": target.get(
                            "original_value_pre_reconciliation",
                            target["value"],
                        ),
                        "variable": target["variable"],
                        "variable_desc": var_desc,
                        "geographic_id": geo_id,
                        "stratum_group_id": target.get(
                            "stratum_group_id", geographic_level
                        ),  # Preserve original group ID
                        "period": target.get("period", self.time_period),
                        "uprating_factor": factor,
                        "reconciliation_factor": target.get(
                            "reconciliation_factor", 1.0
                        ),
                        "undercount_pct": target.get("undercount_pct", 0.0),
                    }
                )

            if geo_target_list:
                targets_df = pd.DataFrame(geo_target_list)
                all_targets.append(targets_df)

                # Build matrix for geo-specific targets
                if sim is not None:
                    household_ids = sim.calculate("household_id").values
                    n_households = len(household_ids)
                    n_targets = len(targets_df)

                    matrix = sparse.lil_matrix(
                        (n_targets, n_households), dtype=np.float32
                    )

                    for j, (_, target) in enumerate(targets_df.iterrows()):
                        constraints = self.get_constraints_for_stratum(
                            target["stratum_id"]
                        )
                        nonzero_indices, nonzero_values = (
                            self.apply_constraints_to_sim_sparse(
                                sim, constraints, target["variable"]
                            )
                        )
                        if len(nonzero_indices) > 0:
                            matrix[j, nonzero_indices] = nonzero_values

                    matrix = matrix.tocsr()
                    geo_matrices.append(matrix)

                    # Store household ID mapping
                    prefix = (
                        "cd"
                        if geographic_level == "congressional_district"
                        else "state"
                    )
                    household_id_mapping[f"{prefix}{geo_id}"] = [
                        f"{hh_id}_{prefix}{geo_id}" for hh_id in household_ids
                    ]

        # If building for congressional districts, add state-level SNAP costs
        state_snap_targets_list = []
        state_snap_matrices = []
        if geographic_level == "congressional_district":
            # Identify unique states from the CDs
            unique_states = set()
            for cd_id in geographic_ids:
                state_fips = self.get_state_fips_for_cd(cd_id)
                unique_states.add(state_fips)

            # Get household info - must match the actual matrix columns
            household_ids = sim.calculate("household_id").values
            n_households = len(household_ids)
            total_cols = n_households * len(geographic_ids)

            # Get SNAP cost target for each state
            for state_fips in sorted(unique_states):
                snap_cost_df = self.get_state_snap_cost(state_fips)
                if not snap_cost_df.empty:
                    for _, target in snap_cost_df.iterrows():
                        # Get uprating info
                        # TODO: why is period showing up as 2022 in my interactive run?
                        period = target.get("period", self.time_period)
                        factor, uprating_type = self._get_uprating_info(
                            target["variable"], period
                        )

                        state_snap_targets_list.append(
                            {
                                "target_id": target["target_id"],
                                "stratum_id": target["stratum_id"],
                                "value": target["value"] * factor,
                                "original_value": target["value"],
                                "variable": target["variable"],
                                "variable_desc": "snap_cost_state",
                                "geographic_id": state_fips,
                                "stratum_group_id": "state_snap_cost",  # Special group for state SNAP costs
                                "period": period,
                                "uprating_factor": factor,
                                "reconciliation_factor": 1.0,
                                "undercount_pct": 0.0,
                            }
                        )

                        # Build matrix row for this state SNAP cost
                        # This row should have SNAP values for households in CDs of this state
                        # Get constraints for this state SNAP stratum to apply to simulation
                        constraints = self.get_constraints_for_stratum(
                            target["stratum_id"]
                        )

                        # Create a sparse row with correct dimensions (1 x total_cols)
                        row_data = []
                        row_indices = []

                        # Calculate SNAP values once for ALL households (geographic isolation via matrix structure)
                        # Note: state_fips constraint is automatically skipped, SNAP values calculated for all
                        # Use state-specific cached values if available
                        nonzero_indices, nonzero_values = (
                            self.apply_constraints_to_sim_sparse(
                                sim, constraints, "snap",
                                target_state_fips=int(state_fips)  # Pass state to use cached values
                            )
                        )

                        # Create a mapping of household indices to SNAP values
                        snap_value_map = dict(
                            zip(nonzero_indices, nonzero_values)
                        )

                        # Place SNAP values in ALL CD columns that belong to this state
                        # This creates the proper geo-stacking structure where state-level targets
                        # span multiple CD columns (all CDs within the state)
                        for cd_idx, cd_id in enumerate(geographic_ids):
                            cd_state_fips = self.get_state_fips_from_cd(cd_id)
                            if cd_state_fips == state_fips:
                                # This CD is in the target state - add SNAP values to its columns
                                col_offset = cd_idx * n_households
                                for hh_idx, snap_val in snap_value_map.items():
                                    row_indices.append(col_offset + hh_idx)
                                    row_data.append(snap_val)

                        # Create sparse matrix row
                        if row_data:
                            row_matrix = sparse.csr_matrix(
                                (row_data, ([0] * len(row_data), row_indices)),
                                shape=(1, total_cols),
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

        # Stack matrices
        if not geo_matrices:
            raise ValueError("No geo_matrices were built - this should not happen")

        # Stack geo-specific targets (block diagonal)
        stacked_geo = sparse.block_diag(geo_matrices)
        logger.info(
            f"Stacked geo-specific matrix: shape {stacked_geo.shape}, nnz={stacked_geo.nnz}"
        )

        # Combine all matrix parts
        matrix_parts = []
        if national_matrix is not None:
            national_copies = [national_matrix] * len(geographic_ids)
            stacked_national = sparse.hstack(national_copies)
            logger.info(
                f"Stacked national matrix: shape {stacked_national.shape}, nnz={stacked_national.nnz}"
            )
            matrix_parts.append(stacked_national)
        matrix_parts.append(stacked_geo)

        # Add state SNAP matrices if we have them (for CD calibration)
        if state_snap_matrices:
            stacked_state_snap = sparse.vstack(state_snap_matrices)
            matrix_parts.append(stacked_state_snap)

        # Combine all parts
        combined_matrix = sparse.vstack(matrix_parts)
        combined_matrix = combined_matrix.tocsr()

        logger.info(
            f"Created stacked sparse matrix: shape {combined_matrix.shape}, nnz={combined_matrix.nnz}"
        )
        return combined_targets, combined_matrix, household_id_mapping


def main():
    """Example usage for California and North Carolina."""
    from policyengine_us import Microsimulation

    # Database path
    db_uri = "sqlite:////home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"

    # Initialize sparse builder
    builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)

    # Create microsimulation with 2024 data
    print("Loading microsimulation...")
    sim = Microsimulation(
        dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
    )

    # Test single state
    print("\nBuilding sparse matrix for California (FIPS 6)...")
    targets_df, matrix, household_ids = (
        builder.build_matrix_for_geography_sparse("state", "6", sim)
    )

    print("\nTarget Summary:")
    print(f"Total targets: {len(targets_df)}")
    print(f"Matrix shape: {matrix.shape}")
    print(
        f"Matrix sparsity: {matrix.nnz} non-zero elements ({100*matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.4f}%)"
    )
    print(
        f"Memory usage: {matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes} bytes"
    )

    # Test stacking multiple states
    print("\n" + "=" * 70)
    print(
        "Testing multi-state stacking: California (6) and North Carolina (37)"
    )
    print("=" * 70)

    targets_df, matrix, hh_mapping = builder.build_stacked_matrix_sparse(
        "state", ["6", "37"], sim
    )

    if matrix is not None:
        print(f"\nStacked matrix shape: {matrix.shape}")
        print(
            f"Stacked matrix sparsity: {matrix.nnz} non-zero elements ({100*matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.4f}%)"
        )
        print(
            f"Memory usage: {matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes} bytes"
        )

        # Compare to dense matrix memory
        dense_memory = (
            matrix.shape[0] * matrix.shape[1] * 4
        )  # 4 bytes per float32
        print(f"Dense matrix would use: {dense_memory} bytes")
        print(
            f"Memory savings: {100*(1 - (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)/dense_memory):.2f}%"
        )


if __name__ == "__main__":
    main()
