"""
Household tracer utility for debugging geo-stacking sparse matrices.

This utility allows tracing a single household through the complex stacked matrix
structure to verify values match sim.calculate results.

USAGE
=====

Basic Setup (from calibration package):

    import pickle
    from household_tracer import HouseholdTracer

    # Load calibration package
    with open('calibration_package.pkl', 'rb') as f:
        data = pickle.load(f)

    # Extract components
    X_sparse = data['X_sparse']
    targets_df = data['targets_df']
    household_id_mapping = data['household_id_mapping']
    cds_to_calibrate = data['cds_to_calibrate']
    # Note: you also need 'sim' (Microsimulation instance)

    # Create tracer
    tracer = HouseholdTracer(
        targets_df, X_sparse, household_id_mapping,
        cds_to_calibrate, sim
    )

Common Operations:

    # 1. Understand what a column represents
    col_info = tracer.get_column_info(100)
    # Returns: {'column_index': 100, 'cd_geoid': '101',
    #           'household_id': 100, 'household_index': 99}

    # 2. Access full column catalog (all column mappings)
    tracer.column_catalog  # DataFrame with all 4.6M column mappings

    # 3. Find where a household appears across all CDs
    positions = tracer.get_household_column_positions(565)
    # Returns: {'101': 564, '102': 11144, '201': 21724, ...}

    # 4. Look up a specific matrix cell with full context
    cell = tracer.lookup_matrix_cell(row_idx=50, col_idx=100)
    # Returns complete info about target, household, and value

    # 5. Get info about a row (target)
    row_info = tracer.get_row_info(50)

    # 6. View matrix structure
    tracer.print_matrix_structure()

    # 7. View column/row catalogs
    tracer.print_column_catalog(max_rows=50)
    tracer.print_row_catalog(max_rows=50)

    # 8. Trace all target values for a specific household
    household_targets = tracer.trace_household_targets(565)

    # 9. Get targets by group
    from calibration_utils import create_target_groups
    tracer.target_groups, _ = create_target_groups(targets_df)
    group_31 = tracer.get_group_rows(31)  # Person count targets

Matrix Structure:

    Columns are organized as: [CD1_households | CD2_households | ... | CD436_households]
    Each CD block has n_households columns (e.g., 10,580 households)

    Formula to find column index:
        column_idx = cd_block_number × n_households + household_index

    Example: Household at index 12 in CD block 371:
        column_idx = 371 × 10580 + 12 = 3,925,192
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import sparse

from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import create_target_groups
from policyengine_us_data.datasets.cps.local_area_calibration.metrics_matrix_geo_stacking_sparse import SparseGeoStackingMatrixBuilder
from policyengine_us import Microsimulation
from sqlalchemy import create_engine, text


logger = logging.getLogger(__name__)


class HouseholdTracer:
    """Trace households through geo-stacked sparse matrices for debugging."""

    def __init__(
        self,
        targets_df: pd.DataFrame,
        matrix: sparse.csr_matrix,
        household_id_mapping: Dict[str, List[str]],
        geographic_ids: List[str],
        sim,
    ):
        """
        Initialize tracer with matrix components.

        Args:
            targets_df: DataFrame of all targets
            matrix: The final stacked sparse matrix
            household_id_mapping: Mapping from geo keys to household ID lists
            geographic_ids: List of geographic IDs in order
            sim: Microsimulation instance
        """
        self.targets_df = targets_df
        self.matrix = matrix
        self.household_id_mapping = household_id_mapping
        self.geographic_ids = geographic_ids
        self.sim = sim

        # Get original household info
        self.original_household_ids = sim.calculate("household_id").values
        self.n_households = len(self.original_household_ids)
        self.n_geographies = len(geographic_ids)

        # Build reverse lookup: original_hh_id -> index in original data
        self.hh_id_to_index = {
            hh_id: idx for idx, hh_id in enumerate(self.original_household_ids)
        }

        # Build column catalog: maps column index -> (cd_geoid, household_id, household_index)
        self.column_catalog = self._build_column_catalog()

        # Build row catalog: maps row index -> target info
        self.row_catalog = self._build_row_catalog()

        logger.info(
            f"Tracer initialized: {self.n_households} households x {self.n_geographies} geographies"
        )
        logger.info(f"Matrix shape: {matrix.shape}")

    def _build_column_catalog(self) -> pd.DataFrame:
        """Build a complete catalog of all matrix columns."""
        catalog = []
        col_idx = 0

        for geo_id in self.geographic_ids:
            for hh_idx, hh_id in enumerate(self.original_household_ids):
                catalog.append(
                    {
                        "column_index": col_idx,
                        "cd_geoid": geo_id,
                        "household_id": hh_id,
                        "household_index": hh_idx,
                    }
                )
                col_idx += 1

        return pd.DataFrame(catalog)

    def _build_row_catalog(self) -> pd.DataFrame:
        """Build a complete catalog of all matrix rows (targets)."""
        catalog = []

        for row_idx, (_, target) in enumerate(self.targets_df.iterrows()):
            catalog.append(
                {
                    "row_index": row_idx,
                    "variable": target["variable"],
                    "variable_desc": target.get(
                        "variable_desc", target["variable"]
                    ),
                    "geographic_id": target.get("geographic_id", "unknown"),
                    "geographic_level": target.get(
                        "geographic_level", "unknown"
                    ),
                    "target_value": target["value"],
                    "stratum_id": target.get("stratum_id"),
                    "stratum_group_id": target.get(
                        "stratum_group_id", "unknown"
                    ),
                }
            )

        return pd.DataFrame(catalog)

    def get_column_info(self, col_idx: int) -> Dict:
        """Get information about a specific column."""
        if col_idx >= len(self.column_catalog):
            raise ValueError(
                f"Column index {col_idx} out of range (max: {len(self.column_catalog)-1})"
            )
        return self.column_catalog.iloc[col_idx].to_dict()

    def get_row_info(self, row_idx: int) -> Dict:
        """Get information about a specific row (target)."""
        if row_idx >= len(self.row_catalog):
            raise ValueError(
                f"Row index {row_idx} out of range (max: {len(self.row_catalog)-1})"
            )
        return self.row_catalog.iloc[row_idx].to_dict()

    def lookup_matrix_cell(self, row_idx: int, col_idx: int) -> Dict:
        """
        Look up a specific matrix cell and return complete context.

        Args:
            row_idx: Row index in matrix
            col_idx: Column index in matrix

        Returns:
            Dict with row info, column info, and matrix value
        """
        row_info = self.get_row_info(row_idx)
        col_info = self.get_column_info(col_idx)
        matrix_value = self.matrix[row_idx, col_idx]

        return {
            "row_index": row_idx,
            "column_index": col_idx,
            "matrix_value": float(matrix_value),
            "target": row_info,
            "household": col_info,
        }

    def print_column_catalog(self, max_rows: int = 50):
        """Print a sample of the column catalog."""
        print(
            f"\nColumn Catalog (showing first {max_rows} of {len(self.column_catalog)}):"
        )
        print(self.column_catalog.head(max_rows).to_string(index=False))

    def print_row_catalog(self, max_rows: int = 50):
        """Print a sample of the row catalog."""
        print(
            f"\nRow Catalog (showing first {max_rows} of {len(self.row_catalog)}):"
        )
        print(self.row_catalog.head(max_rows).to_string(index=False))

    def print_matrix_structure(self, create_groups=True):
        """Print a comprehensive breakdown of the matrix structure."""
        print("\n" + "=" * 80)
        print("MATRIX STRUCTURE BREAKDOWN")
        print("=" * 80)

        print(
            f"\nMatrix dimensions: {self.matrix.shape[0]} rows × {self.matrix.shape[1]} columns"
        )
        print(f"  Rows = {len(self.row_catalog)} targets")
        print(
            f"  Columns = {self.n_households} households × {self.n_geographies} CDs"
        )
        print(
            f"           = {self.n_households:,} × {self.n_geographies} = {self.matrix.shape[1]:,}"
        )

        print("\n" + "-" * 80)
        print("COLUMN STRUCTURE (Households stacked by CD)")
        print("-" * 80)

        # Build column ranges by CD
        col_ranges = []
        cumulative = 0
        for geo_id in self.geographic_ids:
            start_col = cumulative
            end_col = cumulative + self.n_households - 1
            col_ranges.append(
                {
                    "cd_geoid": geo_id,
                    "start_col": start_col,
                    "end_col": end_col,
                    "n_households": self.n_households,
                    "example_household_id": self.original_household_ids[0],
                }
            )
            cumulative += self.n_households

        ranges_df = pd.DataFrame(col_ranges)
        print(f"\nShowing first and last 10 CDs of {len(ranges_df)} total:")
        print("\nFirst 10 CDs:")
        print(ranges_df.head(10).to_string(index=False))
        print("\nLast 10 CDs:")
        print(ranges_df.tail(10).to_string(index=False))

        print("\n" + "-" * 80)
        print("ROW STRUCTURE (Targets by geography and variable)")
        print("-" * 80)

        # Summarize rows by geographic level
        row_summary = (
            self.row_catalog.groupby(["geographic_level", "geographic_id"])
            .size()
            .reset_index(name="n_targets")
        )

        print(f"\nTargets by geographic level:")
        geo_level_summary = (
            self.row_catalog.groupby("geographic_level")
            .size()
            .reset_index(name="n_targets")
        )
        print(geo_level_summary.to_string(index=False))

        print(f"\nTargets by stratum group:")
        stratum_summary = (
            self.row_catalog.groupby("stratum_group_id")
            .agg({"row_index": "count", "variable": lambda x: len(x.unique())})
            .rename(
                columns={"row_index": "n_targets", "variable": "n_unique_vars"}
            )
        )
        print(stratum_summary.to_string())

        # Create and display target groups like calibrate_cds_sparse.py
        if create_groups:
            print("\n" + "-" * 80)
            print("TARGET GROUPS (for loss calculation)")
            print("-" * 80)

            target_groups, group_info = create_target_groups(self.targets_df)

            # Store target groups for later use
            self.target_groups = target_groups

            # Use the improved labels from create_target_groups
            for group_id, info in enumerate(group_info):
                # Get row indices for this group
                group_mask = target_groups == group_id
                row_indices = np.where(group_mask)[0]

                # Format row indices for display
                if len(row_indices) > 6:
                    row_display = f"[{row_indices[0]}, {row_indices[1]}, {row_indices[2]}, '...', {row_indices[-2]}, {row_indices[-1]}]"
                else:
                    row_display = str(row_indices.tolist())

                print(f"  {info} - rows {row_display}")

        print("\n" + "=" * 80)

    def get_group_rows(self, group_id: int) -> pd.DataFrame:
        """
        Get all rows (targets) for a specific target group.

        Args:
            group_id: The target group ID

        Returns:
            DataFrame with all targets in that group
        """
        if not hasattr(self, "target_groups"):
            self.target_groups, _ = create_target_groups(self.targets_df)

        group_mask = self.target_groups == group_id
        group_targets = self.targets_df[group_mask].copy()

        # Add row indices
        row_indices = np.where(group_mask)[0]
        group_targets["row_index"] = row_indices

        # Reorder columns for clarity
        cols = [
            "row_index",
            "variable",
            "geographic_id",
            "value",
            "description",
        ]
        cols = [c for c in cols if c in group_targets.columns]
        group_targets = group_targets[cols]

        return group_targets

    def get_household_column_positions(
        self, original_hh_id: int
    ) -> Dict[str, int]:
        """
        Get all column positions for a household across all geographies.

        Args:
            original_hh_id: Original household ID from simulation

        Returns:
            Dict mapping geo_id to column position in stacked matrix
        """
        if original_hh_id not in self.hh_id_to_index:
            raise ValueError(
                f"Household {original_hh_id} not found in original data"
            )

        # Get the household's index in the original data
        hh_index = self.hh_id_to_index[original_hh_id]

        # Calculate column positions for each geography
        positions = {}
        for geo_idx, geo_id in enumerate(self.geographic_ids):
            # Each geography gets a block of n_households columns
            col_position = geo_idx * self.n_households + hh_index
            positions[geo_id] = col_position

        return positions

    def trace_household_targets(self, original_hh_id: int) -> pd.DataFrame:
        """
        Extract all target values for a household across all geographies.

        Args:
            original_hh_id: Original household ID to trace

        Returns:
            DataFrame with target details and values for this household
        """
        positions = self.get_household_column_positions(original_hh_id)

        results = []

        for target_idx, (_, target) in enumerate(self.targets_df.iterrows()):
            target_result = {
                "target_idx": target_idx,
                "variable": target["variable"],
                "target_value": target["value"],
                "geographic_id": target.get("geographic_id", "unknown"),
                "stratum_group_id": target.get("stratum_group_id", "unknown"),
                "description": target.get("description", ""),
            }

            # Extract values for this target across all geographies
            for geo_id, col_pos in positions.items():
                if col_pos < self.matrix.shape[1]:
                    matrix_value = self.matrix[target_idx, col_pos]
                    target_result[f"matrix_value_{geo_id}"] = matrix_value
                else:
                    target_result[f"matrix_value_{geo_id}"] = np.nan

            results.append(target_result)

        return pd.DataFrame(results)

    def verify_household_target(
        self, original_hh_id: int, target_idx: int, geo_id: str
    ) -> Dict:
        """
        Verify a specific target value for a household by comparing with sim.calculate.

        Args:
            original_hh_id: Original household ID
            target_idx: Target row index in matrix
            geo_id: Geographic ID to check

        Returns:
            Dict with verification results
        """
        # Get target info
        target = self.targets_df.iloc[target_idx]
        variable = target["variable"]
        stratum_id = target["stratum_id"]

        # Get matrix value
        positions = self.get_household_column_positions(original_hh_id)
        col_pos = positions[geo_id]
        matrix_value = self.matrix[target_idx, col_pos]

        # Calculate expected value using sim
        # Import the matrix builder to access constraint methods

        # We need a builder instance to get constraints
        # This is a bit hacky but necessary for verification
        db_uri = "sqlite:////home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
        builder = SparseGeoStackingMatrixBuilder(db_uri)

        # Get constraints for this stratum
        constraints_df = builder.get_constraints_for_stratum(stratum_id)

        # Calculate what the value should be for this household
        expected_value = self._calculate_expected_value(
            original_hh_id, variable, constraints_df
        )

        return {
            "household_id": original_hh_id,
            "target_idx": target_idx,
            "geo_id": geo_id,
            "variable": variable,
            "stratum_id": stratum_id,
            "matrix_value": float(matrix_value),
            "expected_value": float(expected_value),
            "matches": abs(matrix_value - expected_value) < 1e-6,
            "difference": float(matrix_value - expected_value),
            "constraints": (
                constraints_df.to_dict("records")
                if not constraints_df.empty
                else []
            ),
        }

    def _calculate_expected_value(
        self, original_hh_id: int, variable: str, constraints_df: pd.DataFrame
    ) -> float:
        """
        Calculate expected value for a household given variable and constraints.
        """
        # Get household index
        hh_index = self.hh_id_to_index[original_hh_id]

        # Get target entity
        target_entity = self.sim.tax_benefit_system.variables[
            variable
        ].entity.key

        # Check if household satisfies all constraints
        satisfies_constraints = True

        for _, constraint in constraints_df.iterrows():
            var = constraint["constraint_variable"]
            op = constraint["operation"]
            val = constraint["value"]

            # Skip geographic constraints (they're handled by matrix structure)
            if var in ["state_fips", "congressional_district_geoid"]:
                continue

            # Get constraint value for this household
            constraint_entity = self.sim.tax_benefit_system.variables[
                var
            ].entity.key
            if constraint_entity == "person":
                # For person variables, check if any person in household satisfies
                person_values = self.sim.calculate(var, map_to="person").values
                household_ids_person_level = self.sim.calculate(
                    "household_id", map_to="person"
                ).values

                # Get person values for this household
                household_mask = household_ids_person_level == original_hh_id
                household_person_values = person_values[household_mask]

                # Parse constraint value
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

                # Check if any person in household satisfies constraint
                if op == "==" or op == "=":
                    person_satisfies = household_person_values == parsed_val
                elif op == ">":
                    person_satisfies = household_person_values > parsed_val
                elif op == ">=":
                    person_satisfies = household_person_values >= parsed_val
                elif op == "<":
                    person_satisfies = household_person_values < parsed_val
                elif op == "<=":
                    person_satisfies = household_person_values <= parsed_val
                elif op == "!=":
                    person_satisfies = household_person_values != parsed_val
                else:
                    continue

                if not person_satisfies.any():
                    satisfies_constraints = False
                    break

            else:
                # For household/tax_unit variables, get value directly
                if constraint_entity == "household":
                    constraint_value = self.sim.calculate(var).values[hh_index]
                else:
                    # For tax_unit, map to household level
                    constraint_value = self.sim.calculate(
                        var, map_to="household"
                    ).values[hh_index]

                # Parse constraint value
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

                # Check constraint
                if op == "==" or op == "=":
                    if not (constraint_value == parsed_val):
                        satisfies_constraints = False
                        break
                elif op == ">":
                    if not (constraint_value > parsed_val):
                        satisfies_constraints = False
                        break
                elif op == ">=":
                    if not (constraint_value >= parsed_val):
                        satisfies_constraints = False
                        break
                elif op == "<":
                    if not (constraint_value < parsed_val):
                        satisfies_constraints = False
                        break
                elif op == "<=":
                    if not (constraint_value <= parsed_val):
                        satisfies_constraints = False
                        break
                elif op == "!=":
                    if not (constraint_value != parsed_val):
                        satisfies_constraints = False
                        break

        if not satisfies_constraints:
            return 0.0

        # If constraints satisfied, get the target value
        if target_entity == "household":
            target_value = self.sim.calculate(variable).values[hh_index]
        elif target_entity == "person":
            # For person variables, sum over household members
            person_values = self.sim.calculate(
                variable, map_to="person"
            ).values
            household_ids_person_level = self.sim.calculate(
                "household_id", map_to="person"
            ).values
            household_mask = household_ids_person_level == original_hh_id
            target_value = person_values[household_mask].sum()
        else:
            # For tax_unit variables, map to household
            target_value = self.sim.calculate(
                variable, map_to="household"
            ).values[hh_index]

        return float(target_value)

    def audit_household(
        self, original_hh_id: int, max_targets: int = 10
    ) -> Dict:
        """
        Comprehensive audit of a household across all targets and geographies.

        Args:
            original_hh_id: Household ID to audit
            max_targets: Maximum number of targets to verify in detail

        Returns:
            Dict with audit results
        """
        logger.info(f"Auditing household {original_hh_id}")

        # Get basic info
        positions = self.get_household_column_positions(original_hh_id)
        all_values = self.trace_household_targets(original_hh_id)

        # Verify a sample of targets
        verifications = []
        target_sample = min(max_targets, len(self.targets_df))

        for target_idx in range(
            0,
            len(self.targets_df),
            max(1, len(self.targets_df) // target_sample),
        ):
            for geo_id in self.geographic_ids[
                :2
            ]:  # Limit to first 2 geographies
                try:
                    verification = self.verify_household_target(
                        original_hh_id, target_idx, geo_id
                    )
                    verifications.append(verification)
                except Exception as e:
                    logger.warning(
                        f"Could not verify target {target_idx} for geo {geo_id}: {e}"
                    )

        # Summary statistics
        if verifications:
            matches = [v["matches"] for v in verifications]
            match_rate = sum(matches) / len(matches)
            max_diff = max([abs(v["difference"]) for v in verifications])
        else:
            match_rate = 0.0
            max_diff = 0.0

        return {
            "household_id": original_hh_id,
            "column_positions": positions,
            "all_target_values": all_values,
            "verifications": verifications,
            "summary": {
                "total_verifications": len(verifications),
                "match_rate": match_rate,
                "max_difference": max_diff,
                "passes_audit": match_rate > 0.95 and max_diff < 1e-3,
            },
        }


def matrix_tracer():
    """Demo the household tracer."""

    # Setup - match calibrate_cds_sparse.py configuration exactly
    db_uri = "sqlite:////home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/policy_data.db"
    builder = SparseGeoStackingMatrixBuilder(db_uri, time_period=2023)
    sim = Microsimulation(dataset="/home/baogorek/devl/stratified_10k.h5")

    hh_person_rel = pd.DataFrame(
        {
            "household_id": sim.calculate("household_id", map_to="person"),
            "person_id": sim.calculate("person_id", map_to="person"),
        }
    )

    # Get all congressional districts from database (like calibrate_cds_sparse.py does)
    engine = create_engine(db_uri)
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM strata s
    JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
    WHERE s.stratum_group_id = 1
      AND sc.constraint_variable = 'congressional_district_geoid'
    ORDER BY sc.value
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        all_cd_geoids = [row[0] for row in result]

    targets_df, matrix, household_mapping = (
        builder.build_stacked_matrix_sparse(
            "congressional_district", all_cd_geoids, sim
        )
    )
    target_groups, y = create_target_groups(targets_df)

    tracer = HouseholdTracer(
        targets_df, matrix, household_mapping, all_cd_geoids, sim
    )
    tracer.print_matrix_structure()

    # Testing national targets with a test household -----------------
    test_household = sim.calculate("household_id").values[100]
    positions = tracer.get_household_column_positions(test_household)

    # Row 0: Alimony - Row 0
    matrix_hh_position = positions["3910"]
    matrix[0, matrix_hh_position]

    # Row 0: Alimony - Row 0
    matrix_hh_position = positions["3910"]
    matrix[0, matrix_hh_position]

    # Group 32: Medicaid Enrollment (436 targets across 436 geographies) - rows [69, 147, 225, '...', 33921, 33999]
    group_32_mask = target_groups == 32
    group_32_targets = targets_df[group_32_mask].copy()
    group_32_targets["row_index"] = np.where(group_32_mask)[0]
    group_32_targets[
        [
            "target_id",
            "stratum_id",
            "value",
            "original_value",
            "geographic_id",
            "variable_desc",
            "uprating_factor",
            "reconciliation_factor",
        ]
    ]

    # Note that Medicaid reporting in the surveys can sometimes be higher than the administrative totals
    # Alabama is one of the states that has not expanded Medicaid under the Affordable Care Act (ACA).
    # People in the gap might confuse
    group_32_targets.reconciliation_factor.describe()

    cd_101_medicaid = group_32_targets[
        group_32_targets["geographic_id"] == "101"
    ]
    row_idx = cd_101_medicaid["row_index"].values[0]
    target_value = cd_101_medicaid["value"].values[0]

    medicaid_df = sim.calculate_dataframe(
        ["household_id", "medicaid"], map_to="household"
    )
    medicaid_households = medicaid_df[medicaid_df["medicaid"] > 0]

    test_hh = int(medicaid_households.iloc[0]["household_id"])
    medicaid_df.loc[medicaid_df.household_id == test_hh]
    positions = tracer.get_household_column_positions(test_hh)
    col_idx = positions["101"]
    matrix[row_idx, positions["101"]]  # Should be > 0
    matrix[row_idx, positions["102"]]  # Should be zero

    # But Medicaid is a person count concept. In this case, the number is 2.0
    hh_person_rel.loc[hh_person_rel.household_id == test_hh]

    person_medicaid_df = sim.calculate_dataframe(
        ["person_id", "medicaid", "medicaid_enrolled"], map_to="person"
    )
    person_medicaid_df.loc[person_medicaid_df.person_id.isin([56001, 56002])]
    # Note that it's medicaid_enrolled that we're counting for the metrics matrix.

    # Group 43: Tax Units qualified_business_income_deduction>0 (436 targets across 436 geographies) - rows [88, 166, 244, '...', 33940, 34018]
    # Note that this is the COUNT of > 0
    group_43_mask = target_groups == 43
    group_43_targets = targets_df[group_43_mask].copy()
    group_43_targets["row_index"] = np.where(group_43_mask)[0]
    group_43_targets[
        [
            "target_id",
            "stratum_id",
            "value",
            "original_value",
            "geographic_id",
            "variable_desc",
            "uprating_factor",
            "reconciliation_factor",
        ]
    ]

    cd_101_qbid = group_43_targets[group_43_targets["geographic_id"] == "101"]
    row_idx = cd_101_qbid["row_index"].values[0]
    target_value = cd_101_qbid["value"].values[0]

    qbid_df = sim.calculate_dataframe(
        ["household_id", "qualified_business_income_deduction"],
        map_to="household",
    )
    qbid_households = qbid_df[
        qbid_df["qualified_business_income_deduction"] > 0
    ]

    # Check matrix for a specific QBID household
    test_hh = int(qbid_households.iloc[0]["household_id"])
    positions = tracer.get_household_column_positions(test_hh)
    col_idx = positions["101"]
    matrix[row_idx, positions["101"]]  # Should be 1.0
    matrix[row_idx, positions["102"]]  # Should be zero

    qbid_df.loc[qbid_df.household_id == test_hh]
    hh_person_rel.loc[hh_person_rel.household_id == test_hh]

    # Group 66: Qualified Business Income Deduction (436 targets across 436 geographies) - rows [70, 148, 226, '...', 33922, 34000]
    # This is the amount!
    group_66_mask = target_groups == 66
    group_66_targets = targets_df[group_66_mask].copy()
    group_66_targets["row_index"] = np.where(group_66_mask)[0]
    group_66_targets[
        [
            "target_id",
            "stratum_id",
            "value",
            "original_value",
            "geographic_id",
            "variable_desc",
            "uprating_factor",
            "reconciliation_factor",
        ]
    ]

    cd_101_qbid_amount = group_66_targets[
        group_66_targets["geographic_id"] == "101"
    ]
    row_idx = cd_101_qbid_amount["row_index"].values[0]
    target_value = cd_101_qbid_amount["value"].values[0]

    matrix[row_idx, positions["101"]]  # Should > 1.0
    matrix[row_idx, positions["102"]]  # Should be zero

    # Group 60: Household Count (436 targets across 436 geographies) - rows [36, 114, 192, '...', 33888, 33966]
    group_60_mask = target_groups == 60
    group_60_targets = targets_df[group_60_mask].copy()
    group_60_targets["row_index"] = np.where(group_60_mask)[0]
    group_60_targets[
        [
            "target_id",
            "stratum_id",
            "value",
            "original_value",
            "geographic_id",
            "variable_desc",
            "uprating_factor",
            "reconciliation_factor",
        ]
    ]

    cd_101_snap = group_60_targets[group_60_targets["geographic_id"] == "101"]
    row_idx = cd_101_snap["row_index"].values[0]
    target_value = cd_101_snap["value"].values[0]

    # Find households with SNAP > 0
    snap_df = sim.calculate_dataframe(
        ["household_id", "snap"], map_to="household"
    )
    snap_households = snap_df[snap_df["snap"] > 0]

    # Check matrix for a specific SNAP household
    test_hh = int(snap_households.iloc[0]["household_id"])
    positions = tracer.get_household_column_positions(test_hh)
    col_idx = positions["101"]
    matrix[row_idx, positions["101"]]  # Should be > 0
    matrix[row_idx, positions["102"]]  # Should be zero

    # Check non-SNAP household
    non_snap_hh = snap_df[snap_df["snap"] == 0].iloc[0]["household_id"]
    non_snap_positions = tracer.get_household_column_positions(non_snap_hh)
    matrix[row_idx, non_snap_positions["101"]]  # should be 0

    # Group 73: Snap Cost at State Level (51 targets across 51 geographies) - rows 34038-34088   -----------
    group_73_mask = target_groups == 73
    group_73_targets = targets_df[group_73_mask].copy()
    group_73_targets["row_index"] = np.where(group_73_mask)[0]

    state_snap = group_73_targets[
        group_73_targets["geographic_id"] == "1"
    ]  # Delaware
    row_idx = state_snap["row_index"].values[0]
    target_value = state_snap["value"].values[0]

    snap_value = matrix[row_idx, col_idx]
    snap_value

    # AGI target exploration --------
    test_household = 565
    positions = tracer.get_household_column_positions(test_household)
    row_idx = 27268
    one_target = targets_df.iloc[row_idx]
    test_variable = one_target.variable
    print(one_target.variable_desc)
    print(one_target.value)

    # Get value for test household in CD 101
    matrix_hh_position = positions["101"]
    value_correct = matrix[row_idx, matrix_hh_position]
    print(f"Household {test_household} in CD 3910: {value_correct}")

    # Get value for same household but wrong CD (e.g., '1001')
    matrix_hh_position_1001 = positions["1001"]
    value_incorrect = matrix[row_idx_3910, matrix_hh_position_1001]
    print(f"Household {test_household} in CD 1001 (wrong!): {value_incorrect}")

    df = sim.calculate_dataframe(
        ["household_id", test_variable, "adjusted_gross_income"],
        map_to="household",
    )
    df.loc[df.household_id == test_household]

    # Row 78: Taxable Pension Income ---------------------------------------------------------
    group_78 = tracer.get_group_rows(78)
    cd_3910_target = group_78[group_78["geographic_id"] == "3910"]

    row_idx_3910 = cd_3910_target["row_index"].values[0]
    print(f"Taxable Pension Income for CD 3910 is at row {row_idx_3910}")

    # Check here ------
    targets_df.iloc[row_idx_3910]
    cd_3910_target

    test_variable = targets_df.iloc[row_idx_3910].variable

    # Get value for household in CD 3910
    matrix_hh_position_3910 = positions["3910"]
    value_correct = matrix[row_idx_3910, matrix_hh_position_3910]
    print(f"Household {test_household} in CD 3910: {value_correct}")

    # Get value for same household but wrong CD (e.g., '1001')
    matrix_hh_position_1001 = positions["1001"]
    value_incorrect = matrix[row_idx_3910, matrix_hh_position_1001]
    print(f"Household {test_household} in CD 1001 (wrong!): {value_incorrect}")

    df = sim.calculate_dataframe(
        ["household_id", test_variable], map_to="household"
    )
    df.loc[df.household_id == test_household][[test_variable]]

    df.loc[df[test_variable] > 0]

    # Get all target values
    all_values = tracer.trace_household_targets(test_household)
    print(f"\nFound values for {len(all_values)} targets")
    print(all_values.head())

    # Verify a specific target
    verification = tracer.verify_household_target(
        test_household, 0, test_cds[0]
    )
    print(f"\nVerification result: {verification}")

    # Full audit  (TODO: not working, or at least wasn't working, on *_count metrics and targets)
    audit = tracer.audit_household(test_household, max_targets=5)
    print(f"\nAudit summary: {audit['summary']}")


def h5_tracer():
    import pandas as pd
    from policyengine_us import Microsimulation

    # --- 1. Setup: Load simulations and mapping file ---

    # Paths to the datasets and mapping file
    new_dataset_path = "/home/baogorek/devl/policyengine-us-data/policyengine_us_data/datasets/cps/geo_stacking_calibration/temp/RI.h5"
    original_dataset_path = "/home/baogorek/devl/stratified_10k.h5"
    mapping_file_path = "./temp/RI_household_mapping.csv"

    # Initialize the two microsimulations
    sim_new = Microsimulation(dataset=new_dataset_path)
    sim_orig = Microsimulation(dataset=original_dataset_path)

    # Load the household ID mapping file
    mapping_df = pd.read_csv(mapping_file_path)

    # --- 2. Identify households for comparison ---

    # Specify the household ID from the NEW dataset to test
    test_hh_new = 2741169

    # Find the corresponding ORIGINAL household ID using the mapping file
    test_hh_orig = mapping_df.loc[
        mapping_df.new_household_id == test_hh_new
    ].original_household_id.values[0]

    print(
        f"Comparing new household '{test_hh_new}' with original household '{test_hh_orig}'\n"
    )

    # --- 3. Compare household-level data ---

    # Define the variables to analyze at the household level
    household_vars = [
        "household_id",
        "state_fips",
        "congressional_district_geoid",
        "adjusted_gross_income",
    ]

    # Calculate dataframes for both simulations
    df_new = sim_new.calculate_dataframe(household_vars, map_to="household")
    df_orig = sim_orig.calculate_dataframe(household_vars, map_to="household")

    # Filter for the specific households
    household_new_data = df_new.loc[df_new.household_id == test_hh_new]
    household_orig_data = df_orig.loc[df_orig.household_id == test_hh_orig]

    print("--- Household-Level Comparison ---")
    print("\nData from New Simulation (RI.h5):")
    print(household_new_data)
    print("\nData from Original Simulation (stratified_10k.h5):")
    print(household_orig_data)

    # --- 4. Compare person-level data ---

    # A helper function to create a person-level dataframe from a simulation
    def get_person_df(simulation):
        return pd.DataFrame(
            {
                "household_id": simulation.calculate(
                    "household_id", map_to="person"
                ),
                "person_id": simulation.calculate(
                    "person_id", map_to="person"
                ),
                "age": simulation.calculate("age", map_to="person"),
            }
        )

    # Get person-level dataframes
    df_person_new = get_person_df(sim_new)
    df_person_orig = get_person_df(sim_orig)

    # Filter for the members of the specific households
    persons_new = df_person_new.loc[df_person_new.household_id == test_hh_new]
    persons_orig = df_person_orig.loc[
        df_person_orig.household_id == test_hh_orig
    ]

    print("\n\n--- Person-Level Comparison ---")
    print("\nData from New Simulation (RI.h5):")
    print(persons_new)
    print("\nData from Original Simulation (stratified_10k.h5):")
    print(persons_orig)
