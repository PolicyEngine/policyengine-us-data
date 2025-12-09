"""
Matrix tracer utility for debugging geo-stacking sparse matrices.

This utility allows tracing through the complex stacked matrix structure
to verify values match simulation results.

USAGE
=====

Basic Setup:

    from matrix_tracer import MatrixTracer

    tracer = MatrixTracer(
        targets_df, X_sparse, household_id_mapping,
        cds_to_calibrate, sim
    )

Common Operations:

    # 1. Understand what a column represents
    col_info = tracer.get_column_info(100)

    # 2. Find where a household appears across all CDs
    positions = tracer.get_household_column_positions(565)

    # 3. View matrix structure
    tracer.print_matrix_structure()

Matrix Structure:

    Columns are organized as: [CD1_households | CD2_households | ... | CD436_households]
    Each CD block has n_households columns (e.g., 10,580 households)

    Formula to find column index:
        column_idx = cd_block_number * n_households + household_index
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import sparse

from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    create_target_groups,
)


logger = logging.getLogger(__name__)


class MatrixTracer:
    """Trace through geo-stacked sparse matrices for debugging."""

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
            var_name = target["variable"]
            var_desc = ""
            if var_name in self.sim.tax_benefit_system.variables:
                var_obj = self.sim.tax_benefit_system.variables[var_name]
                var_desc = getattr(var_obj, "label", var_name)

            catalog.append(
                {
                    "row_index": row_idx,
                    "variable": var_name,
                    "variable_desc": var_desc,
                    "geographic_id": target.get("geographic_id", "unknown"),
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

    def print_matrix_structure(self, show_groups=True):
        """Print a comprehensive breakdown of the matrix structure."""
        print("\n" + "=" * 80)
        print("MATRIX STRUCTURE BREAKDOWN")
        print("=" * 80)

        print(
            f"\nMatrix dimensions: {self.matrix.shape[0]} rows x "
            f"{self.matrix.shape[1]} columns"
        )
        print(f"  Rows = {len(self.row_catalog)} targets")
        print(
            f"  Columns = {self.n_households} households x "
            f"{self.n_geographies} CDs"
        )
        print(
            f"           = {self.n_households:,} x {self.n_geographies} "
            f"= {self.matrix.shape[1]:,}"
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
                }
            )
            cumulative += self.n_households

        ranges_df = pd.DataFrame(col_ranges)
        print(f"\nShowing first and last 5 CDs of {len(ranges_df)} total:")
        print("\nFirst 5 CDs:")
        print(ranges_df.head(5).to_string(index=False))
        print("\nLast 5 CDs:")
        print(ranges_df.tail(5).to_string(index=False))

        print("\n" + "-" * 80)
        print("ROW STRUCTURE (Targets)")
        print("-" * 80)

        print(f"\nTotal targets: {len(self.row_catalog)}")

        # Summarize by geographic level if column exists
        if "geographic_level" in self.row_catalog.columns:
            print("\nTargets by geographic level:")
            geo_level_summary = (
                self.row_catalog.groupby("geographic_level")
                .size()
                .reset_index(name="n_targets")
            )
            print(geo_level_summary.to_string(index=False))

        print("\nTargets by stratum group:")
        stratum_summary = (
            self.row_catalog.groupby("stratum_group_id")
            .agg({"row_index": "count", "variable": lambda x: len(set(x))})
            .rename(
                columns={"row_index": "n_targets", "variable": "n_unique_vars"}
            )
        )
        print(stratum_summary.to_string())

        # Create and display target groups with row indices
        if show_groups:
            print("\n" + "-" * 80)
            print("TARGET GROUPS (for loss calculation)")
            print("-" * 80)

            target_groups, group_info = create_target_groups(self.targets_df)

            # Store for later use
            self.target_groups = target_groups

            # Print each group with row indices
            for group_id, info in enumerate(group_info):
                group_mask = target_groups == group_id
                row_indices = np.where(group_mask)[0]

                # Format row indices for display
                if len(row_indices) > 6:
                    row_display = (
                        f"[{row_indices[0]}, {row_indices[1]}, "
                        f"{row_indices[2]}, ..., {row_indices[-2]}, "
                        f"{row_indices[-1]}]"
                    )
                else:
                    row_display = str(row_indices.tolist())

                print(f"  {info} - rows {row_display}")

        print("\n" + "=" * 80)

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

    def get_group_rows(self, group_id: int) -> pd.DataFrame:
        """
        Get all rows belonging to a specific target group.

        Args:
            group_id: The group ID to filter by

        Returns:
            DataFrame of row catalog entries for this group
        """
        if not hasattr(self, "target_groups"):
            self.target_groups, self.group_info = create_target_groups(
                self.targets_df
            )

        group_mask = self.target_groups == group_id
        return self.row_catalog[group_mask].copy()

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
