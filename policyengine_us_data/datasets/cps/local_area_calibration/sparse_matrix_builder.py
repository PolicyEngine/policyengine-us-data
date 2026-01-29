"""
Sparse matrix builder for geo-stacking calibration.

Generic, database-driven approach where all constraints (including geographic)
are evaluated as masks. Geographic constraints work because we SET state_fips
before evaluating constraints.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from dataclasses import dataclass
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
    apply_op,
    _get_geo_level,
    build_concept_id,
    extract_constraints_from_row,
)


@dataclass
class ConceptDuplicateWarning:
    """Warning when multiple values exist for the same concept."""

    concept_id: str
    duplicates: List[dict]
    selected: dict
    reason: str


class SparseMatrixBuilder:
    """Build sparse calibration matrices for geo-stacking."""

    def __init__(
        self,
        db_uri: str,
        time_period: int,
        cds_to_calibrate: List[str],
        dataset_path: Optional[str] = None,
    ):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.time_period = time_period
        self.cds_to_calibrate = cds_to_calibrate
        self.dataset_path = dataset_path
        self._entity_rel_cache = None

        # Populated after build_matrix() with deduplicate=True
        self.concept_summary: Optional[pd.DataFrame] = None
        self.dedup_warnings: List[ConceptDuplicateWarning] = []
        self.targets_before_dedup: Optional[pd.DataFrame] = None
        self.targets_after_dedup: Optional[pd.DataFrame] = None

    def _build_entity_relationship(self, sim) -> pd.DataFrame:
        """
        Build entity relationship DataFrame mapping persons to all entity IDs.

        This is used to evaluate constraints at the person level and then
        aggregate to household level, handling variables defined at different
        entity levels (person, tax_unit, household, spm_unit).

        Returns:
            DataFrame with person_id, household_id, tax_unit_id, spm_unit_id
        """
        if self._entity_rel_cache is not None:
            return self._entity_rel_cache

        self._entity_rel_cache = pd.DataFrame(
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
            }
        )
        return self._entity_rel_cache

    def _evaluate_constraints_entity_aware(
        self, state_sim, constraints: List[dict], n_households: int
    ) -> np.ndarray:
        """
        Evaluate non-geographic constraints at person level, aggregate to
        household level using .any().

        This properly handles constraints on variables defined at different
        entity levels (e.g., tax_unit_is_filer at tax_unit level). Instead of
        summing values at household level (which would give 2, 3, etc. for
        households with multiple tax units), we evaluate at person level and
        use .any() aggregation ("does this household have at least one person
        satisfying all constraints?").

        Args:
            state_sim: Microsimulation with state_fips set
            constraints: List of constraint dicts with variable, operation,
                value keys (geographic constraints should be pre-filtered)
            n_households: Number of households

        Returns:
            Boolean mask array of length n_households
        """
        if not constraints:
            return np.ones(n_households, dtype=bool)

        entity_rel = self._build_entity_relationship(state_sim)
        n_persons = len(entity_rel)

        person_mask = np.ones(n_persons, dtype=bool)

        for c in constraints:
            var = c["variable"]
            op = c["operation"]
            val = c["value"]

            # Calculate constraint variable at person level
            constraint_values = state_sim.calculate(
                var, self.time_period, map_to="person"
            ).values

            # Apply operation at person level
            person_mask &= apply_op(constraint_values, op, val)

        # Aggregate to household level using .any()
        # "At least one person in this household satisfies ALL constraints"
        entity_rel_with_mask = entity_rel.copy()
        entity_rel_with_mask["satisfies"] = person_mask

        household_mask_series = entity_rel_with_mask.groupby("household_id")[
            "satisfies"
        ].any()

        # Ensure we return a mask aligned with household order
        household_ids = state_sim.calculate(
            "household_id", map_to="household"
        ).values
        household_mask = np.array(
            [
                household_mask_series.get(hh_id, False)
                for hh_id in household_ids
            ]
        )

        return household_mask

    def _calculate_target_values_entity_aware(
        self,
        state_sim,
        target_variable: str,
        non_geo_constraints: List[dict],
        geo_mask: np.ndarray,
        n_households: int,
    ) -> np.ndarray:
        """
        Calculate target values at household level, handling count targets.

        For count targets (*_count): Count entities per household satisfying
            constraints
        For value targets: Sum values at household level (existing behavior)

        Args:
            state_sim: Microsimulation with state_fips set
            target_variable: The target variable name (e.g., "snap",
                "person_count")
            non_geo_constraints: List of constraint dicts (geographic
                constraints should be pre-filtered)
            geo_mask: Boolean mask array for geographic filtering (household
                level)
            n_households: Number of households

        Returns:
            Float array of target values at household level
        """
        is_count_target = target_variable.endswith("_count")

        if not is_count_target:
            # Value target: use existing entity-aware constraint evaluation
            entity_mask = self._evaluate_constraints_entity_aware(
                state_sim, non_geo_constraints, n_households
            )
            mask = geo_mask & entity_mask

            target_values = state_sim.calculate(
                target_variable, map_to="household"
            ).values
            return (target_values * mask).astype(np.float32)

        # Count target: need to count entities satisfying constraints
        entity_rel = self._build_entity_relationship(state_sim)
        n_persons = len(entity_rel)

        # Evaluate constraints at person level (don't aggregate to HH yet)
        person_mask = np.ones(n_persons, dtype=bool)
        for c in non_geo_constraints:
            constraint_values = state_sim.calculate(
                c["variable"], map_to="person"
            ).values
            person_mask &= apply_op(
                constraint_values, c["operation"], c["value"]
            )

        # Get target entity from variable definition
        target_entity = state_sim.tax_benefit_system.variables[
            target_variable
        ].entity.key

        household_ids = state_sim.calculate(
            "household_id", map_to="household"
        ).values
        geo_mask_map = dict(zip(household_ids, geo_mask))

        if target_entity == "household":
            # household_count: 1 per qualifying household
            if non_geo_constraints:
                entity_mask = self._evaluate_constraints_entity_aware(
                    state_sim, non_geo_constraints, n_households
                )
                return (geo_mask & entity_mask).astype(np.float32)
            return geo_mask.astype(np.float32)

        if target_entity == "person":
            # Count persons satisfying constraints per household
            entity_rel["satisfies"] = person_mask
            entity_rel["geo_ok"] = entity_rel["household_id"].map(geo_mask_map)
            filtered = entity_rel[
                entity_rel["satisfies"] & entity_rel["geo_ok"]
            ]
            counts = filtered.groupby("household_id")["person_id"].nunique()
        else:
            # For tax_unit, spm_unit: aggregate person mask to entity, then
            # count
            entity_id_col = f"{target_entity}_id"
            entity_rel["satisfies"] = person_mask
            entity_satisfies = entity_rel.groupby(entity_id_col)[
                "satisfies"
            ].any()

            entity_rel_unique = entity_rel[
                ["household_id", entity_id_col]
            ].drop_duplicates()
            entity_rel_unique["entity_ok"] = entity_rel_unique[
                entity_id_col
            ].map(entity_satisfies)
            entity_rel_unique["geo_ok"] = entity_rel_unique[
                "household_id"
            ].map(geo_mask_map)
            filtered = entity_rel_unique[
                entity_rel_unique["entity_ok"] & entity_rel_unique["geo_ok"]
            ]
            counts = filtered.groupby("household_id")[entity_id_col].nunique()

        # Build result aligned with household order
        return np.array(
            [counts.get(hh_id, 0) for hh_id in household_ids], dtype=np.float32
        )

    def _query_targets(self, target_filter: dict) -> pd.DataFrame:
        """Query targets based on filter criteria using OR logic."""
        or_conditions = []

        if "stratum_group_ids" in target_filter:
            ids = ",".join(map(str, target_filter["stratum_group_ids"]))
            or_conditions.append(f"s.stratum_group_id IN ({ids})")

        if "variables" in target_filter:
            vars_str = ",".join(f"'{v}'" for v in target_filter["variables"])
            or_conditions.append(f"t.variable IN ({vars_str})")

        if "target_ids" in target_filter:
            ids = ",".join(map(str, target_filter["target_ids"]))
            or_conditions.append(f"t.target_id IN ({ids})")

        if "stratum_ids" in target_filter:
            ids = ",".join(map(str, target_filter["stratum_ids"]))
            or_conditions.append(f"t.stratum_id IN ({ids})")

        if not or_conditions:
            # No filter criteria: fetch all targets
            where_clause = "1=1"
        else:
            where_clause = " OR ".join(f"({c})" for c in or_conditions)

        query = f"""
        SELECT t.target_id, t.stratum_id, t.variable, t.value, t.period,
               s.stratum_group_id
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        WHERE {where_clause}
        ORDER BY t.target_id
        """

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    def _get_constraints(self, stratum_id: int) -> List[dict]:
        """Get all constraints for a stratum (including geographic)."""
        query = """
        SELECT constraint_variable as variable, operation, value
        FROM stratum_constraints
        WHERE stratum_id = :stratum_id
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"stratum_id": stratum_id})
        return df.to_dict("records")

    def _get_geographic_id(self, stratum_id: int) -> str:
        """Extract geographic_id from constraints for targets_df."""
        constraints = self._get_constraints(stratum_id)
        for c in constraints:
            if c["variable"] == "state_fips":
                return c["value"]
            if c["variable"] == "congressional_district_geoid":
                return c["value"]
        return "US"

    def _get_constraint_info(self, stratum_id: int) -> str:
        """Build pipe-separated constraint string for concept identification."""
        constraints = self._get_constraints(stratum_id)
        parts = []
        for c in constraints:
            op = "==" if c["operation"] == "=" else c["operation"]
            parts.append(f"{c['variable']}{op}{c['value']}")
        return "|".join(parts) if parts else None

    def _deduplicate_targets(
        self,
        targets_df: pd.DataFrame,
        mode: str = "within_geography",
        priority_column: str = "geo_priority",
    ) -> pd.DataFrame:
        """
        Deduplicate targets by concept before matrix building.

        Stores results in instance attributes for later inspection:
        - self.concept_summary: DataFrame summarizing concepts
        - self.dedup_warnings: List of ConceptDuplicateWarning
        - self.targets_before_dedup: Original targets DataFrame
        - self.targets_after_dedup: Deduplicated targets DataFrame

        Args:
            targets_df: DataFrame with target rows including geographic_id
                and constraint_info columns
            mode: Deduplication mode ("within_geography" or
                "hierarchical_fallback")
            priority_column: Column to sort by when selecting among
                duplicates. Lower values = higher priority.

        Returns:
            Deduplicated DataFrame with reset index
        """
        df = targets_df.copy()

        # Add geo_priority if not present (CD=1, State=2, National=3)
        if priority_column not in df.columns:
            df["geo_priority"] = df["geographic_id"].apply(
                lambda g: 3 if g == "US" else (1 if int(g) >= 100 else 2)
            )
            priority_column = "geo_priority"

        # Build concept_id for each row
        df["_concept_id"] = df.apply(
            lambda row: build_concept_id(
                row["variable"],
                extract_constraints_from_row(row, exclude_geo=True),
            ),
            axis=1,
        )

        # Store concept summary
        self.concept_summary = df.groupby("_concept_id").agg(
            count=("_concept_id", "size"),
            variable=("variable", "first"),
            geos=("geographic_id", lambda x: list(x.unique())),
        )

        # Store original for comparison
        self.targets_before_dedup = df.copy()

        # Determine deduplication key based on mode
        if mode == "within_geography":
            if "geographic_id" not in df.columns:
                raise ValueError(
                    "Mode 'within_geography' requires 'geographic_id' column"
                )
            dedupe_key = ["_concept_id", "geographic_id"]
        elif mode == "hierarchical_fallback":
            dedupe_key = ["_concept_id"]
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'within_geography' or "
                "'hierarchical_fallback'"
            )

        # Find and process duplicates
        warnings = []
        duplicate_mask = df.duplicated(subset=dedupe_key, keep=False)
        duplicates_df = df[duplicate_mask]

        if len(duplicates_df) > 0:
            for key_vals, group in duplicates_df.groupby(dedupe_key):
                if len(group) <= 1:
                    continue

                dup_list = []
                for _, dup_row in group.iterrows():
                    dup_list.append(
                        {
                            "geographic_id": dup_row.get("geographic_id", "?"),
                            "source": dup_row.get("source_name", "?"),
                            "period": dup_row.get("period", "?"),
                            "value": dup_row.get("value", "?"),
                            "stratum_id": dup_row.get("stratum_id", "?"),
                        }
                    )

                sorted_group = group.sort_values(priority_column)
                selected_row = sorted_group.iloc[0]
                selected = {
                    "geographic_id": selected_row.get("geographic_id", "?"),
                    "source": selected_row.get("source_name", "?"),
                    "period": selected_row.get("period", "?"),
                    "value": selected_row.get("value", "?"),
                }

                concept_id = (
                    key_vals if isinstance(key_vals, str) else key_vals[0]
                )
                warnings.append(
                    ConceptDuplicateWarning(
                        concept_id=concept_id,
                        duplicates=dup_list,
                        selected=selected,
                        reason=f"Selected by lowest {priority_column}",
                    )
                )

        self.dedup_warnings = warnings

        # Deduplicate: sort by key + priority, keep first per key
        sort_cols = (
            dedupe_key + [priority_column]
            if priority_column in df.columns
            else dedupe_key
        )
        df_sorted = df.sort_values(sort_cols)
        df_deduped = df_sorted.drop_duplicates(subset=dedupe_key, keep="first")

        # Clean up temporary column
        df_deduped = df_deduped.drop(columns=["_concept_id"])

        self.targets_after_dedup = df_deduped.copy()

        return df_deduped.reset_index(drop=True)

    def print_concept_summary(self) -> None:
        """
        Print detailed concept summary from the last build_matrix() call.

        Call this after build_matrix() to see what concepts were found.
        """
        if self.concept_summary is None:
            print("No concept summary available. Run build_matrix() first.")
            return

        print("\n" + "=" * 60)
        print("CONCEPT SUMMARY")
        print("=" * 60)

        n_targets = (
            len(self.targets_before_dedup)
            if self.targets_before_dedup is not None
            else 0
        )
        print(
            f"Found {len(self.concept_summary)} unique concepts "
            f"from {n_targets} targets:\n"
        )

        for concept_id, row in self.concept_summary.iterrows():
            n_geos = len(row["geos"])
            print(f"  {concept_id}")
            print(
                f"    Variable: {row['variable']}, "
                f"Targets: {row['count']}, Geographies: {n_geos}"
            )

    def print_dedup_summary(self) -> None:
        """
        Print deduplication summary from the last build_matrix() call.

        Call this after build_matrix() to see what duplicates were removed.
        """
        if self.targets_before_dedup is None:
            print("No dedup summary available. Run build_matrix() first.")
            return

        print("\n" + "=" * 60)
        print("DEDUPLICATION SUMMARY")
        print("=" * 60)

        before = len(self.targets_before_dedup)
        after = (
            len(self.targets_after_dedup)
            if self.targets_after_dedup is not None
            else 0
        )
        removed = before - after

        print(f"Total targets queried: {before}")
        print(f"Targets after deduplication: {after}")
        print(f"Duplicates removed: {removed}")

        if self.dedup_warnings:
            print(f"\nDuplicate groups resolved ({len(self.dedup_warnings)}):")
            for w in self.dedup_warnings:
                print(f"\n  Concept: {w.concept_id}")
                sel_val = w.selected["value"]
                sel_val_str = (
                    f"{sel_val:,.0f}"
                    if isinstance(sel_val, (int, float))
                    else str(sel_val)
                )
                print(
                    f"    Selected: geo={w.selected['geographic_id']}, "
                    f"value={sel_val_str}"
                )
                print(f"    Removed ({len(w.duplicates) - 1}):")
                for dup in w.duplicates:
                    if (
                        dup["value"] != w.selected["value"]
                        or dup["geographic_id"] != w.selected["geographic_id"]
                    ):
                        dup_val = dup["value"]
                        dup_val_str = (
                            f"{dup_val:,.0f}"
                            if isinstance(dup_val, (int, float))
                            else str(dup_val)
                        )
                        print(
                            f"      - geo={dup['geographic_id']}, "
                            f"value={dup_val_str}, "
                            f"source={dup.get('source', '?')}"
                        )

    def _create_state_sim(self, state: int, n_households: int):
        """Create a fresh simulation with state_fips set to given state."""
        from policyengine_us import Microsimulation

        state_sim = Microsimulation(dataset=self.dataset_path)
        state_sim.set_input(
            "state_fips",
            self.time_period,
            np.full(n_households, state, dtype=np.int32),
        )
        for var in get_calculated_variables(state_sim):
            state_sim.delete_arrays(var)
        return state_sim

    def build_matrix(
        self,
        sim,
        target_filter: dict,
        deduplicate: bool = True,
        dedup_mode: str = "within_geography",
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, Dict[str, List[str]]]:
        """
        Build sparse calibration matrix.

        Args:
            sim: Microsimulation instance (used for household_ids, or
                as template)
            target_filter: Dict specifying which targets to include
                - {"stratum_group_ids": [4]} for SNAP targets
                - {"target_ids": [123, 456]} for specific targets
                - an empty dict {} will fetch all targets
            deduplicate: If True, deduplicate targets by concept before
                building the matrix (default True)
            dedup_mode: Deduplication mode - "within_geography" (default)
                removes duplicates with same concept AND geography, or
                "hierarchical_fallback" keeps most specific geography
                per concept

        Returns:
            Tuple of (targets_df, X_sparse, household_id_mapping)

        After calling this method, you can use print_concept_summary() and
        print_dedup_summary() to see details about concepts and deduplication.
        """
        household_ids = sim.calculate(
            "household_id", map_to="household"
        ).values
        n_households = len(household_ids)
        n_cds = len(self.cds_to_calibrate)
        n_cols = n_households * n_cds

        targets_df = self._query_targets(target_filter)

        if len(targets_df) == 0:
            raise ValueError("No targets found matching filter")

        targets_df["geographic_id"] = targets_df["stratum_id"].apply(
            self._get_geographic_id
        )
        targets_df["constraint_info"] = targets_df["stratum_id"].apply(
            self._get_constraint_info
        )

        # Deduplicate targets by concept before building matrix
        if deduplicate:
            targets_df = self._deduplicate_targets(targets_df, mode=dedup_mode)

        n_targets = len(targets_df)

        # Sort by (geo_level, variable, geographic_id) for contiguous group
        targets_df["_geo_level"] = targets_df["geographic_id"].apply(
            _get_geo_level
        )
        targets_df = targets_df.sort_values(
            ["_geo_level", "variable", "geographic_id"]
        )
        targets_df = targets_df.drop(columns=["_geo_level"]).reset_index(
            drop=True
        )

        X = sparse.lil_matrix((n_targets, n_cols), dtype=np.float32)

        # Group CDs by state. CD GEOIDs follow format SSCCC where SS is state
        # FIPS (2 digits) and CCC is CD number (2-3 digits), so state = CD // 100
        cds_by_state = defaultdict(list)
        for cd_idx, cd in enumerate(self.cds_to_calibrate):
            state = int(cd) // 100
            cds_by_state[state].append((cd_idx, cd))

        for state, cd_list in cds_by_state.items():
            # Clear entity relationship cache when creating new simulation
            self._entity_rel_cache = None

            if self.dataset_path:
                state_sim = self._create_state_sim(state, n_households)
            else:
                state_sim = sim
                state_sim.set_input(
                    "state_fips",
                    self.time_period,
                    np.full(n_households, state, dtype=np.int32),
                )
                for var in get_calculated_variables(state_sim):
                    state_sim.delete_arrays(var)

            for cd_idx, cd in cd_list:
                col_start = cd_idx * n_households

                for row_idx, (_, target) in enumerate(targets_df.iterrows()):
                    constraints = self._get_constraints(target["stratum_id"])

                    geo_constraints = []
                    non_geo_constraints = []
                    for c in constraints:
                        if c["variable"] in (
                            "state_fips",
                            "congressional_district_geoid",
                        ):
                            geo_constraints.append(c)
                        else:
                            non_geo_constraints.append(c)

                    # Check geographic constraints first (quick fail)
                    geo_mask = np.ones(n_households, dtype=bool)
                    for c in geo_constraints:
                        if c["variable"] == "congressional_district_geoid":
                            if (
                                c["operation"] in ("==", "=")
                                and c["value"] != cd
                            ):
                                geo_mask[:] = False
                        elif c["variable"] == "state_fips":
                            if (
                                c["operation"] in ("==", "=")
                                and int(c["value"]) != state
                            ):
                                geo_mask[:] = False

                    if not geo_mask.any():
                        continue

                    # Calculate target values with entity-aware handling
                    # This properly handles count targets (*_count) by counting
                    # entities rather than summing values
                    masked_values = self._calculate_target_values_entity_aware(
                        state_sim,
                        target["variable"],
                        non_geo_constraints,
                        geo_mask,
                        n_households,
                    )

                    if not masked_values.any():
                        continue

                    nonzero = np.where(masked_values != 0)[0]
                    if len(nonzero) > 0:
                        X[row_idx, col_start + nonzero] = masked_values[
                            nonzero
                        ]

        household_id_mapping = {}
        for cd in self.cds_to_calibrate:
            key = f"cd{cd}"
            household_id_mapping[key] = [
                f"{hh_id}_{key}" for hh_id in household_ids
            ]

        return targets_df, X.tocsr(), household_id_mapping
