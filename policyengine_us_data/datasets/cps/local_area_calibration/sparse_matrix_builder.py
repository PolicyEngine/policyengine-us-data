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
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
    apply_op,
    _get_geo_level,
)


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
            raise ValueError(
                "target_filter must specify at least one filter criterion"
            )

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
        self, sim, target_filter: dict
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, Dict[str, List[str]]]:
        """
        Build sparse calibration matrix.

        Args:
            sim: Microsimulation instance (used for household_ids, or as template)
            target_filter: Dict specifying which targets to include
                - {"stratum_group_ids": [4]} for SNAP targets
                - {"target_ids": [123, 456]} for specific targets

        Returns:
            Tuple of (targets_df, X_sparse, household_id_mapping)
        """
        household_ids = sim.calculate(
            "household_id", map_to="household"
        ).values
        n_households = len(household_ids)
        n_cds = len(self.cds_to_calibrate)
        n_cols = n_households * n_cds

        targets_df = self._query_targets(target_filter)
        n_targets = len(targets_df)

        if n_targets == 0:
            raise ValueError("No targets found matching filter")

        targets_df["geographic_id"] = targets_df["stratum_id"].apply(
            self._get_geographic_id
        )

        # Sort by (geo_level, variable, geographic_id) for contiguous group rows
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

                    # Evaluate non-geographic constraints at entity level
                    entity_mask = self._evaluate_constraints_entity_aware(
                        state_sim, non_geo_constraints, n_households
                    )

                    # Combine geographic and entity-aware masks
                    mask = geo_mask & entity_mask

                    if not mask.any():
                        continue

                    target_values = state_sim.calculate(
                        target["variable"],
                        self.time_period,
                        map_to="household",
                    ).values
                    masked_values = (target_values * mask).astype(np.float32)

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
