"""
Sparse matrix builder for geo-stacking calibration.

Generic, database-driven approach where all constraints (including geographic)
are evaluated as masks. Geographic constraints work because we SET state_fips
before evaluating constraints.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sqlalchemy import create_engine, text

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

        cds_by_state = defaultdict(list)
        for cd_idx, cd in enumerate(self.cds_to_calibrate):
            state = int(cd) // 100
            cds_by_state[state].append((cd_idx, cd))

        for state, cd_list in cds_by_state.items():
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

                    mask = np.ones(n_households, dtype=bool)
                    for c in constraints:
                        if c["variable"] == "congressional_district_geoid":
                            if (
                                c["operation"] in ("==", "=")
                                and c["value"] != cd
                            ):
                                mask[:] = False
                        elif c["variable"] == "state_fips":
                            if (
                                c["operation"] in ("==", "=")
                                and int(c["value"]) != state
                            ):
                                mask[:] = False
                        else:
                            try:
                                values = state_sim.calculate(
                                    c["variable"], map_to="household"
                                ).values
                                mask &= apply_op(
                                    values, c["operation"], c["value"]
                                )
                            except Exception:
                                pass

                    if not mask.any():
                        continue

                    target_values = state_sim.calculate(
                        target["variable"], map_to="household"
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
