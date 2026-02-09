"""
Unified sparse matrix builder for calibration.

Builds a sparse calibration matrix for cloned+geography-assigned CPS
records. Processes state-by-state for memory efficiency, reusing
base record calculations for all clones assigned to each state.

Matrix shape: (n_targets, n_records * n_clones)
Column ordering: index i = clone_idx * n_records + record_idx
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sqlalchemy import text

from policyengine_us_data.calibration.base_matrix_builder import (
    BaseMatrixBuilder,
)
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
)

logger = logging.getLogger(__name__)

# Geographic constraint variables
_GEO_STATE_VARS = {"state_fips", "state_code"}
_GEO_CD_VARS = {"congressional_district_geoid"}
_GEO_VARS = _GEO_STATE_VARS | _GEO_CD_VARS

# Synthetic constraint variables to skip
_SYNTHETIC_CONSTRAINT_VARS = {"target_category"}

# Count variables (value = 1 per entity satisfying constraints)
COUNT_VARIABLES = {
    "person_count",
    "tax_unit_count",
    "household_count",
    "spm_unit_count",
}


class UnifiedMatrixBuilder(BaseMatrixBuilder):
    """Build sparse calibration matrix for cloned CPS records.

    Each column represents one cloned record with assigned
    geography.  Geographic constraints are checked against the
    assignment, not against simulation state.

    Args:
        db_uri: SQLAlchemy-style database URI.
        time_period: Tax year for the calibration (e.g. 2024).
    """

    def __init__(self, db_uri: str, time_period: int):
        super().__init__(db_uri, time_period)

    # ------------------------------------------------------------------
    # Database queries
    # ------------------------------------------------------------------

    def _query_active_targets(self) -> pd.DataFrame:
        """Query all active, non-zero targets.

        Returns:
            DataFrame with columns: target_id, stratum_id,
            variable, value, period, reform_id, tolerance,
            stratum_group_id, stratum_notes, target_notes.
        """
        query = """
        SELECT t.target_id,
               t.stratum_id,
               t.variable,
               t.value,
               t.period,
               t.reform_id,
               t.tolerance,
               t.notes   AS target_notes,
               s.stratum_group_id,
               s.notes   AS stratum_notes
        FROM targets t
        JOIN strata s ON t.stratum_id = s.stratum_id
        WHERE t.active = 1
        ORDER BY t.target_id
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        # Filter out zero/null target values.
        df = df[
            df["value"].notna()
            & ~np.isclose(df["value"].values, 0.0, atol=0.1)
        ].reset_index(drop=True)
        return df

    def _get_all_constraints(self, stratum_id: int) -> List[dict]:
        """Get constraints for stratum and all ancestors.

        Walks up the ``parent_stratum_id`` chain, collecting
        constraints from each level.  Filters out synthetic
        constraints.

        Args:
            stratum_id: Starting stratum whose full constraint
                chain is needed.

        Returns:
            List of constraint dicts with keys ``variable``,
            ``operation``, ``value``.
        """
        all_constraints: List[dict] = []
        visited: set = set()
        current_id: Optional[int] = stratum_id

        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            constraints = self._get_stratum_constraints(current_id)
            all_constraints.extend(constraints)

            query = """
            SELECT parent_stratum_id FROM strata
            WHERE stratum_id = :sid
            """
            with self.engine.connect() as conn:
                row = conn.execute(
                    text(query), {"sid": int(current_id)}
                ).fetchone()
            current_id = row[0] if row else None

        return [
            c
            for c in all_constraints
            if c["variable"] not in _SYNTHETIC_CONSTRAINT_VARS
        ]

    # ------------------------------------------------------------------
    # Geographic classification
    # ------------------------------------------------------------------

    def _classify_constraint_geo(
        self, constraints: List[dict]
    ) -> Tuple[str, Optional[str]]:
        """Classify geographic level and ID from constraints.

        CD-level takes priority over state-level (a CD target
        always has a parent state constraint too).

        Args:
            constraints: Full constraint list for a target.

        Returns:
            Tuple of ``(geo_level, geo_id)`` where geo_level is
            ``"national"``, ``"state"``, or ``"cd"``, and geo_id
            is the FIPS/GEOID string or ``None`` for national.
        """
        # Check CD first (highest specificity).
        for c in constraints:
            if c["variable"] in _GEO_CD_VARS:
                return "cd", str(c["value"])
        # Then state.
        for c in constraints:
            if c["variable"] in _GEO_STATE_VARS:
                return "state", str(int(float(c["value"])))
        return "national", None

    # ------------------------------------------------------------------
    # State simulation factory
    # ------------------------------------------------------------------

    def _create_state_sim(
        self,
        state_fips: int,
        dataset_path: str,
        n_records: int,
    ):
        """Create simulation with all records set to given state.

        Args:
            state_fips: State FIPS code.
            dataset_path: Path to the base extended CPS h5 file.
            n_records: Number of base household records.

        Returns:
            Microsimulation instance with state overridden and
            calculated variables cleared for recalculation.
        """
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=dataset_path)
        sim.set_input(
            "state_fips",
            self.time_period,
            np.full(n_records, state_fips, dtype=np.int32),
        )
        for var in get_calculated_variables(sim):
            sim.delete_arrays(var)
        return sim

    # ------------------------------------------------------------------
    # Target name generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_target_name(
        variable: str,
        constraints: List[dict],
        reform_id: int = 0,
    ) -> str:
        """Generate human-readable target label.

        Args:
            variable: Target variable name.
            constraints: Resolved constraints for the stratum.
            reform_id: Reform identifier (0 = baseline).

        Returns:
            A slash-separated label string.
        """
        parts: List[str] = []

        # Geographic level
        geo_parts: List[str] = []
        for c in constraints:
            if c["variable"] == "state_fips":
                geo_parts.append(f"state_{c['value']}")
            elif c["variable"] == "state_code":
                geo_parts.append(f"state_{c['value']}")
            elif c["variable"] == "congressional_district_geoid":
                geo_parts.append(f"cd_{c['value']}")

        if geo_parts:
            parts.append("/".join(geo_parts))
        else:
            parts.append("national")

        if reform_id > 0:
            parts.append(f"{variable}_expenditure")
        else:
            parts.append(variable)

        # Non-geo constraint summary
        non_geo = [
            c
            for c in constraints
            if c["variable"]
            not in (
                "state_fips",
                "state_code",
                "congressional_district_geoid",
            )
        ]
        if non_geo:
            constraint_strs = [
                f"{c['variable']}{c['operation']}{c['value']}" for c in non_geo
            ]
            parts.append("[" + ",".join(constraint_strs) + "]")

        return "/".join(parts)

    # ------------------------------------------------------------------
    # Main build method
    # ------------------------------------------------------------------

    def build_matrix(
        self,
        dataset_path: str,
        geography,
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]:
        """Build sparse calibration matrix.

        Args:
            dataset_path: Path to the base extended CPS h5 file.
            geography: Geography assignment object with
                ``state_fips``, ``cd_geoid`` arrays and
                ``n_records``, ``n_clones`` attributes.

        Returns:
            Tuple of ``(targets_df, X_sparse, target_names)``
            where:

            - **targets_df** -- DataFrame of target metadata.
            - **X_sparse** -- sparse CSR matrix of shape
              ``(n_targets, n_records * n_clones)``.
            - **target_names** -- list of human-readable labels.
        """
        n_records = geography.n_records
        n_clones = geography.n_clones
        n_total = n_records * n_clones

        # Build index structures: state -> column indices,
        # CD -> column indices.
        state_to_cols: Dict[int, list] = defaultdict(list)
        cd_to_cols: Dict[str, list] = defaultdict(list)

        for col in range(n_total):
            s = int(geography.state_fips[col])
            cd = str(geography.cd_geoid[col])
            state_to_cols[s].append(col)
            cd_to_cols[cd].append(col)

        # Convert to numpy arrays for vectorized indexing.
        state_to_cols_np: Dict[int, np.ndarray] = {
            s: np.array(cols) for s, cols in state_to_cols.items()
        }
        cd_to_cols_np: Dict[str, np.ndarray] = {
            cd: np.array(cols) for cd, cols in cd_to_cols.items()
        }

        # Query targets from database.
        targets_df = self._query_active_targets()
        n_targets = len(targets_df)

        logger.info(
            "Building unified matrix: %d targets, %d total "
            "columns (%d records x %d clones)",
            n_targets,
            n_total,
            n_records,
            n_clones,
        )

        # Pre-process: resolve constraints, classify geo, build
        # names, separate geo/non-geo constraints.
        constraint_cache: Dict[int, List[dict]] = {}
        target_constraints: List[List[dict]] = []
        target_geo_info: List[Tuple[str, Optional[str]]] = []
        target_names: List[str] = []
        non_geo_constraints_list: List[List[dict]] = []

        for _, row in targets_df.iterrows():
            sid = int(row["stratum_id"])
            if sid not in constraint_cache:
                constraint_cache[sid] = self._get_all_constraints(sid)
            constraints = constraint_cache[sid]
            target_constraints.append(constraints)

            geo_level, geo_id = self._classify_constraint_geo(constraints)
            target_geo_info.append((geo_level, geo_id))

            non_geo = [
                c for c in constraints if c["variable"] not in _GEO_VARS
            ]
            non_geo_constraints_list.append(non_geo)

            target_names.append(
                self._make_target_name(
                    str(row["variable"]),
                    constraints,
                    reform_id=int(row["reform_id"]),
                )
            )

        # Initialize sparse matrix in LIL format for efficient
        # row-by-row construction.
        X = sparse.lil_matrix((n_targets, n_total), dtype=np.float32)

        # Process state-by-state.
        unique_states = sorted(state_to_cols_np.keys())
        for state in unique_states:
            logger.info("Processing state %d...", state)
            # Clear entity relationship cache for new sim.
            self._entity_rel_cache = None

            state_sim = self._create_state_sim(state, dataset_path, n_records)
            state_cols = state_to_cols_np[state]

            for row_idx in range(n_targets):
                target = targets_df.iloc[row_idx]
                geo_level, geo_id = target_geo_info[row_idx]
                non_geo = non_geo_constraints_list[row_idx]
                variable = str(target["variable"])

                # Determine which columns this target applies to
                # within this state's columns.
                if geo_level == "cd":
                    if geo_id not in cd_to_cols_np:
                        continue
                    cd_cols = cd_to_cols_np[geo_id]
                    target_cols = np.intersect1d(cd_cols, state_cols)
                elif geo_level == "state":
                    if int(geo_id) != state:
                        continue
                    target_cols = state_cols
                else:
                    # National: all columns in this state.
                    target_cols = state_cols

                if len(target_cols) == 0:
                    continue

                # Evaluate non-geographic constraints.
                mask = self._evaluate_constraints_entity_aware(
                    state_sim, non_geo, n_records
                )

                # Calculate target variable values.
                try:
                    if variable in COUNT_VARIABLES:
                        values = mask.astype(np.float32)
                    else:
                        values = state_sim.calculate(
                            variable,
                            self.time_period,
                            map_to="household",
                        ).values.astype(np.float32)
                        values = values * mask
                except Exception as exc:
                    logger.warning(
                        "Cannot calculate '%s': %s",
                        variable,
                        exc,
                    )
                    continue

                # Fill matrix columns: column i uses values
                # from record i % n_records.
                rec_idx = target_cols % n_records
                vals = values[rec_idx]
                nonzero = vals != 0
                if nonzero.any():
                    X[row_idx, target_cols[nonzero]] = vals[nonzero]

        X_csr = X.tocsr()
        logger.info(
            "Matrix built: %d targets x %d columns, " "%d nonzero entries",
            X_csr.shape[0],
            X_csr.shape[1],
            X_csr.nnz,
        )
        return targets_df, X_csr, target_names
