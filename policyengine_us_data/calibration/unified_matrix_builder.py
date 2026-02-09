"""
Unified sparse matrix builder for calibration.

Builds a sparse calibration matrix for cloned+geography-assigned CPS
records. Processes clone-by-clone: for each clone, sets each
record's state_fips to its assigned value, simulates, and extracts
variable values. Every simulation result is used.

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

    Processes clone-by-clone: each clone's 111K records get their
    assigned geography, are simulated, and the results fill the
    corresponding columns.  This ensures state-dependent variables
    (state income tax, state benefits) are correct for the assigned
    geography.

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
    # Clone simulation
    # ------------------------------------------------------------------

    def _simulate_clone(
        self,
        dataset_path: str,
        clone_block_geoid: np.ndarray,
        clone_cd_geoid: np.ndarray,
        clone_state_fips: np.ndarray,
        n_records: int,
        variables: set,
        constraint_keys: set,
    ) -> Tuple[Dict[str, np.ndarray], "Microsimulation"]:
        """Simulate one clone with assigned geography.

        Loads the base dataset, overrides all geographic inputs
        derived from the assigned census block, clears calculated
        variables, and computes all needed target/constraint
        variables.

        Args:
            dataset_path: Path to the base extended CPS h5 file.
            clone_block_geoid: Block GEOID (15-char str) for each
                record, shape ``(n_records,)``.
            clone_cd_geoid: Congressional district GEOID for each
                record, shape ``(n_records,)``.
            clone_state_fips: State FIPS for each record,
                shape ``(n_records,)``.
            n_records: Number of base records.
            variables: Set of target variable names to compute.
            constraint_keys: Set of constraint variable names
                needed for mask evaluation.

        Returns:
            Tuple of ``(var_values, sim)`` where var_values maps
            variable name to household-level float32 array.
        """
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=dataset_path)
        sim.default_calculation_period = self.time_period

        # Override all geography from assigned census block.
        sim.set_input(
            "block_geoid",
            self.time_period,
            clone_block_geoid,
        )
        sim.set_input(
            "state_fips",
            self.time_period,
            clone_state_fips.astype(np.int32),
        )
        sim.set_input(
            "congressional_district_geoid",
            self.time_period,
            clone_cd_geoid.astype(np.int64),
        )
        # County FIPS = first 5 chars of block GEOID.
        county_fips = np.array([b[:5] for b in clone_block_geoid])
        sim.set_input(
            "county_fips",
            self.time_period,
            county_fips,
        )

        # Clear calculated variables so they recompute with
        # new geography.
        for var in get_calculated_variables(sim):
            sim.delete_arrays(var)

        # Calculate all target variables.
        var_values: Dict[str, np.ndarray] = {}
        for var in variables:
            if var in COUNT_VARIABLES:
                continue
            try:
                var_values[var] = sim.calculate(
                    var,
                    self.time_period,
                    map_to="household",
                ).values.astype(np.float32)
            except Exception as exc:
                logger.warning("Cannot calculate '%s': %s", var, exc)

        return var_values, sim

    # ------------------------------------------------------------------
    # Main build method
    # ------------------------------------------------------------------

    def build_matrix(
        self,
        dataset_path: str,
        geography,
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]:
        """Build sparse calibration matrix.

        Processes clone-by-clone: for each clone, sets geography
        on the base records, simulates, and fills the
        corresponding matrix columns. This ensures
        state-dependent variables are calculated correctly for
        each record's assigned geography.

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

        # Build column index structures from geography.
        state_to_cols: Dict[int, np.ndarray] = {}
        cd_to_cols: Dict[str, np.ndarray] = {}

        state_col_lists: Dict[int, list] = defaultdict(list)
        cd_col_lists: Dict[str, list] = defaultdict(list)
        for col in range(n_total):
            state_col_lists[int(geography.state_fips[col])].append(col)
            cd_col_lists[str(geography.cd_geoid[col])].append(col)
        state_to_cols = {s: np.array(c) for s, c in state_col_lists.items()}
        cd_to_cols = {cd: np.array(c) for cd, c in cd_col_lists.items()}

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

        # Pre-process targets: resolve constraints, classify geo.
        constraint_cache: Dict[int, List[dict]] = {}
        target_geo_info: List[Tuple[str, Optional[str]]] = []
        target_names: List[str] = []
        non_geo_constraints_list: List[List[dict]] = []

        for _, row in targets_df.iterrows():
            sid = int(row["stratum_id"])
            if sid not in constraint_cache:
                constraint_cache[sid] = self._get_all_constraints(sid)
            constraints = constraint_cache[sid]

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

        # Collect all variables and constraint variables needed.
        unique_variables = set(targets_df["variable"].values)
        constraint_vars = set()
        for non_geo in non_geo_constraints_list:
            for c in non_geo:
                constraint_vars.add(c["variable"])

        # Initialize sparse matrix in LIL format.
        X = sparse.lil_matrix((n_targets, n_total), dtype=np.float32)

        # Process clone-by-clone.
        for clone_idx in range(n_clones):
            col_start = clone_idx * n_records
            col_end = col_start + n_records

            # Geography for this clone's records.
            clone_blocks = geography.block_geoid[col_start:col_end]
            clone_cds = geography.cd_geoid[col_start:col_end]
            clone_states = geography.state_fips[col_start:col_end]

            logger.info(
                "Processing clone %d / %d "
                "(cols %d-%d, %d unique states)...",
                clone_idx + 1,
                n_clones,
                col_start,
                col_end - 1,
                len(np.unique(clone_states)),
            )

            # Simulate this clone with full geography.
            self._entity_rel_cache = None
            var_values, sim = self._simulate_clone(
                dataset_path=dataset_path,
                clone_block_geoid=clone_blocks,
                clone_cd_geoid=clone_cds,
                clone_state_fips=clone_states,
                n_records=n_records,
                variables=unique_variables,
                constraint_keys=constraint_vars,
            )

            # Pre-compute constraint masks for this clone.
            mask_cache: Dict[tuple, np.ndarray] = {}

            def _get_mask(
                constraints_list: List[dict],
            ) -> np.ndarray:
                key = tuple(
                    (c["variable"], c["operation"], c["value"])
                    for c in sorted(
                        constraints_list,
                        key=lambda c: c["variable"],
                    )
                )
                if key not in mask_cache:
                    mask_cache[key] = self._evaluate_constraints_entity_aware(
                        sim, constraints_list, n_records
                    )
                return mask_cache[key]

            # Fill matrix for each target.
            for row_idx in range(n_targets):
                variable = str(targets_df.iloc[row_idx]["variable"])
                geo_level, geo_id = target_geo_info[row_idx]
                non_geo = non_geo_constraints_list[row_idx]

                # Determine which records in this clone match
                # the target's geography.
                if geo_level == "cd":
                    if geo_id not in cd_to_cols:
                        continue
                    cd_cols = cd_to_cols[geo_id]
                    # Intersect with this clone's columns.
                    clone_target_cols = cd_cols[
                        (cd_cols >= col_start) & (cd_cols < col_end)
                    ]
                elif geo_level == "state":
                    state_key = int(geo_id)
                    if state_key not in state_to_cols:
                        continue
                    s_cols = state_to_cols[state_key]
                    clone_target_cols = s_cols[
                        (s_cols >= col_start) & (s_cols < col_end)
                    ]
                else:
                    # National: all records in this clone.
                    clone_target_cols = np.arange(col_start, col_end)

                if len(clone_target_cols) == 0:
                    continue

                # Get constraint mask.
                mask = _get_mask(non_geo)

                # Get variable values.
                if variable in COUNT_VARIABLES:
                    values = mask.astype(np.float32)
                elif variable in var_values:
                    values = var_values[variable] * mask
                else:
                    continue

                # Map columns to base record indices.
                rec_idx = clone_target_cols % n_records
                vals = values[rec_idx]
                nonzero = vals != 0
                if nonzero.any():
                    X[
                        row_idx,
                        clone_target_cols[nonzero],
                    ] = vals[nonzero]

        X_csr = X.tocsr()
        logger.info(
            "Matrix built: %d targets x %d columns, " "%d nonzero entries",
            X_csr.shape[0],
            X_csr.shape[1],
            X_csr.nnz,
        )
        return targets_df, X_csr, target_names
