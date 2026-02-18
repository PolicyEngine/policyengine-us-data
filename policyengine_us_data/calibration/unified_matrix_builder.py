"""
Unified sparse matrix builder for clone-based calibration.

Builds a sparse calibration matrix for cloned+geography-assigned CPS
records. Processes clone-by-clone: for each clone, sets each
record's state_fips to its assigned value, simulates, and extracts
variable values.

Matrix shape: (n_targets, n_records * n_clones)
Column ordering: index i = clone_idx * n_records + record_idx
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sqlalchemy import create_engine, text

from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.utils.census import STATE_NAME_TO_FIPS
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
    apply_op,
    get_geo_level,
)

logger = logging.getLogger(__name__)

_GEO_VARS = {
    "state_fips",
    "state_code",
    "congressional_district_geoid",
}


class UnifiedMatrixBuilder:
    """Build sparse calibration matrix for cloned CPS records.

    Processes clone-by-clone: each clone's records get their
    assigned geography, are simulated, and the results fill
    the corresponding columns.

    Args:
        db_uri: SQLAlchemy database URI.
        time_period: Tax year for calibration (e.g. 2024).
        dataset_path: Path to the base extended CPS h5 file.
    """

    def __init__(
        self,
        db_uri: str,
        time_period: int,
        dataset_path: Optional[str] = None,
    ):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.time_period = time_period
        self.dataset_path = dataset_path
        self._entity_rel_cache = None

    # ---------------------------------------------------------------
    # Entity relationships
    # ---------------------------------------------------------------

    def _build_entity_relationship(self, sim) -> pd.DataFrame:
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

    # ---------------------------------------------------------------
    # Constraint evaluation
    # ---------------------------------------------------------------

    def _evaluate_constraints_entity_aware(
        self,
        sim,
        constraints: List[dict],
        n_households: int,
    ) -> np.ndarray:
        """Evaluate constraints at person level, aggregate to
        household level via .any()."""
        if not constraints:
            return np.ones(n_households, dtype=bool)

        entity_rel = self._build_entity_relationship(sim)
        n_persons = len(entity_rel)
        person_mask = np.ones(n_persons, dtype=bool)

        for c in constraints:
            try:
                vals = sim.calculate(
                    c["variable"],
                    self.time_period,
                    map_to="person",
                ).values
            except Exception as exc:
                logger.warning(
                    "Cannot evaluate constraint '%s': %s",
                    c["variable"],
                    exc,
                )
                return np.zeros(n_households, dtype=bool)
            person_mask &= apply_op(vals, c["operation"], c["value"])

        df = entity_rel.copy()
        df["satisfies"] = person_mask
        hh_mask = df.groupby("household_id")["satisfies"].any()

        household_ids = sim.calculate(
            "household_id", map_to="household"
        ).values
        return np.array([hh_mask.get(hid, False) for hid in household_ids])

    # ---------------------------------------------------------------
    # Database queries
    # ---------------------------------------------------------------

    def _get_stratum_constraints(self, stratum_id: int) -> List[dict]:
        query = """
        SELECT constraint_variable AS variable, operation, value
        FROM stratum_constraints
        WHERE stratum_id = :stratum_id
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"stratum_id": int(stratum_id)},
            )
        return df.to_dict("records")

    def _query_targets(self, target_filter: dict) -> pd.DataFrame:
        """Query targets via target_overview view with
        best-period selection."""
        or_conditions = []

        if "domain_variables" in target_filter:
            dvs = target_filter["domain_variables"]
            ph = ",".join(f"'{dv}'" for dv in dvs)
            or_conditions.append(f"tv.domain_variable IN ({ph})")

        if "variables" in target_filter:
            vs = ",".join(f"'{v}'" for v in target_filter["variables"])
            or_conditions.append(f"tv.variable IN ({vs})")

        if "target_ids" in target_filter:
            ids = ",".join(map(str, target_filter["target_ids"]))
            or_conditions.append(f"tv.target_id IN ({ids})")

        if "stratum_ids" in target_filter:
            ids = ",".join(map(str, target_filter["stratum_ids"]))
            or_conditions.append(f"tv.stratum_id IN ({ids})")

        if not or_conditions:
            where_clause = "1=1"
        else:
            where_clause = " OR ".join(f"({c})" for c in or_conditions)

        query = f"""
        WITH filtered_targets AS (
            SELECT tv.target_id, tv.stratum_id, tv.variable,
                   tv.value, tv.period, tv.geo_level,
                   tv.geographic_id, tv.domain_variable
            FROM target_overview tv
            WHERE {where_clause}
        ),
        best_periods AS (
            SELECT stratum_id, variable,
                CASE
                    WHEN MAX(CASE WHEN period <= :time_period
                             THEN period END) IS NOT NULL
                    THEN MAX(CASE WHEN period <= :time_period
                             THEN period END)
                    ELSE MIN(period)
                END as best_period
            FROM filtered_targets
            GROUP BY stratum_id, variable
        )
        SELECT ft.*
        FROM filtered_targets ft
        JOIN best_periods bp
            ON ft.stratum_id = bp.stratum_id
            AND ft.variable = bp.variable
            AND ft.period = bp.best_period
        ORDER BY ft.target_id
        """

        with self.engine.connect() as conn:
            return pd.read_sql(
                query,
                conn,
                params={"time_period": self.time_period},
            )

    # ---------------------------------------------------------------
    # Uprating
    # ---------------------------------------------------------------

    def _calculate_uprating_factors(self, params) -> dict:
        factors = {}
        query = (
            "SELECT DISTINCT period FROM targets "
            "WHERE period IS NOT NULL ORDER BY period"
        )
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            years_needed = [row[0] for row in result]

        for from_year in years_needed:
            if from_year == self.time_period:
                factors[(from_year, "cpi")] = 1.0
                factors[(from_year, "pop")] = 1.0
                continue

            try:
                cpi_from = params.gov.bls.cpi.cpi_u(from_year)
                cpi_to = params.gov.bls.cpi.cpi_u(self.time_period)
                factors[(from_year, "cpi")] = float(cpi_to / cpi_from)
            except Exception:
                factors[(from_year, "cpi")] = 1.0

            try:
                pop_from = params.calibration.gov.census.populations.total(
                    from_year
                )
                pop_to = params.calibration.gov.census.populations.total(
                    self.time_period
                )
                factors[(from_year, "pop")] = float(pop_to / pop_from)
            except Exception:
                factors[(from_year, "pop")] = 1.0

        return factors

    def _get_uprating_info(
        self,
        variable: str,
        period: int,
        factors: dict,
    ) -> Tuple[float, str]:
        if period == self.time_period:
            return 1.0, "none"

        count_indicators = [
            "count",
            "person",
            "people",
            "households",
            "tax_units",
        ]
        is_count = any(ind in variable.lower() for ind in count_indicators)
        uprating_type = "pop" if is_count else "cpi"
        factor = factors.get((period, uprating_type), 1.0)
        return factor, uprating_type

    def _load_aca_ptc_factors(
        self,
    ) -> Dict[int, Dict[str, float]]:
        csv_path = STORAGE_FOLDER / "aca_ptc_multipliers_2022_2024.csv"
        df = pd.read_csv(csv_path)
        result = {}
        for _, row in df.iterrows():
            fips_str = STATE_NAME_TO_FIPS.get(row["state"])
            if fips_str is None:
                continue
            fips_int = int(fips_str)
            result[fips_int] = {
                "tax_unit_count": row["vol_mult"],
                "aca_ptc": row["vol_mult"] * row["val_mult"],
            }
        return result

    def _get_state_uprating_factors(
        self,
        domain: str,
        targets_df: pd.DataFrame,
        national_factors: dict,
    ) -> Dict[int, Dict[str, float]]:
        state_rows = targets_df[
            (targets_df["domain_variable"] == domain)
            & (targets_df["geo_level"] == "state")
        ]
        state_fips_list = state_rows["geographic_id"].unique()
        variables = state_rows["variable"].unique()

        if domain == "aca_ptc":
            csv_factors = self._load_aca_ptc_factors()
        else:
            csv_factors = None

        result = {}
        for sf in state_fips_list:
            state_int = int(sf)
            var_factors = {}

            if csv_factors and state_int in csv_factors:
                for var in variables:
                    var_factors[var] = csv_factors[state_int].get(var, 1.0)
            else:
                for var in variables:
                    row = state_rows[
                        (state_rows["geographic_id"] == sf)
                        & (state_rows["variable"] == var)
                    ]
                    if row.empty:
                        var_factors[var] = 1.0
                        continue
                    period = row.iloc[0]["period"]
                    factor, _ = self._get_uprating_info(
                        var, period, national_factors
                    )
                    var_factors[var] = factor

            result[state_int] = var_factors

        return result

    def _apply_hierarchical_uprating(
        self,
        targets_df: pd.DataFrame,
        hierarchical_domains: List[str],
        national_factors: dict,
    ) -> pd.DataFrame:
        """Apply state-level uprating and reconcile CDs.

        Two factors per CD row:
        - hif: state_original / sum(cd_originals)
        - uprating_factor: state-specific scaling

        Final CD value = original * hif * uprating_factor.
        """
        df = targets_df.copy()
        df["hif"] = np.nan
        df["state_uprating_factor"] = np.nan
        rows_to_drop = []

        for domain in hierarchical_domains:
            domain_mask = df["domain_variable"] == domain
            state_factors = self._get_state_uprating_factors(
                domain, df, national_factors
            )
            state_mask = domain_mask & (df["geo_level"] == "state")
            district_mask = domain_mask & (df["geo_level"] == "district")

            for sf, var_factors in state_factors.items():
                for var, uf in var_factors.items():
                    state_row = df[
                        state_mask
                        & (df["geographic_id"] == str(sf))
                        & (df["variable"] == var)
                    ]
                    if state_row.empty:
                        continue
                    state_original = state_row.iloc[0]["original_value"]

                    def _cd_in_state(g, s=sf):
                        try:
                            return int(g) // 100 == s
                        except (ValueError, TypeError):
                            return False

                    cd_mask = (
                        district_mask
                        & (df["variable"] == var)
                        & df["geographic_id"].apply(_cd_in_state)
                    )
                    cd_rows = df[cd_mask]
                    if cd_rows.empty:
                        continue

                    cd_original_sum = cd_rows["original_value"].sum()
                    if cd_original_sum == 0:
                        continue

                    hif = state_original / cd_original_sum
                    for cd_idx in cd_rows.index:
                        df.at[cd_idx, "hif"] = hif
                        df.at[cd_idx, "state_uprating_factor"] = uf
                        df.at[cd_idx, "value"] = (
                            df.at[cd_idx, "original_value"] * hif * uf
                        )

            # Drop national/state rows used for reconciliation
            national_mask = domain_mask & (df["geo_level"] == "national")
            for idx in df[national_mask | state_mask].index:
                row = df.loc[idx]
                if row["period"] != self.time_period:
                    rows_to_drop.append(idx)

        if rows_to_drop:
            df = df.drop(index=rows_to_drop).reset_index(drop=True)

        df["target_period"] = self.time_period
        return df

    def print_uprating_summary(self, targets_df: pd.DataFrame) -> None:
        has_state_uf = "state_uprating_factor" in targets_df.columns
        if has_state_uf:
            eff = targets_df["state_uprating_factor"].fillna(
                targets_df["uprating_factor"]
            )
        else:
            eff = targets_df["uprating_factor"]

        uprated = targets_df[eff != 1.0]
        if len(uprated) == 0:
            print("No targets were uprated.")
            return

        print("\n" + "=" * 60)
        print("UPRATING SUMMARY")
        print("=" * 60)
        print(f"Uprated {len(uprated)} of " f"{len(targets_df)} targets")
        period_counts = uprated["period"].value_counts().sort_index()
        for period, count in period_counts.items():
            print(f"  Period {period}: {count} targets")
        factors = eff[eff != 1.0]
        print(
            f"  Factor range: [{factors.min():.4f}, " f"{factors.max():.4f}]"
        )

    # ---------------------------------------------------------------
    # Target naming
    # ---------------------------------------------------------------

    @staticmethod
    def _make_target_name(
        variable: str,
        constraints: List[dict],
        reform_id: int = 0,
    ) -> str:
        geo_parts: List[str] = []
        for c in constraints:
            if c["variable"] == "state_fips":
                geo_parts.append(f"state_{c['value']}")
            elif c["variable"] == "congressional_district_geoid":
                geo_parts.append(f"cd_{c['value']}")

        parts: List[str] = []
        parts.append("/".join(geo_parts) if geo_parts else "national")
        if reform_id > 0:
            parts.append(f"{variable}_expenditure")
        else:
            parts.append(variable)

        non_geo = [c for c in constraints if c["variable"] not in _GEO_VARS]
        if non_geo:
            strs = [
                f"{c['variable']}{c['operation']}{c['value']}" for c in non_geo
            ]
            parts.append("[" + ",".join(strs) + "]")

        return "/".join(parts)

    # ---------------------------------------------------------------
    # Target value calculation
    # ---------------------------------------------------------------

    def _calculate_target_values(
        self,
        sim,
        target_variable: str,
        non_geo_constraints: List[dict],
        n_households: int,
    ) -> np.ndarray:
        """Calculate per-household target values.

        For count targets (*_count): count entities per HH
        satisfying constraints.
        For value targets: multiply values by constraint mask.
        """
        is_count = target_variable.endswith("_count")

        if not is_count:
            mask = self._evaluate_constraints_entity_aware(
                sim, non_geo_constraints, n_households
            )
            vals = sim.calculate(target_variable, map_to="household").values
            return (vals * mask).astype(np.float32)

        # Count target: entity-aware counting
        entity_rel = self._build_entity_relationship(sim)
        n_persons = len(entity_rel)
        person_mask = np.ones(n_persons, dtype=bool)

        for c in non_geo_constraints:
            try:
                cv = sim.calculate(c["variable"], map_to="person").values
            except Exception:
                return np.zeros(n_households, dtype=np.float32)
            person_mask &= apply_op(cv, c["operation"], c["value"])

        target_entity = sim.tax_benefit_system.variables[
            target_variable
        ].entity.key
        household_ids = sim.calculate(
            "household_id", map_to="household"
        ).values

        if target_entity == "household":
            if non_geo_constraints:
                mask = self._evaluate_constraints_entity_aware(
                    sim, non_geo_constraints, n_households
                )
                return mask.astype(np.float32)
            return np.ones(n_households, dtype=np.float32)

        if target_entity == "person":
            er = entity_rel.copy()
            er["satisfies"] = person_mask
            filtered = er[er["satisfies"]]
            counts = filtered.groupby("household_id")["person_id"].nunique()
        else:
            eid_col = f"{target_entity}_id"
            er = entity_rel.copy()
            er["satisfies"] = person_mask
            entity_ok = er.groupby(eid_col)["satisfies"].any()
            unique = er[["household_id", eid_col]].drop_duplicates()
            unique["entity_ok"] = unique[eid_col].map(entity_ok)
            filtered = unique[unique["entity_ok"]]
            counts = filtered.groupby("household_id")[eid_col].nunique()

        return np.array(
            [counts.get(hid, 0) for hid in household_ids],
            dtype=np.float32,
        )

    # ---------------------------------------------------------------
    # Clone simulation
    # ---------------------------------------------------------------

    def _simulate_clone(
        self,
        clone_state_fips: np.ndarray,
        n_records: int,
        variables: set,
        sim_modifier=None,
        post_sim_modifier=None,
        clone_idx: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], object]:
        """Simulate one clone with assigned geography.

        Args:
            clone_state_fips: State FIPS per record, shape
                (n_records,).
            n_records: Number of base records.
            variables: Target variable names to compute.
            sim_modifier: Optional callback(sim, clone_idx)
                called after state_fips is set but before
                cache clearing. Used for simple takeup
                re-randomization.
            post_sim_modifier: Optional callback(sim, clone_idx)
                called after the first cache clear. Used for
                category-dependent takeup re-randomization
                that needs sim-calculated category variables.
                Triggers a second cache clear afterwards.
            clone_idx: Clone index passed to modifiers.

        Returns:
            (var_values, sim) where var_values maps variable
            name to household-level float32 array.
        """
        from policyengine_us import Microsimulation

        sim = Microsimulation(dataset=self.dataset_path)
        sim.set_input(
            "state_fips",
            self.time_period,
            clone_state_fips.astype(np.int32),
        )
        if sim_modifier is not None:
            sim_modifier(sim, clone_idx)
        for var in get_calculated_variables(sim):
            sim.delete_arrays(var)

        # Two-pass: category-dependent takeup needs fresh
        # category variables, then a second cache clear
        if post_sim_modifier is not None:
            post_sim_modifier(sim, clone_idx)
            for var in get_calculated_variables(sim):
                sim.delete_arrays(var)

        var_values: Dict[str, np.ndarray] = {}
        for var in variables:
            if var.endswith("_count"):
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

    # ---------------------------------------------------------------
    # Main build method
    # ---------------------------------------------------------------

    def build_matrix(
        self,
        geography,
        sim,
        target_filter: Optional[dict] = None,
        hierarchical_domains: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        sim_modifier=None,
        post_sim_modifier=None,
    ) -> Tuple[pd.DataFrame, sparse.csr_matrix, List[str]]:
        """Build sparse calibration matrix.

        Two-phase build: (1) simulate each clone and save
        COO entries to disk, (2) assemble CSR from caches.

        Args:
            geography: GeographyAssignment with state_fips,
                cd_geoid, block_geoid arrays and n_records,
                n_clones attributes.
            sim: Microsimulation for parameters and entity
                relationships.
            target_filter: Dict for target_overview filtering.
            hierarchical_domains: Domain names for
                hierarchical uprating + CD reconciliation.
            cache_dir: Directory for per-clone COO caches.
                If None, COO data held in memory.
            sim_modifier: Optional callback(sim, clone_idx)
                called per clone after state_fips is set but
                before cache clearing. Use for simple takeup
                re-randomization.
            post_sim_modifier: Optional callback(sim, clone_idx)
                called after the first cache clear for
                category-dependent takeup re-randomization.

        Returns:
            (targets_df, X_sparse, target_names)
        """
        n_records = geography.n_records
        n_clones = geography.n_clones
        n_total = n_records * n_clones
        self._coo_parts = ([], [], [])

        # 1. Query and uprate targets
        targets_df = self._query_targets(target_filter or {})
        if len(targets_df) == 0:
            raise ValueError("No targets found matching filter")

        params = sim.tax_benefit_system.parameters
        uprating_factors = self._calculate_uprating_factors(params)
        targets_df["original_value"] = targets_df["value"].copy()
        targets_df["uprating_factor"] = targets_df.apply(
            lambda row: self._get_uprating_info(
                row["variable"],
                row["period"],
                uprating_factors,
            )[0],
            axis=1,
        )
        targets_df["value"] = (
            targets_df["original_value"] * targets_df["uprating_factor"]
        )

        if hierarchical_domains:
            targets_df = self._apply_hierarchical_uprating(
                targets_df,
                hierarchical_domains,
                uprating_factors,
            )

        n_targets = len(targets_df)

        # 2. Sort targets by geographic level
        targets_df["_geo_level"] = targets_df["geographic_id"].apply(
            get_geo_level
        )
        targets_df = targets_df.sort_values(
            ["_geo_level", "variable", "geographic_id"]
        )
        targets_df = targets_df.drop(columns=["_geo_level"]).reset_index(
            drop=True
        )

        # 3. Build column index structures from geography
        state_col_lists: Dict[int, list] = defaultdict(list)
        cd_col_lists: Dict[str, list] = defaultdict(list)
        for col in range(n_total):
            state_col_lists[int(geography.state_fips[col])].append(col)
            cd_col_lists[str(geography.cd_geoid[col])].append(col)
        state_to_cols = {s: np.array(c) for s, c in state_col_lists.items()}
        cd_to_cols = {cd: np.array(c) for cd, c in cd_col_lists.items()}

        # 4. Pre-process targets: resolve constraints
        constraint_cache: Dict[int, List[dict]] = {}
        target_geo_info: List[Tuple[str, str]] = []
        target_names: List[str] = []
        non_geo_constraints_list: List[List[dict]] = []

        for _, row in targets_df.iterrows():
            sid = int(row["stratum_id"])
            if sid not in constraint_cache:
                constraint_cache[sid] = self._get_stratum_constraints(sid)
            constraints = constraint_cache[sid]

            geo_level = row["geo_level"]
            geo_id = row["geographic_id"]
            target_geo_info.append((geo_level, geo_id))

            non_geo = [
                c for c in constraints if c["variable"] not in _GEO_VARS
            ]
            non_geo_constraints_list.append(non_geo)

            target_names.append(
                self._make_target_name(str(row["variable"]), constraints)
            )

        unique_variables = set(targets_df["variable"].values)

        # 5. Clone loop
        from pathlib import Path

        clone_dir = Path(cache_dir) if cache_dir else None
        if clone_dir:
            clone_dir.mkdir(parents=True, exist_ok=True)

        self._entity_rel_cache = None

        for clone_idx in range(n_clones):
            if clone_dir:
                coo_path = clone_dir / f"clone_{clone_idx:04d}.npz"
                if coo_path.exists():
                    logger.info(
                        "Clone %d/%d cached, skipping.",
                        clone_idx + 1,
                        n_clones,
                    )
                    continue

            col_start = clone_idx * n_records
            col_end = col_start + n_records
            clone_states = geography.state_fips[col_start:col_end]

            logger.info(
                "Processing clone %d/%d " "(cols %d-%d, %d unique states)...",
                clone_idx + 1,
                n_clones,
                col_start,
                col_end - 1,
                len(np.unique(clone_states)),
            )

            var_values, clone_sim = self._simulate_clone(
                clone_states,
                n_records,
                unique_variables,
                sim_modifier=sim_modifier,
                post_sim_modifier=post_sim_modifier,
                clone_idx=clone_idx,
            )

            mask_cache: Dict[tuple, np.ndarray] = {}
            count_cache: Dict[tuple, np.ndarray] = {}

            rows_list: list = []
            cols_list: list = []
            vals_list: list = []

            for row_idx in range(n_targets):
                variable = str(targets_df.iloc[row_idx]["variable"])
                geo_level, geo_id = target_geo_info[row_idx]
                non_geo = non_geo_constraints_list[row_idx]

                # Geographic column selection
                if geo_level == "district":
                    all_geo_cols = cd_to_cols.get(
                        str(geo_id),
                        np.array([], dtype=np.int64),
                    )
                elif geo_level == "state":
                    all_geo_cols = state_to_cols.get(
                        int(geo_id),
                        np.array([], dtype=np.int64),
                    )
                else:
                    all_geo_cols = np.arange(n_total)

                clone_cols = all_geo_cols[
                    (all_geo_cols >= col_start) & (all_geo_cols < col_end)
                ]
                if len(clone_cols) == 0:
                    continue

                rec_indices = clone_cols - col_start

                constraint_key = tuple(
                    sorted(
                        (
                            c["variable"],
                            c["operation"],
                            c["value"],
                        )
                        for c in non_geo
                    )
                )

                if variable.endswith("_count"):
                    vkey = (variable, constraint_key)
                    if vkey not in count_cache:
                        count_cache[vkey] = self._calculate_target_values(
                            clone_sim,
                            variable,
                            non_geo,
                            n_records,
                        )
                    values = count_cache[vkey]
                else:
                    if variable not in var_values:
                        continue
                    if constraint_key not in mask_cache:
                        mask_cache[constraint_key] = (
                            self._evaluate_constraints_entity_aware(
                                clone_sim,
                                non_geo,
                                n_records,
                            )
                        )
                    mask = mask_cache[constraint_key]
                    values = var_values[variable] * mask

                vals = values[rec_indices]
                nonzero = vals != 0
                if nonzero.any():
                    rows_list.append(
                        np.full(
                            nonzero.sum(),
                            row_idx,
                            dtype=np.int32,
                        )
                    )
                    cols_list.append(clone_cols[nonzero].astype(np.int32))
                    vals_list.append(vals[nonzero])

            # Save COO entries
            if rows_list:
                cr = np.concatenate(rows_list)
                cc = np.concatenate(cols_list)
                cv = np.concatenate(vals_list)
            else:
                cr = np.array([], dtype=np.int32)
                cc = np.array([], dtype=np.int32)
                cv = np.array([], dtype=np.float32)

            if clone_dir:
                np.savez_compressed(
                    str(coo_path),
                    rows=cr,
                    cols=cc,
                    vals=cv,
                )
                logger.info(
                    "Clone %d: %d nonzero entries saved.",
                    clone_idx + 1,
                    len(cv),
                )
                del var_values, clone_sim
            else:
                self._coo_parts[0].append(cr)
                self._coo_parts[1].append(cc)
                self._coo_parts[2].append(cv)

        # 6. Assemble sparse matrix from COO data
        logger.info("Assembling matrix from %d clones...", n_clones)
        if clone_dir:
            all_r, all_c, all_v = [], [], []
            for ci in range(n_clones):
                p = clone_dir / f"clone_{ci:04d}.npz"
                data = np.load(str(p))
                all_r.append(data["rows"])
                all_c.append(data["cols"])
                all_v.append(data["vals"])
            rows = np.concatenate(all_r)
            cols = np.concatenate(all_c)
            vals = np.concatenate(all_v)
        else:
            rows = np.concatenate(self._coo_parts[0])
            cols = np.concatenate(self._coo_parts[1])
            vals = np.concatenate(self._coo_parts[2])
            del self._coo_parts

        X_csr = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_targets, n_total),
            dtype=np.float32,
        )

        logger.info(
            "Matrix: %d targets x %d cols, %d nnz",
            X_csr.shape[0],
            X_csr.shape[1],
            X_csr.nnz,
        )

        return targets_df, X_csr, target_names
