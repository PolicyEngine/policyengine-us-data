"""
National matrix builder for calibration.

Reads ALL active targets from policy_data.db and builds a dense loss
matrix for the full Extended CPS dataset (~200k households).  This
replaces the legacy ``build_loss_matrix()`` in
``policyengine_us_data/utils/loss.py``.

The builder evaluates stratum constraints (geographic, demographic,
filing-status, AGI band, variable-specific) to produce boolean masks,
then computes target variable values under those masks.

Tax expenditure targets (reform_id > 0) trigger counterfactual
simulations that neutralize specific deduction variables so the
matrix column captures the income_tax difference.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from policyengine_us_data.calibration.base_matrix_builder import (
    BaseMatrixBuilder,
)
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    apply_op,
)

logger = logging.getLogger(__name__)

# Variables that indicate the target is a *count* of entities
# (value = 1 per entity satisfying constraints) rather than a sum.
COUNT_VARIABLES = {
    "person_count",
    "tax_unit_count",
    "household_count",
    "spm_unit_count",
}

# Variables evaluated at person level (need person-to-household
# aggregation even for non-count targets).
PERSON_LEVEL_VARIABLES = {
    "person_count",
}

# Variables evaluated at SPM unit level.
SPM_UNIT_VARIABLES = {
    "spm_unit_count",
}

# Mapping from reform_id -> deduction variable to neutralize.
# reform_id 1 is used by the ETL for all JCT tax expenditure targets.
# The target's own ``variable`` column tells us *which* deduction the
# row pertains to; the reform neutralises that deduction and records
# the income_tax delta.
REFORM_ID_NEUTRALIZE: Dict[int, None] = {
    1: None,  # sentinel -- per-target variable is used directly
}

# Pseudo-constraint variables added by the ETL for hash uniqueness
# that do not correspond to real simulation variables.
_SYNTHETIC_CONSTRAINT_VARS = {"target_category"}


class NationalMatrixBuilder(BaseMatrixBuilder):
    """Build a dense calibration matrix for national reweighting.

    Reads all active targets from the database, evaluates their
    stratum constraints against a ``Microsimulation``, and returns
    a matrix suitable for ``SparseCalibrationWeights.fit()`` or the
    legacy ``microcalibrate`` interface.

    Args:
        db_uri: SQLAlchemy-style database URI, e.g.
            ``"sqlite:///path/to/policy_data.db"``.
        time_period: Tax year for the calibration (e.g. 2024).
    """

    def __init__(
        self,
        db_uri: str,
        time_period: int,
    ):
        super().__init__(db_uri, time_period)

    # ------------------------------------------------------------------
    # Database queries
    # ------------------------------------------------------------------

    def _query_active_targets(self) -> pd.DataFrame:
        """Query all active, non-zero targets.

        Returns:
            DataFrame with columns: target_id, stratum_id, variable,
            value, period, reform_id, tolerance, stratum_group_id,
            stratum_notes, target_notes.
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
            return pd.read_sql(query, conn)

    def _get_parent_stratum_id(self, stratum_id: int) -> Optional[int]:
        """Return the parent_stratum_id for a stratum, or None."""
        query = """
        SELECT parent_stratum_id
        FROM strata
        WHERE stratum_id = :stratum_id
        """
        with self.engine.connect() as conn:
            row = conn.execute(
                text(query),
                {"stratum_id": int(stratum_id)},
            ).fetchone()
        if row is None:
            return None
        return row[0]

    def _get_all_constraints(self, stratum_id: int) -> List[dict]:
        """Get constraints for a stratum *and* all its ancestors.

        Walks up the ``parent_stratum_id`` chain, collecting
        constraints from each level.  This ensures that if a parent
        stratum defines geographic or demographic constraints, they
        are applied to the child target as well.

        Synthetic constraint variables (e.g. ``target_category``)
        used only for hash uniqueness are filtered out.

        Args:
            stratum_id: Starting stratum whose full constraint chain
                is needed.

        Returns:
            De-duplicated list of constraint dicts.
        """
        all_constraints: List[dict] = []
        visited: set = set()
        current_id: Optional[int] = stratum_id

        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            constraints = self._get_stratum_constraints(current_id)
            all_constraints.extend(constraints)
            current_id = self._get_parent_stratum_id(current_id)

        # Remove synthetic/non-simulation constraint variables.
        all_constraints = [
            c
            for c in all_constraints
            if c["variable"] not in _SYNTHETIC_CONSTRAINT_VARS
        ]

        return all_constraints

    # ------------------------------------------------------------------
    # Target value computation
    # ------------------------------------------------------------------

    def _compute_target_column(
        self,
        sim,
        variable: str,
        constraints: List[dict],
        n_households: int,
    ) -> np.ndarray:
        """Compute the loss matrix column for a single target.

        For count variables (``person_count``, ``tax_unit_count``,
        ``household_count``), the column value is 1 per entity
        satisfying constraints, summed to household level.  For sum
        variables, the column is the variable value masked by
        constraints, mapped to household level.

        Args:
            sim: Microsimulation instance.
            variable: PolicyEngine variable name.
            constraints: Fully-resolved constraint list.
            n_households: Number of households.

        Returns:
            Array of length *n_households* (float64).
        """
        is_count = variable in COUNT_VARIABLES

        if is_count and variable == "person_count":
            # Count persons satisfying constraints at person level,
            # then sum per household.
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
                except Exception:
                    logger.warning(
                        "Skipping constraint '%s' for "
                        "person_count (variable not found)",
                        c["variable"],
                    )
                    return np.zeros(n_households, dtype=np.float64)

                person_mask &= apply_op(vals, c["operation"], c["value"])

            values = sim.map_result(
                person_mask.astype(float), "person", "household"
            )
            return np.asarray(values, dtype=np.float64)

        if is_count and variable == "tax_unit_count":
            # Count tax units satisfying constraints.  The
            # household-level mask already tells us which households
            # contain at least one qualifying tax unit; for a count
            # target we need the *number* of qualifying tax units per
            # household.  In practice most constraints produce a
            # 0/1-per-household result.
            mask = self._evaluate_constraints_entity_aware(
                sim, constraints, n_households
            )
            return mask.astype(np.float64)

        if is_count and variable == "household_count":
            mask = self._evaluate_constraints_entity_aware(
                sim, constraints, n_households
            )
            return mask.astype(np.float64)

        # Non-count variable: compute value at household level and
        # apply the constraint mask.
        mask = self._evaluate_constraints_entity_aware(
            sim, constraints, n_households
        )

        try:
            values = sim.calculate(
                variable,
                self.time_period,
                map_to="household",
            ).values.astype(np.float64)
        except Exception as exc:
            logger.warning(
                "Cannot calculate target variable '%s': %s "
                "-- returning zeros",
                variable,
                exc,
            )
            return np.zeros(n_households, dtype=np.float64)

        return values * mask

    # ------------------------------------------------------------------
    # Tax expenditure (reform) targets
    # ------------------------------------------------------------------

    def _compute_tax_expenditure_column(
        self,
        sim_baseline,
        variable: str,
        constraints: List[dict],
        n_households: int,
        dataset_class=None,
    ) -> np.ndarray:
        """Compute a tax expenditure column by running a reform.

        The reform neutralises *variable* (the deduction), and the
        column is ``income_tax_reform - income_tax_baseline`` masked
        by the stratum constraints.

        Args:
            sim_baseline: Baseline Microsimulation instance.
            variable: Deduction variable to neutralise
                (e.g. ``"salt_deduction"``).
            constraints: Stratum constraints to evaluate.
            n_households: Number of households.
            dataset_class: Dataset class (or path) to pass to the
                reform Microsimulation constructor.

        Returns:
            Array of length *n_households* (float64).
        """
        from policyengine_core.reforms import Reform
        from policyengine_us import Microsimulation

        # Get baseline income_tax (cached from sim_baseline).
        income_tax_baseline = sim_baseline.calculate(
            "income_tax", map_to="household"
        ).values.astype(np.float64)

        # Build a reform that neutralises the deduction variable.
        def make_repeal_class(deduction_var: str):
            class RepealDeduction(Reform):
                def apply(self):
                    self.neutralize_variable(deduction_var)

            return RepealDeduction

        RepealDeduction = make_repeal_class(variable)

        dataset_arg = dataset_class
        if dataset_arg is None:
            # Fall back to whatever the baseline sim was loaded with.
            dataset_arg = getattr(sim_baseline, "dataset", None)

        sim_reform = Microsimulation(
            dataset=dataset_arg, reform=RepealDeduction
        )
        sim_reform.default_calculation_period = self.time_period

        income_tax_reform = sim_reform.calculate(
            "income_tax", map_to="household"
        ).values.astype(np.float64)

        te_values = income_tax_reform - income_tax_baseline

        # Apply stratum constraints mask.
        mask = self._evaluate_constraints_entity_aware(
            sim_baseline, constraints, n_households
        )
        return te_values * mask

    # ------------------------------------------------------------------
    # Target name generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_target_name(
        variable: str,
        constraints: List[dict],
        stratum_notes: Optional[str] = None,
        reform_id: int = 0,
    ) -> str:
        """Generate a human-readable label for a target.

        Args:
            variable: Target variable name.
            constraints: Resolved constraints for the stratum.
            stratum_notes: Optional notes string from the stratum.
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
        sim,
        include_tax_expenditures: bool = True,
        dataset_class=None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build the national calibration matrix from DB targets.

        For each active target in the database:

        1.  Retrieve the stratum and walk up the parent chain to
            collect **all** constraints.
        2.  Evaluate each constraint variable against the simulation
            to produce a boolean mask.
        3.  Calculate the target variable, apply the mask, and map to
            household level.
        4.  Place the result in one column of the loss matrix.

        Tax expenditure targets (``reform_id > 0``) run a
        counterfactual simulation that neutralises the target's
        deduction variable and records the ``income_tax`` delta.

        Args:
            sim: Microsimulation instance loaded with the dataset.
            include_tax_expenditures: If ``True`` (default), include
                targets with ``reform_id > 0``.  Set to ``False`` to
                skip the expensive counterfactual simulations.
            dataset_class: Dataset class (or path) required for
                tax expenditure counterfactual simulations.  Ignored
                when ``include_tax_expenditures`` is ``False``.

        Returns:
            Tuple of ``(loss_matrix, targets, target_names)`` where:

            - **loss_matrix** -- numpy array, shape
              ``(n_households, n_targets)``, dtype float64.
            - **targets** -- numpy array, shape ``(n_targets,)``,
              dtype float64.
            - **target_names** -- list of human-readable labels.
        """
        household_ids = sim.calculate(
            "household_id", map_to="household"
        ).values
        n_households = len(household_ids)

        targets_df = self._query_active_targets()
        logger.info("Loaded %d active targets from database", len(targets_df))

        if targets_df.empty:
            raise ValueError("No active targets found in database")

        # Filter out targets with zero or null value.
        targets_df = targets_df[
            targets_df["value"].notna()
            & ~np.isclose(targets_df["value"].values, 0.0, atol=0.1)
        ].reset_index(drop=True)

        logger.info(
            "%d targets remain after removing zero/null values",
            len(targets_df),
        )

        if targets_df.empty:
            raise ValueError("All targets have zero or null values")

        # Cache constraints per stratum to avoid repeated DB queries.
        constraint_cache: Dict[int, List[dict]] = {}

        # Pre-allocate outputs.
        columns: List[np.ndarray] = []
        target_values: List[float] = []
        target_names: List[str] = []
        skipped = 0

        # Cache baseline income_tax for tax expenditure targets.
        _income_tax_baseline_cache: Optional[np.ndarray] = None

        for _, row in targets_df.iterrows():
            stratum_id = int(row["stratum_id"])
            variable = str(row["variable"])
            reform_id = int(row["reform_id"])

            # Skip tax expenditure targets if requested.
            if reform_id > 0 and not include_tax_expenditures:
                skipped += 1
                continue

            # Resolve full constraint chain.
            if stratum_id not in constraint_cache:
                constraint_cache[stratum_id] = self._get_all_constraints(
                    stratum_id
                )
            constraints = constraint_cache[stratum_id]

            # -- Tax expenditure target (reform_id > 0) ----------------
            if reform_id > 0:
                logger.info(
                    "Building tax expenditure target: "
                    "neutralize '%s' (target_id=%s)",
                    variable,
                    row["target_id"],
                )
                try:
                    column = self._compute_tax_expenditure_column(
                        sim_baseline=sim,
                        variable=variable,
                        constraints=constraints,
                        n_households=n_households,
                        dataset_class=dataset_class,
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping tax expenditure target '%s' "
                        "(target_id=%s): %s",
                        variable,
                        row["target_id"],
                        exc,
                    )
                    skipped += 1
                    continue
            else:
                # -- Baseline target -----------------------------------
                logger.debug(
                    "Building target: %s (target_id=%s, " "stratum_id=%s)",
                    variable,
                    row["target_id"],
                    stratum_id,
                )
                column = self._compute_target_column(
                    sim, variable, constraints, n_households
                )

            columns.append(column)
            target_values.append(float(row["value"]))
            target_names.append(
                self._make_target_name(
                    variable,
                    constraints,
                    stratum_notes=row.get("stratum_notes", ""),
                    reform_id=reform_id,
                )
            )

        if skipped:
            logger.info("Skipped %d targets", skipped)

        n_targets = len(columns)
        logger.info(
            "Built matrix: %d households x %d targets",
            n_households,
            n_targets,
        )

        if n_targets == 0:
            raise ValueError(
                "No targets could be computed " "(all were skipped or errored)"
            )

        # Stack columns into (n_households, n_targets) matrix.
        matrix = np.column_stack(columns)
        targets = np.array(target_values, dtype=np.float64)

        return matrix, targets, target_names
