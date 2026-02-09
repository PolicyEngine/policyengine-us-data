"""
Base matrix builder with shared logic for calibration.

Provides common functionality used by both ``NationalMatrixBuilder``
(national dense matrix) and ``SparseMatrixBuilder`` (geo-stacking
sparse matrix):

- SQLAlchemy engine setup
- Entity relationship mapping (person -> household/tax_unit/spm_unit)
- Person-level constraint evaluation with household aggregation
- Database query for stratum constraints
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    apply_op,
)

logger = logging.getLogger(__name__)


class BaseMatrixBuilder:
    """Shared base for calibration matrix builders.

    Handles engine creation, entity relationship caching, constraint
    evaluation at person level with household-level aggregation, and
    stratum constraint queries.

    Args:
        db_uri: SQLAlchemy-style database URI, e.g.
            ``"sqlite:///path/to/policy_data.db"``.
        time_period: Tax year for the calibration (e.g. 2024).
    """

    def __init__(self, db_uri: str, time_period: int):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.time_period = time_period
        self._entity_rel_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Entity relationship mapping
    # ------------------------------------------------------------------

    def _build_entity_relationship(self, sim) -> pd.DataFrame:
        """Build entity relationship DataFrame mapping persons to
        all entity IDs.

        Args:
            sim: Microsimulation instance.

        Returns:
            DataFrame with columns person_id, household_id,
            tax_unit_id, spm_unit_id (one row per person).
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

    # ------------------------------------------------------------------
    # Constraint evaluation
    # ------------------------------------------------------------------

    def _evaluate_constraints_entity_aware(
        self,
        sim,
        constraints: List[dict],
        n_households: int,
    ) -> np.ndarray:
        """Evaluate constraints at person level and aggregate to
        household level using ``.any()``.

        Each constraint variable is calculated at person level; the
        boolean intersection is then rolled up so that a household
        passes if *at least one person* satisfies all constraints.

        Args:
            sim: Microsimulation instance.
            constraints: List of constraint dicts with keys
                ``variable``, ``operation``, ``value``.
            n_households: Total number of households.

        Returns:
            Boolean mask array of length *n_households*.
        """
        if not constraints:
            return np.ones(n_households, dtype=bool)

        entity_rel = self._build_entity_relationship(sim)
        n_persons = len(entity_rel)

        person_mask = np.ones(n_persons, dtype=bool)

        for c in constraints:
            var = c["variable"]
            op = c["operation"]
            val = c["value"]

            try:
                constraint_values = sim.calculate(
                    var, self.time_period, map_to="person"
                ).values
            except Exception as exc:
                logger.warning(
                    "Cannot evaluate constraint variable "
                    "'%s': %s -- returning all-False mask",
                    var,
                    exc,
                )
                return np.zeros(n_households, dtype=bool)

            person_mask &= apply_op(constraint_values, op, val)

        # Aggregate to household using .any()
        entity_rel_with_mask = entity_rel.copy()
        entity_rel_with_mask["satisfies"] = person_mask

        household_mask_series = entity_rel_with_mask.groupby("household_id")[
            "satisfies"
        ].any()

        household_ids = sim.calculate(
            "household_id", map_to="household"
        ).values
        household_mask = np.array(
            [
                household_mask_series.get(hh_id, False)
                for hh_id in household_ids
            ]
        )

        return household_mask

    # ------------------------------------------------------------------
    # Database queries
    # ------------------------------------------------------------------

    def _get_stratum_constraints(self, stratum_id: int) -> List[dict]:
        """Get the direct constraints for a single stratum.

        Args:
            stratum_id: Primary key in the ``strata`` table.

        Returns:
            List of dicts with keys ``variable``, ``operation``,
            ``value``.
        """
        query = """
        SELECT constraint_variable AS variable,
               operation,
               value
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
