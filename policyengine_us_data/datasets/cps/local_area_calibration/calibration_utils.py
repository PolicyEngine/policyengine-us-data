"""
Shared utilities for calibration scripts.
"""

from typing import List
import numpy as np


def get_calculated_variables(sim) -> List[str]:
    """
    Return variables that should be cleared for state-swap recalculation.

    Includes variables with formulas, adds, or subtracts.

    Excludes ID variables (person_id, household_id, etc.) because:
    1. They have formulas that generate sequential IDs (0, 1, 2, ...)
    2. We need the original H5 values, not regenerated sequences
    3. PolicyEngine's random() function uses entity IDs as seeds:
       seed = abs(entity_id * 100 + count_random_calls)
       If IDs change, random-dependent variables (SSI resource test,
       WIC nutritional risk, WIC takeup) produce different results.
    """
    exclude_ids = {'person_id', 'household_id', 'tax_unit_id', 'spm_unit_id',
                   'family_id', 'marital_unit_id'}
    return [name for name, var in sim.tax_benefit_system.variables.items()
            if (var.formulas or getattr(var, 'adds', None) or getattr(var, 'subtracts', None))
            and name not in exclude_ids]


def apply_op(values: np.ndarray, op: str, val: str) -> np.ndarray:
    """Apply constraint operation to values array."""
    try:
        parsed = float(val)
        if parsed.is_integer():
            parsed = int(parsed)
    except ValueError:
        if val == 'True':
            parsed = True
        elif val == 'False':
            parsed = False
        else:
            parsed = val

    if op in ('==', '='):
        return values == parsed
    if op == '>':
        return values > parsed
    if op == '>=':
        return values >= parsed
    if op == '<':
        return values < parsed
    if op == '<=':
        return values <= parsed
    if op == '!=':
        return values != parsed
    return np.ones(len(values), dtype=bool)


def _get_geo_level(geo_id) -> int:
    """Return geographic level: 0=National, 1=State, 2=District."""
    if geo_id == 'US':
        return 0
    try:
        val = int(geo_id)
        return 1 if val < 100 else 2
    except (ValueError, TypeError):
        return 3
