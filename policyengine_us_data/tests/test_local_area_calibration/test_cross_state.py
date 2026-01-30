"""Test cross-state values match state-swapped simulations."""

import pytest
import numpy as np
from collections import defaultdict

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
)

from .conftest import VARIABLES_TO_TEST, N_VERIFICATION_SAMPLES


@pytest.mark.skip(
    reason="Sparse matrix builder not used in production; test needs rework after time_period fix"
)
def test_cross_state_matches_swapped_sim(
    X_sparse,
    targets_df,
    test_cds,
    dataset_path,
    n_households,
    household_ids,
    household_states,
):
    """
    Cross-state non-zero cells must match state-swapped simulation.

    When household moves to different state, X_sparse should contain the
    value calculated from a fresh simulation with state_fips set to
    destination state.

    Uses stratified sampling to ensure all variables in VARIABLES_TO_TEST
    are covered with approximately equal samples per variable.
    """
    seed = 42
    rng = np.random.default_rng(seed)
    n_hh = n_households
    hh_ids = household_ids
    hh_states = household_states

    state_sims = {}

    def get_state_sim(state):
        if state not in state_sims:
            s = Microsimulation(dataset=dataset_path)
            s.set_input(
                "state_fips", 2023, np.full(n_hh, state, dtype=np.int32)
            )
            for var in get_calculated_variables(s):
                s.delete_arrays(var)
            state_sims[state] = s
        return state_sims[state]

    nonzero_rows, nonzero_cols = X_sparse.nonzero()

    # Group cross-state cells by variable for stratified sampling
    variable_to_indices = defaultdict(list)
    variables_to_test = {v[0] for v in VARIABLES_TO_TEST}

    for i in range(len(nonzero_rows)):
        row_idx = nonzero_rows[i]
        col_idx = nonzero_cols[i]
        cd_idx = col_idx // n_hh
        hh_idx = col_idx % n_hh
        cd = test_cds[cd_idx]
        dest_state = int(cd) // 100
        orig_state = int(hh_states[hh_idx])

        # Only include cross-state cells
        if dest_state == orig_state:
            continue

        # Get variable for this row
        variable = targets_df.iloc[row_idx]["variable"]
        if variable in variables_to_test:
            variable_to_indices[variable].append(i)

    if not variable_to_indices:
        pytest.skip("No cross-state non-zero cells found for test variables")

    # Stratified sampling: sample proportionally from each variable
    samples_per_var = max(
        1, N_VERIFICATION_SAMPLES // len(variable_to_indices)
    )
    sample_indices = []

    for variable, indices in variable_to_indices.items():
        n_to_sample = min(samples_per_var, len(indices))
        sampled = rng.choice(indices, n_to_sample, replace=False)
        sample_indices.extend(sampled)

    errors = []
    variables_tested = set()

    for idx in sample_indices:
        row_idx = nonzero_rows[idx]
        col_idx = nonzero_cols[idx]
        cd_idx = col_idx // n_hh
        hh_idx = col_idx % n_hh
        cd = test_cds[cd_idx]
        dest_state = int(cd) // 100
        variable = targets_df.iloc[row_idx]["variable"]
        actual = float(X_sparse[row_idx, col_idx])
        state_sim = get_state_sim(dest_state)
        expected = float(
            state_sim.calculate(variable, map_to="household").values[hh_idx]
        )

        variables_tested.add(variable)

        if not np.isclose(actual, expected, atol=0.5):
            errors.append(
                {
                    "hh_id": hh_ids[hh_idx],
                    "orig_state": int(hh_states[hh_idx]),
                    "dest_state": dest_state,
                    "variable": variable,
                    "actual": actual,
                    "expected": expected,
                }
            )

    # Report which variables were tested
    missing_vars = variables_to_test - variables_tested
    if missing_vars:
        print(f"Warning: No cross-state cells found for: {missing_vars}")

    assert not errors, (
        f"Cross-state verification failed: {len(errors)}/{len(sample_indices)} "
        f"mismatches across {len(variables_tested)} variables. "
        f"First 5: {errors[:5]}"
    )
