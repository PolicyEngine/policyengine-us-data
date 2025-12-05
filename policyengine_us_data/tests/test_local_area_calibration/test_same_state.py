"""Test same-state values match fresh simulations."""

import pytest
import numpy as np

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
)


def test_same_state_matches_original(
    X_sparse,
    targets_df,
    tracer,
    sim,
    test_cds,
    dataset_path,
    n_households,
    household_ids,
    household_states,
):
    """
    Same-state non-zero cells must match fresh same-state simulation.

    When household stays in same state, X_sparse should contain the value
    calculated from a fresh simulation with state_fips set to that state.
    """
    n_samples = 200
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

    same_state_indices = []
    for i in range(len(nonzero_rows)):
        col_idx = nonzero_cols[i]
        cd_idx = col_idx // n_hh
        hh_idx = col_idx % n_hh
        cd = test_cds[cd_idx]
        dest_state = int(cd) // 100
        orig_state = int(hh_states[hh_idx])
        if dest_state == orig_state:
            same_state_indices.append(i)

    if not same_state_indices:
        pytest.skip("No same-state non-zero cells found")

    sample_idx = rng.choice(
        same_state_indices,
        min(n_samples, len(same_state_indices)),
        replace=False,
    )
    errors = []

    for idx in sample_idx:
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

        if not np.isclose(actual, expected, atol=0.5):
            errors.append(
                {
                    "hh_id": hh_ids[hh_idx],
                    "variable": variable,
                    "actual": actual,
                    "expected": expected,
                }
            )

    assert not errors, (
        f"Same-state verification failed: {len(errors)}/{len(sample_idx)} "
        f"mismatches. First 5: {errors[:5]}"
    )
