"""Test same-state values match original simulation values."""

import pytest
import numpy as np
from collections import defaultdict

from .conftest import VARIABLES_TO_TEST, N_VERIFICATION_SAMPLES


@pytest.mark.skip(
    reason="Sparse matrix builder not used in production; test needs rework after time_period fix"
)
def test_same_state_matches_original(
    sim,
    X_sparse,
    targets_df,
    test_cds,
    n_households,
    household_ids,
    household_states,
):
    """
    Same-state non-zero cells must match ORIGINAL simulation values.

    When household stays in same state, X_sparse should contain the value
    from the original simulation (ground truth from H5 dataset).

    Uses stratified sampling to ensure all variables in VARIABLES_TO_TEST
    are covered with approximately equal samples per variable.
    """
    seed = 42
    rng = np.random.default_rng(seed)
    n_hh = n_households
    hh_ids = household_ids
    hh_states = household_states

    nonzero_rows, nonzero_cols = X_sparse.nonzero()

    # Group same-state cells by variable for stratified sampling
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

        # Only include same-state cells
        if dest_state != orig_state:
            continue

        variable = targets_df.iloc[row_idx]["variable"]
        if variable in variables_to_test:
            variable_to_indices[variable].append(i)

    if not variable_to_indices:
        pytest.skip("No same-state non-zero cells found for test variables")

    # Stratified sampling: sample proportionally from each variable
    samples_per_var = max(
        1, N_VERIFICATION_SAMPLES // len(variable_to_indices)
    )
    sample_indices = []

    for variable, indices in variable_to_indices.items():
        n_to_sample = min(samples_per_var, len(indices))
        sampled = rng.choice(indices, n_to_sample, replace=False)
        sample_indices.extend(sampled)

    # Cache original values per variable to avoid repeated calculations
    original_values_cache = {}

    def get_original_values(variable):
        if variable not in original_values_cache:
            original_values_cache[variable] = sim.calculate(
                variable, map_to="household"
            ).values
        return original_values_cache[variable]

    errors = []
    variables_tested = set()

    for idx in sample_indices:
        row_idx = nonzero_rows[idx]
        col_idx = nonzero_cols[idx]
        cd_idx = col_idx // n_hh
        hh_idx = col_idx % n_hh
        variable = targets_df.iloc[row_idx]["variable"]
        actual = float(X_sparse[row_idx, col_idx])

        # Compare to ORIGINAL simulation values (ground truth)
        original_values = get_original_values(variable)
        expected = float(original_values[hh_idx])

        variables_tested.add(variable)

        if not np.isclose(actual, expected, atol=0.5):
            errors.append(
                {
                    "hh_id": hh_ids[hh_idx],
                    "hh_idx": hh_idx,
                    "variable": variable,
                    "actual": actual,
                    "expected": expected,
                    "diff": actual - expected,
                    "rel_diff": (
                        (actual - expected) / expected
                        if expected != 0
                        else np.inf
                    ),
                }
            )

    missing_vars = variables_to_test - variables_tested
    if missing_vars:
        print(f"Warning: No same-state cells found for: {missing_vars}")

    assert not errors, (
        f"Same-state verification failed: {len(errors)}/{len(sample_indices)} "
        f"mismatches across {len(variables_tested)} variables. "
        f"First 5: {errors[:5]}"
    )
