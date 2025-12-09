"""Test geographic masking behavior in sparse matrix."""

import pytest
import numpy as np


def test_state_level_zero_masking(
    X_sparse, targets_df, tracer, test_cds, n_households
):
    """
    State-level targets have zeros for wrong-state CD columns.

    For a target with geographic_id=37 (NC), columns for CDs in other states
    (HI, MT, AK) should all be zero.
    """
    seed = 42
    rng = np.random.default_rng(seed)
    n_hh = n_households

    state_targets = []
    for row_idx in range(len(targets_df)):
        geo_id = targets_df.iloc[row_idx].get("geographic_id", "US")
        if geo_id != "US":
            try:
                val = int(geo_id)
                if val < 100:
                    state_targets.append((row_idx, val))
            except (ValueError, TypeError):
                pass

    if not state_targets:
        pytest.skip("No state-level targets found")

    errors = []
    checked = 0
    sample_targets = rng.choice(
        len(state_targets), min(20, len(state_targets)), replace=False
    )

    for idx in sample_targets:
        row_idx, target_state = state_targets[idx]
        other_state_cds = [
            (i, cd)
            for i, cd in enumerate(test_cds)
            if int(cd) // 100 != target_state
        ]
        if not other_state_cds:
            continue

        sample_cds = rng.choice(
            len(other_state_cds), min(5, len(other_state_cds)), replace=False
        )
        for cd_sample_idx in sample_cds:
            cd_idx, cd = other_state_cds[cd_sample_idx]
            sample_hh = rng.choice(n_hh, min(5, n_hh), replace=False)
            for hh_idx in sample_hh:
                col_idx = cd_idx * n_hh + hh_idx
                actual = X_sparse[row_idx, col_idx]
                checked += 1
                if actual != 0:
                    errors.append(
                        {"row": row_idx, "cd": cd, "value": float(actual)}
                    )

    assert (
        not errors
    ), f"State-level masking failed: {len(errors)}/{checked} should be zero"


def test_cd_level_zero_masking(
    X_sparse, targets_df, tracer, test_cds, n_households
):
    """
    CD-level targets have zeros for other CDs, even same-state.

    For a target with geographic_id=3707, columns for CDs 3701-3706, 3708-3714
    should all be zero, even though they're all in NC (state 37).
    """
    seed = 42
    rng = np.random.default_rng(seed)
    n_hh = n_households

    cd_targets_with_same_state = []
    for row_idx in range(len(targets_df)):
        geo_id = targets_df.iloc[row_idx].get("geographic_id", "US")
        if geo_id != "US":
            try:
                val = int(geo_id)
                if val >= 100:
                    target_state = val // 100
                    same_state_other_cds = [
                        cd
                        for cd in test_cds
                        if int(cd) // 100 == target_state and cd != geo_id
                    ]
                    if same_state_other_cds:
                        cd_targets_with_same_state.append(
                            (row_idx, geo_id, same_state_other_cds)
                        )
            except (ValueError, TypeError):
                pass

    if not cd_targets_with_same_state:
        pytest.skip(
            "No CD-level targets with same-state other CDs in test_cds"
        )

    errors = []
    same_state_checks = 0

    for row_idx, target_cd, other_cds in cd_targets_with_same_state[:10]:
        for cd in other_cds:
            cd_idx = test_cds.index(cd)
            for hh_idx in rng.choice(n_hh, 3, replace=False):
                col_idx = cd_idx * n_hh + hh_idx
                actual = X_sparse[row_idx, col_idx]
                same_state_checks += 1
                if actual != 0:
                    errors.append(
                        {
                            "target_cd": target_cd,
                            "other_cd": cd,
                            "value": float(actual),
                        }
                    )

    assert not errors, (
        f"CD-level masking failed: {len(errors)} same-state-different-CD "
        f"non-zero values. First 5: {errors[:5]}"
    )


def test_national_no_geo_masking(
    X_sparse, targets_df, tracer, sim, test_cds, dataset_path, n_households
):
    """
    National targets have no geographic masking.

    National targets (geographic_id='US') can have non-zero values for ANY CD.
    Values differ by destination state because benefits are recalculated
    under each state's rules.
    """
    seed = 42
    rng = np.random.default_rng(seed)
    n_hh = n_households
    hh_ids = tracer.original_household_ids

    national_rows = [
        i
        for i in range(len(targets_df))
        if targets_df.iloc[i].get("geographic_id", "US") == "US"
    ]

    if not national_rows:
        pytest.skip("No national targets found")

    states_in_test = sorted(set(int(cd) // 100 for cd in test_cds))
    cds_by_state = {
        state: [cd for cd in test_cds if int(cd) // 100 == state]
        for state in states_in_test
    }

    for row_idx in national_rows:
        variable = targets_df.iloc[row_idx]["variable"]

        row_data = X_sparse.getrow(row_idx)
        nonzero_cols = row_data.nonzero()[1]

        assert (
            len(nonzero_cols) > 0
        ), f"National target row {row_idx} ({variable}) has no non-zero values"

        sample_cols = rng.choice(
            nonzero_cols, min(5, len(nonzero_cols)), replace=False
        )

        households_checked = 0
        households_with_multi_state_values = 0

        for col_idx in sample_cols:
            hh_idx = col_idx % n_hh

            values_by_state = {}
            for state, cds in cds_by_state.items():
                cd = cds[0]
                cd_idx = test_cds.index(cd)
                state_col = cd_idx * n_hh + hh_idx
                val = float(X_sparse[row_idx, state_col])
                if val != 0:
                    values_by_state[state] = val

            households_checked += 1
            if len(values_by_state) > 1:
                households_with_multi_state_values += 1

        assert households_with_multi_state_values > 0, (
            f"National target {variable}: no households have values in "
            f"multiple states"
        )
