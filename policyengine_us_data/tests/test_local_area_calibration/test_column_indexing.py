"""Test column indexing in sparse matrix."""

import pytest


def test_column_indexing_roundtrip(X_sparse, tracer, test_cds):
    """
    Verify column index = cd_idx * n_households + household_index.

    This is pure math - if this fails, everything else is unreliable.
    """
    n_hh = tracer.n_households
    hh_ids = tracer.original_household_ids
    errors = []

    test_cases = []
    for cd_idx in [0, len(test_cds) // 2, len(test_cds) - 1]:
        for hh_idx in [0, 100, n_hh - 1]:
            test_cases.append((cd_idx, hh_idx))

    for cd_idx, hh_idx in test_cases:
        cd = test_cds[cd_idx]
        hh_id = hh_ids[hh_idx]
        expected_col = cd_idx * n_hh + hh_idx
        col_info = tracer.get_column_info(expected_col)
        positions = tracer.get_household_column_positions(hh_id)
        pos_col = positions[cd]

        if col_info["cd_geoid"] != cd:
            errors.append(f"CD mismatch at col {expected_col}")
        if col_info["household_index"] != hh_idx:
            errors.append(f"HH index mismatch at col {expected_col}")
        if col_info["household_id"] != hh_id:
            errors.append(f"HH ID mismatch at col {expected_col}")
        if pos_col != expected_col:
            errors.append(f"Position mismatch for hh {hh_id}, cd {cd}")

    assert not errors, f"Column indexing errors: {errors}"


def test_matrix_dimensions(X_sparse, tracer, test_cds):
    """Verify matrix width matches expected CD x household count."""
    n_hh = tracer.n_households
    expected_cols = len(test_cds) * n_hh
    assert (
        X_sparse.shape[1] == expected_cols
    ), f"Matrix width mismatch: expected {expected_cols}, got {X_sparse.shape[1]}"
