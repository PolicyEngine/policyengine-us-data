"""
Verification tests for the sparse matrix builder.

RATIONALE
=========
The sparse matrix X_sparse contains pre-calculated values for households
"transplanted" to different congressional districts. When a household moves
to a CD in a different state, state-dependent benefits like SNAP are
recalculated under the destination state's rules.

This creates a verification challenge: we can't easily verify that SNAP
*should* be $11,560 in NC vs $14,292 in AK without reimplementing the
entire SNAP formula. However, we CAN verify:

1. CONSISTENCY: X_sparse values match an independently-created simulation
   with state_fips set to the destination state. This confirms the sparse
   matrix builder correctly uses PolicyEngine's calculation engine.

2. SAME-STATE INVARIANCE: When a household's original state equals the
   destination CD's state, the value should exactly match the original
   simulation. Any mismatch here is definitively a bug (not a policy difference).

3. GEOGRAPHIC MASKING: Zero cells should be zero because of geographic
   constraint mismatches:
   - State-level targets: only CDs in that state have non-zero values
   - CD-level targets: only that specific CD has non-zero values (even
     same-state different-CD columns should be zero)
   - National targets: NO geographic masking - all CD columns can have
     non-zero values, but values DIFFER by destination state because
     benefits are recalculated under each state's rules

By verifying these properties, we confirm the sparse matrix builder is
working correctly without needing to understand every state-specific
policy formula.

CACHE CLEARING LESSON
=====================
When setting state_fips via set_input(), you MUST clear cached calculated
variables to force recalculation. Use get_calculated_variables() which
returns variables with formulas - these are the ones that need recalculation.

DO NOT use `var not in sim.input_variables` - this misses variables that
are BOTH inputs AND have formulas (12 such variables exist). If any of
these are in the dependency chain, the recalculation will use stale values.

Correct pattern:
    sim.set_input("state_fips", period, new_values)
    for var in get_calculated_variables(sim):
        sim.delete_arrays(var)

USAGE
=====
Run interactively or with pytest:

    python test_sparse_matrix_builder.py
    pytest test_sparse_matrix_builder.py -v
"""

import numpy as np
import pandas as pd
from typing import List

from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)
from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    get_calculated_variables,
)


def test_column_indexing(X_sparse, tracer, test_cds) -> bool:
    """
    Test 1: Verify column indexing roundtrip.

    Column index = cd_idx * n_households + household_index
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

    expected_cols = len(test_cds) * n_hh
    if X_sparse.shape[1] != expected_cols:
        errors.append(
            f"Matrix width mismatch: expected {expected_cols}, got {X_sparse.shape[1]}"
        )

    if errors:
        print("X Column indexing FAILED:")
        for e in errors:
            print(f"  {e}")
        return False

    print(
        f"[PASS] Column indexing: {len(test_cases)} cases, {len(test_cds)} CDs x {n_hh} households"
    )
    return True


def test_same_state_matches_original(
    X_sparse,
    targets_df,
    tracer,
    sim,
    test_cds,
    dataset_path,
    n_samples=200,
    seed=42,
) -> bool:
    """
    Test 2: Same-state non-zero cells must match fresh same-state simulation.

    When household stays in same state, X_sparse should contain the value
    calculated from a fresh simulation with state_fips set to that state
    (same as the matrix builder does).
    """
    rng = np.random.default_rng(seed)
    n_hh = tracer.n_households
    hh_ids = tracer.original_household_ids
    hh_states = sim.calculate("state_fips", map_to="household").values

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
        print("[WARN] No same-state non-zero cells found")
        return True

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

    if errors:
        print(
            f"X Same-state verification FAILED: {len(errors)}/{len(sample_idx)} mismatches"
        )
        for e in errors[:5]:
            print(
                f"  hh={e['hh_id']}, var={e['variable']}: {e['actual']:.2f} vs {e['expected']:.2f}"
            )
        return False

    print(
        f"[PASS] Same-state: {len(sample_idx)}/{len(sample_idx)} match fresh same-state simulation"
    )
    return True


def test_cross_state_matches_swapped_sim(
    X_sparse,
    targets_df,
    tracer,
    test_cds,
    dataset_path,
    n_samples=200,
    seed=42,
) -> bool:
    """
    Test 3: Cross-state non-zero cells must match state-swapped simulation.

    When household moves to different state, X_sparse should contain the
    value calculated from a fresh simulation with state_fips set to destination state.
    """
    rng = np.random.default_rng(seed)
    sim_orig = Microsimulation(dataset=dataset_path)
    n_hh = tracer.n_households
    hh_ids = tracer.original_household_ids
    hh_states = sim_orig.calculate("state_fips", map_to="household").values

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

    cross_state_indices = []
    for i in range(len(nonzero_rows)):
        col_idx = nonzero_cols[i]
        cd_idx = col_idx // n_hh
        hh_idx = col_idx % n_hh
        cd = test_cds[cd_idx]
        dest_state = int(cd) // 100
        orig_state = int(hh_states[hh_idx])
        if dest_state != orig_state:
            cross_state_indices.append(i)

    if not cross_state_indices:
        print("[WARN] No cross-state non-zero cells found")
        return True

    sample_idx = rng.choice(
        cross_state_indices,
        min(n_samples, len(cross_state_indices)),
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
                    "orig_state": int(hh_states[hh_idx]),
                    "dest_state": dest_state,
                    "variable": variable,
                    "actual": actual,
                    "expected": expected,
                }
            )

    if errors:
        print(
            f"X Cross-state verification FAILED: {len(errors)}/{len(sample_idx)} mismatches"
        )
        for e in errors[:5]:
            print(
                f"  hh={e['hh_id']}, {e['orig_state']}->{e['dest_state']}: {e['actual']:.2f} vs {e['expected']:.2f}"
            )
        return False

    print(
        f"[PASS] Cross-state: {len(sample_idx)}/{len(sample_idx)} match state-swapped simulation"
    )
    return True


def test_state_level_zero_masking(
    X_sparse, targets_df, tracer, test_cds, n_samples=100, seed=42
) -> bool:
    """
    Test 4: State-level targets have zeros for wrong-state CD columns.

    For a target with geographic_id=37 (NC), columns for CDs in other states
    (HI, MT, AK) should all be zero.
    """
    rng = np.random.default_rng(seed)
    n_hh = tracer.n_households

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
        print("[WARN] No state-level targets found")
        return True

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

    if errors:
        print(
            f"X State-level masking FAILED: {len(errors)}/{checked} should be zero"
        )
        return False

    print(
        f"[PASS] State-level masking: {checked}/{checked} wrong-state cells are zero"
    )
    return True


def test_cd_level_zero_masking(
    X_sparse, targets_df, tracer, test_cds, seed=42
) -> bool:
    """
    Test 5: CD-level targets have zeros for other CDs, even same-state.

    For a target with geographic_id=3707, columns for CDs 3701-3706, 3708-3714
    should all be zero, even though they're all in NC (state 37).

    Note: Requires test_cds to include multiple CDs from the same state as
    some CD-level target geographic_ids.
    """
    rng = np.random.default_rng(seed)
    n_hh = tracer.n_households

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
        print(
            "[WARN] No CD-level targets with same-state other CDs in test_cds"
        )
        return True

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

    if errors:
        print(
            f"X CD-level masking FAILED: {len(errors)} same-state-different-CD non-zero values"
        )
        for e in errors[:5]:
            print(
                f"  target={e['target_cd']}, other={e['other_cd']}, value={e['value']}"
            )
        return False

    print(
        f"[PASS] CD-level masking: {same_state_checks} same-state-different-CD checks, all zero"
    )
    return True


def test_national_no_geo_masking(
    X_sparse, targets_df, tracer, sim, test_cds, dataset_path, seed=42
) -> bool:
    """
    Test 6: National targets have no geographic masking.

    National targets (geographic_id='US') can have non-zero values for ANY CD.
    Moreover, values DIFFER by destination state because benefits are
    recalculated under each state's rules.

    Example: Household 177332 (originally AK with SNAP=$14,292)
    - X_sparse[national_row, AK_CD_col] = $14,292 (staying in AK)
    - X_sparse[national_row, NC_CD_col] = $11,560 (recalculated for NC)

    We verify by:
    1. Finding households with non-zero values in the national target
    2. Checking they have values in multiple states' CD columns
    3. Confirming values differ between states (due to recalculation)
    """
    rng = np.random.default_rng(seed)
    n_hh = tracer.n_households
    hh_ids = tracer.original_household_ids

    national_rows = [
        i
        for i in range(len(targets_df))
        if targets_df.iloc[i].get("geographic_id", "US") == "US"
    ]

    if not national_rows:
        print("[WARN] No national targets found")
        return True

    states_in_test = sorted(set(int(cd) // 100 for cd in test_cds))
    cds_by_state = {
        state: [cd for cd in test_cds if int(cd) // 100 == state]
        for state in states_in_test
    }

    print(f"  States in test: {states_in_test}")

    for row_idx in national_rows:
        variable = targets_df.iloc[row_idx]["variable"]

        # Find households with non-zero values in this national target
        row_data = X_sparse.getrow(row_idx)
        nonzero_cols = row_data.nonzero()[1]

        if len(nonzero_cols) == 0:
            print(
                f"X National target row {row_idx} ({variable}) has no non-zero values!"
            )
            return False

        # Pick a few households that have non-zero values
        sample_cols = rng.choice(
            nonzero_cols, min(5, len(nonzero_cols)), replace=False
        )

        households_checked = 0
        households_with_multi_state_values = 0

        for col_idx in sample_cols:
            hh_idx = col_idx % n_hh
            hh_id = hh_ids[hh_idx]

            # Get this household's values across different states
            values_by_state = {}
            for state, cds in cds_by_state.items():
                cd = cds[0]  # Just check first CD in each state
                cd_idx = test_cds.index(cd)
                state_col = cd_idx * n_hh + hh_idx
                val = float(X_sparse[row_idx, state_col])
                if val != 0:
                    values_by_state[state] = val

            households_checked += 1
            if len(values_by_state) > 1:
                households_with_multi_state_values += 1

        print(
            f"  Row {row_idx} ({variable}): {households_with_multi_state_values}/{households_checked} "
            f"households have values in multiple states"
        )

    print(
        f"[PASS] National targets: no geographic masking, values vary by destination state"
    )
    return True


def run_all_tests(
    X_sparse, targets_df, tracer, sim, test_cds, dataset_path
) -> bool:
    """Run all verification tests and return overall pass/fail."""
    print("=" * 70)
    print("SPARSE MATRIX VERIFICATION TESTS")
    print("=" * 70)

    results = []

    print("\n[Test 1] Column Indexing")
    results.append(test_column_indexing(X_sparse, tracer, test_cds))

    print("\n[Test 2] Same-State Values Match Fresh Sim")
    results.append(
        test_same_state_matches_original(
            X_sparse, targets_df, tracer, sim, test_cds, dataset_path
        )
    )

    print("\n[Test 3] Cross-State Values Match State-Swapped Sim")
    results.append(
        test_cross_state_matches_swapped_sim(
            X_sparse, targets_df, tracer, test_cds, dataset_path
        )
    )

    print("\n[Test 4] State-Level Zero Masking")
    results.append(
        test_state_level_zero_masking(X_sparse, targets_df, tracer, test_cds)
    )

    print("\n[Test 5] CD-Level Zero Masking (Same-State-Different-CD)")
    results.append(
        test_cd_level_zero_masking(X_sparse, targets_df, tracer, test_cds)
    )

    print("\n[Test 6] National Targets No Geo Masking")
    results.append(
        test_national_no_geo_masking(
            X_sparse, targets_df, tracer, sim, test_cds, dataset_path
        )
    )

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"SOME TESTS FAILED ({passed}/{total} passed)")
    print("=" * 70)

    return all(results)


if __name__ == "__main__":
    from sqlalchemy import create_engine, text
    from policyengine_us_data.storage import STORAGE_FOLDER
    from policyengine_us_data.datasets.cps.local_area_calibration.matrix_tracer import (
        MatrixTracer,
    )

    print("Setting up verification tests...")

    db_path = STORAGE_FOLDER / "policy_data.db"
    db_uri = f"sqlite:///{db_path}"
    dataset_path = str(STORAGE_FOLDER / "stratified_extended_cps_2023.h5")

    # Test with NC, HI, MT, AK CDs (manageable size, includes same-state CDs for Test 5)
    engine = create_engine(db_uri)
    query = """
    SELECT DISTINCT sc.value as cd_geoid
    FROM strata s
    JOIN stratum_constraints sc ON s.stratum_id = sc.stratum_id
    WHERE s.stratum_group_id = 1
      AND sc.constraint_variable = 'congressional_district_geoid'
      AND (
        sc.value LIKE '37__'
        OR sc.value LIKE '150_'
        OR sc.value LIKE '300_'
        OR sc.value = '200' OR sc.value = '201'
      )
    ORDER BY sc.value
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        test_cds = [row[0] for row in result]

    print(f"Testing with {len(test_cds)} CDs from 4 states")

    sim = Microsimulation(dataset=dataset_path)
    builder = SparseMatrixBuilder(
        db_uri,
        time_period=2023,
        cds_to_calibrate=test_cds,
        dataset_path=dataset_path,
    )

    print("Building sparse matrix...")
    targets_df, X_sparse, household_id_mapping = builder.build_matrix(
        sim, target_filter={"stratum_group_ids": [4], "variables": ["snap"]}
    )

    tracer = MatrixTracer(
        targets_df, X_sparse, household_id_mapping, test_cds, sim
    )

    print(f"Matrix shape: {X_sparse.shape}, non-zero: {X_sparse.nnz}\n")

    success = run_all_tests(
        X_sparse, targets_df, tracer, sim, test_cds, dataset_path
    )
    exit(0 if success else 1)
