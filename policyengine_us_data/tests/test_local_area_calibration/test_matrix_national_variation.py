"""
Tests for correctness in the sparse matrix builder, particularly for national level contributions.

These tests verify that:
1. Matrix shape and structure are correct
2. Variable aggregation (person to household) preserves totals
3. National-level targets receive contributions from all states (no geographic
   bias)
4. Cross-state recalculation applies state-specific rules
"""

import pytest
import numpy as np
import pandas as pd
from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)

from .conftest import (
    VARIABLES_TO_TEST,
    COMBINED_FILTER_CONFIG,
)

# Variables with state-specific variation (e.g., SNAP eligibility)
VARIABLES_WITH_STATE_VARIATION = [
    "snap",
]


@pytest.fixture(scope="module")
def builder(db_uri, dataset_path, test_cds):
    """SparseMatrixBuilder configured with test CDs."""
    return SparseMatrixBuilder(
        db_uri=db_uri,
        time_period=2023,
        cds_to_calibrate=test_cds,
        dataset_path=dataset_path,
    )


def _get_geo_level(geo_id) -> str:
    """Determine geographic level from geographic_id."""
    if geo_id == "US":
        return "national"
    try:
        val = int(geo_id)
        if 1 <= val <= 56:
            return "state"
        else:
            return "district"
    except (ValueError, TypeError):
        return "unknown"


def test_person_level_aggregation_preserves_totals(sim):
    """Health insurance premiums (person-level) sum correctly to household."""
    var = "health_insurance_premiums_without_medicare_part_b"
    person_total = sim.calculate(var, 2023, map_to="person").values.sum()
    household_total = sim.calculate(var, 2023, map_to="household").values.sum()
    assert np.isclose(person_total, household_total, rtol=1e-6)


def test_matrix_shape(sim, builder):
    """Matrix should have (n_targets, n_households * n_cds) shape."""
    targets_df, X_sparse, _ = builder.build_matrix(
        sim,
        target_filter={
            "variables": ["health_insurance_premiums_without_medicare_part_b"]
        },
    )
    n_households = len(
        sim.calculate("household_id", map_to="household").values
    )
    n_cds = len(builder.cds_to_calibrate)
    assert X_sparse.shape[1] == n_households * n_cds


def test_combined_variables_in_matrix(sim, builder):
    """Matrix should include all configured variables."""
    targets_df, X_sparse, _ = builder.build_matrix(
        sim,
        target_filter=COMBINED_FILTER_CONFIG,
    )
    variables = targets_df["variable"].unique()

    for var_name, _ in VARIABLES_TO_TEST:
        assert var_name in variables, f"Missing variable: {var_name}"


class TestNationalLevelContributions:
    """
    Tests verifying that national-level targets receive contributions from
    households across all states, not just a geographic subset.

    The key insight: for a national target, when we look at a single CD's
    column block, households from ALL original states should potentially
    contribute (subject to meeting eligibility constraints). There should
    be no systematic geographic bias where only households from certain
    states contribute to the national total.
    """

    def test_national_targets_receive_multistate_contributions(
        self, targets_df, X_sparse, household_states, n_households, test_cds
    ):
        """
        Verify that national-level targets have contributions from households
        originally from multiple states.

        For each national target:
        1. Look at the matrix row
        2. For EACH CD's column block, identify which original states have
           non-zero contributions
        3. Verify contributions come from multiple states (not geographically
           biased)
        """
        state_fips = household_states
        cds = test_cds

        # Find national-level targets
        national_targets = targets_df[
            targets_df["geographic_id"].apply(
                lambda x: _get_geo_level(x) == "national"
            )
        ]

        if len(national_targets) == 0:
            pytest.skip("No national-level targets found")

        results = []

        for _, target in national_targets.iterrows():
            row_idx = target.name
            variable = target["variable"]
            row = X_sparse[row_idx, :].toarray().flatten()

            # For each CD block, check which original states contribute
            cd_contribution_stats = []

            for cd_idx, cd in enumerate(cds):
                col_start = cd_idx * n_households
                col_end = col_start + n_households
                cd_values = row[col_start:col_end]

                # Find households with non-zero values in this CD block
                nonzero_mask = cd_values != 0
                nonzero_indices = np.where(nonzero_mask)[0]

                if len(nonzero_indices) == 0:
                    continue

                # Get original states of contributing households
                contributing_states = set(state_fips[nonzero_indices])

                cd_contribution_stats.append(
                    {
                        "cd": cd,
                        "cd_state": int(cd) // 100,
                        "n_contributing": len(nonzero_indices),
                        "n_states": len(contributing_states),
                        "contributing_states": contributing_states,
                    }
                )

            if not cd_contribution_stats:
                results.append(
                    {
                        "variable": variable,
                        "status": "NO_CONTRIBUTIONS",
                        "details": "No non-zero values in any CD block",
                    }
                )
                continue

            # Aggregate stats
            stats_df = pd.DataFrame(cd_contribution_stats)
            avg_states = stats_df["n_states"].mean()
            min_states = stats_df["n_states"].min()

            # Check: on average, contributions should come from multiple states
            # (at least 2, since we have CDs from 4 different states)
            passed = avg_states >= 2 and min_states >= 1

            results.append(
                {
                    "variable": variable,
                    "status": "PASSED" if passed else "FAILED",
                    "avg_contributing_states": avg_states,
                    "min_contributing_states": min_states,
                    "n_cd_blocks_with_data": len(stats_df),
                }
            )

        # Assert no geographic bias
        failed = [r for r in results if r["status"] == "FAILED"]
        assert len(failed) == 0, (
            f"Geographic bias detected in national targets: "
            f"{[r['variable'] for r in failed]}"
        )

    def test_state_distribution_in_national_targets(
        self, targets_df, X_sparse, household_states, n_households, test_cds
    ):
        """
        Verify the distribution of contributing states in national targets
        roughly matches the original data distribution.

        This catches cases where one state dominates the contributions
        disproportionately.
        """
        state_fips = household_states
        cds = test_cds

        # Get original state distribution (count of households per state)
        unique_states, original_counts = np.unique(
            state_fips, return_counts=True
        )
        original_dist = dict(zip(unique_states, original_counts))
        total_hh = len(state_fips)

        # Find national-level targets
        national_targets = targets_df[
            targets_df["geographic_id"].apply(
                lambda x: _get_geo_level(x) == "national"
            )
        ]

        if len(national_targets) == 0:
            pytest.skip("No national-level targets found")

        for _, target in national_targets.iterrows():
            row_idx = target.name
            variable = target["variable"]
            row = X_sparse[row_idx, :].toarray().flatten()

            # Count contributions by original state across ALL CD blocks
            state_contribution_counts = {}

            for cd_idx, cd in enumerate(cds):
                col_start = cd_idx * n_households
                col_end = col_start + n_households
                cd_values = row[col_start:col_end]

                nonzero_mask = cd_values != 0
                nonzero_indices = np.where(nonzero_mask)[0]

                for hh_idx in nonzero_indices:
                    orig_state = state_fips[hh_idx]
                    state_contribution_counts[orig_state] = (
                        state_contribution_counts.get(orig_state, 0) + 1
                    )

            if not state_contribution_counts:
                continue

            # Check that no single state dominates excessively
            total_contributions = sum(state_contribution_counts.values())
            max_contribution = max(state_contribution_counts.values())
            max_state = max(
                state_contribution_counts, key=state_contribution_counts.get
            )
            max_share = max_contribution / total_contributions

            # The max share should not exceed 70% (unless that state has 70%+
            # of households in the original data)
            original_max_share = original_dist.get(max_state, 0) / total_hh

            # Allow 20% margin above original share
            threshold = min(0.7, original_max_share + 0.2)

            assert max_share <= threshold, (
                f"State {max_state} dominates national {variable} target with "
                f"{max_share:.1%} of contributions "
                f"(original share: {original_max_share:.1%})"
            )


@pytest.mark.skip(
    reason="Sparse matrix builder not used in production; test needs rework after time_period fix"
)
class TestCrossStateRecalculation:
    """
    Tests verifying that household values change when borrowed to different
    states, confirming state-specific rules are being applied.

    The key insight: for national-level targets (no state constraint), each
    household appears in every CD block. The value in each CD block represents
    what the variable would be if that household lived in that CD's state.
    For state-dependent variables (like SNAP), values should differ across
    states for at least some households.

    NOTE: This complements test_cross_state.py which verifies exact values.
    These tests verify that variation exists (state rules are applied).
    """

    def test_values_change_across_states_for_national_targets(
        self, targets_df, X_sparse, n_households, test_cds
    ):
        """
        Verify that for national targets, household values vary across CD
        blocks from different states.

        This confirms the matrix builder is correctly recalculating variables
        with state-specific rules when households are "borrowed" to different
        geographic areas.

        The test checks:
        1. For each national target, examine households with non-zero values
        2. Compare each household's value across CD blocks from different states
        3. At least some households should have different values in different
           states (confirming recalculation with different state rules)
        """
        cds = test_cds

        # Group CDs by state
        cds_by_state = {}
        for cd_idx, cd in enumerate(cds):
            state = int(cd) // 100
            if state not in cds_by_state:
                cds_by_state[state] = []
            cds_by_state[state].append((cd_idx, cd))

        states = list(cds_by_state.keys())
        if len(states) < 2:
            pytest.skip("Need at least 2 states to test cross-state variation")

        # Find national-level targets
        national_targets = targets_df[
            targets_df["geographic_id"].apply(
                lambda x: _get_geo_level(x) == "national"
            )
        ]

        if len(national_targets) == 0:
            pytest.skip("No national-level targets found")

        results = []

        for _, target in national_targets.iterrows():
            if target["variable"] not in VARIABLES_WITH_STATE_VARIATION:
                continue
            row_idx = target.name
            variable = target["variable"]
            row = X_sparse[row_idx, :].toarray().flatten()

            # For each household, collect values from different states
            households_with_variation = 0
            households_checked = 0

            # Sample households (check every 10th to keep test fast)
            for hh_idx in range(0, n_households, 10):
                # Get this household's value in each state (use first CD of
                # each state)
                state_values = {}
                for state, cd_list in cds_by_state.items():
                    cd_idx, _ = cd_list[0]  # First CD in this state
                    col_idx = cd_idx * n_households + hh_idx
                    state_values[state] = row[col_idx]

                # Skip if all values are zero (household doesn't qualify for
                # this variable)
                nonzero_values = [v for v in state_values.values() if v != 0]
                if len(nonzero_values) < 2:
                    continue

                households_checked += 1

                # Check if values differ across states
                unique_values = set(nonzero_values)
                if len(unique_values) > 1:
                    households_with_variation += 1

            variation_rate = (
                households_with_variation / households_checked
                if households_checked > 0
                else 0
            )

            results.append(
                {
                    "variable": variable,
                    "households_checked": households_checked,
                    "households_with_variation": households_with_variation,
                    "variation_rate": variation_rate,
                }
            )

        # For state-dependent variables, we expect SOME variation
        # (not all households will vary - some may have $0 or max benefits
        # regardless of state)
        # The key is that variation exists, confirming recalculation occurs
        for r in results:
            if r["households_checked"] > 0:
                # At least 10% of households should show variation for
                # state-dependent variables
                assert (
                    r["variation_rate"] > 0.1 or r["households_checked"] < 10
                ), (
                    f"No cross-state variation found for {r['variable']}. "
                    f"This suggests state-specific rules may not be applied "
                    f"when households are borrowed to different states."
                )

    def test_same_household_different_states_shows_rule_changes(
        self, targets_df, X_sparse, household_states, n_households, test_cds
    ):
        """
        Deep dive test: pick specific households and verify their values
        differ across states in a way consistent with state-specific rules.

        For SNAP specifically, different states have different:
        - Standard deductions
        - Shelter deduction caps
        - Vehicle allowances
        - Categorical eligibility rules

        This test finds households where we can verify the recalculation
        is applying different state rules.
        """
        state_fips_orig = household_states
        cds = test_cds

        # Group CDs by state
        cds_by_state = {}
        for cd_idx, cd in enumerate(cds):
            state = int(cd) // 100
            if state not in cds_by_state:
                cds_by_state[state] = []
            cds_by_state[state].append((cd_idx, cd))

        states = sorted(cds_by_state.keys())
        if len(states) < 2:
            pytest.skip("Need at least 2 states")

        # Find national SNAP target (most state-dependent)
        snap_national = targets_df[
            (targets_df["variable"] == "snap")
            & (
                targets_df["geographic_id"].apply(
                    lambda x: _get_geo_level(x) == "national"
                )
            )
        ]

        if len(snap_national) == 0:
            pytest.skip("No national SNAP target found")

        row_idx = snap_national.iloc[0].name
        row = X_sparse[row_idx, :].toarray().flatten()

        # Find households with interesting variation patterns
        example_households = []

        for hh_idx in range(n_households):
            state_values = {}
            for state, cd_list in cds_by_state.items():
                cd_idx, _ = cd_list[0]
                col_idx = cd_idx * n_households + hh_idx
                state_values[state] = row[col_idx]

            # Look for households where:
            # 1. At least 2 states have non-zero SNAP
            # 2. The values differ significantly (>10% relative difference)
            nonzero_states = {s: v for s, v in state_values.items() if v > 0}

            if len(nonzero_states) >= 2:
                values = list(nonzero_states.values())
                max_val = max(values)
                min_val = min(values)
                if min_val > 0 and (max_val - min_val) / min_val > 0.1:
                    example_households.append(
                        {
                            "hh_idx": hh_idx,
                            "original_state": state_fips_orig[hh_idx],
                            "state_values": nonzero_states,
                            "max_val": max_val,
                            "min_val": min_val,
                            "variation": (max_val - min_val) / min_val,
                        }
                    )

            if len(example_households) >= 5:
                break

        # Assert we found at least one household with variation
        assert len(example_households) > 0, (
            "Expected to find households with >10% SNAP variation across "
            "states, confirming state-specific rules are applied"
        )
