"""
Tests for sparse matrix builder correctness.

These tests verify that:
1. Matrix shape and structure are correct
2. Matrix cell values match simulation calculations for households in their
   original state
3. Variable aggregation (person to household) preserves totals
4. National-level targets receive contributions from all states (no geographic
   bias)

The key verification approach:
- When households are "borrowed" to different geographic areas, state_fips is
  changed and variables are recalculated
- For households borrowed to CDs in their ORIGINAL state, the recalculated
  values should match the original simulation values exactly (since state_fips
  is unchanged)
- This provides a ground-truth verification without needing end-to-end H5
  creation

IMPORTANT NOTE on stochastic eligibility:
Some variables like SNAP have eligibility tests that use PolicyEngine's
random() function. When variables are recalculated in the matrix builder (via
fresh simulations), the random seed sequence may differ, causing ~1-3% of
households to have different eligibility outcomes. This is expected behavior,
so tests allow up to 2% mismatch rate for such variables.
"""

import pytest
import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_us_data.storage import STORAGE_FOLDER
from policyengine_us_data.datasets.cps.local_area_calibration.sparse_matrix_builder import (
    SparseMatrixBuilder,
)


# =============================================================================
# CONFIGURATION - Update these lists as new variables are added
# =============================================================================

# Variables to test for state-level value matching
# Format: (variable_name, rtol) - rtol is relative tolerance for comparison
VARIABLES_TO_TEST = [
    ("snap", 1e-2),
    ("health_insurance_premiums_without_medicare_part_b", 1e-2),
    ("medicaid", 1e-2),
    ("medicare_part_b_premiums", 1e-2),
    ("other_medical_expenses", 1e-2),
    ("over_the_counter_health_expenses", 1e-2),
    ("salt_deduction", 1e-2),
    ("spm_unit_capped_work_childcare_expenses", 1e-2),
    ("spm_unit_capped_housing_subsidy", 1e-2),
    ("ssi", 1e-2),
    ("tanf", 1e-2),
    ("tip_income", 1e-2),
    ("unemployment_compensation", 1e-2),
]

# Combined filter config to build matrix with all variables at once
COMBINED_FILTER_CONFIG = {
    "stratum_group_ids": [
        4,  # SNAP targets
        5,  # Medicaid targets
        112,  # Unemployment compensation targets
    ],
    "variables": [
        "snap",
        "health_insurance_premiums_without_medicare_part_b",
        "medicaid",
        "medicare_part_b_premiums",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "salt_deduction",
        "spm_unit_capped_work_childcare_expenses",
        "spm_unit_capped_housing_subsidy",
        "ssi",
        "tanf",
        "tip_income",
        "unemployment_compensation",
    ],
}

VARIABLES_WITH_STATE_VARIATION = [
    "snap",
]

# Complications:
# (snap)
# (unemployment_compensation)
# income_tax
# qualified_business_income_deduction
# taxable_social_security
# taxable_pension_income
# taxable_ira_distributions
# taxable_interest_income
# tax_exempt_interest_income
# self_employment_income
# salt
# refundable_ctc
# real_estate_taxes
# qualified_dividend_income
# dividend_income
# adjusted_gross_income
# eitc

# Maximum allowed mismatch rate for state-level value comparison
MAX_MISMATCH_RATE = 0.02


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def db_uri():
    """Database URI for calibration targets."""
    db_path = STORAGE_FOLDER / "calibration" / "policy_data.db"
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="module")
def dataset_path():
    """Path to stratified extended CPS dataset."""
    return str(STORAGE_FOLDER / "stratified_extended_cps_2023.h5")


@pytest.fixture(scope="module")
def sim(dataset_path):
    """Base simulation loaded from stratified CPS."""
    return Microsimulation(dataset=dataset_path)


@pytest.fixture(scope="module")
def test_cds():
    """
    Test CDs spanning multiple states for comprehensive testing.

    Selected to include:
    - Small states (1-2 CDs): AL, MT
    - Medium states: NC
    - Large states: CA, TX, NY
    """
    return [
        "101",  # Alabama CD-1 (state_fips=1)
        "102",  # Alabama CD-2
        "601",  # California CD-1 (state_fips=6)
        "602",  # California CD-2
        "3001",  # Montana CD-1 (state_fips=30)
        "3002",  # Montana CD-2
        "3701",  # North Carolina CD-1 (state_fips=37)
        "3702",  # North Carolina CD-2
        "3601",  # New York CD-1 (state_fips=36)
        "3602",  # New York CD-2
        "4801",  # Texas CD-1 (state_fips=48)
        "4802",  # Texas CD-2
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


@pytest.fixture(scope="module")
def combined_matrix_data(sim, builder):
    """
    Build matrix once with all configured variables.

    This fixture is used by the consolidated test to avoid rebuilding
    the matrix for each variable.
    """
    targets_df, X_sparse, hh_mapping = builder.build_matrix(
        sim,
        target_filter=COMBINED_FILTER_CONFIG,
    )

    household_ids = sim.calculate("household_id", map_to="household").values
    state_fips = sim.calculate("state_fips", map_to="household").values

    return {
        "targets_df": targets_df,
        "X_sparse": X_sparse,
        "hh_mapping": hh_mapping,
        "household_ids": household_ids,
        "state_fips": state_fips,
        "cds": builder.cds_to_calibrate,
        "n_households": len(household_ids),
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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


def _verify_state_level_values(
    X_sparse,
    targets_df,
    original_values,
    original_state_fips,
    cds,
    n_households,
    variable_name,
    rtol=1e-2,
):
    """
    Verify that matrix values match original values for households in their
    original state.

    Returns:
        Tuple of (verified_count, mismatches_list, skipped_reason or None)
    """
    # Get state-level targets
    state_targets = targets_df[
        (targets_df["variable"] == variable_name)
        & (targets_df["geographic_id"].apply(lambda x: str(x).isdigit()))
        & (
            targets_df["geographic_id"].apply(
                lambda x: 1 <= int(x) <= 56 if str(x).isdigit() else False
            )
        )
    ]

    if len(state_targets) == 0:
        return 0, [], f"No state-level targets for {variable_name}"

    mismatches = []
    verified_count = 0

    for _, target_row in state_targets.iterrows():
        target_state = int(target_row["geographic_id"])
        row_idx = target_row.name

        # Find all CDs in this state
        state_cds = [
            (cd_idx, cd)
            for cd_idx, cd in enumerate(cds)
            if int(cd) // 100 == target_state
        ]

        if not state_cds:
            continue

        # Find households originally from this state
        hh_from_state_mask = original_state_fips == target_state
        hh_indices_from_state = np.where(hh_from_state_mask)[0]

        if len(hh_indices_from_state) == 0:
            continue

        # For each CD in the state, check matrix values
        for cd_idx, cd in state_cds:
            col_start = cd_idx * n_households

            for hh_idx in hh_indices_from_state:
                col_idx = col_start + hh_idx
                matrix_val = X_sparse[row_idx, col_idx]
                original_val = original_values[hh_idx]

                if original_val == 0 and matrix_val == 0:
                    verified_count += 1
                    continue

                if original_val != 0:
                    rel_diff = abs(matrix_val - original_val) / abs(
                        original_val
                    )
                    if rel_diff > rtol:
                        mismatches.append(
                            {
                                "variable": variable_name,
                                "state": target_state,
                                "cd": cd,
                                "hh_idx": hh_idx,
                                "matrix_val": float(matrix_val),
                                "original_val": float(original_val),
                                "rel_diff": rel_diff,
                            }
                        )
                    else:
                        verified_count += 1
                elif matrix_val != 0:
                    mismatches.append(
                        {
                            "variable": variable_name,
                            "state": target_state,
                            "cd": cd,
                            "hh_idx": hh_idx,
                            "matrix_val": float(matrix_val),
                            "original_val": float(original_val),
                            "rel_diff": float("inf"),
                        }
                    )

    return verified_count, mismatches, None


# =============================================================================
# BASIC STRUCTURE TESTS
# =============================================================================


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


# =============================================================================
# CONSOLIDATED STATE-LEVEL VALUE TEST
# =============================================================================


class TestStateLevelValues:
    """
    Consolidated test for verifying matrix values match original simulation
    values for households in their original state.

    Builds matrix once and iterates through all configured variables.
    """

    def test_all_variables_state_level_match(self, sim, combined_matrix_data):
        """
        Verify all configured variables have correct state-level values.

        For each variable:
        1. Calculate original values from simulation
        2. Compare to matrix values for households in their original state
        3. Allow up to MAX_MISMATCH_RATE due to stochastic eligibility
        """
        results = []
        all_mismatches = []

        for variable_name, rtol in VARIABLES_TO_TEST:
            # Calculate original values for this variable
            original_values = sim.calculate(
                variable_name, map_to="household"
            ).values

            verified, mismatches, skip_reason = _verify_state_level_values(
                X_sparse=combined_matrix_data["X_sparse"],
                targets_df=combined_matrix_data["targets_df"],
                original_values=original_values,
                original_state_fips=combined_matrix_data["state_fips"],
                cds=combined_matrix_data["cds"],
                n_households=combined_matrix_data["n_households"],
                variable_name=variable_name,
                rtol=rtol,
            )

            total_checked = verified + len(mismatches)
            mismatch_rate = (
                len(mismatches) / total_checked if total_checked > 0 else 0
            )

            results.append(
                {
                    "variable": variable_name,
                    "verified": verified,
                    "mismatches": len(mismatches),
                    "total": total_checked,
                    "mismatch_rate": mismatch_rate,
                    "skip_reason": skip_reason,
                    "passed": (
                        skip_reason is not None
                        or mismatch_rate <= MAX_MISMATCH_RATE
                    ),
                }
            )

            all_mismatches.extend(mismatches)

        # Print summary
        print("\n" + "=" * 70)
        print("STATE-LEVEL VALUE VERIFICATION SUMMARY")
        print("=" * 70)

        results_df = pd.DataFrame(results)
        for _, row in results_df.iterrows():
            if row["skip_reason"]:
                status = f"SKIPPED: {row['skip_reason']}"
            elif row["passed"]:
                status = (
                    f"PASSED: {row['verified']:,} verified, "
                    f"{row['mismatch_rate']:.1%} mismatch rate"
                )
            else:
                status = (
                    f"FAILED: {row['mismatches']:,} mismatches, "
                    f"{row['mismatch_rate']:.1%} > {MAX_MISMATCH_RATE:.0%}"
                )
            print(f"  {row['variable']}: {status}")

        # Show sample mismatches if any
        if all_mismatches:
            print(f"\nSample mismatches ({len(all_mismatches)} total):")
            mismatch_df = pd.DataFrame(all_mismatches)
            print(mismatch_df.head(15).to_string())

        mismatch_df.to_csv("state_level_mismatches.csv", index=False)

        # Assert all variables passed
        failed = [r for r in results if not r["passed"]]
        assert len(failed) == 0, (
            f"{len(failed)} variable(s) failed state-level verification: "
            f"{[r['variable'] for r in failed]}"
        )


# =============================================================================
# NATIONAL-LEVEL CONTRIBUTION TEST
# =============================================================================


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
        self, sim, combined_matrix_data
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
        targets_df = combined_matrix_data["targets_df"]
        X_sparse = combined_matrix_data["X_sparse"]
        state_fips = combined_matrix_data["state_fips"]
        n_households = combined_matrix_data["n_households"]
        cds = combined_matrix_data["cds"]

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
            # (at least 2, since we have CDs from 6 different states)
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
        self, sim, combined_matrix_data
    ):
        """
        Verify the distribution of contributing states in national targets
        roughly matches the original data distribution.

        This catches cases where one state dominates the contributions
        disproportionately.
        """
        targets_df = combined_matrix_data["targets_df"]
        X_sparse = combined_matrix_data["X_sparse"]
        state_fips = combined_matrix_data["state_fips"]
        n_households = combined_matrix_data["n_households"]
        cds = combined_matrix_data["cds"]

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


# =============================================================================
# CROSS-STATE RECALCULATION TEST
# =============================================================================


class TestCrossStateRecalculation:
    """
    Tests verifying that household values change when borrowed to different
    states, confirming state-specific rules are being applied.

    The key insight: for national-level targets (no state constraint), each
    household appears in every CD block. The value in each CD block represents
    what the variable would be if that household lived in that CD's state.
    For state-dependent variables (like SNAP), values should differ across
    states for at least some households.
    """

    def test_values_change_across_states_for_national_targets(
        self, combined_matrix_data
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
        targets_df = combined_matrix_data["targets_df"]
        X_sparse = combined_matrix_data["X_sparse"]
        n_households = combined_matrix_data["n_households"]
        cds = combined_matrix_data["cds"]

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
        self, combined_matrix_data
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
        targets_df = combined_matrix_data["targets_df"]
        X_sparse = combined_matrix_data["X_sparse"]
        n_households = combined_matrix_data["n_households"]
        cds = combined_matrix_data["cds"]
        state_fips_orig = combined_matrix_data["state_fips"]

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
