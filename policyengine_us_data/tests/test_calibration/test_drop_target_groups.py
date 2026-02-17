"""Tests for drop_target_groups in calibration_utils."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from policyengine_us_data.datasets.cps.local_area_calibration.calibration_utils import (
    drop_target_groups,
    create_target_groups,
)


@pytest.fixture
def sample_data():
    targets_df = pd.DataFrame(
        {
            "variable": [
                "snap",
                "snap",
                "snap",
                "household_count",
                "household_count",
            ],
            "domain_variable": [
                "snap",
                "snap",
                "snap",
                "snap",
                "snap",
            ],
            "geographic_id": ["US", "6", "37", "6", "37"],
            "value": [1000, 500, 300, 200, 100],
        }
    )
    n_rows = len(targets_df)
    n_cols = 10
    rng = np.random.default_rng(42)
    X = sparse.random(n_rows, n_cols, density=0.5, random_state=rng)
    X = X.tocsr()
    target_groups, group_info = create_target_groups(targets_df)
    return targets_df, X, target_groups, group_info


class TestDropTargetGroups:
    def test_drops_matching_group(self, sample_data):
        targets_df, X, target_groups, group_info = sample_data
        n_before = len(targets_df)
        out_df, out_X = drop_target_groups(
            targets_df,
            X,
            target_groups,
            group_info,
            [("household count", "State")],
        )
        assert len(out_df) < n_before
        assert out_X.shape[0] == len(out_df)
        assert "household_count" not in out_df["variable"].values or not (
            out_df[out_df["variable"] == "household_count"]["geographic_id"]
            .isin(["6", "37"])
            .any()
        )

    def test_keeps_unmatched_groups(self, sample_data):
        targets_df, X, target_groups, group_info = sample_data
        out_df, out_X = drop_target_groups(
            targets_df,
            X,
            target_groups,
            group_info,
            [("household count", "State")],
        )
        assert "snap" in out_df["variable"].values

    def test_matrix_rows_match_df(self, sample_data):
        targets_df, X, target_groups, group_info = sample_data
        out_df, out_X = drop_target_groups(
            targets_df,
            X,
            target_groups,
            group_info,
            [("snap", "National")],
        )
        assert out_X.shape[0] == len(out_df)
        assert out_X.shape[1] == X.shape[1]

    def test_no_match_keeps_all(self, sample_data):
        targets_df, X, target_groups, group_info = sample_data
        out_df, out_X = drop_target_groups(
            targets_df,
            X,
            target_groups,
            group_info,
            [("nonexistent", "National")],
        )
        assert len(out_df) == len(targets_df)
        assert out_X.shape[0] == X.shape[0]

    def test_drop_all_groups(self, sample_data):
        targets_df, X, target_groups, group_info = sample_data
        out_df, out_X = drop_target_groups(
            targets_df,
            X,
            target_groups,
            group_info,
            [
                ("snap", "National"),
                ("snap", "State"),
                ("household count", "State"),
            ],
        )
        assert len(out_df) == 0
        assert out_X.shape[0] == 0

    def test_columns_preserved(self, sample_data):
        targets_df, X, target_groups, group_info = sample_data
        out_df, out_X = drop_target_groups(
            targets_df,
            X,
            target_groups,
            group_info,
            [("snap", "National")],
        )
        assert out_X.shape[1] == X.shape[1]

    def test_case_insensitive_match(self, sample_data):
        targets_df, X, target_groups, group_info = sample_data
        out_df, _ = drop_target_groups(
            targets_df,
            X,
            target_groups,
            group_info,
            [("SNAP", "State")],
        )
        out_df2, _ = drop_target_groups(
            targets_df,
            X,
            target_groups,
            group_info,
            [("snap", "State")],
        )
        assert len(out_df) == len(out_df2)
