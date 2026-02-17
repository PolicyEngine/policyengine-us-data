"""Tests for target config filtering in unified calibration."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from policyengine_us_data.calibration.unified_calibration import (
    apply_target_config,
    load_target_config,
    save_calibration_package,
    load_calibration_package,
)


@pytest.fixture
def sample_targets():
    targets_df = pd.DataFrame(
        {
            "variable": [
                "snap",
                "snap",
                "eitc",
                "eitc",
                "rent",
                "person_count",
            ],
            "geo_level": [
                "national",
                "state",
                "district",
                "state",
                "national",
                "national",
            ],
            "domain_variable": [
                "snap",
                "snap",
                "eitc",
                "eitc",
                "rent",
                "person_count",
            ],
            "geographic_id": ["US", "6", "0601", "6", "US", "US"],
            "value": [1000, 500, 200, 300, 800, 5000],
        }
    )
    n_rows = len(targets_df)
    n_cols = 10
    rng = np.random.default_rng(42)
    X = sparse.random(n_rows, n_cols, density=0.5, random_state=rng)
    X = X.tocsr()
    target_names = [
        f"{r.variable}_{r.geo_level}_{r.geographic_id}"
        for _, r in targets_df.iterrows()
    ]
    return targets_df, X, target_names


class TestApplyTargetConfig:
    def test_empty_config_keeps_all(self, sample_targets):
        df, X, names = sample_targets
        config = {"exclude": []}
        out_df, out_X, out_names = apply_target_config(df, X, names, config)
        assert len(out_df) == len(df)
        assert out_X.shape == X.shape
        assert out_names == names

    def test_single_variable_geo_exclusion(self, sample_targets):
        df, X, names = sample_targets
        config = {"exclude": [{"variable": "rent", "geo_level": "national"}]}
        out_df, out_X, out_names = apply_target_config(df, X, names, config)
        assert len(out_df) == len(df) - 1
        assert "rent" not in out_df["variable"].values

    def test_multiple_exclusions(self, sample_targets):
        df, X, names = sample_targets
        config = {
            "exclude": [
                {"variable": "rent", "geo_level": "national"},
                {"variable": "eitc", "geo_level": "district"},
            ]
        }
        out_df, out_X, out_names = apply_target_config(df, X, names, config)
        assert len(out_df) == len(df) - 2
        kept = set(zip(out_df["variable"], out_df["geo_level"]))
        assert ("rent", "national") not in kept
        assert ("eitc", "district") not in kept
        assert ("eitc", "state") in kept

    def test_domain_variable_matching(self, sample_targets):
        df, X, names = sample_targets
        config = {
            "exclude": [
                {
                    "variable": "snap",
                    "geo_level": "national",
                    "domain_variable": "snap",
                }
            ]
        }
        out_df, out_X, out_names = apply_target_config(df, X, names, config)
        assert len(out_df) == len(df) - 1

    def test_matrix_and_names_stay_in_sync(self, sample_targets):
        df, X, names = sample_targets
        config = {
            "exclude": [{"variable": "person_count", "geo_level": "national"}]
        }
        out_df, out_X, out_names = apply_target_config(df, X, names, config)
        assert out_X.shape[0] == len(out_df)
        assert len(out_names) == len(out_df)
        assert out_X.shape[1] == X.shape[1]

    def test_no_match_keeps_all(self, sample_targets):
        df, X, names = sample_targets
        config = {
            "exclude": [{"variable": "nonexistent", "geo_level": "national"}]
        }
        out_df, out_X, out_names = apply_target_config(df, X, names, config)
        assert len(out_df) == len(df)
        assert out_X.shape[0] == X.shape[0]


class TestLoadTargetConfig:
    def test_load_valid_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "exclude:\n" "  - variable: snap\n" "    geo_level: national\n"
        )
        config = load_target_config(str(config_file))
        assert len(config["exclude"]) == 1
        assert config["exclude"][0]["variable"] == "snap"

    def test_load_empty_config(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_target_config(str(config_file))
        assert config["exclude"] == []


class TestCalibrationPackageRoundTrip:
    def test_round_trip(self, sample_targets, tmp_path):
        df, X, names = sample_targets
        pkg_path = str(tmp_path / "package.pkl")
        metadata = {
            "dataset_path": "/tmp/test.h5",
            "db_path": "/tmp/test.db",
            "n_clones": 5,
            "n_records": X.shape[1],
            "seed": 42,
            "created_at": "2024-01-01T00:00:00",
            "target_config": None,
        }
        save_calibration_package(pkg_path, X, df, names, metadata)
        loaded = load_calibration_package(pkg_path)

        assert loaded["target_names"] == names
        pd.testing.assert_frame_equal(loaded["targets_df"], df)
        assert loaded["X_sparse"].shape == X.shape
        assert loaded["metadata"]["seed"] == 42

    def test_package_then_filter(self, sample_targets, tmp_path):
        df, X, names = sample_targets
        pkg_path = str(tmp_path / "package.pkl")
        metadata = {"n_records": X.shape[1]}
        save_calibration_package(pkg_path, X, df, names, metadata)
        loaded = load_calibration_package(pkg_path)

        config = {"exclude": [{"variable": "rent", "geo_level": "national"}]}
        out_df, out_X, out_names = apply_target_config(
            loaded["targets_df"],
            loaded["X_sparse"],
            loaded["target_names"],
            config,
        )
        assert len(out_df) == len(df) - 1
