"""
Tests for unified L0 calibration pipeline
(unified_calibration.py).

Uses TDD: tests written first, then implementation.
Mocks heavy dependencies (torch, l0, Microsimulation,
clone_and_assign, UnifiedMatrixBuilder) to keep tests fast.
"""

import sys
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

# -------------------------------------------------------------------
# CLI argument parsing tests
# -------------------------------------------------------------------


class TestParseArgs:
    """Test CLI argument parsing for unified calibration."""

    def test_parse_args_defaults(self):
        """Default values when no arguments are passed."""
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
            DEFAULT_EPOCHS,
            DEFAULT_N_CLONES,
        )

        args = parse_args([])
        assert args.dataset is None
        assert args.db_path is None
        assert args.output is None
        assert args.n_clones == DEFAULT_N_CLONES
        assert args.preset is None
        assert args.lambda_l0 is None
        assert args.epochs == DEFAULT_EPOCHS
        assert args.device == "cpu"
        assert args.seed == 42

    def test_parse_args_preset_local(self):
        """--preset local is accepted."""
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
            PRESETS,
        )

        args = parse_args(["--preset", "local"])
        assert args.preset == "local"
        # Verify the preset maps to expected lambda
        assert PRESETS["local"] == 1e-8

    def test_parse_args_preset_national(self):
        """--preset national is accepted."""
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
            PRESETS,
        )

        args = parse_args(["--preset", "national"])
        assert args.preset == "national"
        assert PRESETS["national"] == 1e-4

    def test_parse_args_lambda_overrides_preset(self):
        """--lambda-l0 should be available alongside --preset.

        The actual override logic is in main(), not parse_args.
        Here we verify both values are stored independently.
        """
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(["--preset", "local", "--lambda-l0", "5e-6"])
        assert args.preset == "local"
        assert args.lambda_l0 == 5e-6

    def test_parse_args_all_options(self):
        """All CLI options are parsed correctly."""
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(
            [
                "--dataset",
                "/tmp/data.h5",
                "--db-path",
                "/tmp/db.sqlite",
                "--output",
                "/tmp/weights.npy",
                "--n-clones",
                "50",
                "--preset",
                "national",
                "--lambda-l0",
                "1e-3",
                "--epochs",
                "200",
                "--device",
                "cuda",
                "--seed",
                "99",
            ]
        )
        assert args.dataset == "/tmp/data.h5"
        assert args.db_path == "/tmp/db.sqlite"
        assert args.output == "/tmp/weights.npy"
        assert args.n_clones == 50
        assert args.preset == "national"
        assert args.lambda_l0 == 1e-3
        assert args.epochs == 200
        assert args.device == "cuda"
        assert args.seed == 99

    def test_parse_args_invalid_device(self):
        """Invalid device choice raises SystemExit."""
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        with pytest.raises(SystemExit):
            parse_args(["--device", "tpu"])

    def test_parse_args_invalid_preset(self):
        """Invalid preset choice raises SystemExit."""
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        with pytest.raises(SystemExit):
            parse_args(["--preset", "invalid"])


# -------------------------------------------------------------------
# Helpers for mocking run_calibration dependencies
# -------------------------------------------------------------------


def _make_mock_l0(n_total, return_weights=None):
    """Create a mock SparseCalibrationWeights model.

    Args:
        n_total: Number of features (records * clones).
        return_weights: Optional weight array to return.

    Returns:
        Tuple of (mock_l0_module, mock_model).
    """
    if return_weights is None:
        return_weights = np.ones(n_total) * 95.0

    mock_model = MagicMock()
    mock_model.get_weights.return_value = MagicMock(
        cpu=MagicMock(
            return_value=MagicMock(
                numpy=MagicMock(return_value=return_weights)
            )
        )
    )

    mock_l0_module = MagicMock()
    mock_l0_module.SparseCalibrationWeights = MagicMock(
        return_value=mock_model
    )

    return mock_l0_module, mock_model


def _make_mock_sim(n_records):
    """Create a mock Microsimulation that returns n_records
    household IDs.

    Args:
        n_records: Number of households.

    Returns:
        Mock Microsimulation instance.
    """
    mock_sim = MagicMock()
    mock_calc_result = MagicMock()
    mock_calc_result.values = np.arange(n_records)
    mock_sim.calculate.return_value = mock_calc_result
    return mock_sim


def _make_mock_geography(n_records, n_clones):
    """Create a mock GeographyAssignment.

    Args:
        n_records: Number of base records.
        n_clones: Number of clones.

    Returns:
        Mock GeographyAssignment.
    """
    n_total = n_records * n_clones
    geo = MagicMock()
    geo.n_records = n_records
    geo.n_clones = n_clones
    geo.state_fips = np.ones(n_total, dtype=int) * 6
    geo.cd_geoid = np.array(["0601"] * n_total)
    geo.block_geoid = np.array(["060014001001000"] * n_total)
    return geo


def _make_mock_builder_result(n_targets, n_total):
    """Create mock build_matrix return values.

    Args:
        n_targets: Number of calibration targets.
        n_total: Number of total records (records * clones).

    Returns:
        Tuple of (targets_df, X_sparse, target_names).
    """
    targets_df = pd.DataFrame(
        {
            "variable": [f"var_{i}" for i in range(n_targets)],
            "value": np.random.rand(n_targets) * 1e6 + 1e3,
        }
    )
    X_sparse = scipy.sparse.random(
        n_targets, n_total, density=0.3, format="csr"
    )
    # Ensure no all-zero rows by default
    for i in range(n_targets):
        if X_sparse[i, :].nnz == 0:
            X_sparse[i, 0] = 1.0
    target_names = [f"target_{i}" for i in range(n_targets)]
    return targets_df, X_sparse, target_names


def _setup_module_mocks(
    mock_sim,
    mock_geo,
    mock_builder,
    mock_l0_module,
):
    """Build sys.modules dict for patching imports inside
    run_calibration.

    Since run_calibration uses local imports, we mock the
    source modules in sys.modules so that ``from X import Y``
    resolves to our mocks.

    Args:
        mock_sim: Mock Microsimulation instance.
        mock_geo: Mock GeographyAssignment to return.
        mock_builder: Mock UnifiedMatrixBuilder instance.
        mock_l0_module: Mock l0.calibration module.

    Returns:
        Dict suitable for patch.dict(sys.modules, ...).
    """
    # Mock policyengine_us.Microsimulation
    mock_pe_us = MagicMock()
    mock_pe_us.Microsimulation = MagicMock(return_value=mock_sim)

    # Mock clone_and_assign module
    mock_clone_mod = MagicMock()
    mock_clone_mod.assign_random_geography = MagicMock(return_value=mock_geo)
    mock_clone_mod.double_geography_for_puf = MagicMock(return_value=mock_geo)

    # Mock unified_matrix_builder module
    mock_builder_mod = MagicMock()
    mock_builder_mod.UnifiedMatrixBuilder = MagicMock(
        return_value=mock_builder
    )

    # Mock torch
    mock_torch = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    return {
        "policyengine_us": mock_pe_us,
        "policyengine_us.Microsimulation": mock_pe_us,
        "policyengine_us_data.calibration.clone_and_assign": (mock_clone_mod),
        "policyengine_us_data.calibration."
        "unified_matrix_builder": mock_builder_mod,
        "l0": MagicMock(),
        "l0.calibration": mock_l0_module,
        "torch": mock_torch,
    }


# -------------------------------------------------------------------
# run_calibration tests
# -------------------------------------------------------------------


class TestRunCalibration:
    """Test run_calibration with fully mocked dependencies."""

    def test_returns_weights_array(self):
        """run_calibration returns a numpy array of correct
        shape."""
        # We need to reimport after patching sys.modules
        n_records = 10
        n_clones = 5
        n_total = n_records * n_clones
        n_targets = 8

        expected_weights = np.ones(n_total) * 95.0
        mock_l0_module, mock_model = _make_mock_l0(n_total, expected_weights)
        mock_sim = _make_mock_sim(n_records)
        mock_geo = _make_mock_geography(n_records, n_clones)
        targets_df, X_sparse, target_names = _make_mock_builder_result(
            n_targets, n_total
        )

        mock_builder = MagicMock()
        mock_builder.build_matrix.return_value = (
            targets_df,
            X_sparse,
            target_names,
        )

        modules = _setup_module_mocks(
            mock_sim, mock_geo, mock_builder, mock_l0_module
        )

        with patch.dict(sys.modules, modules):
            # Reimport to pick up patched modules
            import importlib

            mod = importlib.import_module(
                "policyengine_us_data.calibration." "unified_calibration"
            )
            importlib.reload(mod)

            weights = mod.run_calibration(
                dataset_path="/fake/data.h5",
                db_path="/fake/db.sqlite",
                n_clones=n_clones,
                lambda_l0=1e-8,
                epochs=5,
                skip_puf=True,
            )

        assert isinstance(weights, np.ndarray)
        assert weights.shape == (n_total,)
        np.testing.assert_array_equal(weights, expected_weights)

    def test_achievable_target_filtering(self):
        """Targets with all-zero rows in X_sparse are removed
        before L0 fitting."""
        n_records = 5
        n_clones = 2
        n_total = n_records * n_clones

        expected_weights = np.ones(n_total) * 90.0
        mock_l0_module, mock_model = _make_mock_l0(n_total, expected_weights)
        mock_sim = _make_mock_sim(n_records)
        mock_geo = _make_mock_geography(n_records, n_clones)

        # Create sparse matrix with 2 achievable and 2 zero
        # rows
        targets_df = pd.DataFrame(
            {
                "variable": [
                    "achievable_0",
                    "zero_0",
                    "achievable_1",
                    "zero_1",
                ],
                "value": [1e6, 2e6, 3e6, 4e6],
            }
        )
        X_sparse = scipy.sparse.lil_matrix((4, n_total))
        X_sparse[0, 0] = 1.0  # achievable
        X_sparse[1, :] = 0.0  # all-zero row
        X_sparse[2, 1] = 2.0  # achievable
        X_sparse[3, :] = 0.0  # all-zero row
        X_sparse = X_sparse.tocsr()

        target_names = [
            "achievable_0",
            "zero_0",
            "achievable_1",
            "zero_1",
        ]

        mock_builder = MagicMock()
        mock_builder.build_matrix.return_value = (
            targets_df,
            X_sparse,
            target_names,
        )

        # Track what gets passed to model.fit
        fit_call_args = {}

        def capture_fit(**kwargs):
            fit_call_args.update(kwargs)

        mock_model.fit.side_effect = capture_fit

        modules = _setup_module_mocks(
            mock_sim, mock_geo, mock_builder, mock_l0_module
        )

        with patch.dict(sys.modules, modules):
            import importlib

            mod = importlib.import_module(
                "policyengine_us_data.calibration." "unified_calibration"
            )
            importlib.reload(mod)

            mod.run_calibration(
                dataset_path="/fake/data.h5",
                db_path="/fake/db.sqlite",
                n_clones=n_clones,
                lambda_l0=1e-8,
                epochs=5,
                skip_puf=True,
            )

        # model.fit should have been called with only 2
        # achievable targets, not 4
        y_passed = fit_call_args["y"]
        M_passed = fit_call_args["M"]
        assert len(y_passed) == 2
        assert M_passed.shape[0] == 2
        # Values should be the achievable ones
        np.testing.assert_array_almost_equal(y_passed, [1e6, 3e6])

    def test_calls_assign_random_geography(self):
        """assign_random_geography is called with correct
        arguments."""
        n_records = 8
        n_clones = 3
        n_total = n_records * n_clones
        n_targets = 5

        mock_l0_module, _ = _make_mock_l0(n_total)
        mock_sim = _make_mock_sim(n_records)
        mock_geo = _make_mock_geography(n_records, n_clones)
        targets_df, X_sparse, target_names = _make_mock_builder_result(
            n_targets, n_total
        )

        mock_builder = MagicMock()
        mock_builder.build_matrix.return_value = (
            targets_df,
            X_sparse,
            target_names,
        )

        modules = _setup_module_mocks(
            mock_sim, mock_geo, mock_builder, mock_l0_module
        )

        # Get a handle to the mock assign function so we can
        # assert on it after the context manager exits
        mock_assign = modules[
            "policyengine_us_data.calibration.clone_and_assign"
        ].assign_random_geography

        with patch.dict(sys.modules, modules):
            import importlib

            mod = importlib.import_module(
                "policyengine_us_data.calibration." "unified_calibration"
            )
            importlib.reload(mod)

            mod.run_calibration(
                dataset_path="/fake/data.h5",
                db_path="/fake/db.sqlite",
                n_clones=n_clones,
                lambda_l0=1e-8,
                epochs=5,
                seed=123,
                skip_puf=True,
            )

        mock_assign.assert_called_once_with(
            n_records=n_records,
            n_clones=n_clones,
            seed=123,
        )

    def test_l0_import_error(self):
        """ImportError raised when l0-python is missing."""
        n_records = 5
        n_clones = 2
        n_total = n_records * n_clones
        n_targets = 3

        mock_sim = _make_mock_sim(n_records)
        mock_geo = _make_mock_geography(n_records, n_clones)
        targets_df, X_sparse, target_names = _make_mock_builder_result(
            n_targets, n_total
        )

        mock_builder = MagicMock()
        mock_builder.build_matrix.return_value = (
            targets_df,
            X_sparse,
            target_names,
        )

        # Mock policyengine_us
        mock_pe_us = MagicMock()
        mock_pe_us.Microsimulation = MagicMock(return_value=mock_sim)

        # Mock clone_and_assign module
        mock_clone_mod = MagicMock()
        mock_clone_mod.assign_random_geography = MagicMock(
            return_value=mock_geo
        )
        mock_clone_mod.double_geography_for_puf = MagicMock(
            return_value=mock_geo
        )

        # Mock unified_matrix_builder module
        mock_builder_mod = MagicMock()
        mock_builder_mod.UnifiedMatrixBuilder = MagicMock(
            return_value=mock_builder
        )

        modules = {
            "policyengine_us": mock_pe_us,
            "policyengine_us_data.calibration."
            "clone_and_assign": mock_clone_mod,
            "policyengine_us_data.calibration."
            "unified_matrix_builder": mock_builder_mod,
            "l0": None,
            "l0.calibration": None,
        }

        with patch.dict(sys.modules, modules):
            import importlib

            mod = importlib.import_module(
                "policyengine_us_data.calibration." "unified_calibration"
            )
            importlib.reload(mod)

            with pytest.raises(ImportError, match="l0-python"):
                mod.run_calibration(
                    dataset_path="/fake/data.h5",
                    db_path="/fake/db.sqlite",
                    n_clones=n_clones,
                    lambda_l0=1e-8,
                    epochs=5,
                    skip_puf=True,
                )


# -------------------------------------------------------------------
# Constants tests
# -------------------------------------------------------------------


class TestConstants:
    """Test that module constants are defined correctly."""

    def test_presets_defined(self):
        from policyengine_us_data.calibration.unified_calibration import (
            PRESETS,
        )

        assert "local" in PRESETS
        assert "national" in PRESETS
        assert PRESETS["local"] == 1e-8
        assert PRESETS["national"] == 1e-4

    def test_hyperparameters_defined(self):
        from policyengine_us_data.calibration.unified_calibration import (
            BETA,
            GAMMA,
            ZETA,
            INIT_KEEP_PROB,
            LOG_WEIGHT_JITTER_SD,
            LOG_ALPHA_JITTER_SD,
            LAMBDA_L2,
            LEARNING_RATE,
        )

        assert BETA == 0.35
        assert GAMMA == -0.1
        assert ZETA == 1.1
        assert INIT_KEEP_PROB == 0.999
        assert LOG_WEIGHT_JITTER_SD == 0.05
        assert LOG_ALPHA_JITTER_SD == 0.01
        assert LAMBDA_L2 == 1e-12
        assert LEARNING_RATE == 0.15

    def test_default_constants(self):
        from policyengine_us_data.calibration.unified_calibration import (
            DEFAULT_EPOCHS,
            DEFAULT_N_CLONES,
        )

        assert DEFAULT_EPOCHS == 100
        assert DEFAULT_N_CLONES == 10
