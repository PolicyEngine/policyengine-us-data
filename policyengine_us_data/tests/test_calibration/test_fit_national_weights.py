"""
Tests for national L0 calibration script (fit_national_weights.py).

Uses TDD: tests written first, then implementation.
Mocks heavy dependencies (torch, l0, Microsimulation) to keep
tests fast.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest


# -------------------------------------------------------------------
# Import tests
# -------------------------------------------------------------------


class TestImports:
    """Test that the module can be imported."""

    def test_module_imports(self):
        from policyengine_us_data.calibration import (
            fit_national_weights,
        )

        assert hasattr(fit_national_weights, "fit_national_weights")

    def test_public_functions_exist(self):
        from policyengine_us_data.calibration import (
            fit_national_weights as mod,
        )

        for name in [
            "fit_national_weights",
            "initialize_weights",
            "build_calibration_inputs",
            "compute_diagnostics",
            "save_weights_to_h5",
            "run_validation",
            "parse_args",
            "main",
        ]:
            assert hasattr(mod, name), f"Missing: {name}"

    def test_constants_defined(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            LAMBDA_L0,
            LAMBDA_L2,
            LEARNING_RATE,
            DEFAULT_EPOCHS,
            BETA,
            GAMMA,
            ZETA,
            INIT_KEEP_PROB,
            LOG_WEIGHT_JITTER_SD,
            LOG_ALPHA_JITTER_SD,
        )

        assert LAMBDA_L0 == 1e-6
        assert LAMBDA_L2 == 1e-12
        assert LEARNING_RATE == 0.15
        assert DEFAULT_EPOCHS == 1000
        assert BETA == 0.35
        assert GAMMA == -0.1
        assert ZETA == 1.1
        assert INIT_KEEP_PROB == 0.999
        assert LOG_WEIGHT_JITTER_SD == 0.05
        assert LOG_ALPHA_JITTER_SD == 0.01

    def test_weight_floor_constant(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            _WEIGHT_FLOOR,
        )

        assert _WEIGHT_FLOOR > 0
        assert _WEIGHT_FLOOR < 1


# -------------------------------------------------------------------
# initialize_weights tests
# -------------------------------------------------------------------


class TestInitializeWeights:
    """Test weight initialization from household_weight."""

    def test_returns_correct_shape(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
        )

        original = np.array([100.0, 200.0, 50.0, 0.5, 300.0])
        result = initialize_weights(original)
        assert result.shape == original.shape

    def test_returns_float64(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
        )

        original = np.array([100.0, 200.0], dtype=np.float32)
        result = initialize_weights(original)
        assert result.dtype == np.float64

    def test_zero_weights_get_floor(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
            _WEIGHT_FLOOR,
        )

        original = np.array([100.0, 0.0, 50.0, -1.0])
        result = initialize_weights(original)
        assert np.all(result > 0)
        assert result[1] == _WEIGHT_FLOOR
        assert result[3] == _WEIGHT_FLOOR

    def test_preserves_positive_weights(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
        )

        original = np.array([100.0, 200.0, 50.0])
        result = initialize_weights(original)
        np.testing.assert_array_almost_equal(result, original)

    def test_does_not_mutate_input(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
        )

        original = np.array([100.0, 0.0, -5.0])
        original_copy = original.copy()
        initialize_weights(original)
        np.testing.assert_array_equal(original, original_copy)

    def test_all_zero_weights(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
            _WEIGHT_FLOOR,
        )

        original = np.zeros(10)
        result = initialize_weights(original)
        assert np.all(result == _WEIGHT_FLOOR)

    def test_all_negative_weights(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
        )

        original = np.array([-1.0, -100.0, -0.001])
        result = initialize_weights(original)
        assert np.all(result > 0)

    def test_single_element(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
        )

        result = initialize_weights(np.array([42.0]))
        assert result[0] == pytest.approx(42.0)

    def test_large_array(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            initialize_weights,
        )

        original = np.random.uniform(-10, 1000, 200_000)
        result = initialize_weights(original)
        assert result.shape == (200_000,)
        assert np.all(result > 0)


# -------------------------------------------------------------------
# build_calibration_inputs tests
# -------------------------------------------------------------------


class TestBuildCalibrationInputs:
    """Test building the calibration matrix and targets."""

    def _mock_legacy_loss_matrix(self, n_hh=100, n_targets=50):
        """Create a mock loss matrix and targets array."""
        data = np.random.rand(n_hh, n_targets).astype(np.float32)
        cols = [f"target_{i}" for i in range(n_targets)]
        return pd.DataFrame(data, columns=cols)

    def test_fallback_to_legacy_loss_matrix(self):
        """When no DB path given, falls back to
        build_loss_matrix."""
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
        )

        mock_loss_matrix = self._mock_legacy_loss_matrix()
        mock_targets = np.random.rand(50) * 1e9

        with patch(
            "policyengine_us_data.utils.loss.build_loss_matrix",
            return_value=(mock_loss_matrix, mock_targets),
        ):
            matrix, targets, names = build_calibration_inputs(
                dataset_class=MagicMock,
                time_period=2024,
                db_path=None,
            )

        assert matrix.shape[1] == 50
        assert len(targets) == 50
        assert len(names) == 50

    def test_returns_float32_matrix(self):
        """Matrix should be float32 for memory efficiency."""
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
        )

        mock_loss_matrix = self._mock_legacy_loss_matrix(n_hh=10, n_targets=5)
        mock_targets = np.array([1e9, 2e9, 3e9, 4e9, 5e9])

        with patch(
            "policyengine_us_data.utils.loss.build_loss_matrix",
            return_value=(mock_loss_matrix, mock_targets),
        ):
            matrix, _, _ = build_calibration_inputs(
                dataset_class=MagicMock,
                time_period=2024,
            )

        assert matrix.dtype == np.float32

    def test_returns_float64_targets(self):
        """Targets should be float64 for precision."""
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
        )

        mock_loss_matrix = self._mock_legacy_loss_matrix(n_hh=10, n_targets=5)
        mock_targets = np.array([1e9, 2e9, 3e9, 4e9, 5e9])

        with patch(
            "policyengine_us_data.utils.loss.build_loss_matrix",
            return_value=(mock_loss_matrix, mock_targets),
        ):
            _, targets, _ = build_calibration_inputs(
                dataset_class=MagicMock,
                time_period=2024,
            )

        assert targets.dtype == np.float64

    def test_filters_zero_targets(self):
        """Targets with value ~0 should be filtered out."""
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
        )

        data = np.random.rand(100, 5).astype(np.float32)
        cols = [f"target_{i}" for i in range(5)]
        mock_loss_matrix = pd.DataFrame(data, columns=cols)
        # Target at index 2 is near-zero
        mock_targets = np.array([1e9, 2e9, 0.01, 3e9, 4e9])

        with patch(
            "policyengine_us_data.utils.loss.build_loss_matrix",
            return_value=(mock_loss_matrix, mock_targets),
        ):
            matrix, targets, names = build_calibration_inputs(
                dataset_class=MagicMock,
                time_period=2024,
                db_path=None,
            )

        # Near-zero target should be removed
        assert len(targets) == 4
        assert matrix.shape[1] == 4
        assert len(names) == 4

    def test_all_zero_targets_filtered(self):
        """If all targets are near-zero, result should be empty."""
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
        )

        data = np.random.rand(10, 3).astype(np.float32)
        cols = ["a", "b", "c"]
        mock_loss_matrix = pd.DataFrame(data, columns=cols)
        mock_targets = np.array([0.0, 0.05, 0.001])

        with patch(
            "policyengine_us_data.utils.loss.build_loss_matrix",
            return_value=(mock_loss_matrix, mock_targets),
        ):
            matrix, targets, names = build_calibration_inputs(
                dataset_class=MagicMock,
                time_period=2024,
            )

        assert matrix.shape[1] == 0
        assert len(targets) == 0

    def test_matrix_and_targets_consistent_shape(self):
        """Matrix columns must equal targets length."""
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
        )

        mock_loss_matrix = self._mock_legacy_loss_matrix(n_hh=50, n_targets=20)
        mock_targets = np.random.rand(20) * 1e9

        with patch(
            "policyengine_us_data.utils.loss.build_loss_matrix",
            return_value=(mock_loss_matrix, mock_targets),
        ):
            matrix, targets, names = build_calibration_inputs(
                dataset_class=MagicMock,
                time_period=2024,
            )

        assert matrix.shape[1] == len(targets)
        assert matrix.shape[1] == len(names)

    def test_db_path_with_warning(self):
        """When db_path is given but NationalMatrixBuilder is not
        wired up, falls back with warning."""
        from policyengine_us_data.calibration.fit_national_weights import (
            build_calibration_inputs,
        )

        mock_loss_matrix = self._mock_legacy_loss_matrix(n_hh=10, n_targets=3)
        mock_targets = np.array([1e9, 2e9, 3e9])

        with patch(
            "policyengine_us_data.utils.loss.build_loss_matrix",
            return_value=(mock_loss_matrix, mock_targets),
        ):
            # Should not raise, just warn and fall through
            matrix, targets, names = build_calibration_inputs(
                dataset_class=MagicMock,
                time_period=2024,
                db_path="/fake/path/db.sqlite",
            )

        assert matrix.shape[0] == 10
        assert len(targets) == 3


# -------------------------------------------------------------------
# compute_diagnostics tests
# -------------------------------------------------------------------


class TestComputeDiagnostics:
    """Test diagnostic output formatting."""

    def test_basic_diagnostics(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            compute_diagnostics,
        )

        targets = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        estimates = np.array([1050.0, 1500.0, 3100.0, 4000.0])
        names = ["pop", "income", "snap", "ssi"]

        diag = compute_diagnostics(targets, estimates, names)

        assert "pct_within_10" in diag
        assert "worst_targets" in diag
        assert isinstance(diag["pct_within_10"], float)
        assert 0 <= diag["pct_within_10"] <= 100

    def test_worst_targets_sorted(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            compute_diagnostics,
        )

        targets = np.array([1000.0, 2000.0, 3000.0])
        # income has 25% error (worst), pop has 5%
        estimates = np.array([1050.0, 1500.0, 3050.0])
        names = ["pop", "income", "snap"]

        diag = compute_diagnostics(targets, estimates, names)
        worst = diag["worst_targets"]

        # First worst target should be "income" (25% error)
        assert worst[0][0] == "income"

    def test_perfect_match(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            compute_diagnostics,
        )

        targets = np.array([1000.0, 2000.0, 3000.0])
        estimates = targets.copy()
        names = ["a", "b", "c"]

        diag = compute_diagnostics(targets, estimates, names)
        assert diag["pct_within_10"] == pytest.approx(100.0)

    def test_all_outside_threshold(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            compute_diagnostics,
        )

        targets = np.array([100.0, 200.0])
        estimates = np.array([200.0, 400.0])  # 100% error
        names = ["a", "b"]

        diag = compute_diagnostics(targets, estimates, names)
        assert diag["pct_within_10"] == pytest.approx(0.0)

    def test_custom_threshold(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            compute_diagnostics,
        )

        targets = np.array([100.0, 200.0])
        estimates = np.array([110.0, 230.0])
        names = ["a", "b"]

        # With 20% threshold, both should be within
        diag = compute_diagnostics(targets, estimates, names, threshold=0.20)
        assert diag["pct_within_10"] == pytest.approx(100.0)

    def test_custom_n_worst(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            compute_diagnostics,
        )

        targets = np.ones(50) * 1000
        estimates = np.arange(50, dtype=float) * 100
        names = [f"t_{i}" for i in range(50)]

        diag = compute_diagnostics(targets, estimates, names, n_worst=5)
        assert len(diag["worst_targets"]) == 5

    def test_near_zero_targets_handled(self):
        """Targets near zero should not produce inf/nan errors."""
        from policyengine_us_data.calibration.fit_national_weights import (
            compute_diagnostics,
        )

        targets = np.array([0.0, 1e-10, 1000.0])
        estimates = np.array([5.0, 10.0, 1050.0])
        names = ["zero", "tiny", "normal"]

        diag = compute_diagnostics(targets, estimates, names)
        assert not np.isnan(diag["pct_within_10"])
        assert not any(np.isnan(err) for _, err in diag["worst_targets"])

    def test_single_target(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            compute_diagnostics,
        )

        targets = np.array([1000.0])
        estimates = np.array([950.0])
        names = ["only"]

        diag = compute_diagnostics(targets, estimates, names)
        assert diag["pct_within_10"] == pytest.approx(100.0)
        assert len(diag["worst_targets"]) == 1


# -------------------------------------------------------------------
# fit_national_weights tests
# -------------------------------------------------------------------


class TestFitNationalWeights:
    """Test the main fitting function with mocked L0."""

    def _mock_l0(self, n_households, return_weights=None):
        """Create mock l0.calibration.SparseCalibrationWeights."""
        if return_weights is None:
            return_weights = np.ones(n_households) * 95.0

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

    def test_returns_weights_array(self):
        """fit_national_weights returns array with correct shape."""
        from policyengine_us_data.calibration.fit_national_weights import (
            fit_national_weights,
        )

        n_households = 50
        n_targets = 10
        matrix = np.random.rand(n_households, n_targets).astype(np.float32)
        targets = np.random.rand(n_targets).astype(np.float32) * 1e6
        initial_weights = np.ones(n_households) * 100.0

        mock_l0_module, _ = self._mock_l0(n_households)

        with patch.dict(
            sys.modules,
            {
                "l0": MagicMock(),
                "l0.calibration": mock_l0_module,
            },
        ):
            weights = fit_national_weights(
                matrix=matrix,
                targets=targets,
                initial_weights=initial_weights,
                epochs=5,
            )

        assert weights.shape == (n_households,)
        assert np.all(weights > 0)

    def test_calls_sparse_calibration_with_correct_args(self):
        """SparseCalibrationWeights is called with the right
        hyperparameters."""
        from policyengine_us_data.calibration.fit_national_weights import (
            fit_national_weights,
            BETA,
            GAMMA,
            ZETA,
            INIT_KEEP_PROB,
        )

        n = 20
        matrix = np.random.rand(n, 5).astype(np.float32)
        targets = np.random.rand(5) * 1e6
        initial_weights = np.ones(n) * 100.0

        mock_l0_module, mock_model = self._mock_l0(n)

        with patch.dict(
            sys.modules,
            {
                "l0": MagicMock(),
                "l0.calibration": mock_l0_module,
            },
        ):
            fit_national_weights(
                matrix=matrix,
                targets=targets,
                initial_weights=initial_weights,
                epochs=10,
            )

        # Verify SparseCalibrationWeights constructor was called
        constructor = mock_l0_module.SparseCalibrationWeights
        assert constructor.called
        call_kwargs = constructor.call_args[1]
        assert call_kwargs["n_features"] == n
        assert call_kwargs["beta"] == BETA
        assert call_kwargs["gamma"] == GAMMA
        assert call_kwargs["zeta"] == ZETA
        assert call_kwargs["init_keep_prob"] == INIT_KEEP_PROB

    def test_calls_fit_with_correct_args(self):
        """model.fit() is called with the correct parameters."""
        from policyengine_us_data.calibration.fit_national_weights import (
            fit_national_weights,
        )

        n = 20
        matrix = np.random.rand(n, 5).astype(np.float32)
        targets = np.random.rand(5) * 1e6
        initial_weights = np.ones(n) * 100.0
        epochs = 42
        lambda_l0 = 1e-5
        lambda_l2 = 1e-10
        lr = 0.1

        mock_l0_module, mock_model = self._mock_l0(n)

        with patch.dict(
            sys.modules,
            {
                "l0": MagicMock(),
                "l0.calibration": mock_l0_module,
            },
        ):
            fit_national_weights(
                matrix=matrix,
                targets=targets,
                initial_weights=initial_weights,
                epochs=epochs,
                lambda_l0=lambda_l0,
                lambda_l2=lambda_l2,
                learning_rate=lr,
            )

        mock_model.fit.assert_called_once()
        call_kwargs = mock_model.fit.call_args[1]
        assert call_kwargs["epochs"] == epochs
        assert call_kwargs["lambda_l0"] == lambda_l0
        assert call_kwargs["lambda_l2"] == lambda_l2
        assert call_kwargs["lr"] == lr
        assert call_kwargs["loss_type"] == "relative"

    def test_raises_on_missing_l0(self):
        """ImportError raised if l0 is not installed."""
        from policyengine_us_data.calibration.fit_national_weights import (
            fit_national_weights,
        )

        n = 10
        matrix = np.random.rand(n, 3).astype(np.float32)
        targets = np.random.rand(3) * 1e6
        initial_weights = np.ones(n) * 100.0

        # Remove l0 from modules so import fails
        with patch.dict(
            sys.modules,
            {"l0": None, "l0.calibration": None},
        ):
            with pytest.raises(ImportError, match="l0-python"):
                fit_national_weights(
                    matrix=matrix,
                    targets=targets,
                    initial_weights=initial_weights,
                    epochs=1,
                )

    def test_deterministic_weights(self):
        """get_weights is called with deterministic=True."""
        from policyengine_us_data.calibration.fit_national_weights import (
            fit_national_weights,
        )

        n = 10
        matrix = np.random.rand(n, 3).astype(np.float32)
        targets = np.random.rand(3) * 1e6
        initial_weights = np.ones(n) * 100.0

        mock_l0_module, mock_model = self._mock_l0(n)

        with patch.dict(
            sys.modules,
            {
                "l0": MagicMock(),
                "l0.calibration": mock_l0_module,
            },
        ):
            fit_national_weights(
                matrix=matrix,
                targets=targets,
                initial_weights=initial_weights,
                epochs=1,
            )

        mock_model.get_weights.assert_called_once_with(deterministic=True)


# -------------------------------------------------------------------
# save_weights_to_h5 tests
# -------------------------------------------------------------------


class TestSaveWeights:
    """Test saving weights to h5 file."""

    def test_save_to_h5(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            save_weights_to_h5,
        )

        n = 100
        weights = np.random.rand(n) * 200

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_dataset("household_weight/2024", data=np.ones(n))
                f.create_dataset("person_id/2024", data=np.arange(n))

            save_weights_to_h5(tmp.name, weights, year=2024)

            with h5py.File(tmp.name, "r") as f:
                saved = f["household_weight/2024"][:]
                np.testing.assert_array_almost_equal(saved, weights)

    def test_save_preserves_other_data(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            save_weights_to_h5,
        )

        n = 50
        weights = np.random.rand(n) * 200
        other_data = np.arange(n)

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_dataset("household_weight/2024", data=np.ones(n))
                f.create_dataset("person_id/2024", data=other_data)

            save_weights_to_h5(tmp.name, weights, year=2024)

            with h5py.File(tmp.name, "r") as f:
                np.testing.assert_array_equal(
                    f["person_id/2024"][:], other_data
                )

    def test_save_creates_key_if_absent(self):
        """Saving to an h5 file that has no existing weights
        key should create it."""
        from policyengine_us_data.calibration.fit_national_weights import (
            save_weights_to_h5,
        )

        n = 30
        weights = np.random.rand(n) * 100

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            # Create an h5 file with only other data
            with h5py.File(tmp.name, "w") as f:
                f.create_dataset("person_id/2024", data=np.arange(n))

            save_weights_to_h5(tmp.name, weights, year=2024)

            with h5py.File(tmp.name, "r") as f:
                assert "household_weight/2024" in f
                np.testing.assert_array_almost_equal(
                    f["household_weight/2024"][:], weights
                )

    def test_save_different_year(self):
        """Saving for a different year does not overwrite
        other years."""
        from policyengine_us_data.calibration.fit_national_weights import (
            save_weights_to_h5,
        )

        n = 20
        weights_2024 = np.ones(n) * 100
        weights_2025 = np.ones(n) * 200

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_dataset(
                    "household_weight/2024",
                    data=weights_2024,
                )

            save_weights_to_h5(tmp.name, weights_2025, year=2025)

            with h5py.File(tmp.name, "r") as f:
                np.testing.assert_array_almost_equal(
                    f["household_weight/2024"][:],
                    weights_2024,
                )
                np.testing.assert_array_almost_equal(
                    f["household_weight/2025"][:],
                    weights_2025,
                )

    def test_overwrite_existing_weights(self):
        """Saving weights overwrites existing data at the same
        year key."""
        from policyengine_us_data.calibration.fit_national_weights import (
            save_weights_to_h5,
        )

        n = 15
        old_weights = np.ones(n) * 50
        new_weights = np.ones(n) * 150

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_dataset(
                    "household_weight/2024",
                    data=old_weights,
                )

            save_weights_to_h5(tmp.name, new_weights, year=2024)

            with h5py.File(tmp.name, "r") as f:
                np.testing.assert_array_almost_equal(
                    f["household_weight/2024"][:],
                    new_weights,
                )


# -------------------------------------------------------------------
# run_validation tests
# -------------------------------------------------------------------


class TestRunValidation:
    """Test the run_validation convenience function."""

    def test_does_not_raise(self):
        """run_validation should not raise on valid input."""
        from policyengine_us_data.calibration.fit_national_weights import (
            run_validation,
        )

        n = 10
        weights = np.ones(n) * 100
        matrix = np.random.rand(n, 5)
        targets = weights @ matrix
        names = [f"t_{i}" for i in range(5)]

        # Should not raise
        run_validation(weights, matrix, targets, names)

    def test_handles_mismatched_estimates(self):
        """run_validation should handle large errors gracefully."""
        from policyengine_us_data.calibration.fit_national_weights import (
            run_validation,
        )

        n = 5
        weights = np.ones(n)
        matrix = np.eye(n, 3)
        targets = np.array([1e12, 2e12, 3e12])
        names = ["a", "b", "c"]

        # Should not raise
        run_validation(weights, matrix, targets, names)


# -------------------------------------------------------------------
# CLI tests
# -------------------------------------------------------------------


class TestCLI:
    """Test CLI argument parsing."""

    def test_parse_args_defaults(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            parse_args,
        )

        args = parse_args([])
        assert args.epochs == 1000
        assert args.lambda_l0 == 1e-6
        assert args.device == "cpu"
        assert args.dataset is None
        assert args.db_path is None
        assert args.output is None

    def test_parse_args_custom(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            parse_args,
        )

        args = parse_args(
            [
                "--epochs",
                "500",
                "--lambda-l0",
                "1e-5",
                "--device",
                "cuda",
                "--dataset",
                "/tmp/data.h5",
                "--db-path",
                "/tmp/db.sqlite",
                "--output",
                "/tmp/out.h5",
            ]
        )
        assert args.epochs == 500
        assert args.lambda_l0 == 1e-5
        assert args.device == "cuda"
        assert args.dataset == "/tmp/data.h5"
        assert args.db_path == "/tmp/db.sqlite"
        assert args.output == "/tmp/out.h5"

    def test_parse_args_invalid_device(self):
        from policyengine_us_data.calibration.fit_national_weights import (
            parse_args,
        )

        with pytest.raises(SystemExit):
            parse_args(["--device", "tpu"])

    def test_parse_args_negative_epochs(self):
        """Negative epochs is accepted by argparse (validation
        is elsewhere)."""
        from policyengine_us_data.calibration.fit_national_weights import (
            parse_args,
        )

        args = parse_args(["--epochs", "-1"])
        assert args.epochs == -1


# -------------------------------------------------------------------
# Integration test: EnhancedCPS.reweight_l0 interface
# -------------------------------------------------------------------


class TestEnhancedCPSIntegration:
    """Test that EnhancedCPS.reweight_l0 calls the right functions."""

    def test_reweight_l0_calls_pipeline(self):
        """EnhancedCPS.reweight_l0 invokes
        build_calibration_inputs, initialize_weights,
        and fit_national_weights in sequence."""
        from policyengine_us_data.datasets.cps.enhanced_cps import (
            EnhancedCPS,
        )

        # Create a minimal subclass for testing
        class TestEnhancedCPS(EnhancedCPS):
            input_dataset = MagicMock()
            start_year = 2024
            end_year = 2024
            name = "test_enhanced_cps"
            label = "Test Enhanced CPS"
            file_path = "/tmp/test.h5"
            url = None

        instance = TestEnhancedCPS()

        n_hh = 50
        mock_weights = np.ones(n_hh) * 100
        mock_matrix = np.random.rand(n_hh, 10)
        mock_targets = np.random.rand(10) * 1e9
        mock_names = [f"t_{i}" for i in range(10)]
        calibrated = np.ones(n_hh) * 95

        mock_sim = MagicMock()
        mock_sim.calculate.return_value = MagicMock(values=mock_weights)
        mock_sim.dataset.load_dataset.return_value = {"household_weight": {}}

        with (
            patch(
                "policyengine_us_data.datasets.cps."
                "enhanced_cps.Microsimulation",
                return_value=mock_sim,
            ),
            patch(
                "policyengine_us_data.calibration."
                "fit_national_weights."
                "build_calibration_inputs",
                return_value=(
                    mock_matrix,
                    mock_targets,
                    mock_names,
                ),
            ) as mock_build,
            patch(
                "policyengine_us_data.calibration."
                "fit_national_weights."
                "initialize_weights",
                return_value=mock_weights.copy(),
            ) as mock_init,
            patch(
                "policyengine_us_data.calibration."
                "fit_national_weights."
                "fit_national_weights",
                return_value=calibrated,
            ) as mock_fit,
            patch.object(instance, "save_dataset"),
        ):
            instance.reweight_l0(
                db_path="/tmp/fake.db",
                lambda_l0=1e-5,
                epochs=100,
            )

            mock_build.assert_called_once()
            mock_init.assert_called_once()
            mock_fit.assert_called_once()

            # Check fit was called with the right epochs
            fit_kwargs = mock_fit.call_args[1]
            assert fit_kwargs["epochs"] == 100
            assert fit_kwargs["lambda_l0"] == 1e-5
