"""Tests for stochastic variable generation in the data package.

These tests verify that:
1. Take-up rate parameters load correctly
2. Seeded RNG produces deterministic results
3. Take-up rates produce plausible proportions
"""

import pytest
import numpy as np
from policyengine_us_data.parameters import load_take_up_rate


class TestTakeUpRateParameters:
    """Test that take-up rate parameters load correctly."""

    def test_eitc_rate_loads(self):
        """EITC take-up rates should load and be plausible."""
        rates = load_take_up_rate("eitc", 2022)
        # EITC rates are by number of children: 0, 1, 2, 3+
        assert isinstance(rates, dict) or isinstance(rates, float)
        if isinstance(rates, dict):
            for key, rate in rates.items():
                assert 0 < rate <= 1

    def test_snap_rate_loads(self):
        """SNAP take-up rate should load and be plausible."""
        rate = load_take_up_rate("snap", 2022)
        assert 0 < rate <= 1

    def test_medicaid_rate_loads(self):
        """Medicaid take-up rate should load and be plausible."""
        rate = load_take_up_rate("medicaid", 2022)
        assert 0 < rate <= 1

    def test_aca_rate_loads(self):
        """ACA take-up rate should load and be plausible."""
        rate = load_take_up_rate("aca", 2022)
        assert 0 < rate <= 1

    def test_head_start_rate_loads(self):
        """Head Start take-up rate should load and be plausible."""
        rate = load_take_up_rate("head_start", 2022)
        assert 0 < rate <= 1

    def test_early_head_start_rate_loads(self):
        """Early Head Start take-up rate should load and be plausible."""
        rate = load_take_up_rate("early_head_start", 2022)
        assert 0 < rate <= 1

    def test_dc_ptc_rate_loads(self):
        """DC PTC take-up rate should load and be plausible."""
        rate = load_take_up_rate("dc_ptc", 2022)
        assert 0 < rate <= 1


class TestSeededRandomness:
    """Test that stochastic generation is deterministic."""

    def test_same_seed_produces_same_results(self):
        """Using the same seed should produce identical results."""
        seed = 0
        n = 1_000

        generator1 = np.random.default_rng(seed=seed)
        result1 = generator1.random(n)

        generator2 = np.random.default_rng(seed=seed)
        result2 = generator2.random(n)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        n = 1_000

        generator1 = np.random.default_rng(seed=0)
        result1 = generator1.random(n)

        generator2 = np.random.default_rng(seed=1)
        result2 = generator2.random(n)

        assert not np.array_equal(result1, result2)


class TestTakeUpProportions:
    """Test that take-up rates produce plausible proportions."""

    def test_take_up_produces_expected_proportion(self):
        """Simulated take-up should match the rate approximately."""
        rate = 0.7
        n = 10_000
        generator = np.random.default_rng(seed=42)

        take_up = generator.random(n) < rate
        actual_proportion = take_up.mean()

        # Should be within 5 percentage points of the rate
        assert abs(actual_proportion - rate) < 0.05

    def test_boolean_generation(self):
        """Take-up decisions should be boolean."""
        rate = 0.5
        n = 100
        generator = np.random.default_rng(seed=42)

        take_up = generator.random(n) < rate

        assert take_up.dtype == bool
        assert set(take_up).issubset({True, False})
