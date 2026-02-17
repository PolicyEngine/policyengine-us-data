"""Tests for unified_calibration module.

Focuses on rerandomize_takeup: verifies draws differ by
block and are reproducible within the same block.
"""

import numpy as np
import pytest

from policyengine_us_data.utils.randomness import seeded_rng


class TestRerandomizeTakeupSeeding:
    """Verify seeded_rng(var, salt=block) produces
    reproducible, block-dependent draws."""

    def test_same_block_same_draws(self):
        var = "takes_up_snap_if_eligible"
        block = "010010001001001"
        rng1 = seeded_rng(var, salt=block)
        rng2 = seeded_rng(var, salt=block)
        draws1 = rng1.random(100)
        draws2 = rng2.random(100)
        np.testing.assert_array_equal(draws1, draws2)

    def test_different_blocks_different_draws(self):
        var = "takes_up_snap_if_eligible"
        rng1 = seeded_rng(var, salt="010010001001001")
        rng2 = seeded_rng(var, salt="020010001001001")
        draws1 = rng1.random(100)
        draws2 = rng2.random(100)
        assert not np.array_equal(draws1, draws2)

    def test_different_vars_different_draws(self):
        block = "010010001001001"
        rng1 = seeded_rng("takes_up_snap_if_eligible", salt=block)
        rng2 = seeded_rng("takes_up_aca_if_eligible", salt=block)
        draws1 = rng1.random(100)
        draws2 = rng2.random(100)
        assert not np.array_equal(draws1, draws2)

    def test_draws_in_unit_interval(self):
        rng = seeded_rng(
            "takes_up_snap_if_eligible",
            salt="010010001001001",
        )
        draws = rng.random(10000)
        assert draws.min() >= 0.0
        assert draws.max() < 1.0

    def test_rate_comparison_produces_booleans(self):
        rng = seeded_rng(
            "takes_up_snap_if_eligible",
            salt="010010001001001",
        )
        draws = rng.random(10000)
        rate = 0.75
        result = draws < rate
        assert result.dtype == bool
        frac = result.mean()
        assert 0.70 < frac < 0.80


class TestSimpleTakeupConfig:
    """Verify the SIMPLE_TAKEUP_VARS config is well-formed."""

    def test_all_entries_have_required_keys(self):
        from policyengine_us_data.calibration.unified_calibration import (
            SIMPLE_TAKEUP_VARS,
        )

        for entry in SIMPLE_TAKEUP_VARS:
            assert "variable" in entry
            assert "entity" in entry
            assert "rate_key" in entry
            assert entry["entity"] in (
                "person",
                "tax_unit",
                "spm_unit",
            )

    def test_expected_count(self):
        from policyengine_us_data.calibration.unified_calibration import (
            SIMPLE_TAKEUP_VARS,
        )

        assert len(SIMPLE_TAKEUP_VARS) == 8


class TestParseArgsNewFlags:
    """Verify new CLI flags are parsed correctly."""

    def test_target_config_flag(self):
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(["--target-config", "config.yaml"])
        assert args.target_config == "config.yaml"

    def test_build_only_flag(self):
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(["--build-only"])
        assert args.build_only is True

    def test_package_path_flag(self):
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(["--package-path", "pkg.pkl"])
        assert args.package_path == "pkg.pkl"

    def test_hyperparams_flags(self):
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(
            [
                "--beta",
                "0.65",
                "--lambda-l2",
                "1e-8",
                "--learning-rate",
                "0.2",
            ]
        )
        assert args.beta == 0.65
        assert args.lambda_l2 == 1e-8
        assert args.learning_rate == 0.2

    def test_hyperparams_defaults(self):
        from policyengine_us_data.calibration.unified_calibration import (
            BETA,
            LAMBDA_L2,
            LEARNING_RATE,
            parse_args,
        )

        args = parse_args([])
        assert args.beta == BETA
        assert args.lambda_l2 == LAMBDA_L2
        assert args.learning_rate == LEARNING_RATE
