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


class TestCategoryTakeupConfig:
    """Verify the CATEGORY_TAKEUP_VARS config is well-formed."""

    def test_all_entries_have_required_keys(self):
        from policyengine_us_data.calibration.unified_calibration import (
            CATEGORY_TAKEUP_VARS,
        )

        required = {
            "variable",
            "entity",
            "rate_key",
            "category_variable",
            "category_mapper",
        }
        for entry in CATEGORY_TAKEUP_VARS:
            for key in required:
                assert key in entry, f"Missing key '{key}' in {entry}"

    def test_all_mappers_callable(self):
        from policyengine_us_data.calibration.unified_calibration import (
            CATEGORY_TAKEUP_VARS,
        )

        for entry in CATEGORY_TAKEUP_VARS:
            assert callable(entry["category_mapper"])

    def test_all_entities_valid(self):
        from policyengine_us_data.calibration.unified_calibration import (
            CATEGORY_TAKEUP_VARS,
        )

        valid = ("person", "tax_unit", "spm_unit")
        for entry in CATEGORY_TAKEUP_VARS:
            assert entry["entity"] in valid

    def test_expected_count(self):
        from policyengine_us_data.calibration.unified_calibration import (
            CATEGORY_TAKEUP_VARS,
        )

        assert len(CATEGORY_TAKEUP_VARS) == 3


class TestCategoryMappers:
    """Test mapper functions load rates from YAML and map correctly."""

    def test_eitc_mapper_standard_counts(self):
        from policyengine_us_data.calibration.unified_calibration import (
            _eitc_category_mapper,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        rates = load_take_up_rate("eitc")
        categories = np.array([0, 1, 2, 3, 5])
        result = _eitc_category_mapper(categories, rates)
        max_key = max(rates.keys())
        expected = np.array(
            [
                rates[0],
                rates[1],
                rates[2],
                rates[3],
                rates[max_key],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_eitc_mapper_clamps_high_count(self):
        from policyengine_us_data.calibration.unified_calibration import (
            _eitc_category_mapper,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        rates = load_take_up_rate("eitc")
        max_key = max(rates.keys())
        categories = np.array([99])
        result = _eitc_category_mapper(categories, rates)
        assert result[0] == rates[max_key]

    def test_wic_mapper_known_categories(self):
        from policyengine_us_data.calibration.unified_calibration import (
            _wic_category_mapper,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        rates = load_take_up_rate("wic_takeup", 2022)
        categories = np.array(list(rates.keys()))
        result = _wic_category_mapper(categories, rates)
        expected = np.array([rates[c] for c in categories])
        np.testing.assert_array_almost_equal(result, expected)

    def test_wic_mapper_unknown_defaults_zero(self):
        from policyengine_us_data.calibration.unified_calibration import (
            _wic_category_mapper,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        rates = load_take_up_rate("wic_takeup", 2022)
        categories = np.array(["UNKNOWN"])
        result = _wic_category_mapper(categories, rates)
        assert result[0] == 0

    def test_wic_mapper_none_category(self):
        from policyengine_us_data.calibration.unified_calibration import (
            _wic_category_mapper,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        rates = load_take_up_rate("wic_takeup", 2022)
        categories = np.array(["NONE"])
        result = _wic_category_mapper(categories, rates)
        assert result[0] == rates["NONE"]


class _MockResult:
    """Wraps an array to mimic sim.calculate(...).values."""

    def __init__(self, values):
        self.values = values


class _MockSim:
    """Lightweight mock Microsimulation for rerandomize tests.

    Supports sim.calculate(var, map_to=entity).values and
    sim.set_input(var, period, values). No dataset needed.
    """

    def __init__(self, calc_returns: dict):
        """
        Args:
            calc_returns: Dict mapping (var, entity) -> np.ndarray.
        """
        self._calc = calc_returns
        self.inputs_set = {}

    def calculate(self, var, period=None, map_to=None):
        key = (var, map_to) if map_to else (var, None)
        if key not in self._calc:
            raise ValueError(
                f"MockSim has no return for {key}. "
                f"Available: {list(self._calc.keys())}"
            )
        return _MockResult(self._calc[key])

    def set_input(self, var, period, values):
        self.inputs_set[var] = np.asarray(values)


class TestRerandomizeCategoryTakeup:
    """Integration test: call rerandomize_category_takeup()
    with a mock sim and verify the outputs end-to-end."""

    def _build_mock_sim(self):
        """Build a mock sim with 3 households, 5 tax units,
        6 persons, known child counts and WIC categories."""
        # 3 households
        hh_ids_hh = np.array([100, 200, 300])

        # 5 tax units: HH 100 has 2 TUs, HH 200 has 2, HH 300 has 1
        tu_hh_ids = np.array([100, 100, 200, 200, 300])
        # child counts: 0, 2, 1, 3, 0
        eitc_child_counts = np.array([0, 2, 1, 3, 0])

        # 6 persons: HH 100 has 2, HH 200 has 3, HH 300 has 1
        person_hh_ids = np.array([100, 100, 200, 200, 200, 300])
        wic_cats = np.array(
            [
                "CHILD",
                "NONE",
                "INFANT",
                "PREGNANT",
                "CHILD",
                "NONE",
            ]
        )
        # receives_wic: persons 0, 2, 4 receive WIC
        receives_wic = np.array([True, False, True, False, True, False])

        calc_returns = {
            ("household_id", "household"): hh_ids_hh,
            ("eitc_child_count", "tax_unit"): eitc_child_counts,
            ("household_id", "tax_unit"): tu_hh_ids,
            ("wic_category_str", "person"): wic_cats,
            ("household_id", "person"): person_hh_ids,
            ("receives_wic", "person"): receives_wic,
        }
        return _MockSim(calc_returns), {
            "hh_ids": hh_ids_hh,
            "child_counts": eitc_child_counts,
            "tu_hh_ids": tu_hh_ids,
            "wic_cats": wic_cats,
            "person_hh_ids": person_hh_ids,
            "receives_wic": receives_wic,
        }

    def test_sets_all_variables(self):
        """rerandomize_category_takeup sets takes_up_eitc,
        would_claim_wic, and is_wic_at_nutritional_risk."""
        from policyengine_us_data.calibration.unified_calibration import (
            rerandomize_category_takeup,
        )

        sim, _ = self._build_mock_sim()
        blocks = np.array(
            ["010010001001001", "020020002002002", "030030003003003"]
        )
        rerandomize_category_takeup(sim, blocks, 2024)

        assert "takes_up_eitc" in sim.inputs_set
        assert "would_claim_wic" in sim.inputs_set
        assert "is_wic_at_nutritional_risk" in sim.inputs_set

    def test_output_shapes_match_entities(self):
        """Output arrays match entity counts: 5 tax units
        for EITC, 6 persons for WIC and nutritional risk."""
        from policyengine_us_data.calibration.unified_calibration import (
            rerandomize_category_takeup,
        )

        sim, info = self._build_mock_sim()
        blocks = np.array(["blk_a", "blk_b", "blk_c"])
        rerandomize_category_takeup(sim, blocks, 2024)

        assert len(sim.inputs_set["takes_up_eitc"]) == len(
            info["child_counts"]
        )
        assert len(sim.inputs_set["would_claim_wic"]) == len(info["wic_cats"])
        assert len(sim.inputs_set["is_wic_at_nutritional_risk"]) == len(
            info["wic_cats"]
        )

    def test_outputs_are_boolean(self):
        from policyengine_us_data.calibration.unified_calibration import (
            rerandomize_category_takeup,
        )

        sim, _ = self._build_mock_sim()
        blocks = np.array(["blk_a", "blk_b", "blk_c"])
        rerandomize_category_takeup(sim, blocks, 2024)

        assert sim.inputs_set["takes_up_eitc"].dtype == bool
        assert sim.inputs_set["would_claim_wic"].dtype == bool
        assert sim.inputs_set["is_wic_at_nutritional_risk"].dtype == bool

    def test_deterministic_same_blocks(self):
        """Same blocks produce identical takeup values."""
        from policyengine_us_data.calibration.unified_calibration import (
            rerandomize_category_takeup,
        )

        blocks = np.array(["blk_a", "blk_b", "blk_c"])

        sim1, _ = self._build_mock_sim()
        rerandomize_category_takeup(sim1, blocks, 2024)

        sim2, _ = self._build_mock_sim()
        rerandomize_category_takeup(sim2, blocks, 2024)

        np.testing.assert_array_equal(
            sim1.inputs_set["takes_up_eitc"],
            sim2.inputs_set["takes_up_eitc"],
        )
        np.testing.assert_array_equal(
            sim1.inputs_set["would_claim_wic"],
            sim2.inputs_set["would_claim_wic"],
        )
        np.testing.assert_array_equal(
            sim1.inputs_set["is_wic_at_nutritional_risk"],
            sim2.inputs_set["is_wic_at_nutritional_risk"],
        )

    def test_different_blocks_differ(self):
        """Different blocks produce different takeup values."""
        from policyengine_us_data.calibration.unified_calibration import (
            rerandomize_category_takeup,
        )

        sim1, _ = self._build_mock_sim()
        rerandomize_category_takeup(
            sim1,
            np.array(["aaa", "bbb", "ccc"]),
            2024,
        )

        sim2, _ = self._build_mock_sim()
        rerandomize_category_takeup(
            sim2,
            np.array(["xxx", "yyy", "zzz"]),
            2024,
        )

        # At least one of the three variables should differ
        eitc_differ = not np.array_equal(
            sim1.inputs_set["takes_up_eitc"],
            sim2.inputs_set["takes_up_eitc"],
        )
        wic_differ = not np.array_equal(
            sim1.inputs_set["would_claim_wic"],
            sim2.inputs_set["would_claim_wic"],
        )
        risk_differ = not np.array_equal(
            sim1.inputs_set["is_wic_at_nutritional_risk"],
            sim2.inputs_set["is_wic_at_nutritional_risk"],
        )
        assert eitc_differ or wic_differ or risk_differ

    def test_eitc_values_match_manual_calculation(self):
        """Verify takes_up_eitc matches a hand-computed
        result using the same seeded draws and rates."""
        from policyengine_us_data.calibration.unified_calibration import (
            _eitc_category_mapper,
            rerandomize_category_takeup,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        sim, info = self._build_mock_sim()
        blocks = np.array(
            [
                "010010001001001",
                "020020002002002",
                "030030003003003",
            ]
        )
        rerandomize_category_takeup(sim, blocks, 2024)

        # Reproduce manually
        eitc_rates = load_take_up_rate("eitc", 2024)
        child_counts = info["child_counts"]
        per_tu_rates = _eitc_category_mapper(child_counts, eitc_rates)

        hh_to_block = dict(zip(info["hh_ids"], blocks))
        tu_blocks = np.array(
            [hh_to_block.get(hid, "0") for hid in info["tu_hh_ids"]]
        )
        draws = np.zeros(len(child_counts), dtype=np.float64)
        for block in np.unique(tu_blocks):
            mask = tu_blocks == block
            rng = seeded_rng("takes_up_eitc", salt=str(block))
            draws[mask] = rng.random(mask.sum())

        expected = draws < per_tu_rates
        np.testing.assert_array_equal(
            sim.inputs_set["takes_up_eitc"], expected
        )

    def test_wic_values_match_manual_calculation(self):
        """Verify would_claim_wic matches a hand-computed
        result using the same seeded draws and rates."""
        from policyengine_us_data.calibration.unified_calibration import (
            _wic_category_mapper,
            rerandomize_category_takeup,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        sim, info = self._build_mock_sim()
        blocks = np.array(
            [
                "010010001001001",
                "020020002002002",
                "030030003003003",
            ]
        )
        rerandomize_category_takeup(sim, blocks, 2024)

        # Reproduce manually
        wic_rates = load_take_up_rate("wic_takeup", 2024)
        per_person_rates = _wic_category_mapper(info["wic_cats"], wic_rates)

        hh_to_block = dict(zip(info["hh_ids"], blocks))
        person_blocks = np.array(
            [hh_to_block.get(hid, "0") for hid in info["person_hh_ids"]]
        )
        draws = np.zeros(len(info["wic_cats"]), dtype=np.float64)
        for block in np.unique(person_blocks):
            mask = person_blocks == block
            rng = seeded_rng("would_claim_wic", salt=str(block))
            draws[mask] = rng.random(mask.sum())

        expected = draws < per_person_rates
        np.testing.assert_array_equal(
            sim.inputs_set["would_claim_wic"], expected
        )

    def test_none_wic_never_takes_up(self):
        """Persons with wic_category_str='NONE' should never
        take up WIC (rate=0 from YAML)."""
        from policyengine_us_data.calibration.unified_calibration import (
            rerandomize_category_takeup,
        )

        sim, info = self._build_mock_sim()
        blocks = np.array(["blk_a", "blk_b", "blk_c"])
        rerandomize_category_takeup(sim, blocks, 2024)

        wic_result = sim.inputs_set["would_claim_wic"]
        # Persons at indices 1 and 5 have category "NONE"
        none_indices = [
            i for i, c in enumerate(info["wic_cats"]) if c == "NONE"
        ]
        for idx in none_indices:
            assert (
                wic_result[idx] == False
            ), f"Person {idx} (NONE) should not claim WIC"

    def test_nutritional_risk_values_match_manual_calculation(self):
        """Verify is_wic_at_nutritional_risk matches
        receives_wic | (draws < rates) using seeded draws."""
        from policyengine_us_data.calibration.unified_calibration import (
            _wic_category_mapper,
            rerandomize_category_takeup,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        sim, info = self._build_mock_sim()
        blocks = np.array(
            [
                "010010001001001",
                "020020002002002",
                "030030003003003",
            ]
        )
        rerandomize_category_takeup(sim, blocks, 2024)

        # Reproduce manually
        risk_rates = load_take_up_rate("wic_nutritional_risk", 2024)
        per_person_rates = _wic_category_mapper(info["wic_cats"], risk_rates)

        hh_to_block = dict(zip(info["hh_ids"], blocks))
        person_blocks = np.array(
            [hh_to_block.get(hid, "0") for hid in info["person_hh_ids"]]
        )
        draws = np.zeros(len(info["wic_cats"]), dtype=np.float64)
        for block in np.unique(person_blocks):
            mask = person_blocks == block
            rng = seeded_rng("is_wic_at_nutritional_risk", salt=str(block))
            draws[mask] = rng.random(mask.sum())

        expected = info["receives_wic"] | (draws < per_person_rates)
        np.testing.assert_array_equal(
            sim.inputs_set["is_wic_at_nutritional_risk"],
            expected,
        )

    def test_receives_wic_guarantees_nutritional_risk(self):
        """Anyone with receives_wic=True must have
        is_wic_at_nutritional_risk=True."""
        from policyengine_us_data.calibration.unified_calibration import (
            rerandomize_category_takeup,
        )

        sim, info = self._build_mock_sim()
        blocks = np.array(["blk_a", "blk_b", "blk_c"])
        rerandomize_category_takeup(sim, blocks, 2024)

        risk = sim.inputs_set["is_wic_at_nutritional_risk"]
        for i, recv in enumerate(info["receives_wic"]):
            if recv:
                assert risk[i], (
                    f"Person {i} receives WIC but "
                    f"is_wic_at_nutritional_risk is False"
                )

    def test_none_wic_without_receives_never_at_risk(self):
        """Persons with NONE category who don't receive WIC
        are never at nutritional risk (rate=0)."""
        from policyengine_us_data.calibration.unified_calibration import (
            rerandomize_category_takeup,
        )

        sim, info = self._build_mock_sim()
        blocks = np.array(["blk_a", "blk_b", "blk_c"])
        rerandomize_category_takeup(sim, blocks, 2024)

        risk = sim.inputs_set["is_wic_at_nutritional_risk"]
        for i, (cat, recv) in enumerate(
            zip(info["wic_cats"], info["receives_wic"])
        ):
            if cat == "NONE" and not recv:
                assert not risk[i], (
                    f"Person {i} (NONE, no WIC) should "
                    f"not be at nutritional risk"
                )

    def test_zero_child_eitc_rate_is_lower(self):
        """Tax units with 0 children should have a lower
        takeup rate than those with children, verified
        statistically over many blocks."""
        from policyengine_us_data.calibration.unified_calibration import (
            _eitc_category_mapper,
        )
        from policyengine_us_data.parameters import (
            load_take_up_rate,
        )

        eitc_rates = load_take_up_rate("eitc", 2024)
        # 0-child rate should be strictly less than 1-child rate
        assert eitc_rates[0] < eitc_rates[1]

        # Simulate 10k draws to verify the rates manifest
        n = 10000
        child_counts = np.array([0, 1])
        per_tu_rates = _eitc_category_mapper(child_counts, eitc_rates)
        rng = seeded_rng("takes_up_eitc", salt="stat_test")
        draws = rng.random(n)

        frac_0 = (draws[:n] < per_tu_rates[0]).mean()
        frac_1 = (draws[:n] < per_tu_rates[1]).mean()
        assert frac_0 < frac_1


class TestCategoryTakeupSeeding:
    """Verify seeded draws for category takeup are deterministic
    and vary by block/variable."""

    def test_same_block_same_draws(self):
        var = "takes_up_eitc"
        block = "010010001001001"
        rng1 = seeded_rng(var, salt=block)
        rng2 = seeded_rng(var, salt=block)
        np.testing.assert_array_equal(rng1.random(100), rng2.random(100))

    def test_different_blocks_different_draws(self):
        var = "takes_up_eitc"
        rng1 = seeded_rng(var, salt="010010001001001")
        rng2 = seeded_rng(var, salt="020010001001001")
        assert not np.array_equal(rng1.random(100), rng2.random(100))

    def test_different_category_vars_different_draws(self):
        block = "010010001001001"
        rng1 = seeded_rng("takes_up_eitc", salt=block)
        rng2 = seeded_rng("would_claim_wic", salt=block)
        assert not np.array_equal(rng1.random(100), rng2.random(100))
