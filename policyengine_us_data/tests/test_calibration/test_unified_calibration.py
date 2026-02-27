"""Tests for unified_calibration and shared takeup module.

Verifies geo-salted draws are reproducible and vary by geo_id,
SIMPLE_TAKEUP_VARS / TAKEUP_AFFECTED_TARGETS configs are valid,
block-level takeup seeding, county precomputation, and CLI flags.
"""

import numpy as np
import pytest

from policyengine_us_data.utils.randomness import seeded_rng
from policyengine_us_data.utils.takeup import (
    SIMPLE_TAKEUP_VARS,
    TAKEUP_AFFECTED_TARGETS,
    draw_takeup_for_geo,
    compute_entity_takeup_for_geo,
    compute_block_takeup_for_entities,
    _resolve_rate,
)
from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
)


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


class TestGeoSaltedDraws:
    """Verify draw_takeup_for_geo produces reproducible,
    geo-dependent draws using geo: salt prefix."""

    def test_same_geo_same_draws(self):
        d1 = draw_takeup_for_geo("takes_up_snap_if_eligible", "3701", 500)
        d2 = draw_takeup_for_geo("takes_up_snap_if_eligible", "3701", 500)
        np.testing.assert_array_equal(d1, d2)

    def test_different_geos_different_draws(self):
        d1 = draw_takeup_for_geo("takes_up_snap_if_eligible", "3701", 500)
        d2 = draw_takeup_for_geo("takes_up_snap_if_eligible", "4816", 500)
        assert not np.array_equal(d1, d2)

    def test_different_vars_different_draws(self):
        d1 = draw_takeup_for_geo("takes_up_snap_if_eligible", "3701", 500)
        d2 = draw_takeup_for_geo("takes_up_aca_if_eligible", "3701", 500)
        assert not np.array_equal(d1, d2)

    def test_geo_salt_not_collide_with_block_salt(self):
        d_geo = draw_takeup_for_geo("takes_up_snap_if_eligible", "3701", 500)
        rng_block = seeded_rng("takes_up_snap_if_eligible", salt="3701")
        d_block = rng_block.random(500)
        assert not np.array_equal(d_geo, d_block)

    def test_draws_in_unit_interval(self):
        draws = draw_takeup_for_geo("takes_up_snap_if_eligible", "3701", 10000)
        assert draws.min() >= 0.0
        assert draws.max() < 1.0


class TestComputeEntityTakeup:
    """Verify compute_entity_takeup_for_geo returns
    correct boolean arrays."""

    def test_returns_all_takeup_vars(self):
        n = {"person": 100, "tax_unit": 50, "spm_unit": 40}
        result = compute_entity_takeup_for_geo("3701", n, 37, 2024)
        for spec in SIMPLE_TAKEUP_VARS:
            assert spec["variable"] in result
            assert result[spec["variable"]].dtype == bool

    def test_correct_entity_counts(self):
        n = {"person": 200, "tax_unit": 80, "spm_unit": 60}
        result = compute_entity_takeup_for_geo("3701", n, 37, 2024)
        assert len(result["takes_up_snap_if_eligible"]) == 60
        assert len(result["takes_up_aca_if_eligible"]) == 80
        assert len(result["takes_up_ssi_if_eligible"]) == 200

    def test_reproducible(self):
        n = {"person": 100, "tax_unit": 50, "spm_unit": 40}
        r1 = compute_entity_takeup_for_geo("3701", n, 37, 2024)
        r2 = compute_entity_takeup_for_geo("3701", n, 37, 2024)
        for var in r1:
            np.testing.assert_array_equal(r1[var], r2[var])

    def test_different_geo_different_result(self):
        n = {"person": 100, "tax_unit": 50, "spm_unit": 40}
        r1 = compute_entity_takeup_for_geo("3701", n, 37, 2024)
        r2 = compute_entity_takeup_for_geo("4816", n, 48, 2024)
        differs = any(not np.array_equal(r1[v], r2[v]) for v in r1)
        assert differs


class TestResolveRate:
    """Verify _resolve_rate handles scalar and dict rates."""

    def test_scalar_rate(self):
        assert _resolve_rate(0.82, 37) == 0.82

    def test_state_dict_rate(self):
        rates = {"NC": 0.94, "TX": 0.76}
        assert _resolve_rate(rates, 37) == 0.94
        assert _resolve_rate(rates, 48) == 0.76

    def test_unknown_state_fallback(self):
        rates = {"NC": 0.94}
        assert _resolve_rate(rates, 99) == 0.8


class TestSimpleTakeupConfig:
    """Verify the SIMPLE_TAKEUP_VARS config is well-formed."""

    def test_all_entries_have_required_keys(self):
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
        assert len(SIMPLE_TAKEUP_VARS) == 9

    def test_importable_from_unified_calibration(self):
        from policyengine_us_data.calibration.unified_calibration import (
            SIMPLE_TAKEUP_VARS as UC_VARS,
        )

        assert UC_VARS is SIMPLE_TAKEUP_VARS


class TestTakeupAffectedTargets:
    """Verify TAKEUP_AFFECTED_TARGETS is consistent."""

    def test_all_entries_have_required_keys(self):
        for key, info in TAKEUP_AFFECTED_TARGETS.items():
            assert "takeup_var" in info
            assert "entity" in info
            assert "rate_key" in info
            assert info["entity"] in (
                "person",
                "tax_unit",
                "spm_unit",
            )

    def test_takeup_vars_exist_in_simple_vars(self):
        simple_var_names = {s["variable"] for s in SIMPLE_TAKEUP_VARS}
        for info in TAKEUP_AFFECTED_TARGETS.values():
            assert info["takeup_var"] in simple_var_names


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

    def test_skip_takeup_rerandomize_flag(self):
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(["--skip-takeup-rerandomize"])
        assert args.skip_takeup_rerandomize is True

        args_default = parse_args([])
        assert args_default.skip_takeup_rerandomize is False


class TestGeographyAssignmentCountyFips:
    """Verify county_fips field on GeographyAssignment."""

    def test_county_fips_equals_block_prefix(self):
        blocks = np.array(
            ["370010001001001", "480010002002002", "060370003003003"]
        )
        ga = GeographyAssignment(
            block_geoid=blocks,
            cd_geoid=np.array(["3701", "4801", "0613"]),
            county_fips=np.array([b[:5] for b in blocks]),
            state_fips=np.array([37, 48, 6]),
            n_records=3,
            n_clones=1,
        )
        expected = np.array(["37001", "48001", "06037"])
        np.testing.assert_array_equal(ga.county_fips, expected)

    def test_county_fips_length(self):
        blocks = np.array(["370010001001001"] * 5)
        counties = np.array([b[:5] for b in blocks])
        ga = GeographyAssignment(
            block_geoid=blocks,
            cd_geoid=np.array(["3701"] * 5),
            county_fips=counties,
            state_fips=np.array([37] * 5),
            n_records=5,
            n_clones=1,
        )
        assert len(ga.county_fips) == 5
        assert all(len(c) == 5 for c in ga.county_fips)


class TestBlockTakeupSeeding:
    """Verify compute_block_takeup_for_entities is
    reproducible and block-dependent."""

    def test_reproducible(self):
        blocks = np.array(["010010001001001"] * 50 + ["020010001001001"] * 50)
        states = np.array([1] * 50 + [2] * 50)
        r1 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, states
        )
        r2 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, states
        )
        np.testing.assert_array_equal(r1, r2)

    def test_different_blocks_different_draws(self):
        n = 500
        blocks_a = np.array(["010010001001001"] * n)
        blocks_b = np.array(["020010001001001"] * n)
        states = np.array([1] * n)
        r_a = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks_a, states
        )
        r_b = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks_b, states
        )
        assert not np.array_equal(r_a, r_b)

    def test_returns_booleans(self):
        blocks = np.array(["370010001001001"] * 100)
        states = np.array([37] * 100)
        result = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, states
        )
        assert result.dtype == bool

    def test_rate_respected(self):
        n = 10000
        blocks = np.array(["370010001001001"] * n)
        states = np.array([37] * n)
        result = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.75, blocks, states
        )
        frac = result.mean()
        assert 0.70 < frac < 0.80


class TestAssembleCloneValuesCounty:
    """Verify _assemble_clone_values merges state and
    county values correctly."""

    def test_county_var_uses_county_values(self):
        from policyengine_us_data.calibration.unified_matrix_builder import (
            UnifiedMatrixBuilder,
        )

        n = 4
        state_values = {
            1: {
                "hh": {
                    "aca_ptc": np.array([100] * n, dtype=np.float32),
                },
                "person": {},
                "entity": {},
            },
            2: {
                "hh": {
                    "aca_ptc": np.array([200] * n, dtype=np.float32),
                },
                "person": {},
                "entity": {},
            },
        }
        county_values = {
            "01001": {
                "hh": {
                    "aca_ptc": np.array([111] * n, dtype=np.float32),
                },
                "entity": {},
            },
            "02001": {
                "hh": {
                    "aca_ptc": np.array([222] * n, dtype=np.float32),
                },
                "entity": {},
            },
        }
        clone_states = np.array([1, 1, 2, 2])
        clone_counties = np.array(["01001", "01001", "02001", "02001"])
        person_hh_idx = np.array([0, 1, 2, 3])

        builder = UnifiedMatrixBuilder.__new__(UnifiedMatrixBuilder)
        hh_vars, _ = builder._assemble_clone_values(
            state_values,
            clone_states,
            person_hh_idx,
            {"aca_ptc"},
            set(),
            county_values=county_values,
            clone_counties=clone_counties,
            county_dependent_vars={"aca_ptc"},
        )
        expected = np.array([111, 111, 222, 222], dtype=np.float32)
        np.testing.assert_array_equal(hh_vars["aca_ptc"], expected)

    def test_non_county_var_uses_state_values(self):
        from policyengine_us_data.calibration.unified_matrix_builder import (
            UnifiedMatrixBuilder,
        )

        n = 4
        state_values = {
            1: {
                "hh": {
                    "snap": np.array([50] * n, dtype=np.float32),
                },
                "person": {},
                "entity": {},
            },
            2: {
                "hh": {
                    "snap": np.array([60] * n, dtype=np.float32),
                },
                "person": {},
                "entity": {},
            },
        }
        clone_states = np.array([1, 1, 2, 2])
        clone_counties = np.array(["01001", "01001", "02001", "02001"])
        person_hh_idx = np.array([0, 1, 2, 3])

        builder = UnifiedMatrixBuilder.__new__(UnifiedMatrixBuilder)
        hh_vars, _ = builder._assemble_clone_values(
            state_values,
            clone_states,
            person_hh_idx,
            {"snap"},
            set(),
            county_values={},
            clone_counties=clone_counties,
            county_dependent_vars={"aca_ptc"},
        )
        expected = np.array([50, 50, 60, 60], dtype=np.float32)
        np.testing.assert_array_equal(hh_vars["snap"], expected)


class TestConvertBlocksToStackedFormat:
    """Verify convert_blocks_to_stacked_format produces
    correct stacked block arrays."""

    def test_basic_conversion(self):
        from policyengine_us_data.calibration.unified_calibration import (
            convert_blocks_to_stacked_format,
        )

        block_geoid = np.array(
            [
                "370010001001001",
                "370010001001002",
                "480010002002001",
                "480010002002002",
            ]
        )
        cd_geoid = np.array(["3701", "3701", "4801", "4801"])
        base_n_records = 2
        cds_ordered = ["3701", "4801"]

        result = convert_blocks_to_stacked_format(
            block_geoid, cd_geoid, base_n_records, cds_ordered
        )
        assert result.dtype.kind == "U"
        assert len(result) == 4
        assert result[0] == "370010001001001"
        assert result[1] == "370010001001002"
        assert result[2] == "480010002002001"
        assert result[3] == "480010002002002"

    def test_empty_slots(self):
        from policyengine_us_data.calibration.unified_calibration import (
            convert_blocks_to_stacked_format,
        )

        block_geoid = np.array(["370010001001001", "370010001001002"])
        cd_geoid = np.array(["3701", "3701"])
        base_n_records = 2
        cds_ordered = ["3701", "4801"]

        result = convert_blocks_to_stacked_format(
            block_geoid, cd_geoid, base_n_records, cds_ordered
        )
        assert len(result) == 4
        assert result[0] == "370010001001001"
        assert result[1] == "370010001001002"
        assert result[2] == ""
        assert result[3] == ""

    def test_first_clone_wins(self):
        from policyengine_us_data.calibration.unified_calibration import (
            convert_blocks_to_stacked_format,
        )

        block_geoid = np.array(
            [
                "370010001001001",
                "370010001001002",
                "370010001001099",
                "370010001001099",
            ]
        )
        cd_geoid = np.array(["3701", "3701", "3701", "3701"])
        base_n_records = 2
        cds_ordered = ["3701"]

        result = convert_blocks_to_stacked_format(
            block_geoid, cd_geoid, base_n_records, cds_ordered
        )
        assert result[0] == "370010001001001"
        assert result[1] == "370010001001002"


class TestTakeupDrawConsistency:
    """Verify the matrix builder's inline takeup loop and
    compute_block_takeup_for_entities produce identical draws
    when given the same (block, household) inputs."""

    def test_matrix_and_stacked_identical_draws(self):
        """Both paths must produce identical boolean arrays."""
        var = "takes_up_snap_if_eligible"
        rate = 0.75

        # 2 blocks, 3 households, variable entity counts per HH
        # HH0 has 2 entities in block A
        # HH1 has 3 entities in block A
        # HH2 has 1 entity in block B
        blocks = np.array(
            [
                "370010001001001",
                "370010001001001",
                "370010001001001",
                "370010001001001",
                "370010001001001",
                "480010002002002",
            ]
        )
        hh_ids = np.array([100, 100, 200, 200, 200, 300])
        states = np.array([37, 37, 37, 37, 37, 48])

        # Path 1: compute_block_takeup_for_entities (stacked)
        stacked = compute_block_takeup_for_entities(
            var, rate, blocks, states, hh_ids
        )

        # Path 2: reproduce matrix builder inline logic
        n = len(blocks)
        inline_takeup = np.zeros(n, dtype=bool)
        for blk in np.unique(blocks):
            bm = blocks == blk
            for hh_id in np.unique(hh_ids[bm]):
                hh_mask = bm & (hh_ids == hh_id)
                rng = seeded_rng(var, salt=f"{blk}:{int(hh_id)}")
                draws = rng.random(int(hh_mask.sum()))
                inline_takeup[hh_mask] = draws < rate

        np.testing.assert_array_equal(stacked, inline_takeup)

    def test_aggregation_entity_to_household(self):
        """np.add.at aggregation matches manual per-HH sum."""
        n_hh = 3
        n_ent = 6
        ent_hh = np.array([0, 0, 1, 1, 1, 2])
        eligible = np.array(
            [100.0, 200.0, 50.0, 150.0, 100.0, 300.0],
            dtype=np.float32,
        )
        takeup = np.array([True, False, True, True, False, True])

        ent_values = (eligible * takeup).astype(np.float32)
        hh_result = np.zeros(n_hh, dtype=np.float32)
        np.add.at(hh_result, ent_hh, ent_values)

        # Manual: HH0=100, HH1=50+150=200, HH2=300
        expected = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        np.testing.assert_array_equal(hh_result, expected)

    def test_state_specific_rate_resolved_from_block(self):
        """Dict rates are resolved per block's state FIPS."""
        from policyengine_us_data.utils.takeup import _resolve_rate

        var = "takes_up_snap_if_eligible"
        rate_dict = {"NC": 0.9, "TX": 0.6}
        n = 5000

        blocks_nc = np.array(["370010001001001"] * n)
        states_nc = np.array([37] * n)
        result_nc = compute_block_takeup_for_entities(
            var, rate_dict, blocks_nc, states_nc
        )
        # NC rate=0.9, expect ~90%
        frac_nc = result_nc.mean()
        assert 0.85 < frac_nc < 0.95, f"NC frac={frac_nc}"

        blocks_tx = np.array(["480010002002002"] * n)
        states_tx = np.array([48] * n)
        result_tx = compute_block_takeup_for_entities(
            var, rate_dict, blocks_tx, states_tx
        )
        # TX rate=0.6, expect ~60%
        frac_tx = result_tx.mean()
        assert 0.55 < frac_tx < 0.65, f"TX frac={frac_tx}"

        # Verify _resolve_rate actually gives different rates
        assert _resolve_rate(rate_dict, 37) == 0.9
        assert _resolve_rate(rate_dict, 48) == 0.6


class TestDeriveGeographyFromBlocks:
    """Verify derive_geography_from_blocks returns correct
    geography dict from pre-assigned blocks."""

    def test_returns_expected_keys(self):
        from policyengine_us_data.calibration.block_assignment import (
            derive_geography_from_blocks,
        )

        blocks = np.array(["370010001001001"])
        result = derive_geography_from_blocks(blocks)
        expected_keys = {
            "block_geoid",
            "county_fips",
            "tract_geoid",
            "state_fips",
            "cbsa_code",
            "sldu",
            "sldl",
            "place_fips",
            "vtd",
            "puma",
            "zcta",
            "county_index",
        }
        assert set(result.keys()) == expected_keys

    def test_county_fips_derived(self):
        from policyengine_us_data.calibration.block_assignment import (
            derive_geography_from_blocks,
        )

        blocks = np.array(["370010001001001", "480010002002002"])
        result = derive_geography_from_blocks(blocks)
        np.testing.assert_array_equal(
            result["county_fips"],
            np.array(["37001", "48001"]),
        )

    def test_state_fips_derived(self):
        from policyengine_us_data.calibration.block_assignment import (
            derive_geography_from_blocks,
        )

        blocks = np.array(["370010001001001", "060370003003003"])
        result = derive_geography_from_blocks(blocks)
        np.testing.assert_array_equal(
            result["state_fips"],
            np.array(["37", "06"]),
        )

    def test_tract_geoid_derived(self):
        from policyengine_us_data.calibration.block_assignment import (
            derive_geography_from_blocks,
        )

        blocks = np.array(["370010001001001"])
        result = derive_geography_from_blocks(blocks)
        assert result["tract_geoid"][0] == "37001000100"

    def test_block_geoid_passthrough(self):
        from policyengine_us_data.calibration.block_assignment import (
            derive_geography_from_blocks,
        )

        blocks = np.array(["370010001001001"])
        result = derive_geography_from_blocks(blocks)
        assert result["block_geoid"][0] == "370010001001001"
