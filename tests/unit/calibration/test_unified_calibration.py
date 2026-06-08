"""Tests for unified_calibration and shared takeup module.

Verifies geo-salted draws are reproducible and vary by geo_id,
SIMPLE_TAKEUP_VARS / TAKEUP_AFFECTED_TARGETS configs are valid,
block-level takeup seeding, county precomputation, and CLI flags.
"""

import sys
import types

import numpy as np
import pytest
import scipy.sparse as sp
from types import SimpleNamespace
from unittest.mock import patch

# Ensure `l0.calibration` is importable so patch() can traverse the path
# even when the real l0-python package is not installed (e.g. CI).
if "l0" not in sys.modules:
    _l0 = types.ModuleType("l0")
    _l0.calibration = types.ModuleType("l0.calibration")
    _l0.calibration.SparseCalibrationWeights = None
    sys.modules["l0"] = _l0
    sys.modules["l0.calibration"] = _l0.calibration

from policyengine_us_data.utils.randomness import seeded_rng
from policyengine_us_data.utils.takeup import (
    SIMPLE_TAKEUP_VARS,
    TAKEUP_AFFECTED_TARGETS,
    adjust_aca_takeup_to_match_enrollment_and_spending_targets,
    adjust_aca_takeup_to_match_target,
    adjust_aca_takeup_to_state_targets,
    apply_block_takeup_to_arrays,
    compute_voluntary_filing_takeup_for_tax_units,
    compute_block_takeup_draws_for_entities,
    compute_block_takeup_for_entities,
    extend_aca_takeup_to_match_target,
    _resolve_rate,
)
from policyengine_us_data.calibration.clone_and_assign import (
    GeographyAssignment,
)
from policyengine_us_data.calibration.unified_calibration import (
    PRESETS,
    _calibration_package_contract_parameters,
    _target_config_identity_for_metadata,
    check_package_staleness,
    run_calibration,
)
from policyengine_us_data.stage_contracts.calibration_package import (
    CalibrationPackageParameters,
)

TARGET_CONFIG_SHA256 = "sha256:" + "a" * 64


def test_national_preset_has_no_l0_penalty():
    assert PRESETS["national"] == pytest.approx(0.0)
    assert PRESETS["local"] > 0


def test_calibration_package_contract_parameters_track_effective_matrix_mode():
    params = _calibration_package_contract_parameters(
        workers=8,
        n_clones=430,
        target_config_path="policyengine_us_data/calibration/target_config.yaml",
        target_config_sha256=TARGET_CONFIG_SHA256,
        target_config_mode="default",
        skip_county=True,
        skip_source_impute=True,
        skip_takeup_rerandomize=False,
        chunked_matrix=True,
        chunk_size=25_000,
        parallel=True,
        num_matrix_workers=50,
    )

    assert isinstance(params, CalibrationPackageParameters)
    assert params.to_dict() == {
        "workers": None,
        "n_clones": 430,
        "target_config": "policyengine_us_data/calibration/target_config.yaml",
        "target_config_sha256": TARGET_CONFIG_SHA256,
        "target_config_mode": "default",
        "skip_county": True,
        "skip_source_impute": True,
        "skip_takeup_rerandomize": False,
        "chunked_matrix": True,
        "chunk_size": 25_000,
        "parallel_matrix": True,
        "num_matrix_workers": 50,
    }


def test_calibration_package_contract_parameters_ignore_unused_chunk_options():
    params = _calibration_package_contract_parameters(
        workers=8,
        n_clones=430,
        target_config_path=None,
        target_config_sha256=None,
        target_config_mode="all_active_targets",
        skip_county=True,
        skip_source_impute=True,
        skip_takeup_rerandomize=False,
        chunked_matrix=False,
        chunk_size=25_000,
        parallel=True,
        num_matrix_workers=50,
    )

    assert params.to_dict()["workers"] == 8
    assert params.to_dict()["chunk_size"] is None
    assert params.to_dict()["parallel_matrix"] is False
    assert params.to_dict()["num_matrix_workers"] is None


def test_target_config_identity_for_metadata_requires_identity_for_parsed_config():
    with pytest.raises(
        ValueError, match="target_config_path or target_config_identity"
    ):
        _target_config_identity_for_metadata(
            target_config={"include": []},
            target_config_path=None,
            target_config_identity=None,
        )


def test_run_calibration_validates_target_identity_before_dataset_loading():
    with (
        patch.dict(sys.modules, {"policyengine_us": None}),
        pytest.raises(
            ValueError,
            match="target_config_path or target_config_identity",
        ),
    ):
        run_calibration(
            dataset_path="/missing/source.h5",
            db_path="/missing/policy_data.db",
            target_config={"include": []},
            target_config_path=None,
        )


def test_check_package_staleness_warns_for_old_utc_timestamp(
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.unified_calibration.get_git_provenance",
        lambda: {"git_branch": "main"},
    )

    check_package_staleness(
        {
            "created_at": "2000-01-01T00:00:00Z",
            "git_branch": "main",
        }
    )

    assert "WARNING: Package is" in capsys.readouterr().out


class TestForbesStateOverrides:
    def test_extracts_only_synthetic_puf_state_fips(self):
        from policyengine_us_data.calibration.unified_calibration import (
            _extract_forbes_state_fips_overrides,
        )

        raw_dataset = {
            "household_id": {2024: np.array([10, 1_000_000, 1_000_001])},
            "forbes_state_fips": {2024: np.array([6, 36, 0])},
        }

        result = _extract_forbes_state_fips_overrides(
            raw_dataset=raw_dataset,
            time_period=2024,
            n_records=3,
        )

        np.testing.assert_array_equal(result, np.array([0, 36, 0]))

    def test_ignores_ordinary_positive_state_fips(self):
        from policyengine_us_data.calibration.unified_calibration import (
            _extract_forbes_state_fips_overrides,
        )

        raw_dataset = {
            "household_id": {2024: np.array([10, 20, 30])},
            "forbes_state_fips": {2024: np.array([6, 36, 48])},
        }

        result = _extract_forbes_state_fips_overrides(
            raw_dataset=raw_dataset,
            time_period=2024,
            n_records=3,
        )

        assert result is None


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


class TestBlockSaltedDraws:
    """Verify compute_block_takeup_for_entities produces
    reproducible, clone-dependent draws."""

    def test_same_inputs_same_results(self):
        n = 500
        blocks = np.array(["370010001001001"] * n)
        hh_ids = np.arange(n, dtype=np.int64)
        ci = np.zeros(n, dtype=np.int64)
        d1 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_ids, ci
        )
        d2 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_ids, ci
        )
        np.testing.assert_array_equal(d1, d2)

    def test_different_clone_idx_different_results(self):
        n = 500
        blocks = np.array(["370010001001001"] * n)
        hh_ids = np.arange(n, dtype=np.int64)
        ci0 = np.zeros(n, dtype=np.int64)
        ci1 = np.ones(n, dtype=np.int64)
        d1 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_ids, ci0
        )
        d2 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_ids, ci1
        )
        assert not np.array_equal(d1, d2)

    def test_different_vars_different_results(self):
        n = 500
        blocks = np.array(["370010001001001"] * n)
        hh_ids = np.arange(n, dtype=np.int64)
        ci = np.zeros(n, dtype=np.int64)
        d1 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_ids, ci
        )
        d2 = compute_block_takeup_for_entities(
            "takes_up_aca_if_eligible", 0.8, blocks, hh_ids, ci
        )
        assert not np.array_equal(d1, d2)

    def test_different_hh_ids_different_results(self):
        n = 500
        blocks = np.array(["370010001001001"] * n)
        ci = np.zeros(n, dtype=np.int64)
        hh_a = np.arange(n, dtype=np.int64)
        hh_b = np.arange(n, dtype=np.int64) + 1000
        d1 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_a, ci
        )
        d2 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_b, ci
        )
        assert not np.array_equal(d1, d2)

    def test_empty_blocks_do_not_crash_state_rate_takeup(self):
        result = compute_block_takeup_for_entities(
            "takes_up_medicaid_if_eligible",
            {"NC": 0.9},
            np.array(["", "370010001001001"]),
            np.array([1, 2], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        )

        assert result.dtype == bool
        assert len(result) == 2
        assert result[0]

    def test_nested_rate_table_cannot_use_state_rate_fallback(self):
        with pytest.raises(ValueError, match="Cell-based take-up rates"):
            compute_block_takeup_for_entities(
                "would_file_taxes_voluntarily",
                {"no_children": {"low": {"under_65": 0.24}}},
                np.array(["370010001001001"]),
                np.array([1], dtype=np.int64),
                np.array([0], dtype=np.int64),
            )

    def test_voluntary_filing_uses_demographic_rates(self):
        rates = {
            "no_children": {
                "zero": {"under_65": 0.0, "age_65_plus": 0.0},
                "low": {"under_65": 0.0, "age_65_plus": 0.0},
                "medium": {"under_65": 0.0, "age_65_plus": 0.0},
                "high": {"under_65": 0.0, "age_65_plus": 0.0},
            },
            "with_children": {
                "zero": {"under_65": 1.0, "age_65_plus": 1.0},
                "low": {"under_65": 1.0, "age_65_plus": 1.0},
                "medium": {"under_65": 1.0, "age_65_plus": 1.0},
                "high": {"under_65": 1.0, "age_65_plus": 1.0},
            },
        }

        result = compute_voluntary_filing_takeup_for_tax_units(
            rates,
            np.array(["370010001001001", "370010001001001"]),
            np.array([1, 2], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
            tax_unit_child_dependents=np.array([0, 1]),
            tax_unit_wage_income=np.array([10_000, 10_000], dtype=np.float32),
            age_head=np.array([40, 40]),
        )

        np.testing.assert_array_equal(result, np.array([False, True]))

    def test_eligible_mask_preserves_reported_and_limits_others(self):
        result = compute_block_takeup_for_entities(
            "takes_up_housing_assistance_if_eligible",
            1.0,
            np.array(["370010001001001"] * 4),
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            reported_mask=np.array([True, False, False, False]),
            eligible_mask=np.array([False, False, True, False]),
        )

        np.testing.assert_array_equal(
            result,
            np.array([True, False, True, False]),
        )


class TestApplyBlockTakeupToArrays:
    """Verify apply_block_takeup_to_arrays returns correct
    boolean arrays for all entity levels."""

    def _make_arrays(self, n_hh, persons_per_hh, tu_per_hh, spm_per_hh):
        """Build test arrays for n_hh households."""
        n_p = n_hh * persons_per_hh
        n_tu = n_hh * tu_per_hh
        n_spm = n_hh * spm_per_hh
        hh_blocks = np.array(["370010001001001"] * n_hh)
        hh_state_fips = np.array([37] * n_hh, dtype=np.int32)
        hh_ids = np.arange(n_hh, dtype=np.int64)
        hh_clone_indices = np.zeros(n_hh, dtype=np.int64)
        entity_hh_indices = {
            "person": np.repeat(np.arange(n_hh), persons_per_hh),
            "tax_unit": np.repeat(np.arange(n_hh), tu_per_hh),
            "spm_unit": np.repeat(np.arange(n_hh), spm_per_hh),
        }
        entity_counts = {
            "person": n_p,
            "tax_unit": n_tu,
            "spm_unit": n_spm,
        }
        return (
            hh_blocks,
            hh_state_fips,
            hh_ids,
            hh_clone_indices,
            entity_hh_indices,
            entity_counts,
        )

    def test_returns_all_takeup_vars(self):
        args = self._make_arrays(10, 3, 2, 1)
        result = apply_block_takeup_to_arrays(
            *args,
            time_period=2024,
            voluntary_filing_inputs={
                "tax_unit_child_dependents": np.zeros(20),
                "tax_unit_wage_income": np.zeros(20, dtype=np.float32),
                "age_head": np.full(20, 40),
            },
        )
        for spec in SIMPLE_TAKEUP_VARS:
            assert spec["variable"] in result
            assert result[spec["variable"]].dtype == bool

    def test_correct_entity_counts(self):
        args = self._make_arrays(20, 10, 4, 3)
        result = apply_block_takeup_to_arrays(
            *args,
            time_period=2024,
            voluntary_filing_inputs={
                "tax_unit_child_dependents": np.zeros(80),
                "tax_unit_wage_income": np.zeros(80, dtype=np.float32),
                "age_head": np.full(80, 40),
            },
        )
        assert len(result["takes_up_snap_if_eligible"]) == 60
        assert len(result["takes_up_aca_if_eligible"]) == 80
        assert len(result["takes_up_ssi_if_eligible"]) == 200

    def test_reproducible(self):
        args = self._make_arrays(10, 3, 2, 1)
        voluntary_filing_inputs = {
            "tax_unit_child_dependents": np.zeros(20),
            "tax_unit_wage_income": np.zeros(20, dtype=np.float32),
            "age_head": np.full(20, 40),
        }
        r1 = apply_block_takeup_to_arrays(
            *args,
            time_period=2024,
            voluntary_filing_inputs=voluntary_filing_inputs,
        )
        r2 = apply_block_takeup_to_arrays(
            *args,
            time_period=2024,
            voluntary_filing_inputs=voluntary_filing_inputs,
        )
        for var in r1:
            np.testing.assert_array_equal(r1[var], r2[var])

    def test_different_blocks_different_result(self):
        args_a = self._make_arrays(10, 3, 2, 1)
        voluntary_filing_inputs = {
            "tax_unit_child_dependents": np.zeros(20),
            "tax_unit_wage_income": np.zeros(20, dtype=np.float32),
            "age_head": np.full(20, 40),
        }
        r1 = apply_block_takeup_to_arrays(
            *args_a,
            time_period=2024,
            voluntary_filing_inputs=voluntary_filing_inputs,
        )

        args_b = list(self._make_arrays(10, 3, 2, 1))
        args_b[0] = np.array(["480010002002002"] * 10)
        args_b[1] = np.array([48] * 10, dtype=np.int32)
        r2 = apply_block_takeup_to_arrays(
            *args_b,
            time_period=2024,
            voluntary_filing_inputs=voluntary_filing_inputs,
        )

        differs = any(not np.array_equal(r1[v], r2[v]) for v in r1)
        assert differs

    def test_reported_anchors_feed_through_for_aca(self):
        args = self._make_arrays(4, 1, 1, 1)
        result = apply_block_takeup_to_arrays(
            *args,
            time_period=2024,
            takeup_filter=["takes_up_aca_if_eligible"],
            precomputed_rates={"aca": 0.25},
            reported_anchors={
                "takes_up_aca_if_eligible": np.array([True, False, False, False])
            },
        )
        np.testing.assert_array_equal(
            result["takes_up_aca_if_eligible"],
            [True, False, False, False],
        )

    def test_voluntary_filing_requires_demographics(self):
        args = self._make_arrays(4, 1, 1, 1)

        with pytest.raises(ValueError, match="voluntary_filing_inputs"):
            apply_block_takeup_to_arrays(
                *args,
                time_period=2024,
                takeup_filter=["would_file_taxes_voluntarily"],
                precomputed_rates={
                    "voluntary_filing": {
                        "no_children": {"zero": {"under_65": 0.0}},
                    },
                },
            )

    def test_apply_block_takeup_uses_voluntary_filing_demographics(self):
        args = self._make_arrays(2, 1, 1, 1)
        rates = {
            "no_children": {
                "zero": {"under_65": 0.0, "age_65_plus": 0.0},
                "low": {"under_65": 0.0, "age_65_plus": 0.0},
                "medium": {"under_65": 0.0, "age_65_plus": 0.0},
                "high": {"under_65": 0.0, "age_65_plus": 0.0},
            },
            "with_children": {
                "zero": {"under_65": 1.0, "age_65_plus": 1.0},
                "low": {"under_65": 1.0, "age_65_plus": 1.0},
                "medium": {"under_65": 1.0, "age_65_plus": 1.0},
                "high": {"under_65": 1.0, "age_65_plus": 1.0},
            },
        }

        result = apply_block_takeup_to_arrays(
            *args,
            time_period=2024,
            takeup_filter=["would_file_taxes_voluntarily"],
            precomputed_rates={"voluntary_filing": rates},
            voluntary_filing_inputs={
                "tax_unit_child_dependents": np.array([0, 1]),
                "tax_unit_wage_income": np.array([10_000, 10_000], dtype=np.float32),
                "age_head": np.array([40, 40]),
            },
        )

        np.testing.assert_array_equal(
            result["would_file_taxes_voluntarily"],
            np.array([False, True]),
        )

    def test_apply_block_takeup_passes_eligibility_masks(self):
        args = self._make_arrays(4, 1, 1, 1)

        result = apply_block_takeup_to_arrays(
            *args,
            time_period=2024,
            takeup_filter=["takes_up_housing_assistance_if_eligible"],
            precomputed_rates={"housing_assistance": 1.0},
            reported_anchors={
                "takes_up_housing_assistance_if_eligible": np.array(
                    [True, False, False, False]
                )
            },
            eligibility_masks={
                "takes_up_housing_assistance_if_eligible": np.array(
                    [False, False, True, False]
                )
            },
        )

        np.testing.assert_array_equal(
            result["takes_up_housing_assistance_if_eligible"],
            np.array([True, False, True, False]),
        )


class TestAcaTakeupTargeting:
    """Verify ACA post-calibration targeting helpers."""

    def test_draw_helper_matches_boolean_helper(self):
        blocks = np.array(["370010001001001"] * 25)
        hh_ids = np.arange(25, dtype=np.int64)
        draws = compute_block_takeup_draws_for_entities(
            "takes_up_aca_if_eligible",
            blocks,
            hh_ids,
        )
        result = compute_block_takeup_for_entities(
            "takes_up_aca_if_eligible",
            0.7,
            blocks,
            hh_ids,
        )
        np.testing.assert_array_equal(result, draws < 0.7)

    def test_extend_only_adds_true_values_until_target(self):
        base_takeup = np.array([True, False, False, False], dtype=bool)
        entity_draws = np.array([0.10, 0.40, 0.20, 0.30], dtype=np.float64)
        enrolled_person_weights = np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float64)

        result = extend_aca_takeup_to_match_target(
            base_takeup,
            entity_draws,
            enrolled_person_weights,
            target_people=6.0,
        )

        np.testing.assert_array_equal(
            result,
            np.array([True, False, True, True], dtype=bool),
        )

    def test_adjust_removes_high_draw_takers_when_above_target(self):
        base_takeup = np.array([True, True, True, False], dtype=bool)
        entity_draws = np.array([0.10, 0.90, 0.20, 0.30], dtype=np.float64)
        enrolled_person_weights = np.array([2.0, 5.0, 3.0, 4.0], dtype=np.float64)

        result = adjust_aca_takeup_to_match_target(
            base_takeup,
            entity_draws,
            enrolled_person_weights,
            target_people=5.0,
        )

        np.testing.assert_array_equal(
            result,
            np.array([True, False, True, False], dtype=bool),
        )

    def test_adjust_state_targets_adds_and_removes_independently(self):
        base_takeup = np.array([True, True, False, False], dtype=bool)
        entity_draws = np.array([0.90, 0.10, 0.20, 0.30], dtype=np.float64)
        enrolled_person_weights = np.array([5.0, 4.0, 7.0, 3.0], dtype=np.float64)
        state_codes = np.array(["NY", "NY", "FL", "FL"])

        result = adjust_aca_takeup_to_state_targets(
            base_takeup,
            entity_draws,
            enrolled_person_weights,
            entity_state_codes=state_codes,
            target_people_by_state={"NY": 4.0, "FL": 10.0},
        )

        np.testing.assert_array_equal(
            result,
            np.array([False, True, True, True], dtype=bool),
        )

    def test_adjust_targets_spending_per_person_when_provided(self):
        base_takeup = np.array([True, True, True], dtype=bool)
        entity_draws = np.array([0.30, 0.10, 0.20], dtype=np.float64)
        enrolled_person_weights = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        assigned_spending_weights = np.array(
            [100.0, 500.0, 1_000.0],
            dtype=np.float64,
        )

        result = adjust_aca_takeup_to_match_enrollment_and_spending_targets(
            base_takeup,
            entity_draws,
            enrolled_person_weights,
            assigned_spending_weights,
            target_people=100.0,
            target_spending=1_000.0,
        )

        np.testing.assert_array_equal(
            result,
            np.array([False, False, True], dtype=bool),
        )

    def test_state_targets_use_spending_when_available(self):
        base_takeup = np.array([False, False, False, False], dtype=bool)
        entity_draws = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
        enrolled_person_weights = np.array([100.0, 100.0, 100.0, 100.0])
        assigned_spending_weights = np.array([100.0, 1_000.0, 500.0, 100.0])
        state_codes = np.array(["NY", "NY", "FL", "FL"])

        result = adjust_aca_takeup_to_state_targets(
            base_takeup,
            entity_draws,
            enrolled_person_weights,
            entity_state_codes=state_codes,
            target_people_by_state={"NY": 100.0, "FL": 100.0},
            assigned_spending_weights=assigned_spending_weights,
            target_spending_by_state={"NY": 1_000.0, "FL": 100.0},
        )

        np.testing.assert_array_equal(
            result,
            np.array([False, True, False, True], dtype=bool),
        )


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
        assert len(SIMPLE_TAKEUP_VARS) == 10


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

    def test_all_active_targets_flag(self):
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(["--all-active-targets"])
        assert args.all_active_targets is True

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

    def test_resume_flags(self):
        from policyengine_us_data.calibration.unified_calibration import (
            parse_args,
        )

        args = parse_args(
            [
                "--resume-from",
                "weights.npy",
                "--checkpoint-output",
                "weights.checkpoint.pt",
            ]
        )
        assert args.resume_from == "weights.npy"
        assert args.checkpoint_output == "weights.checkpoint.pt"

        args_default = parse_args([])
        assert args_default.checkpoint_output is None


class FakeSparseCalibrationWeights:
    fit_calls = []

    def __init__(
        self,
        n_features,
        beta=None,
        gamma=None,
        zeta=None,
        init_keep_prob=None,
        init_weights=None,
        log_weight_jitter_sd=0.0,
        log_alpha_jitter_sd=0.0,
        device="cpu",
    ):
        import torch

        self.n_features = n_features
        self.device = device
        self.log_weight_jitter_sd = log_weight_jitter_sd
        weight_values = (
            np.ones(n_features, dtype=np.float32)
            if init_weights is None
            else np.asarray(init_weights, dtype=np.float32)
        )
        self.weights = torch.tensor(weight_values, dtype=torch.float32)
        self.alpha = torch.zeros(n_features, dtype=torch.float32)

    def fit(
        self,
        M,
        y,
        lambda_l0=0.0,
        lambda_l2=0.0,
        lr=0.0,
        epochs=1,
        loss_type="relative",
        verbose=False,
        verbose_freq=1,
        target_groups=None,
    ):
        type(self).fit_calls.append({"target_groups": target_groups})
        increment = float(epochs) + (self.alpha / 10.0)
        self.weights = self.weights + increment
        self.alpha = self.alpha + (10.0 * float(epochs))
        return self

    def predict(self, M):
        import torch

        weights = self.get_weights(deterministic=True).cpu().numpy()
        return torch.tensor(M.dot(weights), dtype=torch.float32)

    def get_weights(self, deterministic=True):
        return self.weights.clone()

    def state_dict(self):
        return {
            "weights": self.weights.clone(),
            "alpha": self.alpha.clone(),
        }

    def load_state_dict(self, state_dict):
        self.weights = state_dict["weights"].clone()
        self.alpha = state_dict["alpha"].clone()


class TestFitTargetGroups:
    def test_passes_target_groups_to_l0_model(self, tmp_path):
        from policyengine_us_data.calibration.unified_calibration import (
            fit_l0_weights,
        )

        target_groups = np.array([0, 1], dtype=np.int64)
        FakeSparseCalibrationWeights.fit_calls = []

        with patch(
            "l0.calibration.SparseCalibrationWeights",
            FakeSparseCalibrationWeights,
        ):
            weights = fit_l0_weights(
                X_sparse=sp.csr_matrix(np.eye(2, dtype=np.float32)),
                targets=np.array([1.0, 2.0], dtype=np.float64),
                lambda_l0=1e-4,
                epochs=1,
                device="cpu",
                target_names=["target_a", "target_b"],
                initial_weights=np.array([1.0, 2.0], dtype=np.float64),
                log_path=str(tmp_path / "calibration_log.csv"),
                target_groups=target_groups,
            )

        np.testing.assert_allclose(weights, np.array([2.0, 3.0]))
        np.testing.assert_array_equal(
            FakeSparseCalibrationWeights.fit_calls[-1]["target_groups"],
            target_groups,
        )

    def test_passes_target_groups_to_logged_l0_fit(self, tmp_path):
        from policyengine_us_data.calibration.unified_calibration import (
            fit_l0_weights,
        )

        target_groups = np.array([0, 1], dtype=np.int64)
        FakeSparseCalibrationWeights.fit_calls = []

        with patch(
            "l0.calibration.SparseCalibrationWeights",
            FakeSparseCalibrationWeights,
        ):
            weights = fit_l0_weights(
                X_sparse=sp.csr_matrix(np.eye(2, dtype=np.float32)),
                targets=np.array([1.0, 2.0], dtype=np.float64),
                lambda_l0=1e-4,
                epochs=2,
                device="cpu",
                target_names=["target_a", "target_b"],
                initial_weights=np.array([1.0, 2.0], dtype=np.float64),
                log_freq=1,
                log_path=str(tmp_path / "calibration_log.csv"),
                target_groups=target_groups,
            )

        np.testing.assert_allclose(weights, np.array([4.0, 5.0]))
        assert len(FakeSparseCalibrationWeights.fit_calls) == 2
        for fit_call in FakeSparseCalibrationWeights.fit_calls:
            np.testing.assert_array_equal(
                fit_call["target_groups"],
                target_groups,
            )


class TestFitResume:
    def _fit_kwargs(self, tmp_path):
        return {
            "X_sparse": sp.csr_matrix(np.eye(2, dtype=np.float32)),
            "targets": np.array([1.0, 2.0], dtype=np.float64),
            "lambda_l0": 1e-4,
            "epochs": 1,
            "device": "cpu",
            "beta": 0.65,
            "lambda_l2": 1e-12,
            "learning_rate": 0.15,
            "log_freq": 1,
            "log_path": str(tmp_path / "calibration_log.csv"),
            "target_names": ["target_a", "target_b"],
            "initial_weights": np.array([1.0, 2.0], dtype=np.float64),
            "achievable": np.array([True, True]),
        }

    def test_resume_from_weights_prefers_sibling_checkpoint(self, tmp_path):
        from policyengine_us_data.calibration.unified_calibration import (
            default_checkpoint_path,
            fit_l0_weights,
        )

        weights_path = tmp_path / "weights.npy"
        checkpoint_path = default_checkpoint_path(str(weights_path))
        kwargs = self._fit_kwargs(tmp_path)
        kwargs["checkpoint_path"] = str(checkpoint_path)

        with patch(
            "l0.calibration.SparseCalibrationWeights",
            FakeSparseCalibrationWeights,
        ):
            first_weights = fit_l0_weights(**kwargs)
            np.save(weights_path, first_weights)

            resumed_weights = fit_l0_weights(
                **{
                    **kwargs,
                    "resume_from": str(weights_path),
                }
            )

        np.testing.assert_allclose(first_weights, np.array([2.0, 3.0]))
        np.testing.assert_allclose(resumed_weights, np.array([4.0, 5.0]))

        with open(kwargs["log_path"]) as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 5
        assert lines[1].split(",")[3] == "1"
        assert lines[3].split(",")[3] == "2"

    def test_resume_from_weights_falls_back_when_checkpoint_missing(self, tmp_path):
        from policyengine_us_data.calibration.unified_calibration import fit_l0_weights

        weights_path = tmp_path / "weights.npy"
        np.save(weights_path, np.array([2.0, 3.0], dtype=np.float64))
        kwargs = self._fit_kwargs(tmp_path)

        with patch(
            "l0.calibration.SparseCalibrationWeights",
            FakeSparseCalibrationWeights,
        ):
            resumed_weights = fit_l0_weights(
                **{
                    **kwargs,
                    "resume_from": str(weights_path),
                }
            )

        np.testing.assert_allclose(resumed_weights, np.array([3.0, 4.0]))

    def test_resume_checkpoint_warns_on_hyperparameter_change(self, tmp_path, caplog):
        import logging

        from policyengine_us_data.calibration.unified_calibration import (
            default_checkpoint_path,
            fit_l0_weights,
        )

        weights_path = tmp_path / "weights.npy"
        checkpoint_path = default_checkpoint_path(str(weights_path))
        kwargs = self._fit_kwargs(tmp_path)
        kwargs["checkpoint_path"] = str(checkpoint_path)

        with patch(
            "l0.calibration.SparseCalibrationWeights",
            FakeSparseCalibrationWeights,
        ):
            first_weights = fit_l0_weights(**kwargs)
            np.save(weights_path, first_weights)

            with caplog.at_level(logging.WARNING):
                resumed_weights = fit_l0_weights(
                    **{
                        **kwargs,
                        "lambda_l0": 9e-4,
                        "resume_from": str(checkpoint_path),
                    }
                )

        assert resumed_weights is not None
        assert any(
            "Resuming with hyperparameter change" in record.message
            and "lambda_l0" in record.message
            for record in caplog.records
        )

    def test_resume_checkpoint_rejects_changed_matrix_with_same_shape(self, tmp_path):
        from policyengine_us_data.calibration.unified_calibration import (
            default_checkpoint_path,
            fit_l0_weights,
        )

        weights_path = tmp_path / "weights.npy"
        checkpoint_path = default_checkpoint_path(str(weights_path))
        kwargs = self._fit_kwargs(tmp_path)
        kwargs["checkpoint_path"] = str(checkpoint_path)

        with patch(
            "l0.calibration.SparseCalibrationWeights",
            FakeSparseCalibrationWeights,
        ):
            first_weights = fit_l0_weights(**kwargs)
            np.save(weights_path, first_weights)

            changed_matrix = sp.csr_matrix(
                np.array(
                    [
                        [0.0, 1.0],
                        [1.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            )

            with pytest.raises(
                ValueError, match="Checkpoint is structurally incompatible"
            ):
                fit_l0_weights(
                    **{
                        **kwargs,
                        "X_sparse": changed_matrix,
                        "resume_from": str(checkpoint_path),
                    }
                )

    def test_resume_checkpoint_rejects_changed_target_groups(self, tmp_path):
        from policyengine_us_data.calibration.unified_calibration import (
            default_checkpoint_path,
            fit_l0_weights,
        )

        weights_path = tmp_path / "weights.npy"
        checkpoint_path = default_checkpoint_path(str(weights_path))
        kwargs = self._fit_kwargs(tmp_path)
        kwargs["checkpoint_path"] = str(checkpoint_path)
        kwargs["target_groups"] = np.array([0, 1], dtype=np.int64)

        with patch(
            "l0.calibration.SparseCalibrationWeights",
            FakeSparseCalibrationWeights,
        ):
            first_weights = fit_l0_weights(**kwargs)
            np.save(weights_path, first_weights)

            with pytest.raises(ValueError, match="target_groups_sha256"):
                fit_l0_weights(
                    **{
                        **kwargs,
                        "target_groups": np.array([1, 0], dtype=np.int64),
                        "resume_from": str(checkpoint_path),
                    }
                )

    def test_resume_checkpoint_rejects_missing_matrix_fingerprint(self, tmp_path):
        import torch
        from policyengine_us_data.calibration.unified_calibration import (
            default_checkpoint_path,
            fit_l0_weights,
        )

        weights_path = tmp_path / "weights.npy"
        checkpoint_path = default_checkpoint_path(str(weights_path))
        kwargs = self._fit_kwargs(tmp_path)
        kwargs["checkpoint_path"] = str(checkpoint_path)

        with patch(
            "l0.calibration.SparseCalibrationWeights",
            FakeSparseCalibrationWeights,
        ):
            first_weights = fit_l0_weights(**kwargs)
            np.save(weights_path, first_weights)

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint["signature"].pop("x_sparse_sha256")
            torch.save(checkpoint, checkpoint_path)

            with pytest.raises(ValueError, match="x_sparse_sha256"):
                fit_l0_weights(
                    **{
                        **kwargs,
                        "resume_from": str(checkpoint_path),
                    }
                )


class TestGeographyAssignmentCountyFips:
    """Verify county_fips field on GeographyAssignment."""

    def test_county_fips_equals_block_prefix(self):
        blocks = np.array(["370010001001001", "480010002002002", "060370003003003"])
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


class TestRunCalibrationAgiTargets:
    def test_uses_requested_db_for_district_agi_targets(self):
        from policyengine_us_data.calibration.unified_calibration import (
            run_calibration,
        )

        captured = {}

        class StopAfterAssignment(RuntimeError):
            pass

        class FakeMicrosimulation:
            def __init__(self, dataset, reform=None):
                self.dataset = SimpleNamespace(
                    load_dataset=lambda: {"household_id": {2024: np.array([1, 2])}}
                )

            def calculate(self, variable, *args, **kwargs):
                if variable == "household_id":
                    return SimpleNamespace(values=np.array([1, 2], dtype=np.int64))
                if variable == "adjusted_gross_income":
                    return SimpleNamespace(
                        values=np.array([100.0, 200.0], dtype=np.float64)
                    )
                raise AssertionError(f"Unexpected calculate({variable!r})")

        class FakeBuilder:
            def __init__(self, db_uri, time_period, dataset_path=None):
                captured["db_uri"] = db_uri
                captured["time_period"] = time_period
                captured["dataset_path_at_init"] = dataset_path

            def get_district_agi_targets(self):
                return {"601": 123.0}

        def fake_assign_random_geography(**kwargs):
            captured["assign_kwargs"] = kwargs
            raise StopAfterAssignment

        with (
            patch("policyengine_us.Microsimulation", FakeMicrosimulation),
            patch(
                "policyengine_us_data.calibration.unified_matrix_builder.UnifiedMatrixBuilder",
                FakeBuilder,
            ),
            patch(
                "policyengine_us_data.calibration.clone_and_assign.assign_random_geography",
                fake_assign_random_geography,
            ),
        ):
            with pytest.raises(StopAfterAssignment):
                run_calibration(
                    dataset_path="input.h5",
                    db_path="/tmp/custom-policy-data.db",
                    n_clones=2,
                )

        assert captured["db_uri"] == "sqlite:////tmp/custom-policy-data.db"
        assert captured["time_period"] == 2024
        assert captured["assign_kwargs"]["cd_agi_targets"] == {"601": 123.0}


class TestBlockTakeupSeeding:
    """Verify compute_block_takeup_for_entities is
    reproducible and clone-dependent."""

    def test_reproducible(self):
        n = 100
        blocks = np.array(["010010001001001"] * 50 + ["020010001001001"] * 50)
        hh_ids = np.arange(n, dtype=np.int64)
        ci = np.zeros(n, dtype=np.int64)
        r1 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_ids, ci
        )
        r2 = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_ids, ci
        )
        np.testing.assert_array_equal(r1, r2)

    def test_different_blocks_different_rates(self):
        """With state-dependent rates, different blocks yield
        different takeup because rate thresholds differ."""
        n = 500
        hh_ids = np.arange(n, dtype=np.int64)
        ci = np.zeros(n, dtype=np.int64)
        rate_dict = {"AL": 0.9, "AK": 0.3}
        r_a = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible",
            rate_dict,
            np.array(["010010001001001"] * n),
            hh_ids,
            ci,
        )
        r_b = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible",
            rate_dict,
            np.array(["020010001001001"] * n),
            hh_ids,
            ci,
        )
        assert not np.array_equal(r_a, r_b)

    def test_returns_booleans(self):
        n = 100
        blocks = np.array(["370010001001001"] * n)
        hh_ids = np.arange(n, dtype=np.int64)
        ci = np.zeros(n, dtype=np.int64)
        result = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.8, blocks, hh_ids, ci
        )
        assert result.dtype == bool

    def test_rate_respected(self):
        n = 10000
        blocks = np.array(["370010001001001"] * n)
        hh_ids = np.arange(n, dtype=np.int64)
        ci = np.zeros(n, dtype=np.int64)
        result = compute_block_takeup_for_entities(
            "takes_up_snap_if_eligible", 0.75, blocks, hh_ids, ci
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
        hh_vars, _, _ = builder._assemble_clone_values(
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
        hh_vars, _, _ = builder._assemble_clone_values(
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


class TestAssembleCloneValuesStringConstraint:
    """Verify string constraint vars (e.g. ssn_card_type) are
    assembled without crashing on float32 conversion."""

    def _make_state_values(self):
        n = 4
        return {
            1: {
                "hh": {"snap": np.array([50] * n, dtype=np.float32)},
                "person": {
                    "ssn_card_type": np.array(
                        ["CITIZEN", "CITIZEN", "UNDOCUMENTED", "CITIZEN"],
                        dtype=object,
                    ),
                },
                "entity": {},
            },
            2: {
                "hh": {"snap": np.array([60] * n, dtype=np.float32)},
                "person": {
                    "ssn_card_type": np.array(
                        ["UNDOCUMENTED", "CITIZEN", "CITIZEN", "UNDOCUMENTED"],
                        dtype=object,
                    ),
                },
                "entity": {},
            },
        }

    def test_string_constraint_var_assembled(self):
        from policyengine_us_data.calibration.unified_matrix_builder import (
            UnifiedMatrixBuilder,
        )

        state_values = self._make_state_values()
        clone_states = np.array([1, 1, 2, 2])
        person_hh_idx = np.array([0, 1, 2, 3])

        builder = UnifiedMatrixBuilder.__new__(UnifiedMatrixBuilder)
        _, person_vars, _ = builder._assemble_clone_values(
            state_values,
            clone_states,
            person_hh_idx,
            {"snap"},
            {"ssn_card_type"},
        )
        assert "ssn_card_type" in person_vars
        arr = person_vars["ssn_card_type"]
        assert arr.dtype == object
        expected = np.array(
            ["CITIZEN", "CITIZEN", "CITIZEN", "UNDOCUMENTED"], dtype=object
        )
        np.testing.assert_array_equal(arr, expected)

    def test_string_constraint_var_standalone(self):
        from policyengine_us_data.calibration.unified_matrix_builder import (
            _assemble_clone_values_standalone,
        )

        state_values = self._make_state_values()
        clone_states = np.array([1, 1, 2, 2])
        person_hh_idx = np.array([0, 1, 2, 3])

        _, person_vars, _ = _assemble_clone_values_standalone(
            state_values,
            clone_states,
            person_hh_idx,
            {"snap"},
            {"ssn_card_type"},
        )
        assert "ssn_card_type" in person_vars
        arr = person_vars["ssn_card_type"]
        assert arr.dtype == object
        expected = np.array(
            ["CITIZEN", "CITIZEN", "CITIZEN", "UNDOCUMENTED"], dtype=object
        )
        np.testing.assert_array_equal(arr, expected)

    def test_string_constraint_with_equality_op(self):
        import pandas as pd

        from policyengine_us_data.calibration.unified_matrix_builder import (
            _assemble_clone_values_standalone,
            _evaluate_constraints_standalone,
        )

        state_values = self._make_state_values()
        clone_states = np.array([1, 1, 2, 2])
        person_hh_idx = np.array([0, 1, 2, 3])

        _, person_vars, _ = _assemble_clone_values_standalone(
            state_values,
            clone_states,
            person_hh_idx,
            {"snap"},
            {"ssn_card_type"},
        )

        household_ids = np.array([0, 1, 2, 3])
        entity_rel = pd.DataFrame(
            {"household_id": person_hh_idx, "person_id": np.arange(4)}
        )
        constraints = [
            {"variable": "ssn_card_type", "operation": "==", "value": "CITIZEN"}
        ]
        mask = _evaluate_constraints_standalone(
            constraints, person_vars, entity_rel, household_ids, 4
        )
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(mask, expected)


class TestTakeupDrawConsistency:
    """Verify the matrix builder's inline takeup loop and
    compute_block_takeup_for_entities produce identical draws
    when given the same (block, household) inputs."""

    def test_matrix_and_stacked_identical_draws(self):
        """Both paths must produce identical boolean arrays."""
        var = "takes_up_snap_if_eligible"
        rate = 0.75
        clone_idx = 5

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
        ci = np.full(len(blocks), clone_idx, dtype=np.int64)

        # Path 1: compute_block_takeup_for_entities
        stacked = compute_block_takeup_for_entities(var, rate, blocks, hh_ids, ci)

        # Path 2: reproduce inline logic with hh_id:clone_idx salt
        n = len(blocks)
        inline_takeup = np.zeros(n, dtype=bool)
        for hh_id in np.unique(hh_ids):
            hh_mask = hh_ids == hh_id
            rng = seeded_rng(var, salt=f"{int(hh_id)}:{clone_idx}")
            draws = rng.random(int(hh_mask.sum()))
            # Rate from block's state FIPS
            blk = blocks[hh_mask][0]
            sf = int(str(blk)[:2])
            r = _resolve_rate(rate, sf)
            inline_takeup[hh_mask] = draws < r

        np.testing.assert_array_equal(stacked, inline_takeup)

    def test_aggregation_entity_to_household(self):
        """np.add.at aggregation matches manual per-HH sum."""
        n_hh = 3
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
        hh_ids_nc = np.arange(n, dtype=np.int64)
        ci = np.zeros(n, dtype=np.int64)
        result_nc = compute_block_takeup_for_entities(
            var, rate_dict, blocks_nc, hh_ids_nc, ci
        )
        frac_nc = result_nc.mean()
        assert 0.85 < frac_nc < 0.95, f"NC frac={frac_nc}"

        blocks_tx = np.array(["480010002002002"] * n)
        hh_ids_tx = np.arange(n, dtype=np.int64)
        result_tx = compute_block_takeup_for_entities(
            var, rate_dict, blocks_tx, hh_ids_tx, ci
        )
        frac_tx = result_tx.mean()
        assert 0.55 < frac_tx < 0.65, f"TX frac={frac_tx}"

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
