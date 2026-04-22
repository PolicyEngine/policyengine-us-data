"""Tests for stochastic variable generation in the data package."""

import numpy as np
from policyengine_us_data.parameters import load_take_up_rate
from policyengine_us_data.utils.takeup import (
    any_person_flag_by_entity,
    assign_takeup_with_reported_anchors,
    reported_subsidized_marketplace_by_tax_unit,
)
from policyengine_us_data.utils.randomness import (
    _stable_string_hash,
    seeded_rng,
)


class TestTakeUpRateParameters:
    def test_eitc_rate_loads(self):
        rates = load_take_up_rate("eitc", 2022)
        assert isinstance(rates, dict)
        for key, rate in rates.items():
            assert 0 < rate <= 1

    def test_snap_rate_loads(self):
        rate = load_take_up_rate("snap", 2022)
        assert 0 < rate <= 1

    def test_medicaid_rate_loads_state_specific(self):
        rates = load_take_up_rate("medicaid", 2022)
        assert isinstance(rates, dict)
        assert len(rates) == 51  # 50 states + DC
        for state, rate in rates.items():
            assert 0 < rate <= 1, f"{state}: {rate}"
        assert rates["UT"] == 0.53
        assert rates["CO"] == 0.99

    def test_aca_rate_loads(self):
        rate = load_take_up_rate("aca", 2022)
        assert 0 < rate <= 1

    def test_head_start_rate_loads(self):
        rate = load_take_up_rate("head_start", 2022)
        assert 0 < rate <= 1

    def test_early_head_start_rate_loads(self):
        rate = load_take_up_rate("early_head_start", 2022)
        assert 0 < rate <= 1

    def test_dc_ptc_rate_loads(self):
        rate = load_take_up_rate("dc_ptc", 2022)
        assert 0 < rate <= 1

    def test_ssi_takeup_rate_loads(self):
        rate = load_take_up_rate("ssi", 2022)
        assert rate == 0.50

    def test_voluntary_filing_table_loads(self):
        rates = load_take_up_rate("voluntary_filing", 2024)
        assert isinstance(rates, dict)
        assert rates["no_children"]["zero"]["under_65"] == 0.2
        assert rates["with_children"]["low"]["under_65"] == 0.6
        for children_rates in rates.values():
            for wage_rates in children_rates.values():
                for rate in wage_rates.values():
                    assert 0 <= rate <= 1


class TestStableStringHash:
    def test_deterministic(self):
        h1 = _stable_string_hash("takes_up_snap_if_eligible")
        h2 = _stable_string_hash("takes_up_snap_if_eligible")
        assert h1 == h2

    def test_different_strings_differ(self):
        h1 = _stable_string_hash("takes_up_snap_if_eligible")
        h2 = _stable_string_hash("takes_up_aca_if_eligible")
        assert h1 != h2

    def test_returns_uint64(self):
        h = _stable_string_hash("test")
        assert h.dtype == np.uint64


class TestSeededRng:
    def test_same_name_same_results(self):
        rng1 = seeded_rng("takes_up_snap_if_eligible")
        result1 = rng1.random(1000)
        rng2 = seeded_rng("takes_up_snap_if_eligible")
        result2 = rng2.random(1000)
        np.testing.assert_array_equal(result1, result2)

    def test_different_names_different_results(self):
        rng1 = seeded_rng("takes_up_snap_if_eligible")
        result1 = rng1.random(1000)
        rng2 = seeded_rng("takes_up_aca_if_eligible")
        result2 = rng2.random(1000)
        assert not np.array_equal(result1, result2)

    def test_order_independence(self):
        """Generating variables in different order produces same values."""
        # Order A: SNAP then ACA
        rng_snap_a = seeded_rng("takes_up_snap_if_eligible")
        snap_a = rng_snap_a.random(1000)
        rng_aca_a = seeded_rng("takes_up_aca_if_eligible")
        aca_a = rng_aca_a.random(1000)

        # Order B: ACA then SNAP
        rng_aca_b = seeded_rng("takes_up_aca_if_eligible")
        aca_b = rng_aca_b.random(1000)
        rng_snap_b = seeded_rng("takes_up_snap_if_eligible")
        snap_b = rng_snap_b.random(1000)

        np.testing.assert_array_equal(snap_a, snap_b)
        np.testing.assert_array_equal(aca_a, aca_b)


class TestTakeUpProportions:
    def test_take_up_produces_expected_proportion(self):
        rate = 0.7
        n = 10_000
        rng = seeded_rng("test_variable")
        take_up = rng.random(n) < rate
        assert abs(take_up.mean() - rate) < 0.05

    def test_boolean_generation(self):
        rng = seeded_rng("test_bool")
        take_up = rng.random(100) < 0.5
        assert take_up.dtype == bool
        assert set(take_up).issubset({True, False})

    def test_wic_takeup_rates_load(self):
        rates = load_take_up_rate("wic_takeup", 2022)
        assert isinstance(rates, dict)
        assert rates["PREGNANT"] == 0.456
        assert rates["INFANT"] == 0.784
        assert rates["NONE"] == 0

    def test_wic_nutritional_risk_rates_load(self):
        rates = load_take_up_rate("wic_nutritional_risk", 2022)
        assert isinstance(rates, dict)
        assert rates["INFANT"] == 0.95
        assert rates["CHILD"] == 0.752
        assert rates["NONE"] == 0

    def test_wic_category_specific_proportions(self):
        rates = load_take_up_rate("wic_takeup", 2022)
        n = 10_000
        rng = seeded_rng("would_claim_wic")
        draws = rng.random(n)
        for category, expected_rate in [
            ("INFANT", 0.784),
            ("CHILD", 0.46),
        ]:
            take_up = draws[:n] < expected_rate
            assert abs(take_up.mean() - expected_rate) < 0.05

    def test_state_specific_medicaid_proportions(self):
        rates = load_take_up_rate("medicaid", 2022)
        rng = seeded_rng("takes_up_medicaid_if_eligible")
        n = 50_000
        draws = rng.random(n)
        # Test a few states
        for state, expected_rate in [("UT", 0.53), ("CO", 0.99)]:
            take_up = draws[:10_000] < expected_rate
            assert abs(take_up.mean() - expected_rate) < 0.05


class TestReportedTakeupAnchors:
    def test_global_anchor_preserves_reported_and_fills_remaining(self):
        draws = np.array([0.9, 0.2, 0.6, 0.9])
        reported = np.array([True, False, False, False])
        result = assign_takeup_with_reported_anchors(
            draws,
            0.5,
            reported_mask=reported,
        )
        np.testing.assert_array_equal(result, [True, True, False, False])

    def test_grouped_anchor_applies_within_each_group(self):
        draws = np.array([0.9, 0.2, 0.1, 0.9])
        rates = np.array([0.5, 0.5, 0.5, 0.5])
        reported = np.array([True, False, False, False])
        groups = np.array(["A", "A", "B", "B"])
        result = assign_takeup_with_reported_anchors(
            draws,
            rates,
            reported_mask=reported,
            group_keys=groups,
        )
        np.testing.assert_array_equal(result, [True, False, True, False])

    def test_any_person_flag_by_entity_aggregates_correctly(self):
        person_tax_unit_ids = np.array([10, 10, 20, 30])
        tax_unit_ids = np.array([10, 20, 30])
        person_marketplace = np.array([False, True, False, True])
        result = any_person_flag_by_entity(
            person_tax_unit_ids,
            tax_unit_ids,
            person_marketplace,
        )
        np.testing.assert_array_equal(result, [True, False, True])

    def test_subsidized_marketplace_anchor_excludes_unsubsidized_only(self):
        person_tax_unit_ids = np.array([10, 10, 20, 30])
        tax_unit_ids = np.array([10, 20, 30])
        person_subsidized_marketplace = np.array([False, False, False, True])
        result = reported_subsidized_marketplace_by_tax_unit(
            person_tax_unit_ids,
            tax_unit_ids,
            person_subsidized_marketplace,
        )
        np.testing.assert_array_equal(result, [False, False, True])
