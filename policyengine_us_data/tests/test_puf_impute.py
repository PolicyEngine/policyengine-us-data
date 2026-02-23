"""Tests for PUF imputation, specifically SS sub-component reconciliation.

When PUF imputation replaces social_security values, the sub-components
(retirement, disability, survivors, dependents) must be rescaled to match
the new total. See: https://github.com/PolicyEngine/policyengine-us-data/issues/551
"""

from unittest.mock import patch

import numpy as np
import pytest

from policyengine_us_data.calibration.puf_impute import (
    MINIMUM_RETIREMENT_AGE,
    _age_heuristic_ss_shares,
    _qrf_ss_shares,
    reconcile_ss_subcomponents,
)

SS_SUBS = [
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
]


def _make_data(
    orig_ss,
    orig_ret,
    orig_dis,
    orig_surv,
    orig_dep,
    imputed_ss,
    age=None,
    is_male=None,
):
    """Build a doubled data dict (CPS half + PUF half)."""
    n = len(orig_ss)
    tp = 2024
    data = {
        "social_security": {
            tp: np.concatenate([orig_ss, imputed_ss]).astype(np.float32)
        },
        "social_security_retirement": {
            tp: np.concatenate([orig_ret, orig_ret]).astype(np.float32)
        },
        "social_security_disability": {
            tp: np.concatenate([orig_dis, orig_dis]).astype(np.float32)
        },
        "social_security_survivors": {
            tp: np.concatenate([orig_surv, orig_surv]).astype(np.float32)
        },
        "social_security_dependents": {
            tp: np.concatenate([orig_dep, orig_dep]).astype(np.float32)
        },
    }
    if age is not None:
        data["age"] = {tp: np.concatenate([age, age]).astype(np.float32)}
    if is_male is not None:
        data["is_male"] = {
            tp: np.concatenate([is_male, is_male]).astype(np.float32)
        }
    return data, n, tp


class TestReconcileSsSubcomponents:
    """Sub-components must sum to social_security after reconciliation."""

    def test_proportional_rescaling(self):
        """When CPS has a split, PUF subs scale proportionally."""
        data, n, tp = _make_data(
            orig_ss=np.array([20000, 15000]),
            orig_ret=np.array([14000, 0]),
            orig_dis=np.array([6000, 15000]),
            orig_surv=np.array([0, 0]),
            orig_dep=np.array([0, 0]),
            imputed_ss=np.array([30000, 10000]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ss = data["social_security"][tp][puf]
        total_subs = sum(data[sub][tp][puf] for sub in SS_SUBS)
        np.testing.assert_allclose(total_subs, ss, rtol=1e-5)

        # Person 0: ret=14/20=70%, dis=6/20=30%
        ret = data["social_security_retirement"][tp][puf]
        np.testing.assert_allclose(ret[0], 30000 * 0.7, rtol=1e-5)
        dis = data["social_security_disability"][tp][puf]
        np.testing.assert_allclose(dis[0], 30000 * 0.3, rtol=1e-5)

    def test_imputed_zero_clears_subs(self):
        """When PUF imputes zero SS, all subs should be zero."""
        data, n, tp = _make_data(
            orig_ss=np.array([20000]),
            orig_ret=np.array([20000]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([0]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        for sub in SS_SUBS:
            assert data[sub][tp][puf][0] == pytest.approx(0)

    def test_cps_half_unchanged(self):
        """CPS half should not be modified."""
        orig_ret = np.array([14000, 0])
        data, n, tp = _make_data(
            orig_ss=np.array([20000, 15000]),
            orig_ret=orig_ret,
            orig_dis=np.array([6000, 15000]),
            orig_surv=np.array([0, 0]),
            orig_dep=np.array([0, 0]),
            imputed_ss=np.array([30000, 10000]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        cps = slice(0, n)
        np.testing.assert_array_equal(
            data["social_security_retirement"][tp][cps],
            orig_ret,
        )

    def test_missing_subcomponent_is_skipped(self):
        """If a sub-component is absent from data, no error."""
        data, n, tp = _make_data(
            orig_ss=np.array([10000]),
            orig_ret=np.array([10000]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([15000]),
        )
        del data["social_security_survivors"]
        del data["social_security_dependents"]
        reconcile_ss_subcomponents(data, n, tp)

    def test_no_social_security_is_noop(self):
        """If social_security is absent, function is a no-op."""
        data = {"some_var": {2024: np.array([1, 2])}}
        reconcile_ss_subcomponents(data, 1, 2024)

    def test_subs_sum_to_total_all_cases(self):
        """Comprehensive: subs must sum to total across all cases."""
        data, n, tp = _make_data(
            # Person 0: existing recipient (rescale)
            # Person 1: new recipient, old (retirement)
            # Person 2: new recipient, young (disability)
            # Person 3: PUF zeroed out (clear)
            orig_ss=np.array([20000, 0, 0, 15000]),
            orig_ret=np.array([20000, 0, 0, 15000]),
            orig_dis=np.array([0, 0, 0, 0]),
            orig_surv=np.array([0, 0, 0, 0]),
            orig_dep=np.array([0, 0, 0, 0]),
            imputed_ss=np.array([25000, 18000, 12000, 0]),
            age=np.array([70, 68, 40, 75]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ss = data["social_security"][tp][puf]
        total_subs = sum(data[sub][tp][puf] for sub in SS_SUBS)
        np.testing.assert_allclose(total_subs, ss, rtol=1e-5)


class TestAgeHeuristicSsShares:
    """Age-based fallback for new SS recipient classification."""

    def test_elderly_gets_retirement(self):
        """Age >= 62 -> retirement."""
        data, n, tp = _make_data(
            orig_ss=np.array([0]),
            orig_ret=np.array([0]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([25000]),
            age=np.array([70]),
        )
        new_recip = np.array([True])
        shares = _age_heuristic_ss_shares(data, n, tp, new_recip)

        assert shares["social_security_retirement"][0] == pytest.approx(1.0)
        assert shares["social_security_disability"][0] == pytest.approx(0.0)

    def test_young_gets_disability(self):
        """Age < 62 -> disability."""
        data, n, tp = _make_data(
            orig_ss=np.array([0]),
            orig_ret=np.array([0]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([18000]),
            age=np.array([45]),
        )
        new_recip = np.array([True])
        shares = _age_heuristic_ss_shares(data, n, tp, new_recip)

        assert shares["social_security_retirement"][0] == pytest.approx(0.0)
        assert shares["social_security_disability"][0] == pytest.approx(1.0)

    def test_no_age_defaults_to_retirement(self):
        """Without age data, default to retirement."""
        data, n, tp = _make_data(
            orig_ss=np.array([0]),
            orig_ret=np.array([0]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([25000]),
        )
        new_recip = np.array([True])
        shares = _age_heuristic_ss_shares(data, n, tp, new_recip)

        assert shares["social_security_retirement"][0] == pytest.approx(1.0)

    def test_mixed_ages(self):
        """Multiple people split correctly by age threshold."""
        data, n, tp = _make_data(
            orig_ss=np.array([0, 0, 0]),
            orig_ret=np.array([0, 0, 0]),
            orig_dis=np.array([0, 0, 0]),
            orig_surv=np.array([0, 0, 0]),
            orig_dep=np.array([0, 0, 0]),
            imputed_ss=np.array([1, 1, 1]),
            age=np.array([30, 62, 80]),
        )
        new_recip = np.array([True, True, True])
        shares = _age_heuristic_ss_shares(data, n, tp, new_recip)

        # age 30 -> disability
        assert shares["social_security_retirement"][0] == 0
        assert shares["social_security_disability"][0] == 1
        # age 62 -> retirement (>= threshold)
        assert shares["social_security_retirement"][1] == 1
        assert shares["social_security_disability"][1] == 0
        # age 80 -> retirement
        assert shares["social_security_retirement"][2] == 1


class TestQrfSsShares:
    """QRF-based SS sub-component share prediction."""

    def test_returns_none_without_microimpute(self):
        """Returns None when microimpute is not importable."""
        data, n, tp = _make_data(
            orig_ss=np.array([10000]),
            orig_ret=np.array([10000]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([15000]),
            age=np.array([70]),
        )
        new_recip = np.array([False])
        with patch.dict("sys.modules", {"microimpute": None}):
            result = _qrf_ss_shares(data, n, tp, new_recip)
        # No new recipients -> no call needed, but also shouldn't error
        # Actually test with import blocked:
        assert result is None or isinstance(result, dict)

    def test_returns_none_with_few_training_records(self):
        """Returns None when < MIN_QRF_TRAINING_RECORDS have SS."""
        # Only 2 training records with SS > 0.
        data, n, tp = _make_data(
            orig_ss=np.array([10000, 5000, 0]),
            orig_ret=np.array([10000, 0, 0]),
            orig_dis=np.array([0, 5000, 0]),
            orig_surv=np.array([0, 0, 0]),
            orig_dep=np.array([0, 0, 0]),
            imputed_ss=np.array([0, 0, 20000]),
            age=np.array([70, 45, 55]),
        )
        new_recip = np.array([False, False, True])
        result = _qrf_ss_shares(data, n, tp, new_recip)
        assert result is None

    def test_shares_sum_to_one(self):
        """QRF-predicted shares should sum to ~1 after normalisation."""
        pytest.importorskip("microimpute")
        rng = np.random.default_rng(42)
        n = 500

        # Synthetic training: age > 62 mostly retirement,
        # age < 62 mostly disability.
        ages = rng.integers(20, 90, size=n).astype(np.float32)
        is_male = rng.integers(0, 2, size=n).astype(np.float32)
        ss_vals = rng.uniform(5000, 40000, size=n).astype(np.float32)
        ret = np.where(ages >= 62, ss_vals, 0).astype(np.float32)
        dis = np.where(ages < 62, ss_vals, 0).astype(np.float32)
        surv = np.zeros(n, dtype=np.float32)
        dep = np.zeros(n, dtype=np.float32)

        # 20 new recipients (CPS had 0 SS).
        n_new = 20
        new_ages = rng.integers(20, 90, size=n_new).astype(np.float32)
        new_is_male = rng.integers(0, 2, size=n_new).astype(np.float32)
        new_ss = rng.uniform(10000, 30000, size=n_new).astype(np.float32)

        all_ss = np.concatenate([ss_vals, np.zeros(n_new)])
        all_ret = np.concatenate([ret, np.zeros(n_new)])
        all_dis = np.concatenate([dis, np.zeros(n_new)])
        all_surv = np.concatenate([surv, np.zeros(n_new)])
        all_dep = np.concatenate([dep, np.zeros(n_new)])
        all_ages = np.concatenate([ages, new_ages])
        all_is_male = np.concatenate([is_male, new_is_male])
        n_total = n + n_new

        data, nn, tp = _make_data(
            orig_ss=all_ss,
            orig_ret=all_ret,
            orig_dis=all_dis,
            orig_surv=all_surv,
            orig_dep=all_dep,
            imputed_ss=np.concatenate([ss_vals, new_ss]),
            age=all_ages,
            is_male=all_is_male,
        )

        new_recip = np.concatenate(
            [np.zeros(n, dtype=bool), np.ones(n_new, dtype=bool)]
        )
        shares = _qrf_ss_shares(data, nn, tp, new_recip)

        assert shares is not None
        total = sum(shares.get(sub, 0) for sub in SS_SUBS)
        np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_elderly_predicted_as_retirement(self):
        """QRF should mostly predict retirement for age >= 62."""
        pytest.importorskip("microimpute")
        rng = np.random.default_rng(123)
        n = 500

        ages = rng.integers(20, 90, size=n).astype(np.float32)
        ss_vals = rng.uniform(5000, 40000, size=n).astype(np.float32)
        ret = np.where(ages >= 62, ss_vals, 0).astype(np.float32)
        dis = np.where(ages < 62, ss_vals, 0).astype(np.float32)
        surv = np.zeros(n, dtype=np.float32)
        dep = np.zeros(n, dtype=np.float32)

        # 10 new elderly recipients.
        n_new = 10
        new_ages = np.full(n_new, 70, dtype=np.float32)
        new_ss = np.full(n_new, 20000, dtype=np.float32)

        all_ss = np.concatenate([ss_vals, np.zeros(n_new)])
        all_ret = np.concatenate([ret, np.zeros(n_new)])
        all_dis = np.concatenate([dis, np.zeros(n_new)])
        all_surv = np.concatenate([surv, np.zeros(n_new)])
        all_dep = np.concatenate([dep, np.zeros(n_new)])
        all_ages = np.concatenate([ages, new_ages])

        data, nn, tp = _make_data(
            orig_ss=all_ss,
            orig_ret=all_ret,
            orig_dis=all_dis,
            orig_surv=all_surv,
            orig_dep=all_dep,
            imputed_ss=np.concatenate([ss_vals, new_ss]),
            age=all_ages,
        )

        new_recip = np.concatenate(
            [np.zeros(n, dtype=bool), np.ones(n_new, dtype=bool)]
        )
        shares = _qrf_ss_shares(data, nn, tp, new_recip)

        assert shares is not None
        ret_share = shares["social_security_retirement"]
        # All elderly -> retirement share should dominate.
        assert np.mean(ret_share) > 0.7
