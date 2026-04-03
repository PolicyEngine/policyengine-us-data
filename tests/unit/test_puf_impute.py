"""Tests for PUF imputation, specifically SS sub-component reconciliation.

When PUF imputation replaces social_security values, the sub-components
(retirement, disability, survivors, dependents) must be predicted from
demographics (not copied from the statistically-linked CPS record).
See: https://github.com/PolicyEngine/policyengine-us-data/issues/551
"""

import numpy as np
import pytest

from policyengine_us_data.calibration.puf_impute import (
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
        data["is_male"] = {tp: np.concatenate([is_male, is_male]).astype(np.float32)}
    return data, n, tp


class TestReconcileSsSubcomponents:
    """Sub-components must sum to social_security after reconciliation."""

    def test_subs_sum_to_total(self):
        """PUF subs must sum to imputed SS total."""
        data, n, tp = _make_data(
            orig_ss=np.array([20000, 15000]),
            orig_ret=np.array([14000, 0]),
            orig_dis=np.array([6000, 15000]),
            orig_surv=np.array([0, 0]),
            orig_dep=np.array([0, 0]),
            imputed_ss=np.array([30000, 10000]),
            age=np.array([70, 45]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ss = data["social_security"][tp][puf]
        total_subs = sum(data[sub][tp][puf] for sub in SS_SUBS)
        np.testing.assert_allclose(total_subs, ss, rtol=1e-5)

    def test_age_determines_split(self):
        """Age heuristic: >= 62 -> retirement, < 62 -> disability."""
        data, n, tp = _make_data(
            orig_ss=np.array([20000, 15000]),
            orig_ret=np.array([14000, 0]),
            orig_dis=np.array([6000, 15000]),
            orig_surv=np.array([0, 0]),
            orig_dep=np.array([0, 0]),
            imputed_ss=np.array([30000, 10000]),
            age=np.array([70, 45]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ret = data["social_security_retirement"][tp][puf]
        dis = data["social_security_disability"][tp][puf]
        # Person 0 is 70 -> retirement
        assert ret[0] == pytest.approx(30000)
        assert dis[0] == pytest.approx(0)
        # Person 1 is 45 -> disability
        assert ret[1] == pytest.approx(0)
        assert dis[1] == pytest.approx(10000)

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

    def test_subs_sum_to_total_mixed_cases(self):
        """Comprehensive: subs sum to total across mixed cases."""
        data, n, tp = _make_data(
            # Person 0: has CPS SS, has PUF SS (old)
            # Person 1: no CPS SS, has PUF SS (old)
            # Person 2: no CPS SS, has PUF SS (young)
            # Person 3: has CPS SS, PUF zeroed out
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
    """Age-based fallback for SS type classification."""

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
        puf_has_ss = np.array([True])
        shares = _age_heuristic_ss_shares(data, n, tp, puf_has_ss)

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
        puf_has_ss = np.array([True])
        shares = _age_heuristic_ss_shares(data, n, tp, puf_has_ss)

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
        puf_has_ss = np.array([True])
        shares = _age_heuristic_ss_shares(data, n, tp, puf_has_ss)

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
        puf_has_ss = np.array([True, True, True])
        shares = _age_heuristic_ss_shares(data, n, tp, puf_has_ss)

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
        puf_has_ss = np.array([False, False, True])
        result = _qrf_ss_shares(data, n, tp, puf_has_ss)
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

        # PUF SS for all records.
        puf_ss = rng.uniform(5000, 40000, size=n).astype(np.float32)

        data, nn, tp = _make_data(
            orig_ss=ss_vals,
            orig_ret=ret,
            orig_dis=dis,
            orig_surv=surv,
            orig_dep=dep,
            imputed_ss=puf_ss,
            age=ages,
            is_male=is_male,
        )

        puf_has_ss = puf_ss > 0
        shares = _qrf_ss_shares(data, nn, tp, puf_has_ss)

        assert shares is not None
        total = sum(shares.get(sub, 0) for sub in SS_SUBS)
        np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_elderly_predicted_as_retirement(self):
        """QRF should mostly predict retirement for age >= 62."""
        pytest.importorskip("microimpute")
        rng = np.random.default_rng(123)
        n_train = 500

        ages = rng.integers(20, 90, size=n_train).astype(np.float32)
        ss_vals = rng.uniform(5000, 40000, size=n_train).astype(np.float32)
        ret = np.where(ages >= 62, ss_vals, 0).astype(np.float32)
        dis = np.where(ages < 62, ss_vals, 0).astype(np.float32)
        surv = np.zeros(n_train, dtype=np.float32)
        dep = np.zeros(n_train, dtype=np.float32)

        # Only elderly records get PUF SS > 0; training records
        # get PUF SS = 0 so puf_has_ss selects just the elderly.
        n_pred = 10
        pred_ages = np.full(n_pred, 70, dtype=np.float32)

        all_ages = np.concatenate([ages, pred_ages])
        all_ss = np.concatenate([ss_vals, np.zeros(n_pred)])
        all_ret = np.concatenate([ret, np.zeros(n_pred)])
        all_dis = np.concatenate([dis, np.zeros(n_pred)])
        all_surv = np.concatenate([surv, np.zeros(n_pred)])
        all_dep = np.concatenate([dep, np.zeros(n_pred)])
        puf_ss = np.concatenate(
            [
                np.zeros(n_train, dtype=np.float32),
                np.full(n_pred, 20000, dtype=np.float32),
            ]
        )

        data, nn, tp = _make_data(
            orig_ss=all_ss,
            orig_ret=all_ret,
            orig_dis=all_dis,
            orig_surv=all_surv,
            orig_dep=all_dep,
            imputed_ss=puf_ss,
            age=all_ages,
        )

        puf_has_ss = data["social_security"][tp][nn:] > 0
        shares = _qrf_ss_shares(data, nn, tp, puf_has_ss)

        assert shares is not None
        ret_share = shares["social_security_retirement"]
        # All elderly -> retirement share should dominate.
        assert np.mean(ret_share) > 0.7
