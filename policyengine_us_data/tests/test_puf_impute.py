"""Tests for PUF imputation, specifically SS sub-component reconciliation.

When PUF imputation replaces social_security values, the sub-components
(retirement, disability, survivors, dependents) must be rescaled to match
the new total. See: https://github.com/PolicyEngine/policyengine-us-data/issues/551
"""

import numpy as np
import pytest

from policyengine_us_data.calibration.puf_impute import (
    MINIMUM_RETIREMENT_AGE,
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
):
    """Build a doubled data dict (CPS half + PUF half).

    CPS half keeps original values; PUF half has imputed
    social_security but original (stale) sub-components.
    """
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

    def test_new_recipient_elderly_gets_retirement(self):
        """New recipient aged >= 62 -> retirement."""
        data, n, tp = _make_data(
            orig_ss=np.array([0]),
            orig_ret=np.array([0]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([25000]),
            age=np.array([70]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ret = data["social_security_retirement"][tp][puf]
        dis = data["social_security_disability"][tp][puf]
        assert ret[0] == pytest.approx(25000, rel=1e-5)
        assert dis[0] == pytest.approx(0)

    def test_new_recipient_young_gets_disability(self):
        """New recipient aged < 62 -> disability."""
        data, n, tp = _make_data(
            orig_ss=np.array([0]),
            orig_ret=np.array([0]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([18000]),
            age=np.array([45]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ret = data["social_security_retirement"][tp][puf]
        dis = data["social_security_disability"][tp][puf]
        assert ret[0] == pytest.approx(0)
        assert dis[0] == pytest.approx(18000, rel=1e-5)

    def test_new_recipient_mixed_ages(self):
        """Multiple new recipients split by age threshold."""
        data, n, tp = _make_data(
            orig_ss=np.array([0, 0, 10000]),
            orig_ret=np.array([0, 0, 10000]),
            orig_dis=np.array([0, 0, 0]),
            orig_surv=np.array([0, 0, 0]),
            orig_dep=np.array([0, 0, 0]),
            imputed_ss=np.array([20000, 15000, 12000]),
            age=np.array([30, 65, 50]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ret = data["social_security_retirement"][tp][puf]
        dis = data["social_security_disability"][tp][puf]

        # Person 0: age 30, new -> disability
        assert ret[0] == pytest.approx(0)
        assert dis[0] == pytest.approx(20000, rel=1e-5)
        # Person 1: age 65, new -> retirement
        assert ret[1] == pytest.approx(15000, rel=1e-5)
        assert dis[1] == pytest.approx(0)
        # Person 2: existing CPS recipient, just rescaled
        assert ret[2] == pytest.approx(12000, rel=1e-5)

    def test_new_recipient_no_age_defaults_to_retirement(self):
        """Without age data, new recipients default to retirement."""
        data, n, tp = _make_data(
            orig_ss=np.array([0]),
            orig_ret=np.array([0]),
            orig_dis=np.array([0]),
            orig_surv=np.array([0]),
            orig_dep=np.array([0]),
            imputed_ss=np.array([25000]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ret = data["social_security_retirement"][tp][puf]
        assert ret[0] == pytest.approx(25000, rel=1e-5)

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
        """Comprehensive: subs must sum to total across all case types."""
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
