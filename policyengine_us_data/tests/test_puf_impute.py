"""Tests for PUF imputation, specifically SS sub-component reconciliation.

When PUF imputation replaces social_security values, the sub-components
(retirement, disability, survivors, dependents) must be rescaled to match
the new total. See: https://github.com/PolicyEngine/policyengine-us-data/issues/551
"""

import numpy as np
import pytest

from policyengine_us_data.calibration.puf_impute import (
    reconcile_ss_subcomponents,
)

SS_SUBS = [
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
]


def _make_data(orig_ss, orig_ret, orig_dis, orig_surv, orig_dep, imputed_ss):
    """Build a doubled data dict (CPS half + PUF half).

    CPS half keeps original values; PUF half has imputed
    social_security but original (stale) sub-components.
    """
    n = len(orig_ss)
    tp = 2024
    return (
        {
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
        },
        n,
        tp,
    )


class TestReconcileSsSubcomponents:
    """Sub-components must sum to social_security after reconciliation."""

    def test_proportional_rescaling(self):
        """When CPS has a split, PUF subs should scale proportionally."""
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

        # Check proportions preserved
        # Person 0: ret=14/20=70%, dis=6/20=30%
        ret = data["social_security_retirement"][tp][puf]
        np.testing.assert_allclose(ret[0], 30000 * 0.7, rtol=1e-5)
        dis = data["social_security_disability"][tp][puf]
        np.testing.assert_allclose(dis[0], 30000 * 0.3, rtol=1e-5)

    def test_new_recipients_default_to_retirement(self):
        """When CPS had zero SS but PUF imputes positive, assign to retirement."""
        data, n, tp = _make_data(
            orig_ss=np.array([0, 10000]),
            orig_ret=np.array([0, 10000]),
            orig_dis=np.array([0, 0]),
            orig_surv=np.array([0, 0]),
            orig_dep=np.array([0, 0]),
            imputed_ss=np.array([25000, 12000]),
        )
        reconcile_ss_subcomponents(data, n, tp)

        puf = slice(n, 2 * n)
        ret = data["social_security_retirement"][tp][puf]
        dis = data["social_security_disability"][tp][puf]

        # Person 0: new recipient -> all goes to retirement
        assert ret[0] == pytest.approx(25000, rel=1e-5)
        assert dis[0] == pytest.approx(0)

    def test_imputed_zero_clears_subs(self):
        """When PUF imputes zero SS, all sub-components should be zero."""
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
        # Should not raise
        reconcile_ss_subcomponents(data, n, tp)

    def test_no_social_security_is_noop(self):
        """If social_security is absent, function is a no-op."""
        data = {"some_var": {2024: np.array([1, 2])}}
        reconcile_ss_subcomponents(data, 1, 2024)
