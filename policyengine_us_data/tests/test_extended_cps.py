"""Tests for extended CPS QRF imputation functions.

Uses synthetic data to verify that:
1. Sequential QRF preserves covariance between imputed variables
2. CPS-only imputation uses PUF-imputed income (not CPS originals)
3. Variable lists don't overlap (no double-imputation)
"""

import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.calibration.puf_impute import (
    IMPUTED_VARIABLES,
    OVERRIDDEN_IMPUTED_VARIABLES,
)
from policyengine_us_data.datasets.cps.extended_cps import (
    CPS_ONLY_IMPUTED_VARIABLES,
    CPS_STAGE2_INCOME_PREDICTORS,
)


class TestVariableListConsistency:
    """Variable lists should not overlap — no variable should be
    imputed by two different mechanisms."""

    def test_no_overlap_imputed_and_cps_only(self):
        overlap = set(IMPUTED_VARIABLES) & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert overlap == set(), f"Variables in both IMPUTED and CPS_ONLY: {overlap}"

    def test_no_overlap_overridden_and_cps_only(self):
        overlap = set(OVERRIDDEN_IMPUTED_VARIABLES) & set(CPS_ONLY_IMPUTED_VARIABLES)
        assert overlap == set(), f"Variables in both OVERRIDDEN and CPS_ONLY: {overlap}"

    def test_overridden_is_subset_of_imputed(self):
        not_in_imputed = set(OVERRIDDEN_IMPUTED_VARIABLES) - set(IMPUTED_VARIABLES)
        assert not_in_imputed == set(), (
            f"OVERRIDDEN vars not in IMPUTED: {not_in_imputed}"
        )

    def test_stage2_income_predictors_in_imputed(self):
        """Stage-2 income predictors must come from stage-1 imputation."""
        for var in CPS_STAGE2_INCOME_PREDICTORS:
            assert var in IMPUTED_VARIABLES, (
                f"Stage-2 income predictor '{var}' not in "
                f"IMPUTED_VARIABLES — won't have PUF-imputed values"
            )

    def test_cps_only_vars_mostly_exist_in_tbs(self):
        """Most CPS-only variables should exist in policyengine-us.
        A few may be missing if upstream hasn't added them yet."""
        from policyengine_us import CountryTaxBenefitSystem

        tbs = CountryTaxBenefitSystem()
        valid = [v for v in CPS_ONLY_IMPUTED_VARIABLES if v in tbs.variables]
        assert len(valid) >= len(CPS_ONLY_IMPUTED_VARIABLES) * 0.9, (
            f"Only {len(valid)}/{len(CPS_ONLY_IMPUTED_VARIABLES)} "
            f"CPS-only vars exist in tax-benefit system"
        )


class TestSequentialQRF:
    """Verify that sequential QRF produces correlated outputs,
    unlike independent imputation."""

    @pytest.fixture
    def correlated_training_data(self):
        """Synthetic data where y1 and y2 are correlated through x."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.normal(50, 15, n)
        # y1 strongly correlated with x
        y1 = 0.8 * x + rng.normal(0, 5, n)
        # y2 correlated with both x and y1
        y2 = 0.3 * x + 0.5 * y1 + rng.normal(0, 3, n)
        return pd.DataFrame({"x": x, "y1": y1, "y2": y2})

    def test_sequential_qrf_preserves_correlation(self, correlated_training_data):
        """When y2 depends on y1, sequential QRF should produce
        y1-y2 correlation closer to training data than independent
        imputation would."""
        from microimpute.models.qrf import QRF

        df = correlated_training_data
        train = df.sample(1500, random_state=0)
        test_x = df.drop(train.index)[["x"]]

        # Sequential: y2 conditions on y1
        qrf = QRF(log_level="ERROR", memory_efficient=True)
        result = qrf.fit_predict(
            X_train=train,
            X_test=test_x,
            predictors=["x"],
            imputed_variables=["y1", "y2"],
            n_jobs=1,
        )

        # The imputed y1 and y2 should be positively correlated
        corr = result["y1"].corr(result["y2"])
        assert corr > 0.5, (
            f"Sequential QRF y1-y2 correlation = {corr:.3f}, expected > 0.5"
        )

    def test_single_call_vs_separate_calls_differ(self, correlated_training_data):
        """Imputing y1 and y2 in a single sequential call should
        produce different y2 values than imputing them separately
        (the old batched approach)."""
        from microimpute.models.qrf import QRF

        df = correlated_training_data
        train = df.sample(1500, random_state=0)
        test_x = df.drop(train.index)[["x"]]

        # Sequential (single call)
        qrf_seq = QRF(log_level="ERROR", memory_efficient=True)
        result_seq = qrf_seq.fit_predict(
            X_train=train,
            X_test=test_x,
            predictors=["x"],
            imputed_variables=["y1", "y2"],
            n_jobs=1,
        )

        # Independent (separate calls, like old batched approach)
        qrf_y1 = QRF(log_level="ERROR", memory_efficient=True)
        result_y1 = qrf_y1.fit_predict(
            X_train=train[["x", "y1"]],
            X_test=test_x,
            predictors=["x"],
            imputed_variables=["y1"],
            n_jobs=1,
        )

        qrf_y2 = QRF(log_level="ERROR", memory_efficient=True)
        result_y2 = qrf_y2.fit_predict(
            X_train=train[["x", "y2"]],
            X_test=test_x,
            predictors=["x"],
            imputed_variables=["y2"],
            n_jobs=1,
        )

        # The sequential y1-y2 correlation should be higher than
        # the independent one
        corr_seq = result_seq["y1"].corr(result_seq["y2"])
        corr_indep = result_y1["y1"].corr(result_y2["y2"])

        assert corr_seq > corr_indep, (
            f"Sequential corr ({corr_seq:.3f}) should exceed "
            f"independent corr ({corr_indep:.3f})"
        )
