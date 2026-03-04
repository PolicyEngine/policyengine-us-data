"""Tests for retirement contribution QRF imputation.

Covers:
- Constants & list membership
- _get_retirement_limits() logic
- _impute_retirement_contributions() with mocked QRF/Microsimulation
- puf_clone_dataset() integration routing for retirement variables
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.calibration.puf_impute import (
    CPS_RETIREMENT_VARIABLES,
    IMPUTED_VARIABLES,
    OVERRIDDEN_IMPUTED_VARIABLES,
    RETIREMENT_DEMOGRAPHIC_PREDICTORS,
    RETIREMENT_INCOME_PREDICTORS,
    RETIREMENT_PREDICTORS,
    _get_retirement_limits,
    _impute_retirement_contributions,
    puf_clone_dataset,
)

# The function imports Microsimulation and QRF locally via:
#   from policyengine_us import Microsimulation
#   from microimpute.models.qrf import QRF
# We must patch those source modules so the local import picks
# up mocks, not real objects.
_MSIM_PATCH = "policyengine_us.Microsimulation"
_QRF_PATCH = "microimpute.models.qrf.QRF"


# ── helpers ──────────────────────────────────────────────────────────


def _make_mock_data(n_persons=20, n_households=5, time_period=2024):
    """Minimal CPS data dict with retirement variables."""
    rng = np.random.default_rng(42)
    person_ids = np.arange(1, n_persons + 1)
    hh_ids_person = np.repeat(
        np.arange(1, n_households + 1),
        n_persons // n_households,
    )

    data = {
        "person_id": {time_period: person_ids},
        "household_id": {time_period: np.arange(1, n_households + 1)},
        "tax_unit_id": {time_period: np.arange(1, n_households + 1)},
        "spm_unit_id": {time_period: np.arange(1, n_households + 1)},
        "person_household_id": {time_period: hh_ids_person},
        "person_tax_unit_id": {time_period: hh_ids_person.copy()},
        "person_spm_unit_id": {time_period: hh_ids_person.copy()},
        "age": {
            time_period: rng.integers(18, 80, size=n_persons).astype(
                np.float32
            )
        },
        "is_male": {
            time_period: rng.integers(0, 2, size=n_persons).astype(np.float32)
        },
        "household_weight": {time_period: np.ones(n_households) * 1000},
        "employment_income": {
            time_period: rng.uniform(0, 100_000, n_persons).astype(np.float32)
        },
        "self_employment_income": {
            time_period: rng.uniform(0, 50_000, n_persons).astype(np.float32)
        },
    }
    for var in CPS_RETIREMENT_VARIABLES:
        data[var] = {
            time_period: rng.uniform(0, 5000, n_persons).astype(np.float32)
        }
    return data


def _make_cps_df(n, rng):
    """Build a mock CPS DataFrame with all needed columns."""
    return pd.DataFrame(
        {
            # Demographics
            "age": rng.integers(18, 80, n).astype(float),
            "is_male": rng.integers(0, 2, n).astype(float),
            "tax_unit_is_joint": rng.integers(0, 2, n).astype(float),
            "tax_unit_count_dependents": rng.integers(0, 4, n).astype(float),
            "is_tax_unit_head": rng.integers(0, 2, n).astype(float),
            "is_tax_unit_spouse": np.zeros(n),
            "is_tax_unit_dependent": np.zeros(n),
            # Income predictors
            "employment_income": rng.uniform(0, 100_000, n),
            "self_employment_income": rng.uniform(0, 50_000, n),
            "taxable_interest_income": rng.uniform(0, 5_000, n),
            "qualified_dividend_income": rng.uniform(0, 3_000, n),
            "taxable_pension_income": rng.uniform(0, 20_000, n),
            "social_security": rng.uniform(0, 15_000, n),
            # Targets
            "traditional_401k_contributions": rng.uniform(0, 5000, n),
            "roth_401k_contributions": rng.uniform(0, 3000, n),
            "traditional_ira_contributions": rng.uniform(0, 2000, n),
            "roth_ira_contributions": rng.uniform(0, 2000, n),
            "self_employed_pension_contributions": rng.uniform(0, 10_000, n),
        }
    )


def _make_mock_sim(cps_df):
    """Build a mock Microsimulation object."""
    sim = MagicMock()

    def calc_df(cols):
        return cps_df[cols].copy()

    def calc_single(var):
        result = MagicMock()
        result.values = cps_df[var].values
        return result

    sim.calculate_dataframe = calc_df
    sim.calculate = calc_single
    return sim


def _make_mock_qrf_class(predictions_df):
    """Build a mock QRF class whose predict returns predictions_df."""
    mock_cls = MagicMock()
    fitted = MagicMock()
    fitted.predict.return_value = predictions_df
    mock_cls.return_value.fit.return_value = fitted
    return mock_cls


# ── TestConstants ────────────────────────────────────────────────────


class TestConstants:
    def test_retirement_vars_not_in_imputed(self):
        """Retirement vars must NOT be in IMPUTED_VARIABLES."""
        for var in CPS_RETIREMENT_VARIABLES:
            assert (
                var not in IMPUTED_VARIABLES
            ), f"{var} should not be in IMPUTED_VARIABLES"

    def test_retirement_vars_not_in_overridden(self):
        for var in CPS_RETIREMENT_VARIABLES:
            assert var not in OVERRIDDEN_IMPUTED_VARIABLES

    def test_five_retirement_variables(self):
        assert len(CPS_RETIREMENT_VARIABLES) == 5

    def test_retirement_variable_names(self):
        expected = {
            "traditional_401k_contributions",
            "roth_401k_contributions",
            "traditional_ira_contributions",
            "roth_ira_contributions",
            "self_employed_pension_contributions",
        }
        assert set(CPS_RETIREMENT_VARIABLES) == expected

    def test_retirement_predictors_include_income(self):
        for var in RETIREMENT_INCOME_PREDICTORS:
            assert var in RETIREMENT_PREDICTORS

    def test_retirement_predictors_include_demographics(self):
        for pred in RETIREMENT_DEMOGRAPHIC_PREDICTORS:
            assert pred in RETIREMENT_PREDICTORS

    def test_income_predictors_in_imputed_variables(self):
        """All income predictors must be available from PUF QRF."""
        for var in RETIREMENT_INCOME_PREDICTORS:
            assert (
                var in IMPUTED_VARIABLES
            ), f"{var} not in IMPUTED_VARIABLES — won't be in puf_imputations"

    def test_predictors_are_combined_lists(self):
        expected = (
            RETIREMENT_DEMOGRAPHIC_PREDICTORS + RETIREMENT_INCOME_PREDICTORS
        )
        assert RETIREMENT_PREDICTORS == expected


# ── TestGetRetirementLimits ──────────────────────────────────────────


class TestGetRetirementLimits:
    def test_known_year_2024(self):
        lim = _get_retirement_limits(2024)
        assert lim["401k"] == 23_000
        assert lim["401k_catch_up"] == 7_500
        assert lim["ira"] == 7_000
        assert lim["ira_catch_up"] == 1_000

    def test_known_year_2020(self):
        lim = _get_retirement_limits(2020)
        assert lim["401k"] == 19_500
        assert lim["ira"] == 6_000

    def test_known_year_2025(self):
        lim = _get_retirement_limits(2025)
        assert lim["401k"] == 23_500

    def test_clamps_below_min_year(self):
        assert _get_retirement_limits(2015) == _get_retirement_limits(2020)

    def test_clamps_above_max_year(self):
        assert _get_retirement_limits(2030) == _get_retirement_limits(2025)

    def test_all_years_have_expected_keys(self):
        for year in range(2020, 2026):
            lim = _get_retirement_limits(year)
            assert set(lim.keys()) == {
                "401k",
                "401k_catch_up",
                "ira",
                "ira_catch_up",
                "se_pension_rate",
                "se_pension_dollar_limit",
            }

    def test_limits_increase_monotonically(self):
        prev = _get_retirement_limits(2020)["401k"]
        for year in range(2021, 2026):
            cur = _get_retirement_limits(year)["401k"]
            assert cur >= prev
            prev = cur

    def test_ira_limits_increase_monotonically(self):
        prev = _get_retirement_limits(2020)["ira"]
        for year in range(2021, 2026):
            cur = _get_retirement_limits(year)["ira"]
            assert cur >= prev
            prev = cur


# ── TestImputeRetirementContributions ────────────────────────────────


class TestImputeRetirementContributions:
    """Tests with mocked Microsimulation and QRF."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.n = 50
        self.time_period = 2024
        rng = np.random.default_rng(42)

        self.data = {
            "person_id": {self.time_period: np.arange(1, self.n + 1)},
        }

        emp = rng.uniform(0, 150_000, self.n).astype(np.float32)
        emp[:10] = 0  # 10 records with $0 wages
        se = rng.uniform(0, 80_000, self.n).astype(np.float32)
        se[:20] = 0  # 20 records with $0 SE income

        self.puf_imputations = {
            "employment_income": emp,
            "self_employment_income": se,
            "taxable_interest_income": rng.uniform(0, 5_000, self.n).astype(
                np.float32
            ),
            "qualified_dividend_income": rng.uniform(0, 3_000, self.n).astype(
                np.float32
            ),
            "taxable_pension_income": rng.uniform(0, 20_000, self.n).astype(
                np.float32
            ),
            "social_security": rng.uniform(0, 15_000, self.n).astype(
                np.float32
            ),
        }

        self.cps_df = _make_cps_df(self.n, rng)

    def _call_with_mocks(self, pred_df):
        """Run _impute_retirement_contributions with mocked deps."""
        import sys

        mock_sim = _make_mock_sim(self.cps_df)
        mock_qrf_cls = _make_mock_qrf_class(pred_df)

        # patch() doesn't work reliably on MagicMock modules,
        # so we set the QRF attribute directly.
        qrf_mod = sys.modules["microimpute.models.qrf"]
        old_qrf = getattr(qrf_mod, "QRF", None)
        qrf_mod.QRF = mock_qrf_cls
        try:
            with patch(_MSIM_PATCH, return_value=mock_sim):
                return _impute_retirement_contributions(
                    self.data,
                    self.puf_imputations,
                    self.time_period,
                    "/fake/path.h5",
                )
        finally:
            if old_qrf is not None:
                qrf_mod.QRF = old_qrf

    def _uniform_preds(self, value):
        """Build a DataFrame predicting `value` for all vars."""
        return pd.DataFrame(
            {var: np.full(self.n, value) for var in CPS_RETIREMENT_VARIABLES}
        )

    def _random_preds(self, low, high, seed=99):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                var: rng.uniform(low, high, self.n)
                for var in CPS_RETIREMENT_VARIABLES
            }
        )

    def test_returns_all_retirement_vars(self):
        result = self._call_with_mocks(self._random_preds(0, 8000))
        for var in CPS_RETIREMENT_VARIABLES:
            assert var in result
            assert len(result[var]) == self.n

    def test_nonnegative_output(self):
        result = self._call_with_mocks(self._random_preds(-5000, 8000, seed=7))
        for var in CPS_RETIREMENT_VARIABLES:
            assert np.all(result[var] >= 0), f"{var} has negative values"

    def test_401k_capped(self):
        result = self._call_with_mocks(self._uniform_preds(50_000.0))
        lim = _get_retirement_limits(self.time_period)
        max_401k = lim["401k"] + lim["401k_catch_up"]

        for var in (
            "traditional_401k_contributions",
            "roth_401k_contributions",
        ):
            assert np.all(result[var] <= max_401k), f"{var} exceeds 401k limit"

    def test_ira_capped(self):
        result = self._call_with_mocks(self._uniform_preds(50_000.0))
        lim = _get_retirement_limits(self.time_period)
        max_ira = lim["ira"] + lim["ira_catch_up"]

        for var in (
            "traditional_ira_contributions",
            "roth_ira_contributions",
        ):
            assert np.all(result[var] <= max_ira), f"{var} exceeds IRA limit"

    def test_401k_zero_when_no_wages(self):
        result = self._call_with_mocks(self._uniform_preds(5_000.0))
        zero_wage = self.puf_imputations["employment_income"] == 0
        assert zero_wage.sum() == 10

        for var in (
            "traditional_401k_contributions",
            "roth_401k_contributions",
        ):
            assert np.all(
                result[var][zero_wage] == 0
            ), f"{var} should be 0 when employment_income is 0"

    def test_se_pension_zero_when_no_se_income(self):
        result = self._call_with_mocks(self._uniform_preds(5_000.0))
        zero_se = self.puf_imputations["self_employment_income"] == 0
        assert zero_se.sum() == 20
        assert np.all(
            result["self_employed_pension_contributions"][zero_se] == 0
        )

    def test_catch_up_age_threshold(self):
        """Records age >= 50 get higher caps than younger."""
        self.cps_df["age"] = np.concatenate(
            [np.full(25, 30.0), np.full(25, 55.0)]
        )
        # All have positive income
        self.puf_imputations["employment_income"] = np.full(
            self.n, 100_000.0
        ).astype(np.float32)

        lim = _get_retirement_limits(self.time_period)
        val = float(lim["401k"]) + 1000  # 24000

        result = self._call_with_mocks(self._uniform_preds(val))

        young_401k = result["traditional_401k_contributions"][:25]
        old_401k = result["traditional_401k_contributions"][25:]

        # Young capped at base limit
        assert np.all(young_401k == lim["401k"])
        # Old get full value (within catch-up limit)
        assert np.all(old_401k == val)

    def test_ira_catch_up_threshold(self):
        """IRA catch-up also works for age >= 50."""
        self.cps_df["age"] = np.concatenate(
            [np.full(25, 30.0), np.full(25, 55.0)]
        )
        lim = _get_retirement_limits(self.time_period)
        val = float(lim["ira"]) + 500  # 7500

        result = self._call_with_mocks(self._uniform_preds(val))

        young_ira = result["traditional_ira_contributions"][:25]
        old_ira = result["traditional_ira_contributions"][25:]

        assert np.all(young_ira == lim["ira"])
        assert np.all(old_ira == val)

    def test_401k_nonzero_for_positive_wages(self):
        """Records with positive wages should keep their
        predicted 401k (not zeroed out)."""
        result = self._call_with_mocks(self._uniform_preds(5_000.0))
        pos_wage = self.puf_imputations["employment_income"] > 0
        for var in (
            "traditional_401k_contributions",
            "roth_401k_contributions",
        ):
            assert np.all(result[var][pos_wage] > 0)

    def test_se_pension_nonzero_for_positive_se(self):
        result = self._call_with_mocks(self._uniform_preds(5_000.0))
        pos_se = self.puf_imputations["self_employment_income"] > 0
        assert np.all(
            result["self_employed_pension_contributions"][pos_se] > 0
        )

    def test_se_pension_capped_at_rate_times_income(self):
        """SE pension should not exceed 25% of SE income."""
        # Predict a large value that would exceed the SE cap
        result = self._call_with_mocks(self._uniform_preds(50_000.0))
        lim = _get_retirement_limits(self.time_period)
        se_income = self.puf_imputations["self_employment_income"]
        se_cap = np.minimum(
            se_income * lim["se_pension_rate"],
            lim["se_pension_dollar_limit"],
        )
        pos_se = se_income > 0
        assert np.all(
            result["self_employed_pension_contributions"][pos_se]
            <= se_cap[pos_se] + 0.01
        ), "SE pension exceeds 25%-of-income cap"

    def test_qrf_failure_returns_zeros(self):
        """When QRF fit/predict throws, should return all zeros."""
        import sys

        mock_sim = _make_mock_sim(self.cps_df)

        # Make a QRF that crashes on fit
        mock_qrf_cls = MagicMock()
        mock_qrf_cls.return_value.fit.side_effect = RuntimeError(
            "QRF exploded"
        )

        qrf_mod = sys.modules["microimpute.models.qrf"]
        old_qrf = getattr(qrf_mod, "QRF", None)
        qrf_mod.QRF = mock_qrf_cls
        try:
            with patch(_MSIM_PATCH, return_value=mock_sim):
                result = _impute_retirement_contributions(
                    self.data,
                    self.puf_imputations,
                    self.time_period,
                    "/fake/path.h5",
                )
        finally:
            if old_qrf is not None:
                qrf_mod.QRF = old_qrf

        for var in CPS_RETIREMENT_VARIABLES:
            assert var in result
            assert np.all(result[var] == 0)

    def test_training_data_failure_returns_zeros(self):
        """When CPS calculate_dataframe fails, returns zeros."""
        import sys

        mock_sim = MagicMock()
        mock_sim.calculate_dataframe.side_effect = ValueError(
            "missing variable"
        )

        qrf_mod = sys.modules["microimpute.models.qrf"]
        old_qrf = getattr(qrf_mod, "QRF", None)
        qrf_mod.QRF = MagicMock()
        try:
            with patch(_MSIM_PATCH, return_value=mock_sim):
                result = _impute_retirement_contributions(
                    self.data,
                    self.puf_imputations,
                    self.time_period,
                    "/fake/path.h5",
                )
        finally:
            if old_qrf is not None:
                qrf_mod.QRF = old_qrf

        for var in CPS_RETIREMENT_VARIABLES:
            assert var in result
            assert np.all(result[var] == 0)


# ── TestPufCloneRetirementRouting ────────────────────────────────────


class TestPufCloneRetirementRouting:
    """Test puf_clone_dataset() routes retirement vars correctly."""

    def test_retirement_vars_duplicated_when_skip_qrf(self):
        data = _make_mock_data()
        state_fips = np.array([1, 2, 36, 6, 48])
        n = 20

        result = puf_clone_dataset(
            data=data,
            state_fips=state_fips,
            time_period=2024,
            skip_qrf=True,
        )

        for var in CPS_RETIREMENT_VARIABLES:
            vals = result[var][2024]
            assert len(vals) == n * 2
            np.testing.assert_array_equal(vals[:n], vals[n:])

    def test_retirement_vars_use_imputed_when_available(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])
        n = 20

        fake_retirement = {
            var: np.full(n, 999.0) for var in CPS_RETIREMENT_VARIABLES
        }

        with (
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._impute_retirement_contributions",
                return_value=fake_retirement,
            ),
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._run_qrf_imputation",
                return_value=(
                    {v: np.zeros(n) for v in IMPUTED_VARIABLES},
                    {},
                ),
            ),
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._impute_weeks_unemployed",
                return_value=np.zeros(n),
            ),
            patch(_MSIM_PATCH),
        ):
            result = puf_clone_dataset(
                data=data,
                state_fips=state_fips,
                time_period=2024,
                skip_qrf=False,
                puf_dataset="/fake/puf.h5",
                dataset_path="/fake/cps.h5",
            )

        for var in CPS_RETIREMENT_VARIABLES:
            vals = result[var][2024]
            assert len(vals) == n * 2
            np.testing.assert_array_equal(vals[:n], data[var][2024])
            np.testing.assert_array_equal(vals[n:], np.full(n, 999.0))

    def test_cps_half_unchanged_with_imputation(self):
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])
        n = 20

        originals = {
            var: data[var][2024].copy() for var in CPS_RETIREMENT_VARIABLES
        }
        fake_retirement = {
            var: np.zeros(n) for var in CPS_RETIREMENT_VARIABLES
        }

        with (
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._impute_retirement_contributions",
                return_value=fake_retirement,
            ),
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._run_qrf_imputation",
                return_value=(
                    {v: np.zeros(n) for v in IMPUTED_VARIABLES},
                    {},
                ),
            ),
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._impute_weeks_unemployed",
                return_value=np.zeros(n),
            ),
            patch(_MSIM_PATCH),
        ):
            result = puf_clone_dataset(
                data=data,
                state_fips=state_fips,
                time_period=2024,
                skip_qrf=False,
                puf_dataset="/fake/puf.h5",
                dataset_path="/fake/cps.h5",
            )

        for var in CPS_RETIREMENT_VARIABLES:
            np.testing.assert_array_equal(
                result[var][2024][:n], originals[var]
            )

    def test_puf_half_gets_zero_retirement_for_zero_imputed(self):
        """When imputation returns zeros, PUF half should be zero."""
        data = _make_mock_data(n_persons=20, n_households=5)
        state_fips = np.array([1, 2, 36, 6, 48])
        n = 20

        fake_retirement = {
            var: np.zeros(n) for var in CPS_RETIREMENT_VARIABLES
        }

        with (
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._impute_retirement_contributions",
                return_value=fake_retirement,
            ),
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._run_qrf_imputation",
                return_value=(
                    {v: np.zeros(n) for v in IMPUTED_VARIABLES},
                    {},
                ),
            ),
            patch(
                "policyengine_us_data.calibration.puf_impute"
                "._impute_weeks_unemployed",
                return_value=np.zeros(n),
            ),
            patch(_MSIM_PATCH),
        ):
            result = puf_clone_dataset(
                data=data,
                state_fips=state_fips,
                time_period=2024,
                skip_qrf=False,
                puf_dataset="/fake/puf.h5",
                dataset_path="/fake/cps.h5",
            )

        for var in CPS_RETIREMENT_VARIABLES:
            puf_half = result[var][2024][n:]
            assert np.all(puf_half == 0)


# ── TestLimitsMatchCps ───────────────────────────────────────────────


class TestLimitsMatchYaml:
    """Cross-check _get_retirement_limits() against the YAML source."""

    @pytest.fixture(autouse=True)
    def _load_yaml(self):
        from importlib.resources import files as pkg_files

        import yaml

        yaml_path = (
            pkg_files("policyengine_us_data")
            / "datasets"
            / "cps"
            / "imputation_parameters.yaml"
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        self.yaml_limits = params["retirement_contribution_limits"]

    def test_all_years_match_yaml(self):
        """Every year in the YAML must match _get_retirement_limits()."""
        for year, expected in self.yaml_limits.items():
            actual = _get_retirement_limits(year)
            assert actual == expected, f"Year {year}: {actual} != {expected}"

    def test_yaml_has_expected_years(self):
        assert set(self.yaml_limits.keys()) == set(range(2020, 2026))
