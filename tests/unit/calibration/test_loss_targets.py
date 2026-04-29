import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.loss import (
    _get_aca_national_targets,
    _add_education_credit_targets,
    _add_aotc_targets,
    _add_ctc_targets,
    _get_medicaid_national_targets,
    _load_aca_spending_and_enrollment_targets,
    _load_medicaid_enrollment_targets,
    HARD_CODED_TOTALS,
)


def test_aca_targets_roll_forward_to_2025():
    targets, data_year = _load_aca_spending_and_enrollment_targets(2025)

    assert data_year == 2025
    assert len(targets) == 51
    assert int(targets["enrollment"].sum()) == 21_822_894


def test_aca_targets_use_latest_available_year():
    _, data_year = _load_aca_spending_and_enrollment_targets(2026)
    assert data_year == 2025


def test_aca_targets_fall_back_to_earliest_available_year():
    _, data_year = _load_aca_spending_and_enrollment_targets(2023)
    assert data_year == 2024


def test_aca_national_targets_annualize_2025_state_file():
    spending, enrollment, data_year = _get_aca_national_targets(2025)

    assert data_year == 2025
    assert enrollment == 21_822_894
    assert spending == pytest.approx(143_951_057_388.72)


def test_medicaid_targets_roll_forward_to_2025():
    targets, data_year = _load_medicaid_enrollment_targets(2025)

    assert data_year == 2025
    assert len(targets) == 51
    assert int(targets["enrollment"].sum()) == 69_185_225


def test_medicaid_targets_fall_back_to_earliest_available_year():
    _, data_year = _load_medicaid_enrollment_targets(2023)
    assert data_year == 2024


def test_medicaid_national_targets_use_2025_values():
    spending, enrollment, data_year = _get_medicaid_national_targets(2025)

    assert data_year == 2025
    assert enrollment == 69_185_225
    assert spending == pytest.approx(1_000_645_800_000.0001)


class _FakeArrayResult:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)


class _FakeSimulation:
    def __init__(self):
        self.calculate_calls = []
        self.map_result_calls = []

    def calculate(self, variable, map_to=None, period=None):
        self.calculate_calls.append((variable, map_to, period))
        values = {
            "education_tax_credits": [500.0, 0.0, 300.0],
            "refundable_american_opportunity_credit": [400.0, 0.0, 250.0],
            "refundable_ctc": [100.0, 0.0, 50.0],
            "non_refundable_ctc": [80.0, 10.0, 0.0],
        }
        if variable not in values:
            raise AssertionError(f"Unexpected variable {variable!r}")
        if map_to == "household":
            return _FakeArrayResult(values[variable])
        if map_to is None:
            return _FakeArrayResult(values[variable])
        raise AssertionError(f"Unexpected map_to {map_to!r}")

    def map_result(self, values, source_entity, target_entity, how=None):
        self.map_result_calls.append((source_entity, target_entity, how))
        assert source_entity == "tax_unit"
        assert target_entity == "household"
        return np.asarray(values, dtype=np.float32)


def test_add_ctc_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_national_geography_soi_target",
        lambda variable, year: {
            "refundable_ctc": {"amount": 33_000.0, "count": 17.0},
            "non_refundable_ctc": {"amount": 81_000.0, "count": 37.0},
        }[variable],
    )
    sim = _FakeSimulation()

    targets, loss_matrix = _add_ctc_targets(
        pd.DataFrame(),
        [],
        sim,
        2024,
    )

    assert targets == [33_000.0, 17.0, 81_000.0, 37.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/refundable_ctc"],
        np.array([100.0, 0.0, 50.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/refundable_ctc_count"],
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/non_refundable_ctc"],
        np.array([80.0, 10.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/non_refundable_ctc_count"],
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_add_aotc_targets(monkeypatch):
    def fake_get_tracked_soi_row(variable, requested_year, *, count, **kwargs):
        assert variable == "refundable_american_opportunity_credit"
        assert requested_year == 2024
        return pd.Series(
            {
                "Year": 2023,
                "Value": 5_821_688.0 if count else 5_090_364_000.0,
                "SOI table": "Table 3.3",
            }
        )

    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_tracked_soi_row",
        fake_get_tracked_soi_row,
    )
    sim = _FakeSimulation()

    targets, loss_matrix = _add_aotc_targets(
        pd.DataFrame(),
        [],
        sim,
        2024,
    )

    assert targets == [5_090_364_000.0, 5_821_688.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/refundable_american_opportunity_credit"],
        np.array([400.0, 0.0, 250.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/refundable_american_opportunity_credit_count"],
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )


def test_add_education_credit_targets(monkeypatch):
    def fake_get_tracked_soi_row(variable, requested_year, *, count, **kwargs):
        assert variable == "education_tax_credits"
        assert requested_year == 2024
        return pd.Series(
            {
                "Year": 2023,
                "Value": 7_211_349.0 if count else 7_554_668_000.0,
                "SOI table": "Table 3.3",
            }
        )

    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_tracked_soi_row",
        fake_get_tracked_soi_row,
    )
    sim = _FakeSimulation()

    targets, loss_matrix = _add_education_credit_targets(
        pd.DataFrame(),
        [],
        sim,
        2024,
    )

    assert targets == [7_554_668_000.0, 7_211_349.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/education_tax_credits"],
        np.array([500.0, 0.0, 300.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/education_tax_credits_count"],
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )


def test_tanf_hardcoded_target_uses_fy2024_basic_assistance_total():
    assert HARD_CODED_TOTALS["tanf"] == pytest.approx(7_788_317_474.55)
