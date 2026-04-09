import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.loss import (
    _get_aca_national_targets,
    _add_refundable_ctc_targets,
    _get_medicaid_national_targets,
    _load_aca_spending_and_enrollment_targets,
    _load_medicaid_enrollment_targets,
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
        if variable != "refundable_ctc":
            raise AssertionError(f"Unexpected variable {variable!r}")
        if map_to == "household":
            return _FakeArrayResult([100.0, 0.0, 50.0])
        if map_to is None:
            return _FakeArrayResult([100.0, 0.0, 50.0])
        raise AssertionError(f"Unexpected map_to {map_to!r}")

    def map_result(self, values, source_entity, target_entity, how=None):
        self.map_result_calls.append((source_entity, target_entity, how))
        assert source_entity == "tax_unit"
        assert target_entity == "household"
        return np.asarray(values, dtype=np.float32)


def test_add_refundable_ctc_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_national_geography_soi_target",
        lambda variable, year: {"amount": 33_000.0, "count": 17.0},
    )
    sim = _FakeSimulation()

    targets, loss_matrix = _add_refundable_ctc_targets(
        pd.DataFrame(),
        [],
        sim,
        2024,
    )

    assert targets == [33_000.0, 17.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/refundable_ctc"],
        np.array([100.0, 0.0, 50.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/refundable_ctc_count"],
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )
