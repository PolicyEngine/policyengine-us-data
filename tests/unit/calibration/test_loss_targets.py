import pytest

from policyengine_us_data.utils.loss import (
    _get_aca_national_targets,
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
