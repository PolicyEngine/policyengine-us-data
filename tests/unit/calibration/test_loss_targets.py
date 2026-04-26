import inspect

import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.loss import (
    ABSOLUTE_ERROR_SCALE_TARGETS,
    AGE_BUCKETED_HEALTH_TARGETS,
    BLS_CE_TOTALS,
    TRANSFER_BALANCE_TARGETS,
    _get_aca_national_targets,
    _add_acs_housing_cost_targets,
    _add_bls_ce_targets,
    _add_ctc_targets,
    _add_real_estate_tax_targets,
    _add_transfer_balance_targets,
    get_target_error_normalisation,
    _get_medicaid_national_targets,
    _load_aca_spending_and_enrollment_targets,
    _load_medicaid_enrollment_targets,
    HARD_CODED_TOTALS,
    build_loss_matrix,
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
        self.values = np.asarray(values)


class _FakeSimulation:
    def __init__(self):
        self.calculate_calls = []
        self.map_result_calls = []

    def calculate(self, variable, map_to=None, period=None):
        self.calculate_calls.append((variable, map_to, period))
        values = {
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


class _FakeRealEstateTaxSimulation:
    def calculate(self, variable, map_to=None, period=None):
        values = {
            ("real_estate_taxes", None): [100.0, 0.0, 50.0, 0.0],
            ("tax_unit_is_filer", None): [1.0, 1.0],
            ("tax_unit_itemizes", None): [1.0, 0.0],
            ("state_code", "household"): ["CA", "NY"],
        }
        key = (variable, map_to)
        if key not in values:
            raise AssertionError(f"Unexpected calculate call {key!r}")
        return _FakeArrayResult(values[key])

    def map_result(self, values, source_entity, target_entity, how=None):
        arr = np.asarray(values, dtype=np.float32)
        if (source_entity, target_entity) == ("person", "tax_unit"):
            return np.array([arr[:2].sum(), arr[2:].sum()], dtype=np.float32)
        if (source_entity, target_entity) == ("tax_unit", "household"):
            return arr.astype(np.float32)
        raise AssertionError(
            f"Unexpected map_result call {(source_entity, target_entity, how)!r}"
        )


def test_add_real_estate_tax_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_national_geography_soi_target",
        lambda variable, year: {"amount": 123_000.0, "count": 17.0},
    )
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_state_geography_soi_targets",
        lambda variable, year: [
            {"state_code": "CA", "amount": 100_000.0, "count": 10.0},
            {"state_code": "NY", "amount": 50_000.0, "count": 5.0},
        ],
    )

    targets, loss_matrix = _add_real_estate_tax_targets(
        pd.DataFrame(),
        [],
        _FakeRealEstateTaxSimulation(),
        2024,
    )

    assert targets == [123_000.0, 17.0, 100_000.0, 10.0, 50_000.0, 5.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/real_estate_taxes"],
        np.array([100.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/real_estate_taxes_count"],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/irs/real_estate_taxes/CA"],
        np.array([100.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/irs/real_estate_taxes/NY"],
        np.array([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/irs/real_estate_taxes_count/CA"],
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/irs/real_estate_taxes_count/NY"],
        np.array([0.0, 0.0], dtype=np.float32),
    )


class _FakeAcsHousingCostSimulation:
    def calculate(self, variable, map_to=None, period=None):
        values = {
            ("state_code", "household"): ["CA", "NY", "CA"],
            ("rent", "household"): [10.0, 20.0, 30.0],
            ("real_estate_taxes", "household"): [1.0, 2.0, 3.0],
            ("childcare_expenses", "household"): [4.0, 0.0, 6.0],
        }
        key = (variable, map_to)
        if key not in values:
            raise AssertionError(f"Unexpected calculate call {key!r}")
        return _FakeArrayResult(values[key])


def test_add_acs_housing_cost_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss._load_yeared_target_csv",
        lambda prefix, year: (
            pd.DataFrame(
                {
                    "state_code": ["CA", "NY"],
                    "annual_contract_rent": [100.0, 200.0],
                    "real_estate_taxes": [30.0, 40.0],
                }
            ),
            2024,
        ),
    )

    targets, loss_matrix = _add_acs_housing_cost_targets(
        pd.DataFrame(),
        [],
        _FakeAcsHousingCostSimulation(),
        2024,
    )

    assert targets == [300.0, 100.0, 200.0, 70.0, 30.0, 40.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/census/acs/rent"],
        np.array([10.0, 20.0, 30.0]),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/census/acs/rent/CA"],
        np.array([10.0, 0.0, 30.0]),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/census/acs/real_estate_taxes/NY"],
        np.array([0.0, 2.0, 0.0]),
    )


def test_bls_ce_childcare_target():
    assert BLS_CE_TOTALS["childcare_expenses"] == pytest.approx(63_092e6)

    targets, loss_matrix = _add_bls_ce_targets(
        pd.DataFrame(),
        [],
        _FakeAcsHousingCostSimulation(),
        2024,
    )

    assert targets == [63_092e6]
    np.testing.assert_array_equal(
        loss_matrix["nation/bls/ce/childcare_expenses"],
        np.array([4.0, 0.0, 6.0]),
    )


class _FakeTransferBalanceSimulation:
    def calculate(self, variable, map_to=None, period=None):
        values = {
            "alimony_expense": [100.0, 0.0, 20.0],
            "alimony_income": [30.0, 40.0, 0.0],
            "child_support_expense": [0.0, 50.0, 10.0],
            "child_support_received": [20.0, 10.0, 40.0],
        }
        if variable not in values:
            raise AssertionError(f"Unexpected variable {variable!r}")
        assert map_to == "household"
        assert period == 2024
        return _FakeArrayResult(values[variable])


def test_transfer_balance_targets_are_net_zero_accounting_constraints():
    targets, loss_matrix = _add_transfer_balance_targets(
        pd.DataFrame(),
        [],
        _FakeTransferBalanceSimulation(),
        2024,
    )

    assert targets == [0.0, 0.0]
    assert set(TRANSFER_BALANCE_TARGETS) == {
        "nation/accounting/alimony_paid_minus_received",
        "nation/accounting/child_support_paid_minus_received",
    }
    np.testing.assert_array_equal(
        loss_matrix["nation/accounting/alimony_paid_minus_received"],
        np.array([70.0, -40.0, 20.0]),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/accounting/child_support_paid_minus_received"],
        np.array([-20.0, 40.0, -30.0]),
    )


def test_transfer_balance_targets_use_absolute_error_scale():
    target_names = np.array(
        [
            "nation/accounting/alimony_paid_minus_received",
            "nation/census/snap",
        ]
    )
    numerator_shift, denominator = get_target_error_normalisation(
        target_names,
        np.array([0.0, 10.0]),
    )

    assert ABSOLUTE_ERROR_SCALE_TARGETS[
        "nation/accounting/alimony_paid_minus_received"
    ] == pytest.approx(1e9)
    np.testing.assert_array_equal(numerator_shift, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(denominator, np.array([1e9, 11.0]))


def test_tanf_hardcoded_target_uses_fy2024_basic_assistance_total():
    assert HARD_CODED_TOTALS["tanf"] == pytest.approx(7_788_317_474.55)


def test_hardcoded_totals_drop_survey_spm_targets():
    removed_targets = {
        "alimony_income",
        "alimony_expense",
        "child_support_expense",
        "child_support_received",
        "health_insurance_premiums_without_medicare_part_b",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "spm_unit_spm_threshold",
        "spm_unit_capped_housing_subsidy",
        "spm_unit_capped_work_childcare_expenses",
    }

    assert removed_targets.isdisjoint(HARD_CODED_TOTALS)


def test_age_bucketed_health_targets_keep_only_medicare_part_b():
    assert AGE_BUCKETED_HEALTH_TARGETS == ("medicare_part_b_premiums",)


def test_national_loss_excludes_survey_spm_threshold_decile_targets():
    source = inspect.getsource(build_loss_matrix)

    assert "spm_threshold_agi.csv" not in source
    assert "agi_in_spm_threshold_decile" not in source
    assert "count_in_spm_threshold_decile" not in source
