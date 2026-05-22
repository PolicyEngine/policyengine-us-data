import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.loss import (
    ABSOLUTE_ERROR_SCALE_TARGETS,
    AGE_BUCKETED_HEALTH_TARGETS,
    AGGREGATE_LEVEL_TARGETED_VARIABLES,
    AGI_LEVEL_TARGETED_VARIABLES,
    BEA_NIPA_DIRECT_SUM_TARGETS,
    BEA_NIPA_DIRECT_SUM_LOSS_WEIGHT,
    BEA_WAGES_AND_SALARIES_LOSS_WEIGHT,
    BLS_CE_TOTALS,
    HARD_CODED_TOTALS,
    TRANSFER_BALANCE_TARGETS,
    _add_bea_state_wage_targets,
    _add_agi_metric_columns,
    _add_acs_housing_cost_targets,
    _add_aotc_targets,
    _add_bls_ce_targets,
    _add_ctc_targets,
    _add_education_credit_targets,
    _add_irs_soi_aggregate_targets,
    _add_medicare_enrollment_target,
    _add_real_estate_tax_targets,
    _add_ssi_recipient_targets,
    _add_transfer_balance_targets,
    _cbo_program_target_value,
    _get_medicaid_national_targets,
    _get_aca_national_targets,
    _load_aca_spending_and_enrollment_targets,
    _load_medicaid_enrollment_targets,
    _should_skip_soi_agi_row,
    _should_skip_soi_taxability_row,
    build_loss_matrix,
    get_target_error_normalisation,
    get_target_loss_weights,
)
from policyengine_us_data.db import etl_national_targets
from policyengine_us_data.utils.ssi_targets import (
    SSI_RECIPIENT_TARGETS_2024,
    get_ssi_fiscal_year_payment_count,
    get_ssi_single_year_available_payment_count,
    scale_ssi_fiscal_year_target_for_single_year_data,
)


def test_legacy_loss_targets_include_aggregate_qbi_deduction():
    assert "qualified_business_income_deduction" in AGGREGATE_LEVEL_TARGETED_VARIABLES
    assert "qualified_business_income_deduction" not in AGI_LEVEL_TARGETED_VARIABLES


def test_bea_nipa_direct_sum_targets_match_targets_db():
    loss_targets_by_variable = {
        variable: target for _, variable, target in BEA_NIPA_DIRECT_SUM_TARGETS
    }

    assert loss_targets_by_variable == {
        "employment_income_before_lsr": (
            etl_national_targets.BEA_NIPA_WAGES_AND_SALARIES_2024
        ),
        etl_national_targets.NIPA_PROPRIETORS_INCOME_VARIABLE: (
            etl_national_targets.BEA_NIPA_PROPRIETORS_INCOME_2024
        ),
    }


def test_bea_nipa_direct_sum_targets_get_higher_loss_weight():
    target_names = np.array(
        [
            "nation/bea/nipa_wages_and_salaries",
            "state/bea/wages_and_salaries/CA",
            "nation/bea/nipa_proprietors_income",
            "nation/bea/nipa_personal_interest_income",
            "nation/bea/nipa_personal_dividend_income",
            "state/CA/adjusted_gross_income/amount/1000000_inf",
        ]
    )

    weights = get_target_loss_weights(target_names)

    assert weights.tolist() == [
        BEA_WAGES_AND_SALARIES_LOSS_WEIGHT,
        BEA_WAGES_AND_SALARIES_LOSS_WEIGHT,
        BEA_NIPA_DIRECT_SUM_LOSS_WEIGHT,
        1.0,
        1.0,
        1.0,
    ]


def test_aca_targets_roll_forward_to_2025():
    targets, data_year = _load_aca_spending_and_enrollment_targets(2025)

    assert data_year == 2025
    assert len(targets) == 51
    assert int(targets["enrollment"].sum()) == 21_822_894


def test_aca_targets_use_latest_available_year():
    _, data_year = _load_aca_spending_and_enrollment_targets(2026)
    assert data_year == 2026


def test_aca_targets_fall_back_to_earliest_available_year():
    _, data_year = _load_aca_spending_and_enrollment_targets(2023)
    assert data_year == 2024


def test_aca_national_targets_use_uprated_soi_total_ptc_amount():
    spending, enrollment, data_year = _get_aca_national_targets(2025)

    assert data_year == 2025
    assert enrollment == 21_822_894
    assert spending == pytest.approx(101_191_587_487.48738)


def test_aca_national_targets_reuse_latest_uprated_soi_total_ptc_amount():
    spending, enrollment, data_year = _get_aca_national_targets(2026)

    assert data_year == 2026
    assert enrollment == 20_035_756
    assert spending == pytest.approx(101_191_587_487.48738)


def test_medicaid_targets_roll_forward_to_2025():
    targets, data_year = _load_medicaid_enrollment_targets(2025)

    assert data_year == 2025
    assert len(targets) == 51
    assert int(targets["enrollment"].sum()) == 68_925_023


def test_medicaid_targets_roll_forward_to_2026():
    targets, data_year = _load_medicaid_enrollment_targets(2026)

    assert data_year == 2026
    assert len(targets) == 51
    assert int(targets["enrollment"].sum()) == 68_022_529


def test_medicaid_targets_fall_back_to_earliest_available_year():
    _, data_year = _load_medicaid_enrollment_targets(2023)
    assert data_year == 2024


def test_medicaid_national_targets_use_2025_values():
    spending, enrollment, data_year = _get_medicaid_national_targets(2025)

    assert data_year == 2025
    assert enrollment == 68_925_023
    assert spending == pytest.approx(1_000_645_800_000.0001)


def test_medicaid_national_targets_use_2026_enrollment():
    spending, enrollment, data_year = _get_medicaid_national_targets(2026)

    assert data_year == 2026
    assert enrollment == 68_022_529
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


class _FakeMedicareEnrollmentSimulation:
    def __init__(self):
        self.calculate_calls = []
        self.map_result_calls = []

    def calculate(self, variable, map_to=None, period=None):
        self.calculate_calls.append((variable, map_to, period))
        if variable != "medicare_enrolled":
            raise AssertionError(f"Unexpected variable {variable!r}")
        if map_to != "person":
            raise AssertionError(f"Unexpected map_to {map_to!r}")
        return _FakeArrayResult([1.0, 0.0, 1.0])

    def map_result(self, values, source_entity, target_entity, how=None):
        self.map_result_calls.append((source_entity, target_entity, how))
        assert source_entity == "person"
        assert target_entity == "household"
        return np.asarray(values, dtype=np.float32)


class _FakeSSIRecipientSimulation:
    def calculate(self, variable, map_to=None, period=None):
        values = {
            "ssi": [100.0, 50.0, 0.0, 75.0],
            "age": [10.0, 40.0, 80.0, 70.0],
        }
        if variable not in values:
            raise AssertionError(f"Unexpected variable {variable!r}")
        assert map_to == "person"
        assert period == 2024
        return _FakeArrayResult(values[variable])

    def map_result(self, values, source_entity, target_entity, how=None):
        assert source_entity == "person"
        assert target_entity == "household"
        return np.asarray(values, dtype=np.float32)


class _FakeCBOProgramTargetSimulation:
    def __init__(self):
        self.tax_benefit_system = SimpleNamespace(
            parameters=lambda period: SimpleNamespace(
                calibration=SimpleNamespace(
                    gov=SimpleNamespace(
                        cbo=SimpleNamespace(
                            _children={
                                "income_tax": 2_000.0,
                                "snap": 1_000.0,
                                "social_security": 3_000.0,
                                "ssi": 57_000_000_000.0,
                                "unemployment_compensation": 4_000.0,
                            }
                        )
                    )
                )
            )
        )


class _FakeCapitalGainsSimulation:
    def __init__(self):
        self.calculate_calls = []
        self.tax_benefit_system = SimpleNamespace(
            parameters=lambda period: SimpleNamespace(
                calibration=SimpleNamespace(
                    gov=SimpleNamespace(
                        irs=SimpleNamespace(
                            soi=SimpleNamespace(
                                _children={
                                    "long_term_capital_gains": 1_650.0,
                                }
                            )
                        )
                    )
                )
            )
        )

    def calculate(self, variable, map_to=None, period=None):
        self.calculate_calls.append((variable, map_to, period))
        values = {
            "long_term_capital_gains": [100.0, 0.0, 50.0],
        }
        if variable not in values:
            raise AssertionError(f"Unexpected variable {variable!r}")
        assert map_to == "household"
        return _FakeArrayResult(values[variable])


class _FakeStateAgiSimulation:
    def calculate(self, variable, map_to=None, period=None):
        values = {
            "adjusted_gross_income": [-100.0, -50.0, 5_000.0, 7_000.0],
            "tax_unit_is_filer": [1.0, 0.0, 1.0, 1.0],
            "state_code": ["CA", "CA", "CA", "NY"],
        }
        if variable not in values:
            raise AssertionError(f"Unexpected variable {variable!r}")
        if variable == "state_code":
            assert map_to == "person"
            return SimpleNamespace(values=np.asarray(values[variable], dtype=object))
        else:
            assert map_to is None
        return _FakeArrayResult(values[variable])

    def map_result(self, values, source_entity, target_entity, how=None):
        if source_entity == "person":
            assert target_entity == "tax_unit"
            assert how == "value_from_first_person"
            return np.asarray(values)
        assert source_entity == "tax_unit"
        assert target_entity == "household"
        return np.asarray(values)


def test_state_agi_targets_are_limited_to_filers(tmp_path, monkeypatch):
    calibration_folder = tmp_path
    (calibration_folder / "agi_state.csv").write_text(
        "\n".join(
            [
                "GEO_ID,GEO_NAME,AGI_LOWER_BOUND,AGI_UPPER_BOUND,VALUE,IS_COUNT,VARIABLE",
                "0400000US06,CA,-inf,1.0,1,1,adjusted_gross_income/count",
                "0400000US06,CA,-inf,1.0,-100,0,adjusted_gross_income/amount",
                "0400000US06,CA,1.0,10000.0,1,1,adjusted_gross_income/count",
                "0400000US06,CA,1.0,10000.0,5000,0,adjusted_gross_income/amount",
            ]
        )
    )

    from policyengine_us_data.utils import loss as loss_module

    monkeypatch.setattr(loss_module, "CALIBRATION_FOLDER", calibration_folder)

    loss_matrix = _add_agi_metric_columns(
        pd.DataFrame(),
        _FakeStateAgiSimulation(),
    )

    np.testing.assert_array_equal(
        loss_matrix["state/CA/adjusted_gross_income/count/-inf_1"],
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/CA/adjusted_gross_income/amount/-inf_1"],
        np.array([-100.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/CA/adjusted_gross_income/count/1_10000"],
        np.array([0.0, 0.0, 1.0, 0.0]),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/CA/adjusted_gross_income/amount/1_10000"],
        np.array([0.0, 0.0, 5_000.0, 0.0]),
    )


def test_add_ssi_recipient_targets_adds_total_and_age_counts():
    targets, loss_matrix = _add_ssi_recipient_targets(
        pd.DataFrame(),
        [],
        _FakeSSIRecipientSimulation(),
        2024,
    )

    assert targets == [
        spec["person_count"] for spec in SSI_RECIPIENT_TARGETS_2024.values()
    ]
    np.testing.assert_array_equal(
        loss_matrix["nation/ssa/ssi_recipients/all"],
        np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/ssa/ssi_recipients/under_18"],
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/ssa/ssi_recipients/18_64"],
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["nation/ssa/ssi_recipients/65_plus"],
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_ssi_payment_targets_scale_to_single_year_fiscal_year_coverage():
    assert get_ssi_fiscal_year_payment_count(2024) == 11
    assert get_ssi_single_year_available_payment_count(2024) == 9
    assert get_ssi_fiscal_year_payment_count(2025) == 12
    assert get_ssi_single_year_available_payment_count(2025) == 9
    assert get_ssi_fiscal_year_payment_count(2028) == 13
    assert get_ssi_single_year_available_payment_count(2028) == 10

    assert scale_ssi_fiscal_year_target_for_single_year_data(
        57_000_000_000, 2024
    ) == pytest.approx(57_000_000_000 * 9 / 11)
    assert scale_ssi_fiscal_year_target_for_single_year_data(
        75_400_000_000, 2028
    ) == pytest.approx(75_400_000_000 * 10 / 13)


def test_legacy_cbo_ssi_target_uses_single_year_fiscal_year_coverage():
    sim = _FakeCBOProgramTargetSimulation()

    assert _cbo_program_target_value(
        sim, "ssi_federal_fiscal_year_outlays", 2024
    ) == pytest.approx(57_000_000_000 * 9 / 11)
    assert _cbo_program_target_value(sim, "snap", 2024) == 1_000.0


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


class _FakeBeaStateWageSimulation:
    tax_benefit_system = SimpleNamespace(
        variables={
            "employment_income_before_lsr": SimpleNamespace(
                entity=SimpleNamespace(key="person")
            )
        }
    )

    def calculate(self, variable, map_to=None, period=None):
        values = {
            ("state_code", "household"): ["CA", "NY", "CA"],
            ("employment_income_before_lsr", "household"): [10.0, 20.0, 30.0],
        }
        key = (variable, map_to)
        if key not in values:
            raise AssertionError(f"Unexpected calculate call {key!r}")
        assert period == 2024
        return _FakeArrayResult(values[key])


def test_add_bea_state_wage_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_bea_state_wage_targets",
        lambda year, *, national_total: (
            pd.DataFrame(
                {
                    "state_code": ["CA", "NY"],
                    "employment_income_before_lsr": [100.0, 200.0],
                }
            ),
            2024,
        ),
    )

    targets, loss_matrix = _add_bea_state_wage_targets(
        pd.DataFrame(),
        [],
        _FakeBeaStateWageSimulation(),
        2024,
    )

    assert targets == [100.0, 200.0]
    np.testing.assert_array_equal(
        loss_matrix["state/bea/wages_and_salaries/CA"],
        np.array([10.0, 0.0, 30.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loss_matrix["state/bea/wages_and_salaries/NY"],
        np.array([0.0, 20.0, 0.0], dtype=np.float32),
    )


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


def test_add_irs_soi_capital_gains_targets():
    sim = _FakeCapitalGainsSimulation()

    targets, loss_matrix = _add_irs_soi_aggregate_targets(
        pd.DataFrame(),
        [],
        sim,
        2026,
    )

    assert targets == [1_650.0]
    np.testing.assert_array_equal(
        loss_matrix["nation/irs/soi/long_term_capital_gains"],
        np.array([100.0, 0.0, 50.0], dtype=np.float32),
    )
    assert sim.calculate_calls == [
        ("long_term_capital_gains", "household", None),
    ]


def test_low_agi_soi_skip_keeps_investment_income_targets():
    ordinary_low_agi_row = pd.Series(
        {"Variable": "employment_income", "AGI upper bound": 10_000.0}
    )
    capital_income_low_agi_row = pd.Series(
        {"Variable": "capital_gains_gross", "AGI upper bound": 10_000.0}
    )
    ordinary_higher_agi_row = pd.Series(
        {"Variable": "employment_income", "AGI upper bound": 25_000.0}
    )

    assert _should_skip_soi_agi_row(ordinary_low_agi_row)
    assert not _should_skip_soi_agi_row(capital_income_low_agi_row)
    assert not _should_skip_soi_agi_row(ordinary_higher_agi_row)


def test_all_return_soi_skip_keeps_investment_income_targets():
    ordinary_all_return_row = pd.Series(
        {"Variable": "employment_income", "Taxable only": False}
    )
    capital_income_all_return_row = pd.Series(
        {"Variable": "capital_gains_gross", "Taxable only": False}
    )
    ordinary_taxable_row = pd.Series(
        {"Variable": "employment_income", "Taxable only": True}
    )
    qbi_taxable_row = pd.Series(
        {
            "Variable": "qualified_business_income_deduction",
            "Taxable only": True,
        }
    )
    capital_income_taxable_row = pd.Series(
        {"Variable": "capital_gains_gross", "Taxable only": True}
    )

    assert _should_skip_soi_taxability_row(ordinary_all_return_row)
    assert not _should_skip_soi_taxability_row(capital_income_all_return_row)
    assert not _should_skip_soi_taxability_row(ordinary_taxable_row)
    assert not _should_skip_soi_taxability_row(qbi_taxable_row)
    assert _should_skip_soi_taxability_row(capital_income_taxable_row)


def test_tanf_hardcoded_target_uses_fy2024_basic_assistance_total():
    assert HARD_CODED_TOTALS["tanf"] == pytest.approx(7_788_317_474.55)


def test_hardcoded_totals_drop_survey_spm_targets():
    removed_targets = {
        "alimony_income",
        "alimony_expense",
        "child_support_expense",
        "child_support_received",
        "employer_sponsored_insurance_premiums",
        "health_insurance_premiums_without_medicare_part_b",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "spm_unit_spm_threshold",
        "spm_unit_capped_housing_subsidy",
        "spm_unit_capped_work_childcare_expenses",
    }

    assert removed_targets.isdisjoint(HARD_CODED_TOTALS)


def test_age_bucketed_health_targets_keep_only_medicare_part_b():
    assert AGE_BUCKETED_HEALTH_TARGETS == (
        ("medicare_part_b_premium", "medicare_part_b_premiums"),
    )


def test_national_loss_excludes_survey_spm_threshold_decile_targets():
    source = inspect.getsource(build_loss_matrix)

    assert "spm_threshold_agi.csv" not in source
    assert "agi_in_spm_threshold_decile" not in source
    assert "count_in_spm_threshold_decile" not in source


def test_add_medicare_enrollment_target(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.utils.loss.get_medicare_enrollment_target",
        lambda year: 68_030_000.0,
    )
    sim = _FakeMedicareEnrollmentSimulation()

    targets, loss_matrix = _add_medicare_enrollment_target(
        pd.DataFrame(),
        [],
        sim,
        2024,
    )

    assert targets == [68_030_000.0]
    assert sim.calculate_calls == [("medicare_enrolled", "person", 2024)]
    np.testing.assert_array_equal(
        loss_matrix["nation/cms/medicare_enrollment"],
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )
