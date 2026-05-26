import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps import cps as cps_module
from policyengine_us_data.datasets.cps.cps import (
    FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES,
    FLSA_OVERTIME_OCCUPATION_CODES,
    add_personal_income_variables,
    derive_flsa_overtime_premium,
    _flsa_overtime_policy_for_year,
    _flsa_overtime_thresholds_for_year,
)


def _minimal_person_income_frame() -> pd.DataFrame:
    columns = [
        "WSAL_VAL",
        "HRSWK",
        "A_HRS1",
        "WKSWORK",
        "INT_VAL",
        "SEMP_VAL",
        "FRSE_VAL",
        "DIV_VAL",
        "RNT_VAL",
        "RESNSS1",
        "RESNSS2",
        "SS_VAL",
        "A_AGE",
        "UC_VAL",
        "LKWEEKS",
        "PNSN_VAL",
        "ANN_VAL",
        "DST_SC1",
        "DST_VAL1",
        "DST_SC2",
        "DST_VAL2",
        "DST_SC1_YNG",
        "DST_VAL1_YNG",
        "DST_SC2_YNG",
        "DST_VAL2_YNG",
        "OI_OFF",
        "OI_VAL",
        "ED_VAL",
        "FIN_VAL",
        "SRVS_VAL",
        "CSP_VAL",
        "PAW_VAL",
        "SSI_VAL",
        "RETCB_VAL",
        "CAP_VAL",
        "WICYN",
        "VET_VAL",
        "WC_VAL",
        "DIS_VAL1",
        "DIS_SC1",
        "DIS_VAL2",
        "DIS_SC2",
        "CHSP_VAL",
        "PHIP_VAL",
        "POTC_VAL",
        "PMED_VAL",
        "MCARE",
        "PEMCPREM",
    ]
    person = pd.DataFrame({column: [0.0, 0.0] for column in columns})
    person["A_AGE"] = [30, 45]
    person["LKWEEKS"] = [0, 0]
    person["WKSWORK"] = [0, 0]
    return person


def test_add_personal_income_variables_maps_farm_self_employment_to_operations():
    person = _minimal_person_income_frame()
    person["FRSE_VAL"] = [1_000.0, -500.0]
    cps = {}

    add_personal_income_variables(cps, person, 2024)

    np.testing.assert_array_equal(cps["farm_operations_income"], [1_000.0, -500.0])
    assert "farm_income" not in cps


def test_add_personal_income_variables_maps_spm_income_leaves():
    person = pd.concat(
        [_minimal_person_income_frame(), _minimal_person_income_frame().iloc[[0]]],
        ignore_index=True,
    )
    person["OI_OFF"] = [0, 20, 12]
    person["OI_VAL"] = [50.0, 70.0, 90.0]
    person["ED_VAL"] = [10.0, 11.0, 12.0]
    person["FIN_VAL"] = [20.0, 21.0, 22.0]
    person["SRVS_VAL"] = [30.0, 31.0, 32.0]
    cps = {}

    add_personal_income_variables(cps, person, 2024)

    np.testing.assert_array_equal(cps["miscellaneous_income"], [50.0, 0.0, 0.0])
    np.testing.assert_array_equal(cps["alimony_income"], [0.0, 70.0, 0.0])
    np.testing.assert_array_equal(cps["strike_benefits"], [0.0, 0.0, 90.0])
    np.testing.assert_array_equal(cps["educational_assistance"], [10.0, 11.0, 12.0])
    np.testing.assert_array_equal(cps["financial_assistance"], [20.0, 21.0, 22.0])
    np.testing.assert_array_equal(cps["survivor_benefits"], [30.0, 31.0, 32.0])


def test_retirement_contributions_write_desired_without_se_rate_cap():
    person = _minimal_person_income_frame()
    person["SEMP_VAL"] = [100.0, 0.0]
    person["WSAL_VAL"] = [0.0, 100_000.0]
    person["RETCB_VAL"] = [100_000.0, 100_000.0]
    cps = {}

    add_personal_income_variables(cps, person, 2024)

    assert cps["self_employed_pension_contributions_desired"][0] > 100 * 0.25
    assert cps["self_employed_pension_contributions_desired"][1] == 0
    assert cps["traditional_ira_contributions_desired"][0] > 0
    assert cps["traditional_401k_contributions_desired"][1] > 0


def test_derive_flsa_overtime_premium_uses_wage_share_and_exemption_screen():
    premium = derive_flsa_overtime_premium(
        time_period=2024,
        employment_income=np.array(
            [57_200.0, 100_000.0, 30_000.0, 60_000.0, 50_000.0, 50_000.0]
        ),
        hours_worked_last_week=np.array([50.0, 50.0, 50.0, 40.0, 50.0, 50.0]),
        weeks_worked=np.array([52.0, 52.0, 52.0, 52.0, 52.0, 52.0]),
        is_paid_hourly=np.array([True, False, False, True, True, True]),
        has_never_worked=np.array([False, False, False, False, True, False]),
        is_military=np.array([False, False, False, False, False, True]),
        is_executive_administrative_professional=np.array(
            [False, True, True, False, False, False]
        ),
        is_farmer_fisher=np.array([False, False, False, False, False, False]),
        is_computer_scientist=np.array([False, False, False, False, False, False]),
    )

    np.testing.assert_allclose(
        premium,
        np.array(
            [5_200.0, 0.0, 30_000 * 5 / 55, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_derive_flsa_overtime_premium_uses_historical_salary_thresholds():
    inputs = dict(
        employment_income=np.array([30_000.0, 40_000.0]),
        hours_worked_last_week=np.array([50.0, 50.0]),
        weeks_worked=np.array([52.0, 52.0]),
        is_paid_hourly=np.array([False, False]),
        has_never_worked=np.array([False, False]),
        is_military=np.array([False, False]),
        is_executive_administrative_professional=np.array([True, True]),
        is_farmer_fisher=np.array([False, False]),
        is_computer_scientist=np.array([False, False]),
    )
    premium_2019 = derive_flsa_overtime_premium(
        time_period=2019,
        **inputs,
    )
    premium_2024 = derive_flsa_overtime_premium(
        time_period=2024,
        **inputs,
    )

    np.testing.assert_allclose(premium_2019, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(
        premium_2024,
        np.array([30_000 * 5 / 55, 0.0], dtype=np.float32),
    )


def test_derive_flsa_overtime_premium_uses_policy_hours_and_rate(monkeypatch):
    monkeypatch.setattr(
        cps_module,
        "_flsa_overtime_policy_for_year",
        lambda _year: (
            np.float32(100_000),
            np.float32(100_000),
            np.float32(100_000),
            np.float32(35),
            np.float32(2),
        ),
    )

    premium = derive_flsa_overtime_premium(
        time_period=2024,
        employment_income=np.array([60_000.0]),
        hours_worked_last_week=np.array([45.0]),
        weeks_worked=np.array([52.0]),
        is_paid_hourly=np.array([True]),
        has_never_worked=np.array([False]),
        is_military=np.array([False]),
        is_executive_administrative_professional=np.array([False]),
        is_farmer_fisher=np.array([False]),
        is_computer_scientist=np.array([False]),
    )

    np.testing.assert_allclose(
        premium,
        np.array([60_000 * 10 / 55], dtype=np.float32),
    )


def test_flsa_overtime_thresholds_match_policyengine_us_parameters():
    assert _flsa_overtime_thresholds_for_year(2019)[:2] == (
        np.float32(100_000),
        np.float32(455 * 52),
    )
    assert _flsa_overtime_thresholds_for_year(2024)[:2] == (
        np.float32(107_432),
        np.float32(684 * 52),
    )


def test_flsa_overtime_hours_and_rate_match_policyengine_us_parameters():
    from policyengine_us import CountryTaxBenefitSystem

    policy = _flsa_overtime_policy_for_year(2024)
    overtime = (
        CountryTaxBenefitSystem()
        .parameters("2024-01-01")
        .gov.irs.income.exemption.overtime
    )

    assert policy[3:] == (
        np.float32(overtime.hours_threshold),
        np.float32(overtime.rate_multiplier),
    )


def test_flsa_overtime_occupation_codes_match_policyengine_us_when_available():
    from policyengine_us.data import cps as policyengine_us_cps

    np.testing.assert_array_equal(
        FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES,
        policyengine_us_cps.CPS_FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES,
    )
    assert (
        FLSA_OVERTIME_OCCUPATION_CODES
        == policyengine_us_cps.CPS_FLSA_OVERTIME_OCCUPATION_CODES
    )
