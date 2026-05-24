import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.cps import add_personal_income_variables


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
