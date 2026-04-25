import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.cps import add_personal_income_variables


def _minimal_person_income_frame() -> pd.DataFrame:
    columns = [
        "WSAL_VAL",
        "HRSWK",
        "A_HRS1",
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
    return person


def test_add_personal_income_variables_maps_farm_self_employment_to_operations():
    person = _minimal_person_income_frame()
    person["FRSE_VAL"] = [1_000.0, -500.0]
    cps = {}

    add_personal_income_variables(cps, person, 2024)

    np.testing.assert_array_equal(cps["farm_operations_income"], [1_000.0, -500.0])
    assert "farm_income" not in cps
