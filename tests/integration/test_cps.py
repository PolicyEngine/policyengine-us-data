"""Integration tests for CPS dataset (requires cps_2024.h5)."""

import numpy as np
import pytest
import pandas as pd


@pytest.fixture(scope="module")
def cps_sim():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    return Microsimulation(dataset=CPS_2024)


def test_add_personal_variables_maps_current_health_coverage_flags():
    from policyengine_us_data.datasets.cps.cps import add_personal_variables

    person = pd.DataFrame(
        {
            "A_AGE": [30, 45, 28],
            "A_SEX": [2, 1, 2],
            "PEDISEYE": [0, 1, 0],
            "PEDISDRS": [0, 0, 0],
            "PEDISEAR": [0, 0, 0],
            "PEDISOUT": [0, 0, 0],
            "PEDISPHY": [0, 0, 0],
            "PEDISREM": [0, 0, 0],
            "PEPAR1": [0, 0, 0],
            "PEPAR2": [0, 0, 0],
            "PH_SEQ": [1, 1, 2],
            "A_LINENO": [1, 2, 1],
            "NOW_COV": [1, 1, 0],
            "NOW_DIR": [1, 0, 0],
            "NOW_MRK": [1, 0, 0],
            "NOW_MRKS": [1, 0, 0],
            "NOW_MRKUN": [0, 0, 0],
            "NOW_NONM": [0, 0, 0],
            "NOW_PRIV": [1, 0, 0],
            "NOW_PUB": [0, 1, 0],
            "NOW_GRP": [0, 1, 0],
            "NOW_CAID": [0, 0, 0],
            "NOW_MCAID": [0, 1, 0],
            "NOW_PCHIP": [0, 0, 0],
            "NOW_OTHMT": [0, 1, 0],
            "NOW_MCARE": [0, 0, 0],
            "NOW_MIL": [0, 0, 0],
            "NOW_CHAMPVA": [0, 0, 0],
            "NOW_VACARE": [0, 0, 0],
            "NOW_IHSFLG": [0, 0, 0],
            "PRDTRACE": [1, 2, 3],
            "PRDTHSP": [0, 1, 0],
            "A_MARITL": [1, 4, 1],
            "A_HSCOL": [0, 2, 0],
            "POCCU2": [39, 52, 29],
            "PEIOOCC": [4040, 9999, 4020],
        }
    )
    cps = {}

    add_personal_variables(cps, person)

    np.testing.assert_array_equal(
        cps["reported_has_marketplace_health_coverage_at_interview"],
        [True, False, False],
    )
    np.testing.assert_array_equal(
        cps["has_marketplace_health_coverage_at_interview"],
        [True, False, False],
    )
    np.testing.assert_array_equal(
        cps["has_other_means_tested_health_coverage_at_interview"],
        [False, True, False],
    )
    np.testing.assert_array_equal(
        cps["has_medicaid_health_coverage_at_interview"],
        [False, False, False],
    )
    np.testing.assert_array_equal(
        cps["reported_has_means_tested_health_coverage_at_interview"],
        [False, True, False],
    )
    np.testing.assert_array_equal(
        cps["reported_is_uninsured_at_interview"],
        [False, False, True],
    )
    np.testing.assert_array_equal(
        cps["reported_has_multiple_health_coverage_at_interview"],
        [False, True, False],
    )
    np.testing.assert_array_equal(
        cps["has_marketplace_health_coverage"], [True, False, False]
    )
    np.testing.assert_array_equal(cps["has_esi"], [False, True, False])


def test_add_personal_variables_uses_full_time_flag():
    from policyengine_us_data.datasets.cps.cps import add_personal_variables

    person = pd.DataFrame(
        {
            "A_AGE": [19, 20, 45],
            "A_SEX": [2, 1, 2],
            "PEDISEYE": [0, 0, 0],
            "PEDISDRS": [0, 0, 0],
            "PEDISEAR": [0, 0, 0],
            "PEDISOUT": [0, 0, 0],
            "PEDISPHY": [0, 0, 0],
            "PEDISREM": [0, 0, 0],
            "PEPAR1": [0, 0, 0],
            "PEPAR2": [0, 0, 0],
            "PH_SEQ": [1, 1, 1],
            "A_LINENO": [1, 2, 3],
            "NOW_COV": [0, 0, 0],
            "NOW_DIR": [0, 0, 0],
            "NOW_MRK": [0, 0, 0],
            "NOW_MRKS": [0, 0, 0],
            "NOW_MRKUN": [0, 0, 0],
            "NOW_NONM": [0, 0, 0],
            "NOW_PRIV": [0, 0, 0],
            "NOW_PUB": [0, 0, 0],
            "NOW_GRP": [0, 0, 0],
            "NOW_CAID": [0, 0, 0],
            "NOW_MCAID": [0, 0, 0],
            "NOW_PCHIP": [0, 0, 0],
            "NOW_OTHMT": [0, 0, 0],
            "NOW_MCARE": [0, 0, 0],
            "NOW_MIL": [0, 0, 0],
            "NOW_CHAMPVA": [0, 0, 0],
            "NOW_VACARE": [0, 0, 0],
            "NOW_IHSFLG": [0, 0, 0],
            "PRDTRACE": [1, 2, 3],
            "PRDTHSP": [0, 0, 0],
            "A_MARITL": [7, 7, 7],
            "A_HSCOL": [2, 2, 0],
            "A_FTPT": [1, 0, 0],
            "POCCU2": [39, 52, 29],
            "PEIOOCC": [4040, 9999, 4020],
        }
    )
    cps = {}

    add_personal_variables(cps, person)

    np.testing.assert_array_equal(
        cps["is_full_time_college_student"],
        [True, False, False],
    )
    assert "tax_unit_role_input" not in cps
    assert "is_related_to_head_or_spouse" not in cps


def test_add_id_variables_copies_constructed_tax_unit_ids_only():
    from policyengine_us_data.datasets.cps.cps import add_id_variables

    cps = {}
    person = pd.DataFrame(
        {
            "PH_SEQ": [1, 1],
            "PF_SEQ": [1, 1],
            "P_SEQ": [1, 2],
            "TAX_ID": [10, 10],
            "SPM_ID": [20, 20],
            "A_LINENO": [1, 2],
            "A_SPOUSE": [0, 0],
        }
    )
    tax_unit = pd.DataFrame({"TAX_ID": [10]})
    family = pd.DataFrame({"FH_SEQ": [1], "FFPOS": [1]})
    spm_unit = pd.DataFrame({"SPM_ID": [20]})
    household = pd.DataFrame({"H_SEQ": [1], "HSUP_WGT": [12_345]})

    add_id_variables(cps, person, tax_unit, family, spm_unit, household)

    assert cps["person_tax_unit_id"].tolist() == [10, 10]
    assert cps["tax_unit_id"].tolist() == [10]
    assert "filing_status_input" not in cps


def test_validate_raw_cps_schema_rejects_stale_raw_tables():
    from policyengine_us_data.datasets.cps.cps import _validate_raw_cps_schema

    person = pd.DataFrame({"PH_SEQ": [1], "TAX_ID": [1]})
    tax_unit = pd.DataFrame({"TAX_ID": [1]})

    with pytest.raises(ValueError) as error:
        _validate_raw_cps_schema(person, tax_unit, "census_cps_2024")

    message = str(error.value)
    assert "census_cps_2024" in message
    assert "CENSUS_TAX_ID" in message


def test_validate_raw_cps_schema_accepts_constructed_tax_unit_id_column():
    from policyengine_us_data.datasets.cps.cps import _validate_raw_cps_schema

    person = pd.DataFrame({"CENSUS_TAX_ID": [123]})
    tax_unit = pd.DataFrame({"TAX_ID": [1]})

    _validate_raw_cps_schema(person, tax_unit, "census_cps_2024")


# ── Sanity checks ─────────────────────────────────────────────


def test_cps_employment_income_positive(cps_sim):
    total = cps_sim.calculate("employment_income").sum()
    assert total > 5e12, f"CPS employment_income sum is {total:.2e}, expected > 5T."


def test_cps_household_count(cps_sim):
    total_hh = cps_sim.calculate("household_weight").values.sum()
    assert 100e6 < total_hh < 200e6, f"CPS total households = {total_hh:.2e}."


# ── Calibration checks ────────────────────────────────────────


def test_cps_has_auto_loan_interest():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2024)
    # Ensure we impute around $85 billion in overtime premium with 25% error bounds.
    AUTO_LOAN_INTEREST_TARGET = 85e9
    AUTO_LOAN_BALANCE_TARGET = 1550e9
    RELATIVE_TOLERANCE = 0.4

    assert (
        abs(sim.calculate("auto_loan_interest").sum() / AUTO_LOAN_INTEREST_TARGET - 1)
        < RELATIVE_TOLERANCE
    )
    assert (
        abs(sim.calculate("auto_loan_balance").sum() / AUTO_LOAN_BALANCE_TARGET - 1)
        < RELATIVE_TOLERANCE
    )


def test_cps_has_fsla_overtime_premium():
    from policyengine_us_data.datasets.cps import CPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2024)
    # ORG-backed hourly-pay data materially increases modeled overtime premium.
    # Keep a broad sanity band around the new CPS aggregate level.
    OVERTIME_PREMIUM_TARGET = 130e9
    RELATIVE_TOLERANCE = 0.2
    assert (
        abs(sim.calculate("fsla_overtime_premium").sum() / OVERTIME_PREMIUM_TARGET - 1)
        < RELATIVE_TOLERANCE
    )


def test_cps_has_net_worth():
    from policyengine_us_data.datasets.cps import CPS_2022
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2022)
    # Ensure we impute around 160 trillion in net worth with 25% error bounds.
    # https://fred.stlouisfed.org/series/BOGZ1FL192090005Q
    NET_WORTH_TARGET = 160e12
    RELATIVE_TOLERANCE = 0.25
    np.random.seed(42)
    assert (
        abs(sim.calculate("net_worth").sum() / NET_WORTH_TARGET - 1)
        < RELATIVE_TOLERANCE
    )
