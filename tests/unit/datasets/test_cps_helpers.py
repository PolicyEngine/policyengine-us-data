import numpy as np
import pandas as pd
import pytest


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

    person = pd.DataFrame(
        {
            "CENSUS_TAX_ID": [123],
            "PERRP": [43],
            "NOW_GRPFTYP": [1],
            "NOW_HIPAID": [1],
            "NOW_OWNGRP": [1],
        }
    )
    tax_unit = pd.DataFrame({"TAX_ID": [1]})

    _validate_raw_cps_schema(person, tax_unit, "census_cps_2024")


def test_validate_raw_cps_schema_requires_reference_partner_column():
    from policyengine_us_data.datasets.cps.cps import _validate_raw_cps_schema

    person = pd.DataFrame(
        {
            "CENSUS_TAX_ID": [123],
            "NOW_GRPFTYP": [1],
            "NOW_HIPAID": [1],
            "NOW_OWNGRP": [1],
        }
    )
    tax_unit = pd.DataFrame({"TAX_ID": [1]})

    with pytest.raises(ValueError) as error:
        _validate_raw_cps_schema(person, tax_unit, "census_cps_2024")

    assert "PERRP" in str(error.value)
