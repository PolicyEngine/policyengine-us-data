import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.census_cps import (
    PERSON_COLUMNS,
    TAX_UNIT_COLUMNS,
    _resolve_person_usecols,
)
from policyengine_us_data.datasets.cps.cps import (
    ESI_POLICYHOLDER_VARIABLE,
    ESI_SOURCE_COLUMNS,
    _EMPLOYER_PAYS_ALL,
    _EMPLOYER_PAYS_SOME,
    _ESI_PLAN_PRIORS_2024,
    _validate_raw_cps_schema,
    impute_employer_sponsored_insurance_premiums,
)
from policyengine_us_data.datasets.cps.extended_cps import (
    CPS_ONLY_IMPUTED_VARIABLES,
)


def test_resolve_person_usecols_requests_optional_esi_columns_when_available():
    available = (
        PERSON_COLUMNS
        + TAX_UNIT_COLUMNS
        + [
            "NOW_OWNGRP",
            "NOW_HIPAID",
            "NOW_GRPFTYP",
        ]
    )
    usecols = _resolve_person_usecols(available, spm_unit_columns=[])

    for column in ["NOW_OWNGRP", "NOW_HIPAID", "NOW_GRPFTYP"]:
        assert column in usecols


def test_impute_employer_sponsored_insurance_premiums():
    person = pd.DataFrame(
        {
            "NOW_OWNGRP": [1, 1, 1, 0, 1],
            "NOW_HIPAID": [1, 2, 2, 1, 2],
            "NOW_GRPFTYP": [2, 2, 1, 2, 1],
            "PHIP_VAL": [0, 1_200, 0, 0, 50_000],
        }
    )

    result = impute_employer_sponsored_insurance_premiums(person)

    np.testing.assert_allclose(
        result[0],
        _ESI_PLAN_PRIORS_2024["self_only"]["total_premium"],
    )
    np.testing.assert_allclose(
        result[1],
        _ESI_PLAN_PRIORS_2024["self_only"]["total_premium"] - 1_200,
    )
    np.testing.assert_allclose(
        result[2],
        _ESI_PLAN_PRIORS_2024["family"]["total_premium"]
        - _ESI_PLAN_PRIORS_2024["family"]["employee_contribution"],
    )
    assert result[3] == 0
    assert result[4] == 0


def test_impute_employer_sponsored_insurance_premiums_tolerates_missing_esi_columns():
    person = pd.DataFrame({"PHIP_VAL": [1_000, 2_000]})

    result = impute_employer_sponsored_insurance_premiums(person)

    np.testing.assert_array_equal(result, np.zeros(2))


def test_imputation_status_codes_remain_stable():
    assert _EMPLOYER_PAYS_ALL == 1
    assert _EMPLOYER_PAYS_SOME == 2


def test_extended_cps_imputes_esi_premiums_for_clone_half():
    assert "employer_sponsored_insurance_premiums" in CPS_ONLY_IMPUTED_VARIABLES


def test_policyholder_variable_name_remains_stable():
    assert (
        ESI_POLICYHOLDER_VARIABLE
        == "reported_owns_employer_sponsored_health_insurance_at_interview"
    )


def test_raw_cps_schema_requires_esi_source_columns():
    person = pd.DataFrame(
        {
            "CENSUS_TAX_ID": [1],
            "PERRP": [43],
            **{column: [1] for column in ESI_SOURCE_COLUMNS},
        }
    )
    tax_unit = pd.DataFrame()

    _validate_raw_cps_schema(person, tax_unit, "raw")

    stale_person = person.drop(columns=["NOW_OWNGRP"])
    try:
        _validate_raw_cps_schema(stale_person, tax_unit, "raw")
    except ValueError as error:
        assert "NOW_OWNGRP" in str(error)
    else:
        raise AssertionError("Expected missing ESI source column to fail validation")
