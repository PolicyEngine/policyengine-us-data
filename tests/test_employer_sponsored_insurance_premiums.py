from pathlib import Path

import numpy as np
import pandas as pd

from policyengine_us_data.datasets.cps.census_cps import (
    PERSON_COLUMNS,
    TAX_UNIT_COLUMNS,
    _resolve_person_usecols,
)
from policyengine_us_data.datasets.cps.cps import (
    _EMPLOYER_PAYS_ALL,
    _EMPLOYER_PAYS_SOME,
    _ESI_PLAN_PRIORS_2024,
    impute_employer_sponsored_insurance_premiums,
)
from policyengine_us_data.datasets.cps.extended_cps import (
    CPS_ONLY_IMPUTED_VARIABLES,
)
from policyengine_us_data.storage.calibration_targets.pull_hardcoded_targets import (
    HARD_CODED_TOTALS,
)


def test_resolve_person_usecols_requests_optional_esi_columns_when_available():
    available = PERSON_COLUMNS + TAX_UNIT_COLUMNS + [
        "NOW_OWNGRP",
        "NOW_HIPAID",
        "NOW_GRPFTYP",
    ]
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


def test_impute_employer_sponsored_insurance_premiums_defaults_missing_columns():
    person = pd.DataFrame({"PHIP_VAL": [1_200, 0]})

    result = impute_employer_sponsored_insurance_premiums(person)

    np.testing.assert_allclose(result, np.array([0.0, 0.0]))


def test_imputation_status_codes_remain_stable():
    assert _EMPLOYER_PAYS_ALL == 1
    assert _EMPLOYER_PAYS_SOME == 2


def test_extended_cps_imputes_esi_premiums_for_clone_half():
    assert "employer_sponsored_insurance_premiums" in CPS_ONLY_IMPUTED_VARIABLES


def test_hardcoded_targets_include_total_esi_premiums():
    assert HARD_CODED_TOTALS["employer_sponsored_insurance_premiums"] == 1_002.9e9


def test_target_config_includes_total_esi_premiums():
    target_config_path = Path(__file__).parent.parent / (
        "policyengine_us_data/calibration/target_config.yaml"
    )
    content = target_config_path.read_text()

    assert "employer_sponsored_insurance_premiums" in content
