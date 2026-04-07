from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from policyengine_us import CountryTaxBenefitSystem
from policyengine_us_data.calibration.calibration_utils import (
    calculate_spm_thresholds_vectorized,
    load_cd_geoadj_values,
)

SYSTEM = CountryTaxBenefitSystem()
CPI_U = SYSTEM.parameters.gov.bls.cpi.cpi_u


def test_load_cd_geoadj_values_returns_tenure_specific_lookup(monkeypatch):
    rent_df = pd.DataFrame(
        {
            "cd_id": ["0101"],
            "median_2br_rent": [1_500.0],
            "national_median_2br_rent": [1_000.0],
        }
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.calibration_utils.pd.read_csv",
        lambda *args, **kwargs: rent_df,
    )

    geoadj_lookup = load_cd_geoadj_values(["101"])

    assert geoadj_lookup["101"]["renter"] == pytest.approx(1.2215)
    assert geoadj_lookup["101"]["owner_with_mortgage"] == pytest.approx(1.217)
    assert geoadj_lookup["101"]["owner_without_mortgage"] == pytest.approx(
        1.1615
    )


def test_calculate_spm_thresholds_vectorized_matches_policyengine_us_future_path():
    thresholds = calculate_spm_thresholds_vectorized(
        person_ages=np.array([40, 35, 10, 8]),
        person_spm_unit_ids=np.array([0, 0, 0, 0]),
        spm_unit_tenure_types=np.array([b"RENTER"]),
        spm_unit_geoadj=np.array([1.1]),
        year=2027,
    )

    cpi_ratio = float(CPI_U("2027-02-01") / CPI_U("2024-02-01"))
    expected = 39_430.0 * cpi_ratio * 1.1

    assert thresholds[0] == pytest.approx(expected)
