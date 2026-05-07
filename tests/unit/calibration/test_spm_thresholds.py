import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.calibration.calibration_utils import (
    load_cd_geoadj_values,
)
from policyengine_us_data.utils.spm import geoadj_for_tenure


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
    assert geoadj_lookup["101"]["owner_without_mortgage"] == pytest.approx(1.1615)


def test_geoadj_for_tenure_accepts_policyengine_tenure_bytes():
    geoadj_lookup = {
        "renter": 1.1,
        "owner_with_mortgage": 1.2,
        "owner_without_mortgage": 1.3,
    }

    assert geoadj_for_tenure(geoadj_lookup, np.bytes_("RENTER")) == pytest.approx(1.1)
    assert geoadj_for_tenure(
        geoadj_lookup,
        np.bytes_("OWNER_WITH_MORTGAGE"),
    ) == pytest.approx(1.2)
