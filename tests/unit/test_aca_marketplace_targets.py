import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.db.etl_aca_marketplace import (
    build_state_marketplace_bronze_aptc_targets,
)


def test_build_state_marketplace_bronze_aptc_targets_uses_hcgov_aptc_counts():
    df = pd.DataFrame(
        [
            {
                "state_code": "AL",
                "platform": "HC.gov",
                "metal_level": "All",
                "enrollment_status": "01-atv",
                "consumers": 100.0,
                "aptc_consumers": 90.0,
            },
            {
                "state_code": "AL",
                "platform": "HC.gov",
                "metal_level": "All",
                "enrollment_status": "02-aut",
                "consumers": 50.0,
                "aptc_consumers": 40.0,
            },
            {
                "state_code": "AL",
                "platform": "HC.gov",
                "metal_level": "B",
                "enrollment_status": "All",
                "consumers": 80.0,
                "aptc_consumers": 60.0,
            },
            {
                "state_code": "CA",
                "platform": "SBM",
                "metal_level": "All",
                "enrollment_status": "01-atv",
                "consumers": 200.0,
                "aptc_consumers": np.nan,
            },
            {
                "state_code": "CA",
                "platform": "SBM",
                "metal_level": "B",
                "enrollment_status": "All",
                "consumers": 90.0,
                "aptc_consumers": np.nan,
            },
        ]
    )

    result = build_state_marketplace_bronze_aptc_targets(df)

    assert list(result["state_code"]) == ["AL"]
    row = result.iloc[0]
    assert row["marketplace_aptc_consumers"] == pytest.approx(130.0)
    assert row["marketplace_consumers"] == pytest.approx(150.0)
    assert row["bronze_aptc_consumers"] == pytest.approx(60.0)
    assert row["bronze_consumers"] == pytest.approx(80.0)
    assert row["bronze_aptc_share"] == pytest.approx(60.0 / 130.0)
