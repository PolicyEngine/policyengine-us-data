import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.db.etl_aca_marketplace import (
    STATE_METAL_SELECTION_PATH,
    build_state_marketplace_bronze_aptc_targets,
    extract_aca_marketplace_state_metal_data,
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


def test_checked_in_csv_produces_hcgov_targets_with_consistent_bronze_counts():
    """Regression: the real checked-in CMS CSV must produce HC.gov targets
    with bronze APTC consumers ≤ total APTC consumers for every state, and
    must not silently include SBM states."""
    df = extract_aca_marketplace_state_metal_data(STATE_METAL_SELECTION_PATH)
    result = build_state_marketplace_bronze_aptc_targets(df)

    # Non-trivial number of HC.gov states (32 in 2024; 27+ leaves a cushion).
    assert len(result) >= 27, f"Expected 27+ HC.gov states, got {len(result)}"

    # Bronze is always a subset of total APTC.
    assert (
        result["bronze_aptc_consumers"] <= result["marketplace_aptc_consumers"]
    ).all()

    # State-based marketplaces must be excluded (CA, NY, WA, etc.).
    sbm_states = {"CA", "NY", "WA", "CO", "CT", "MA", "MD", "MN", "NJ", "RI", "VT"}
    assert sbm_states.isdisjoint(result["state_code"])

    # state_fips is always a valid positive integer.
    assert result["state_fips"].between(1, 56).all()


def test_build_targets_raises_when_bronze_exceeds_total():
    """ETL-level sanity check: bronze APTC > total APTC signals bad data."""
    df = pd.DataFrame(
        [
            {
                "state_code": "AL",
                "platform": "HC.gov",
                "metal_level": "All",
                "enrollment_status": "01-atv",
                "consumers": 100.0,
                "aptc_consumers": 40.0,
            },
            {
                "state_code": "AL",
                "platform": "HC.gov",
                "metal_level": "B",
                "enrollment_status": "All",
                "consumers": 80.0,
                "aptc_consumers": 60.0,  # exceeds total
            },
        ]
    )
    with pytest.raises(ValueError, match="Bronze APTC consumers exceed"):
        build_state_marketplace_bronze_aptc_targets(df)
