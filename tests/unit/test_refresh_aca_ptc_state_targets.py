import pandas as pd

from policyengine_us_data.storage.calibration_targets.refresh_aca_ptc_state_targets import (
    extract_state_aca_ptc,
)


def test_extract_state_aca_ptc_uses_total_ptc_columns(tmp_path):
    source = tmp_path / "22in55cmcsv.csv"
    pd.DataFrame(
        [
            {"STATE": "US", "AGI_STUB": 0, "N85770": 30, "A85770": 300},
            {"STATE": "AL", "AGI_STUB": 0, "N85770": 10, "A85770": 100},
            {"STATE": "AK", "AGI_STUB": 0, "N85770": 20, "A85770": 200},
            {"STATE": "AL", "AGI_STUB": 1, "N85770": 999, "A85770": 999},
        ]
    ).to_csv(source, index=False)

    result = extract_state_aca_ptc(2022, source_file=source)

    assert result.to_dict("records") == [
        {"GEO_ID": "0400000US01", "Returns": 10, "TotalPTCAmount": 100_000},
        {"GEO_ID": "0400000US02", "Returns": 20, "TotalPTCAmount": 200_000},
    ]
