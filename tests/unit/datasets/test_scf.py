import pandas as pd

from policyengine_us_data.datasets.scf.scf import (
    SCF_AUTO_LOAN_COLUMNS,
    _clean_auto_loan_columns,
)


def test_clean_auto_loan_columns_clips_negative_missing_codes():
    auto_df = pd.DataFrame(
        {
            "yy1": [1],
            "y1": [2],
            "x2209": [-1.0],
            "x2309": [10_000.0],
            "x2409": [0.0],
            "x7158": [5_000.0],
            "x2219": [-1.0],
            "x2319": [500.0],
            "x2419": [0.0],
            "x7170": [750.0],
        }
    )

    cleaned = _clean_auto_loan_columns(auto_df)

    assert cleaned.loc[0, "yy1"] == 1
    assert cleaned.loc[0, "y1"] == 2
    assert (cleaned[SCF_AUTO_LOAN_COLUMNS].to_numpy() >= 0).all()
