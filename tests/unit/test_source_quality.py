import numpy as np
import pandas as pd

from policyengine_us_data.utils.source_quality import (
    observed_source_mask,
    sipp_allocation_flag_for,
    target_observed_source_masks,
)


def test_sipp_allocation_flag_for_source_column():
    assert sipp_allocation_flag_for("TVAL_BANK") == "AVAL_BANK"
    assert sipp_allocation_flag_for("RSSI_YRYN") == "ASSI_YRYN"
    assert sipp_allocation_flag_for("TJB1_TXAMT") == "AJB1_TXAMT"


def test_observed_source_mask_excludes_nonzero_allocation_flags():
    df = pd.DataFrame(
        {
            "TVAL_BANK": [100.0, 200.0, 300.0],
            "AVAL_BANK": [0, 1, 2],
        }
    )

    result = observed_source_mask(
        df,
        source_columns=["TVAL_BANK"],
        allocation_flag_columns=["AVAL_BANK"],
    )

    np.testing.assert_array_equal(result.values, [True, False, False])


def test_observed_source_mask_is_target_specific():
    df = pd.DataFrame(
        {
            "tip_income": [10.0, 20.0],
            "bank_account_assets": [100.0, 200.0],
            "AJB1_TXAMT": [0, 0],
            "AVAL_BANK": [1, 0],
        }
    )

    tip_mask = observed_source_mask(
        df,
        source_columns=["tip_income"],
        allocation_flag_columns=["AJB1_TXAMT"],
    )
    bank_mask = observed_source_mask(
        df,
        source_columns=["bank_account_assets"],
        allocation_flag_columns=["AVAL_BANK"],
    )

    np.testing.assert_array_equal(tip_mask.values, [True, True])
    np.testing.assert_array_equal(bank_mask.values, [False, True])


def test_observed_source_mask_allows_missing_tip_components_when_requested():
    df = pd.DataFrame(
        {
            "TJB1_TXAMT": [np.nan, 5.0],
            "AJB1_TXAMT": [0, 0],
        }
    )

    result = observed_source_mask(
        df,
        source_columns=["TJB1_TXAMT"],
        allocation_flag_columns=["AJB1_TXAMT"],
        require_nonmissing_source=False,
    )

    np.testing.assert_array_equal(result.values, [True, True])


def test_target_observed_source_masks_are_target_specific():
    df = pd.DataFrame(
        {
            "tip_income": [10.0, 20.0],
            "bank_account_assets": [100.0, 200.0],
            "AJB1_TXAMT": [0, 0],
            "AVAL_BANK": [1, 0],
        }
    )

    result = target_observed_source_masks(
        df,
        targets=["tip_income", "bank_account_assets"],
        target_allocation_flag_columns={
            "tip_income": ["AJB1_TXAMT"],
            "bank_account_assets": ["AVAL_BANK"],
        },
    )

    np.testing.assert_array_equal(result["tip_income"].values, [True, True])
    np.testing.assert_array_equal(result["bank_account_assets"].values, [False, True])
