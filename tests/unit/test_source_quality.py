import numpy as np
import pandas as pd

from policyengine_us_data.utils.source_quality import (
    cap_training_sample,
    filter_positive_finite_weight_rows,
    is_sipp_status_flag_column,
    observed_source_mask,
    require_columns_present,
    sipp_allocation_flag_for,
    target_observed_source_masks,
)


def test_sipp_allocation_flag_for_source_column():
    assert sipp_allocation_flag_for("TVAL_BANK") == "AVAL_BANK"
    assert sipp_allocation_flag_for("RSSI_YRYN") == "ASSI_YRYN"
    assert sipp_allocation_flag_for("TJB1_TXAMT") == "AJB1_TXAMT"


def test_require_columns_present_accepts_available_columns():
    require_columns_present(
        {"rent_is_allocated", "real_estate_taxes_is_allocated"},
        ["rent_is_allocated", "real_estate_taxes_is_allocated"],
        source_name="ACS",
    )


def test_require_columns_present_raises_for_missing_columns():
    try:
        require_columns_present(
            {"rent_is_allocated"},
            ["rent_is_allocated", "real_estate_taxes_is_allocated"],
            source_name="ACS",
        )
    except KeyError as error:
        message = str(error)
    else:
        raise AssertionError("Expected missing source-quality columns to fail")

    assert "real_estate_taxes_is_allocated" in message
    assert "Regenerate the donor artifact" in message


def test_observed_source_mask_excludes_nonzero_binary_allocation_flags():
    df = pd.DataFrame(
        {
            "TVAL_BANK": [100.0, 200.0, 300.0],
            "asset_is_allocated": [0, 1, 2],
        }
    )

    result = observed_source_mask(
        df,
        source_columns=["TVAL_BANK"],
        allocation_flag_columns=["asset_is_allocated"],
    )

    np.testing.assert_array_equal(result.values, [True, False, False])


def test_observed_source_mask_uses_sipp_status_flag_semantics():
    df = pd.DataFrame(
        {
            "TJB1_TXAMT": [np.nan, 10.0, 20.0, 30.0, 40.0],
            "AJB1_TXAMT": [0, 1, 2, 9, 6],
        }
    )

    result = observed_source_mask(
        df,
        source_columns=["TJB1_TXAMT"],
        allocation_flag_columns=["AJB1_TXAMT"],
        require_nonmissing_source=False,
    )

    assert is_sipp_status_flag_column("AJB1_TXAMT")
    assert is_sipp_status_flag_column("ASSI_YRYN")
    assert not is_sipp_status_flag_column("ACS_ALLOCATED")
    np.testing.assert_array_equal(result.values, [True, True, False, True, False])


def test_observed_source_mask_is_target_specific():
    df = pd.DataFrame(
        {
            "tip_income": [10.0, 20.0],
            "bank_account_assets": [100.0, 200.0],
            "tip_is_allocated": [0, 0],
            "asset_is_allocated": [1, 0],
        }
    )

    tip_mask = observed_source_mask(
        df,
        source_columns=["tip_income"],
        allocation_flag_columns=["tip_is_allocated"],
    )
    bank_mask = observed_source_mask(
        df,
        source_columns=["bank_account_assets"],
        allocation_flag_columns=["asset_is_allocated"],
    )

    np.testing.assert_array_equal(tip_mask.values, [True, True])
    np.testing.assert_array_equal(bank_mask.values, [False, True])


def test_observed_source_mask_allows_missing_tip_components_when_requested():
    df = pd.DataFrame(
        {
            "TJB1_TXAMT": [np.nan, 5.0],
            "AJB1_TXAMT": [0, 1],
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
            "tip_is_allocated": [0, 0],
            "asset_is_allocated": [1, 0],
        }
    )

    result = target_observed_source_masks(
        df,
        targets=["tip_income", "bank_account_assets"],
        target_allocation_flag_columns={
            "tip_income": ["tip_is_allocated"],
            "bank_account_assets": ["asset_is_allocated"],
        },
    )

    np.testing.assert_array_equal(result["tip_income"].values, [True, True])
    np.testing.assert_array_equal(result["bank_account_assets"].values, [False, True])


def test_cap_training_sample_keeps_target_filters_aligned():
    df = pd.DataFrame({"value": np.arange(20)}, index=np.arange(100, 120))
    filters = {
        "value": pd.Series(
            [i % 2 == 0 for i in range(20)],
            index=df.index,
        )
    }

    sampled, sampled_filters = cap_training_sample(
        df,
        max_train_samples=5,
        seed_name="unit_test_cap_training_sample",
        target_filters=filters,
    )

    assert len(sampled) == 5
    assert list(sampled.index) == list(sampled_filters["value"].index)
    assert sampled["value"].mod(2).eq(0).all()
    np.testing.assert_array_equal(sampled_filters["value"].values, [True] * 5)


def test_cap_training_sample_uses_observed_union_before_capping():
    df = pd.DataFrame({"value": np.arange(100)})
    filters = {
        "value": pd.Series(
            [False] * 97 + [True] * 3,
            index=df.index,
        )
    }

    sampled, sampled_filters = cap_training_sample(
        df,
        max_train_samples=5,
        seed_name="unit_test_sparse_cap_training_sample",
        target_filters=filters,
    )

    assert sampled["value"].tolist() == [97, 98, 99]
    np.testing.assert_array_equal(sampled_filters["value"].values, [True, True, True])


def test_cap_training_sample_preserves_each_target_when_capping():
    df = pd.DataFrame({"value": np.arange(100)})
    filters = {
        "rare_a": pd.Series([True, True] + [False] * 98, index=df.index),
        "rare_b": pd.Series([False] * 98 + [True, True], index=df.index),
    }

    sampled, sampled_filters = cap_training_sample(
        df,
        max_train_samples=4,
        seed_name="unit_test_target_coverage_cap_training_sample",
        target_filters=filters,
    )

    assert set(sampled["value"]) == {0, 1, 98, 99}
    assert sampled_filters["rare_a"].sum() == 2
    assert sampled_filters["rare_b"].sum() == 2


def test_cap_training_sample_rejects_misaligned_filters():
    df = pd.DataFrame({"value": [1, 2]}, index=[10, 11])
    filters = {"value": pd.Series([True, False], index=[0, 1])}

    try:
        cap_training_sample(
            df,
            max_train_samples=1,
            seed_name="unit_test_misaligned_cap_training_sample",
            target_filters=filters,
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Expected misaligned target filters to fail")

    assert "target_filters['value']" in message


def test_filter_positive_finite_weight_rows_reindexes_target_filters():
    df = pd.DataFrame(
        {
            "value": [10, 20, 30, 40, 50],
            "household_weight": [1.0, 0.0, np.nan, np.inf, 5.0],
        },
        index=[10, 11, 12, 13, 14],
    )
    filters = {
        "value": pd.Series(
            [True, True, False, True, True],
            index=df.index,
        )
    }

    filtered, filtered_filters = filter_positive_finite_weight_rows(
        df,
        weight_col="household_weight",
        target_filters=filters,
        context_name="unit-test donor",
    )

    assert filtered["value"].tolist() == [10, 50]
    assert filtered.index.tolist() == [0, 1]
    np.testing.assert_array_equal(filtered_filters["value"].values, [True, True])
    assert filtered_filters["value"].index.tolist() == [0, 1]


def test_filter_positive_finite_weight_rows_requires_observed_target_rows():
    df = pd.DataFrame(
        {
            "value": [10, 20],
            "household_weight": [0.0, 1.0],
        }
    )
    filters = {"value": pd.Series([True, False], index=df.index)}

    try:
        filter_positive_finite_weight_rows(
            df,
            weight_col="household_weight",
            target_filters=filters,
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Expected all invalid observed weights to fail")

    assert "No observed donor rows with positive finite household_weight" in message
