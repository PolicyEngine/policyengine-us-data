"""Regression test for SIPP tip income column matching.

Previously `str.contains("TXAMT")` matched both `TJB*_TXAMT` (dollar
amounts) and `AJB*_TXAMT` (Census allocation flags). The fix narrows
to explicit `TJB*_TXAMT` dollar-amount columns only.
"""

import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.datasets.sipp.sipp import (
    SIPP_JOB_OCCUPATION_COLUMNS,
    SIPP_TIP_AMOUNT_COLUMNS,
)
import policyengine_us_data.datasets.sipp.sipp as sipp_module


def test_tip_regex_matches_dollar_amounts_only():
    # SIPP column naming: TJB<N>_TXAMT is the dollar amount for job N,
    # AJB<N>_TXAMT is the allocation flag for the same field.
    # Include several distractors that the old `contains("TXAMT")` regex
    # would have caught.
    columns = pd.Index(
        [
            "TJB1_TXAMT",
            "TJB2_TXAMT",
            "AJB1_TXAMT",  # allocation flag — should NOT be summed
            "AJB2_TXAMT",  # allocation flag — should NOT be summed
            "SOME_TXAMT_OTHER",  # unrelated non-numbered column
            "TPTOTINC",  # unrelated
        ]
    )

    matches = [column for column in SIPP_TIP_AMOUNT_COLUMNS if column in columns]

    assert list(matches) == ["TJB1_TXAMT", "TJB2_TXAMT"]


def test_tip_sum_excludes_allocation_flags():
    df = pd.DataFrame(
        {
            "TJB1_TXAMT": [100.0, 200.0],
            "TJB2_TXAMT": [50.0, 75.0],
            "AJB1_TXAMT": [1, 2],  # allocation flags: small ints
            "AJB2_TXAMT": [0, 1],
        }
    )
    # Mirror the sipp.py computation using the new regex.
    tip_cols = [column for column in SIPP_TIP_AMOUNT_COLUMNS if column in df]
    tip_income_monthly = df[tip_cols].fillna(0).sum(axis=1)
    assert list(tip_income_monthly) == [150.0, 275.0]

    # Sanity check: the buggy regex would have included AJB flags.
    buggy_tip_income_monthly = (
        df[df.columns[df.columns.str.contains("TXAMT")]].fillna(0).sum(axis=1)
    )
    assert list(buggy_tip_income_monthly) == [151.0, 278.0]


def test_train_tip_model_requires_allocation_flags_for_present_tip_columns(
    monkeypatch,
):
    monkeypatch.setattr(sipp_module, "hf_hub_download", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sipp_module.pd,
        "read_csv",
        lambda *args, **kwargs: pd.DataFrame({"TJB1_TXAMT": [10.0]}),
    )

    with pytest.raises(KeyError, match="AJB1_TXAMT"):
        sipp_module.train_tip_model()


def test_train_tip_model_drops_non_positive_weights(monkeypatch):
    monkeypatch.setattr(sipp_module, "hf_hub_download", lambda *args, **kwargs: None)

    data = {
        "SSUID": [1, 2, 3, 4],
        "MONTHCODE": [12, 12, 12, 12],
        "TAGE": [30, 31, 32, 33],
        "WPFINWGT": [100.0, 0.0, -5.0, 200.0],
        "TPTOTINC": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
        "TJB1_TXAMT": [10.0, 20.0, 30.0, 40.0],
        "AJB1_TXAMT": [0, 0, 0, 0],
    }
    for column in SIPP_JOB_OCCUPATION_COLUMNS:
        data[column] = [0, 0, 0, 0]
    monkeypatch.setattr(
        sipp_module.pd,
        "read_csv",
        lambda *args, **kwargs: pd.DataFrame(data),
    )

    captured = {}

    class FakeQRF:
        def fit(
            self,
            *,
            X_train,
            predictors,
            imputed_variables,
            target_filters,
            weight_col,
        ):
            captured["weights"] = X_train[weight_col].to_numpy()
            captured["target_filter"] = target_filters["tip_income"].to_numpy()
            return self

    monkeypatch.setattr(sipp_module, "QRF", FakeQRF)

    sipp_module.train_tip_model()

    np.testing.assert_array_equal(captured["weights"], [100.0, 200.0])
    np.testing.assert_array_equal(captured["target_filter"], [True, True])


def test_train_tip_model_keeps_reported_sipp_status_flags(monkeypatch):
    monkeypatch.setattr(sipp_module, "hf_hub_download", lambda *args, **kwargs: None)

    data = {
        "SSUID": [1, 2, 3, 4],
        "MONTHCODE": [12, 12, 12, 12],
        "TAGE": [30, 31, 32, 33],
        "WPFINWGT": [100.0, 100.0, 100.0, 100.0],
        "TPTOTINC": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
        "TJB1_TXAMT": [10.0, 20.0, 30.0, 40.0],
        "AJB1_TXAMT": [1, 2, 6, 9],
    }
    for column in SIPP_JOB_OCCUPATION_COLUMNS:
        data[column] = [0, 0, 0, 0]
    monkeypatch.setattr(
        sipp_module.pd,
        "read_csv",
        lambda *args, **kwargs: pd.DataFrame(data),
    )

    captured = {}

    class FakeQRF:
        def fit(
            self,
            *,
            X_train,
            predictors,
            imputed_variables,
            target_filters,
            weight_col,
        ):
            captured["tip_income"] = X_train["tip_income"].to_numpy()
            captured["target_filter"] = target_filters["tip_income"].to_numpy()
            return self

    monkeypatch.setattr(sipp_module, "QRF", FakeQRF)

    sipp_module.train_tip_model()

    np.testing.assert_array_equal(captured["tip_income"], [120.0, 480.0])
    np.testing.assert_array_equal(captured["target_filter"], [True, True])
