import numpy as np
import pandas as pd

from policyengine_us_data.datasets.sipp import (
    SSI_DISABILITY_MODEL_PREDICTORS,
    SSI_DISABILITY_MODEL_VARIABLE,
    apply_ssi_disability_signal_screen,
    apply_ssi_sga_screen,
    build_ssi_disability_training_frame,
    coerce_ssi_disability_predictions,
    prepare_ssi_disability_receiver,
)


def _base_sipp_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SSUID": [1, 2, 3, 4],
            "PNUM": [1, 1, 1, 1],
            "MONTHCODE": [12, 12, 12, 12],
            "WPFINWGT": [1.0, 1.0, 1.0, 1.0],
            "TAGE": [40, 70, 40, 40],
            "ESEX": [1, 2, 1, 1],
            "EMS": [2, 2, 2, 2],
            "TPTOTINC": [500.0, 500.0, 500.0, 8_000.0],
            "TVAL_BANK": [100.0, 100.0, 100.0, 100_000.0],
            "TVAL_STMF": [0.0, 0.0, 0.0, 0.0],
            "TVAL_BOND": [0.0, 0.0, 0.0, 0.0],
            "TINC_BANK": [0.0, 0.0, 0.0, 0.0],
            "TINC_STMF": [0.0, 0.0, 0.0, 0.0],
            "TINC_BOND": [0.0, 0.0, 0.0, 0.0],
            "TINC_RENT": [0.0, 0.0, 0.0, 0.0],
            "RSSI_YRYN": [1, 1, 2, 2],
            "ESSI_BRSN": [1, 2, -9, -9],
            "EDISABL": [1, 1, 1, 1],
            "EHLTHCOND": [1, 1, 1, 1],
            "RDIS": [1, 1, 1, 1],
            "RDIS_ALT": [1, 1, 1, 1],
            "EDISANY": [2, 2, 2, 2],
            "ENJ_NOWRK3": [2, 2, 2, 2],
            "ESSRSN2YN": [2, 2, 2, 2],
        }
    )


def test_build_ssi_disability_training_frame_screens_financially():
    result = build_ssi_disability_training_frame(_base_sipp_frame())

    np.testing.assert_array_equal(
        result[SSI_DISABILITY_MODEL_VARIABLE].values,
        np.array([True, False, False, False]),
    )
    np.testing.assert_array_equal(
        result["ssi_disability_training_candidate"].values,
        np.array([True, False, True, False]),
    )


def test_prepare_ssi_disability_receiver_fills_missing_predictors():
    result = prepare_ssi_disability_receiver(
        pd.DataFrame(
            {
                "age": [40],
                "employment_income": [0],
            }
        )
    )

    assert list(result.columns) == SSI_DISABILITY_MODEL_PREDICTORS
    assert result.shape == (1, len(SSI_DISABILITY_MODEL_PREDICTORS))
    assert result["age"].iloc[0] == 40
    assert result["is_disabled"].iloc[0] == 0


def test_apply_ssi_sga_screen_excludes_high_earners():
    result = apply_ssi_sga_screen(
        np.array([True, True, False]),
        np.array([0, 60_000, 0]),
    )

    np.testing.assert_array_equal(result, np.array([True, False, False]))


def test_apply_ssi_disability_signal_screen_excludes_records_without_signal():
    result = apply_ssi_disability_signal_screen(
        np.array([True, True, True, False]),
        is_disabled=np.array([True, False, False, True]),
        social_security_disability=np.array([False, True, False, False]),
        has_disability_income=np.array([False, False, False, True]),
    )

    np.testing.assert_array_equal(result, np.array([True, True, False, False]))


def test_coerce_ssi_disability_predictions_handles_string_false():
    result = coerce_ssi_disability_predictions(
        pd.Series(["False", "True", "0", "1", False, True, 0, 1])
    )

    np.testing.assert_array_equal(
        result,
        np.array([False, True, False, True, False, True, False, True]),
    )
