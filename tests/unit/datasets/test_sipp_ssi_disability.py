import numpy as np
import pandas as pd

from policyengine_us_data.datasets.sipp import (
    SSI_DISABILITY_DIFFICULTY_PREDICTORS,
    SSI_DISABILITY_MODEL_PREDICTORS,
    SSI_DISABILITY_MODEL_VARIABLE,
    SSI_DISABILITY_CRITERIA_VARIABLE,
    apply_ssi_disability_signal_screen,
    build_ssi_disability_training_frame,
    coerce_ssi_disability_predictions,
    predict_ssi_disability_criteria,
    preserve_under_65_ssi_disability_criteria,
    prepare_ssi_disability_receiver,
)
from policyengine_us_data.datasets.sipp.sipp import (
    SSI_DISABILITY_COLUMNS,
    SSI_DISABILITY_MODEL_VERSION,
    _ssi_disability_model_path,
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
            "TSSSAMT": [0.0, 0.0, 0.0, 0.0],
            "ESELFCARE": [1, 1, 1, 1],
            "EHEARING": [2, 2, 2, 2],
            "ESEEING": [2, 2, 2, 2],
            "EERRANDS": [2, 2, 2, 2],
            "EAMBULAT": [2, 2, 2, 2],
            "ECOGNIT": [2, 2, 2, 2],
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


def test_build_ssi_disability_training_frame_uses_all_disability_amounts():
    frame = _base_sipp_frame().iloc[[2]].copy()
    frame["TDIS6AMT"] = 100

    result = build_ssi_disability_training_frame(frame)

    assert result["has_disability_income"].iloc[0]


def test_ssi_disability_training_usecols_include_label_and_income_columns():
    assert {"TPTOTINC", "RSSI_YRYN"} <= set(SSI_DISABILITY_COLUMNS)
    assert {"ASSI_YRYN", "ASSI_BRSN"} <= set(SSI_DISABILITY_COLUMNS)
    assert {
        "ESELFCARE",
        "EHEARING",
        "ESEEING",
        "EERRANDS",
        "EAMBULAT",
        "ECOGNIT",
    } <= set(SSI_DISABILITY_COLUMNS)


def test_ssi_disability_predictors_use_six_comparable_difficulty_items():
    assert set(SSI_DISABILITY_DIFFICULTY_PREDICTORS) <= set(
        SSI_DISABILITY_MODEL_PREDICTORS
    )
    assert "is_disabled" not in SSI_DISABILITY_MODEL_PREDICTORS


def test_ssi_disability_model_cache_version_tracks_predictor_schema():
    assert SSI_DISABILITY_MODEL_VERSION == 6
    assert _ssi_disability_model_path(2024).name == (
        "ssi_disability_criteria_v6_2024.pkl"
    )


def test_build_ssi_disability_training_frame_annualizes_ssdi_amount():
    frame = _base_sipp_frame().iloc[[2]].copy()
    frame["ESSRSN2YN"] = 1
    frame["TSSSAMT"] = 125.0

    result = build_ssi_disability_training_frame(frame)

    assert result["social_security_disability"].iloc[0] == 1_500.0


def test_build_ssi_disability_training_frame_excludes_allocated_label_source():
    frame = _base_sipp_frame()
    frame.loc[0, "ASSI_YRYN"] = 3
    frame.loc[1:, "ASSI_YRYN"] = 0
    frame["ASSI_BRSN"] = 0

    result = build_ssi_disability_training_frame(frame)

    assert len(result) == 3
    np.testing.assert_array_equal(
        result[SSI_DISABILITY_MODEL_VARIABLE].values,
        np.array([False, False, False]),
    )


def test_build_ssi_disability_training_frame_keeps_non_ssi_without_reason_source():
    frame = _base_sipp_frame()
    frame["ASSI_YRYN"] = 0
    frame["ASSI_BRSN"] = 3

    result = build_ssi_disability_training_frame(frame)

    assert len(result) == 2
    np.testing.assert_array_equal(
        result[SSI_DISABILITY_MODEL_VARIABLE].values,
        np.array([False, False]),
    )


def test_build_ssi_disability_training_frame_excludes_ssi_with_missing_reason_source():
    frame = _base_sipp_frame()
    frame.loc[0, "ESSI_BRSN"] = -9
    frame["ASSI_YRYN"] = 0
    frame["ASSI_BRSN"] = 0

    result = build_ssi_disability_training_frame(frame)

    assert len(result) == 3
    np.testing.assert_array_equal(
        result[SSI_DISABILITY_MODEL_VARIABLE].values,
        np.array([False, False, False]),
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
    assert result["difficulty_hearing"].iloc[0] == 0


def test_apply_ssi_disability_signal_screen_excludes_records_without_signal():
    result = apply_ssi_disability_signal_screen(
        np.array([True, True, True, False]),
        disability_difficulty_signal=np.array([True, False, False, True]),
        social_security_disability=np.array([False, True, False, False]),
        has_disability_income=np.array([False, False, False, True]),
    )

    np.testing.assert_array_equal(result, np.array([True, True, False, False]))


def test_apply_ssi_disability_signal_screen_treats_missing_as_false():
    result = apply_ssi_disability_signal_screen(
        np.array([True, True, True]),
        disability_difficulty_signal=np.array([np.nan, 0, 0]),
        social_security_disability=np.array([0, np.nan, 0]),
        has_disability_income=np.array([0, 0, np.nan]),
    )

    np.testing.assert_array_equal(result, np.array([False, False, False]))


def test_preserve_under_65_ssi_disability_criteria_keeps_observed_anchors():
    result = preserve_under_65_ssi_disability_criteria(
        np.array([False, False, False, False]),
        age=np.array([40, 64, 70, 30]),
        ssi_reported=np.array([0, 100, 100, np.nan]),
        existing_meets_ssi_disability_criteria=np.array([True, False, True, np.nan]),
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


def test_predict_ssi_disability_criteria_does_not_apply_sga_screen():
    class AlwaysTrueModel:
        def predict(self, X_test):
            return pd.DataFrame(
                {SSI_DISABILITY_CRITERIA_VARIABLE: np.ones(len(X_test), dtype=bool)}
            )

    receiver = pd.DataFrame(
        {
            "age": [40],
            "employment_income": [60_000],
            "difficulty_walking_or_climbing_stairs": [True],
            "social_security_disability": [False],
            "has_disability_income": [False],
        }
    )

    result = predict_ssi_disability_criteria(AlwaysTrueModel(), receiver)

    np.testing.assert_array_equal(result, np.array([True]))
