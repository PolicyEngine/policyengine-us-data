import numpy as np
import pytest

from policyengine_us_data.datasets.cps.takeup import (
    align_reported_ssi_disability,
    prioritize_reported_recipients,
)


def test_prioritize_reported_recipients_preserves_reporters():
    reported = np.array([True, False, False, True, False])
    draws = np.array([0.9, 0.1, 0.7, 0.8, 0.9])

    result = prioritize_reported_recipients(reported, 0.6, draws)

    np.testing.assert_array_equal(
        result,
        np.array([True, True, False, True, False]),
    )


def test_prioritize_reported_recipients_caps_non_reporters_at_zero():
    reported = np.array([True, False, True, True])
    draws = np.array([0.2, 0.1, 0.3, 0.4])

    result = prioritize_reported_recipients(reported, 0.5, draws)

    np.testing.assert_array_equal(
        result,
        np.array([True, False, True, True]),
    )


def test_prioritize_reported_recipients_requires_matching_shapes():
    with pytest.raises(ValueError):
        prioritize_reported_recipients(
            np.array([True, False]),
            0.5,
            np.array([0.1]),
        )


def test_align_reported_ssi_disability_marks_under_65_reporters_disabled():
    result = align_reported_ssi_disability(
        is_disabled=np.array([False, False, True, False]),
        reported_ssi=np.array([True, True, False, False]),
        ages=np.array([40, 70, 30, 20]),
    )

    np.testing.assert_array_equal(
        result,
        np.array([True, False, True, False]),
    )
