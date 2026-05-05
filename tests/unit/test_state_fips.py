"""Tests for the shared STATE_ABBR_TO_FIPS dict used across calibration code."""

import pytest


def test_dc_fips_is_string():
    """DC's FIPS in the canonical dict is a string '11' (matches GEO_ID suffix)."""
    from policyengine_us_data.storage.calibration_targets.pull_soi_targets import (
        STATE_ABBR_TO_FIPS,
    )

    assert STATE_ABBR_TO_FIPS["DC"] == "11"
    assert isinstance(STATE_ABBR_TO_FIPS["DC"], str)


def test_all_state_fips_are_strings():
    """All entries in STATE_ABBR_TO_FIPS are strings — downstream code compares
    against ``r.GEO_ID[-2:]`` which is a string slice, so any int entry would
    silently fail comparison and drop that state's targets."""
    from policyengine_us_data.storage.calibration_targets.pull_soi_targets import (
        STATE_ABBR_TO_FIPS,
    )

    for abbr, fips in STATE_ABBR_TO_FIPS.items():
        assert isinstance(fips, str), f"{abbr} FIPS {fips!r} is not a string"
        assert len(fips) == 2, f"{abbr} FIPS {fips!r} is not a 2-character string"


def test_loss_module_does_not_mutate_state_fips_on_import():
    """Regression for the DC SNAP calibration drop bug.

    Previously ``_add_snap_metric_columns`` wrote ``STATE_ABBR_TO_FIPS["DC"] = 11``
    (int) inline, corrupting the shared dict the moment the function ran. The fix
    removed that line. Even at import time the dict should be untouched.
    """
    from policyengine_us_data.storage.calibration_targets.pull_soi_targets import (
        STATE_ABBR_TO_FIPS,
    )

    before = dict(STATE_ABBR_TO_FIPS)
    import policyengine_us_data.utils.loss  # noqa: F401

    assert STATE_ABBR_TO_FIPS == before
