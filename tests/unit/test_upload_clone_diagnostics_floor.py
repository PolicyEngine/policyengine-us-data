"""Upload-time enforcement of the PUF-clone weight-share floor and tax bound.

The generation-time guard in ``enhanced_cps.validate_clone_diagnostics`` rejects a
degraded Enhanced CPS build whose PUF clones are starved below the weight-share
floor or whose clones carry taxes that wildly exceed their market income. These
tests pin the matching enforcement in the dataset *upload* validator
(``_clone_diagnostics_errors`` / ``validate_clone_diagnostics``), which
previously only checked that each metric was finite and within ``[0, 100]`` and
so could publish a degraded artifact.
"""

import json

import pytest

from policyengine_us_data.datasets.cps.enhanced_cps import (
    MAX_PUF_CLONE_TAXES_EXCEED_MARKET_INCOME_SHARE_PCT,
    MIN_PUF_CLONE_HOUSEHOLD_WEIGHT_SHARE_PCT,
)
from policyengine_us_data.storage.upload_completed_datasets import (
    CLONE_DIAGNOSTICS_METRICS,
    DatasetValidationError,
    _clone_diagnostics_errors,
    validate_clone_diagnostics,
)


def _full_diagnostics(**overrides):
    """A complete, healthy clone-diagnostics payload.

    Every required metric is present and well inside ``[0, 100]`` so the
    baseline payload passes; individual tests override one field to exercise a
    single threshold.
    """

    payload = {metric: 10.0 for metric in CLONE_DIAGNOSTICS_METRICS}
    payload["clone_household_weight_share_pct"] = 10.0
    payload["clone_taxes_exceed_market_income_share_pct"] = 5.0
    payload.update(overrides)
    return payload


def _write_sidecar(tmp_path, diagnostics, *, period="2024"):
    # The upload validator accepts either a single-period payload (with a
    # "period" key) or a multi-period {"periods": {year: metrics}} payload.
    # Use the multi-period shape so the per-period metrics are validated.
    payload = {"periods": {period: diagnostics}}
    path = tmp_path / "enhanced_cps_2024.clone_diagnostics.json"
    path.write_text(json.dumps(payload))
    return path


# --- _clone_diagnostics_errors: unit-level threshold checks ------------------


def test_errors_flag_weight_share_below_floor():
    errors = _clone_diagnostics_errors(
        _full_diagnostics(clone_household_weight_share_pct=2.0),
        context="period 2024",
    )
    assert any("floor" in e and "weight share" in e for e in errors), errors


def test_errors_flag_clone_tax_share_above_bound():
    errors = _clone_diagnostics_errors(
        _full_diagnostics(
            clone_taxes_exceed_market_income_share_pct=66.0,
        ),
        context="period 2024",
    )
    assert any("limit" in e and "market income" in e for e in errors), errors


def test_errors_pass_for_healthy_payload():
    assert _clone_diagnostics_errors(_full_diagnostics(), context="period 2024") == []


def test_errors_at_floor_boundary_pass():
    # Exactly at the floor is acceptable; only strictly-below is rejected.
    assert (
        _clone_diagnostics_errors(
            _full_diagnostics(
                clone_household_weight_share_pct=(
                    MIN_PUF_CLONE_HOUSEHOLD_WEIGHT_SHARE_PCT
                )
            ),
            context="period 2024",
        )
        == []
    )


def test_errors_at_tax_bound_boundary_pass():
    # Exactly at the bound is acceptable; only strictly-above is rejected.
    assert (
        _clone_diagnostics_errors(
            _full_diagnostics(
                clone_taxes_exceed_market_income_share_pct=(
                    MAX_PUF_CLONE_TAXES_EXCEED_MARKET_INCOME_SHARE_PCT
                )
            ),
            context="period 2024",
        )
        == []
    )


def test_errors_no_crash_when_share_fields_absent():
    # Back-compat: an older payload that does not carry the share fields must
    # not crash the new threshold logic. (Required-metric presence is enforced
    # separately; here we confirm the floor/bound checks are absence-safe.)
    errors = _clone_diagnostics_errors(
        {"some_other_metric": 1.0}, context="period 2024"
    )
    assert all("floor" not in e for e in errors), errors
    assert all("market income" not in e for e in errors), errors


# --- validate_clone_diagnostics: end-to-end sidecar checks -------------------


def test_validate_rejects_weight_share_below_floor(tmp_path):
    path = _write_sidecar(
        tmp_path,
        _full_diagnostics(clone_household_weight_share_pct=2.0),
    )
    with pytest.raises(DatasetValidationError, match="floor"):
        validate_clone_diagnostics(path)


def test_validate_rejects_clone_tax_share_above_bound(tmp_path):
    path = _write_sidecar(
        tmp_path,
        _full_diagnostics(
            clone_taxes_exceed_market_income_share_pct=66.0,
        ),
    )
    with pytest.raises(DatasetValidationError, match="limit"):
        validate_clone_diagnostics(path)


def test_validate_accepts_healthy_sidecar(tmp_path):
    path = _write_sidecar(tmp_path, _full_diagnostics())
    validate_clone_diagnostics(path)
