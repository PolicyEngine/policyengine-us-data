"""Tests for the production eCPS baseline resolver and integrity gate."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from policyengine_us_data.utils import production_baseline as pb


def _write_h5(path, columns):
    """Write a minimal PolicyEngine-style HDF5 dataset.

    Args:
        path: Output path.
        columns: Mapping of column name -> 1-D array. A value that is itself
            a mapping is written as a period-keyed group.
    """
    with h5py.File(path, "w") as handle:
        for name, value in columns.items():
            if isinstance(value, dict):
                group = handle.create_group(name)
                for period, arr in value.items():
                    group.create_dataset(period, data=np.asarray(arr))
            else:
                handle.create_dataset(name, data=np.asarray(value))


def _healthy_columns():
    return {col: np.array([0.0, 1.0, 2.0]) for col in pb.REQUIRED_NONZERO_COLUMNS}


def test_assert_intact_passes_on_healthy_dataset(tmp_path):
    path = tmp_path / "good.h5"
    _write_h5(path, _healthy_columns())

    checks = pb.assert_baseline_intact(path)

    assert checks["passed"] is True
    assert checks["missing"] == []
    assert checks["all_zero"] == []
    assert checks["nonzero_counts"]["social_security_retirement"] == 2


def test_assert_intact_raises_on_missing_column(tmp_path):
    columns = _healthy_columns()
    del columns["social_security_retirement"]
    path = tmp_path / "missing.h5"
    _write_h5(path, columns)

    with pytest.raises(pb.BaselineIntegrityError) as excinfo:
        pb.assert_baseline_intact(path)

    assert "social_security_retirement" in str(excinfo.value)


def test_assert_intact_raises_on_all_zero_column(tmp_path):
    # The exact recurring failure: the column exists but is all zeros.
    columns = _healthy_columns()
    columns["social_security_retirement"] = np.zeros(3)
    path = tmp_path / "zeroed.h5"
    _write_h5(path, columns)

    with pytest.raises(pb.BaselineIntegrityError):
        pb.assert_baseline_intact(path)


def test_assert_intact_handles_period_keyed_groups(tmp_path):
    columns = {
        col: {"2024": np.array([0.0, 3.0])} for col in pb.REQUIRED_NONZERO_COLUMNS
    }
    path = tmp_path / "grouped.h5"
    _write_h5(path, columns)

    checks = pb.assert_baseline_intact(path)

    assert checks["passed"] is True


def test_production_pin_prefers_explicit_then_env(monkeypatch):
    monkeypatch.setenv(pb.VERSION_ENV_VAR, "from-env")
    assert pb.production_pin("explicit") == "explicit"
    assert pb.production_pin() == "from-env"

    monkeypatch.delenv(pb.VERSION_ENV_VAR, raising=False)
    # Falls back to the installed package version (a non-empty string).
    assert pb.production_pin()


def test_resolve_returns_provenance_and_verifies(tmp_path, monkeypatch):
    path = tmp_path / "resolved.h5"
    _write_h5(path, _healthy_columns())

    import huggingface_hub

    def fake_download(**kwargs):
        assert kwargs["revision"] == "1.2.3"
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    baseline = pb.resolve_production_ecps(version="1.2.3")

    assert baseline.path == str(path)
    assert baseline.revision == "1.2.3"
    assert baseline.repo == pb.DEFAULT_REPO
    assert len(baseline.sha256) == 64
    assert baseline.checks["passed"] is True
    assert baseline.to_dict()["sha256"] == baseline.sha256


def test_resolve_propagates_integrity_failure(tmp_path, monkeypatch):
    columns = _healthy_columns()
    columns["social_security_retirement"] = np.zeros(3)
    path = tmp_path / "broken.h5"
    _write_h5(path, columns)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: str(path))

    with pytest.raises(pb.BaselineIntegrityError):
        pb.resolve_production_ecps(version="1.2.3")
