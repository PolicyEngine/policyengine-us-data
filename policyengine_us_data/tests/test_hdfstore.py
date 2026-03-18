"""Unit tests for policyengine_us_data.utils.hdfstore utilities."""

import h5py
import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.utils.hdfstore import (
    ENTITIES,
    DatasetResult,
    _resolve_period_key,
    save_h5,
    save_hdfstore,
)


# -------------------------------------------------------------------
# _resolve_period_key
# -------------------------------------------------------------------


class TestResolvePeriodKey:
    def test_int_key(self):
        assert _resolve_period_key({2024: "a"}, 2024) == 2024

    def test_str_key(self):
        assert _resolve_period_key({"2024": "a"}, 2024) == "2024"

    def test_int_preferred_over_str(self):
        assert _resolve_period_key({2024: "a", "2024": "b"}, 2024) == 2024

    def test_eternity_fallback(self):
        assert _resolve_period_key({"ETERNITY": "a"}, 2024) == "ETERNITY"

    def test_arbitrary_fallback(self):
        assert _resolve_period_key({"2024-01": "a"}, 2024) == "2024-01"

    def test_empty_returns_none(self):
        assert _resolve_period_key({}, 2024) is None


# -------------------------------------------------------------------
# save_h5 / save_hdfstore round-trip
# -------------------------------------------------------------------


class _FakeVariable:
    """Minimal stand-in for a PolicyEngine variable."""

    def __init__(self, entity_key, uprating=""):
        self.entity = type("E", (), {"key": entity_key})()
        self.uprating = uprating


class _FakeSystem:
    """Minimal stand-in for a TaxBenefitSystem."""

    def __init__(self, var_map):
        self.variables = var_map


def _make_tiny_result(time_period=2024):
    """Build a DatasetResult with a handful of rows per entity."""
    n = 5
    data = {
        "person_id": {time_period: np.arange(n, dtype=np.int64)},
        "household_id": {time_period: np.arange(n, dtype=np.int64)},
        "tax_unit_id": {time_period: np.arange(n, dtype=np.int64)},
        "spm_unit_id": {time_period: np.arange(n, dtype=np.int64)},
        "family_id": {time_period: np.arange(n, dtype=np.int64)},
        "marital_unit_id": {time_period: np.arange(n, dtype=np.int64)},
        "person_household_id": {time_period: np.arange(n, dtype=np.int64)},
        "person_tax_unit_id": {time_period: np.arange(n, dtype=np.int64)},
        "person_spm_unit_id": {time_period: np.arange(n, dtype=np.int64)},
        "person_family_id": {time_period: np.arange(n, dtype=np.int64)},
        "person_marital_unit_id": {time_period: np.arange(n, dtype=np.int64)},
        "age": {time_period: np.array([25, 30, 5, 60, 45])},
        "household_weight": {time_period: np.ones(n)},
    }

    variables = {
        "person_id": _FakeVariable("person"),
        "household_id": _FakeVariable("household"),
        "tax_unit_id": _FakeVariable("tax_unit"),
        "spm_unit_id": _FakeVariable("spm_unit"),
        "family_id": _FakeVariable("family"),
        "marital_unit_id": _FakeVariable("marital_unit"),
        "age": _FakeVariable("person"),
        "household_weight": _FakeVariable("household", uprating="cpi"),
    }

    return DatasetResult(
        data=data,
        time_period=time_period,
        system=_FakeSystem(variables),
    )


def test_save_h5_roundtrip(tmp_path):
    """save_h5 writes a file that h5py can read back identically."""
    result = _make_tiny_result()
    output_base = str(tmp_path / "test")
    h5_path = save_h5(result, output_base)

    assert h5_path == output_base + ".h5"

    with h5py.File(h5_path, "r") as f:
        assert "age" in f
        assert "2024" in f["age"]
        np.testing.assert_array_equal(f["age"]["2024"][:], result.data["age"][2024])


def test_save_hdfstore_roundtrip(tmp_path):
    """save_hdfstore writes entity tables readable by pd.HDFStore."""
    result = _make_tiny_result()
    output_base = str(tmp_path / "test")
    hdfstore_path = save_hdfstore(result, output_base)

    assert hdfstore_path == output_base + ".hdfstore.h5"

    with pd.HDFStore(hdfstore_path, "r") as store:
        keys = {k.lstrip("/") for k in store.keys()}
        for entity in ENTITIES:
            assert entity in keys, f"Missing entity: {entity}"
            df = store[f"/{entity}"]
            assert f"{entity}_id" in df.columns

        assert "_variable_metadata" in keys
        manifest = store["/_variable_metadata"]
        assert "variable" in manifest.columns
        assert "entity" in manifest.columns
        assert "uprating" in manifest.columns
        assert len(manifest) > 0

        assert "_time_period" in keys
        tp = store["/_time_period"]
        assert tp.iloc[0] == 2024


def test_save_hdfstore_does_not_mutate_input(tmp_path):
    """save_hdfstore should not modify the DatasetResult's data."""
    result = _make_tiny_result()
    original_age = result.data["age"][2024].copy()

    save_hdfstore(result, str(tmp_path / "test"))

    np.testing.assert_array_equal(result.data["age"][2024], original_age)


def test_eternity_variable_included(tmp_path):
    """Variables keyed by ETERNITY should appear in the HDFStore."""
    n = 3
    data = {
        "person_id": {2024: np.arange(n, dtype=np.int64)},
        "household_id": {2024: np.arange(n, dtype=np.int64)},
        "tax_unit_id": {2024: np.arange(n, dtype=np.int64)},
        "spm_unit_id": {2024: np.arange(n, dtype=np.int64)},
        "family_id": {2024: np.arange(n, dtype=np.int64)},
        "marital_unit_id": {2024: np.arange(n, dtype=np.int64)},
        "person_household_id": {2024: np.arange(n, dtype=np.int64)},
        "is_male": {"ETERNITY": np.array([True, False, True])},
    }
    variables = {
        "person_id": _FakeVariable("person"),
        "household_id": _FakeVariable("household"),
        "tax_unit_id": _FakeVariable("tax_unit"),
        "spm_unit_id": _FakeVariable("spm_unit"),
        "family_id": _FakeVariable("family"),
        "marital_unit_id": _FakeVariable("marital_unit"),
        "is_male": _FakeVariable("person"),
    }
    result = DatasetResult(data=data, time_period=2024, system=_FakeSystem(variables))

    hdfstore_path = save_hdfstore(result, str(tmp_path / "test"))

    with pd.HDFStore(hdfstore_path, "r") as store:
        person_df = store["/person"]
        assert "is_male" in person_df.columns, (
            "ETERNITY-keyed variable missing from person entity"
        )
