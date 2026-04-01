"""Tests for H5 pre-publish validation."""

from unittest.mock import patch, MagicMock

import h5py
import numpy as np
import pytest

from policyengine_us_data.utils.validate_h5 import (
    validate_h5_entity_dimensions,
    validate_h5_or_raise,
)


def _make_mock_tbs(variable_entities: dict[str, str]):
    """Build a mock CountryTaxBenefitSystem with given variable->entity mappings."""
    tbs = MagicMock()
    variables = {}
    for var_name, entity_key in variable_entities.items():
        var_mock = MagicMock()
        var_mock.entity.key = entity_key
        variables[var_name] = var_mock
    tbs.variables = variables
    return tbs


def _write_h5_flat(path, datasets: dict[str, np.ndarray]):
    """Flat layout: datasets at the top level (storage files)."""
    with h5py.File(path, "w") as f:
        for name, arr in datasets.items():
            f.create_dataset(name, data=arr)


def _write_h5_nested(path, period, datasets: dict[str, np.ndarray]):
    """Nested layout: variable/period (pipeline-built files)."""
    with h5py.File(path, "w") as f:
        for name, arr in datasets.items():
            grp = f.create_group(name)
            grp.create_dataset(str(period), data=arr)


PERIOD = 2024
N_PERSONS = 10
N_HOUSEHOLDS = 5

GOOD_DATA = {
    "person_id": np.arange(N_PERSONS),
    "household_id": np.arange(N_HOUSEHOLDS),
    "age": np.ones(N_PERSONS),
    "income": np.ones(N_PERSONS),
    "household_weight": np.ones(N_HOUSEHOLDS),
}


@pytest.fixture
def mock_tbs():
    return _make_mock_tbs(
        {
            "person_id": "person",
            "household_id": "household",
            "age": "person",
            "household_weight": "household",
            "income": "person",
        }
    )


class TestFlatLayout:
    def test_all_correct(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "good.h5"
        _write_h5_flat(h5_path, GOOD_DATA)
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        assert results == []

    def test_wrong_person_length(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "bad.h5"
        data = {**GOOD_DATA, "age": np.ones(N_PERSONS + 99)}
        _write_h5_flat(h5_path, data)
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        dim_fails = [r for r in results if r["check"] == "dimension"]
        assert len(dim_fails) == 1
        assert "age" in dim_fails[0]["detail"]


class TestNestedLayout:
    def test_all_correct(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "good_nested.h5"
        _write_h5_nested(h5_path, PERIOD, GOOD_DATA)
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        assert results == []

    def test_wrong_person_length(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "bad_nested.h5"
        data = {**GOOD_DATA, "age": np.ones(N_PERSONS + 99)}
        _write_h5_nested(h5_path, PERIOD, data)
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        dim_fails = [r for r in results if r["check"] == "dimension"]
        assert len(dim_fails) == 1
        assert "age" in dim_fails[0]["detail"]


class TestOrRaise:
    def test_passes(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "good.h5"
        _write_h5_flat(h5_path, GOOD_DATA)
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            validate_h5_or_raise(h5_path, period=PERIOD)

    def test_raises_on_mismatch(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "bad.h5"
        data = {**GOOD_DATA, "age": np.ones(N_PERSONS + 99)}
        _write_h5_flat(h5_path, data)
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            with pytest.raises(ValueError, match="age"):
                validate_h5_or_raise(h5_path, period=PERIOD)


class TestMissingHouseholdWeight:
    def test_missing_weight(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "no_weight.h5"
        data = {k: v for k, v in GOOD_DATA.items() if k != "household_weight"}
        _write_h5_flat(h5_path, data)
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        checks = [r["check"] for r in results]
        assert "household_weight_exists" in checks


class TestAllZeroWeights:
    def test_zero_weights(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "zero_weight.h5"
        data = {**GOOD_DATA, "household_weight": np.zeros(N_HOUSEHOLDS)}
        _write_h5_flat(h5_path, data)
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        checks = [r["check"] for r in results]
        assert "household_weight_nonzero" in checks
