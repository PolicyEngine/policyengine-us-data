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
    """Build a mock CountryTaxBenefitSystem with given variable→entity mappings."""
    tbs = MagicMock()
    variables = {}
    for var_name, entity_key in variable_entities.items():
        var_mock = MagicMock()
        var_mock.entity.key = entity_key
        variables[var_name] = var_mock
    tbs.variables = variables
    return tbs


def _write_h5(path, period, datasets: dict[str, np.ndarray]):
    with h5py.File(path, "w") as f:
        grp = f.create_group(str(period))
        for name, arr in datasets.items():
            grp.create_dataset(name, data=arr)


PERIOD = 2024
N_PERSONS = 10
N_HOUSEHOLDS = 5


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


class TestDimensionsMatch:
    def test_all_correct(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "good.h5"
        _write_h5(
            h5_path,
            PERIOD,
            {
                "person_id": np.arange(N_PERSONS),
                "household_id": np.arange(N_HOUSEHOLDS),
                "age": np.ones(N_PERSONS),
                "income": np.ones(N_PERSONS),
                "household_weight": np.ones(N_HOUSEHOLDS),
            },
        )
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        assert results == []

    def test_or_raise_passes(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "good.h5"
        _write_h5(
            h5_path,
            PERIOD,
            {
                "person_id": np.arange(N_PERSONS),
                "household_id": np.arange(N_HOUSEHOLDS),
                "age": np.ones(N_PERSONS),
                "income": np.ones(N_PERSONS),
                "household_weight": np.ones(N_HOUSEHOLDS),
            },
        )
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            validate_h5_or_raise(h5_path, period=PERIOD)


class TestPersonDimensionMismatch:
    def test_wrong_person_variable_length(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "bad_dim.h5"
        _write_h5(
            h5_path,
            PERIOD,
            {
                "person_id": np.arange(N_PERSONS),
                "household_id": np.arange(N_HOUSEHOLDS),
                "age": np.ones(N_PERSONS + 99),  # wrong length
                "income": np.ones(N_PERSONS),
                "household_weight": np.ones(N_HOUSEHOLDS),
            },
        )
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        fails = [r for r in results if r["status"] == "FAIL"]
        assert len(fails) == 1
        assert "age" in fails[0]["detail"]

    def test_or_raise_raises(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "bad_dim.h5"
        _write_h5(
            h5_path,
            PERIOD,
            {
                "person_id": np.arange(N_PERSONS),
                "household_id": np.arange(N_HOUSEHOLDS),
                "age": np.ones(N_PERSONS + 99),
                "income": np.ones(N_PERSONS),
                "household_weight": np.ones(N_HOUSEHOLDS),
            },
        )
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            with pytest.raises(ValueError, match="age"):
                validate_h5_or_raise(h5_path, period=PERIOD)


class TestMissingHouseholdWeight:
    def test_missing_weight(self, tmp_path, mock_tbs):
        h5_path = tmp_path / "no_weight.h5"
        _write_h5(
            h5_path,
            PERIOD,
            {
                "person_id": np.arange(N_PERSONS),
                "household_id": np.arange(N_HOUSEHOLDS),
                "age": np.ones(N_PERSONS),
                "income": np.ones(N_PERSONS),
            },
        )
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
        _write_h5(
            h5_path,
            PERIOD,
            {
                "person_id": np.arange(N_PERSONS),
                "household_id": np.arange(N_HOUSEHOLDS),
                "age": np.ones(N_PERSONS),
                "income": np.ones(N_PERSONS),
                "household_weight": np.zeros(N_HOUSEHOLDS),
            },
        )
        with patch(
            "policyengine_us.CountryTaxBenefitSystem",
            return_value=mock_tbs,
        ):
            results = validate_h5_entity_dimensions(h5_path, period=PERIOD)
        checks = [r["check"] for r in results]
        assert "household_weight_nonzero" in checks
