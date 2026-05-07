import h5py
import numpy as np
import pandas as pd
import pytest

from policyengine_us_data.datasets.cps.small_enhanced_cps import (
    _attach_clone_origin_flags,
)


def _write_period_group(
    h5_file, variable_name: str, values_by_period: dict[int, np.ndarray]
):
    group = h5_file.create_group(variable_name)
    for period, values in values_by_period.items():
        group.create_dataset(str(period), data=values)


def test_attach_clone_origin_flags_maps_sampled_entity_ids(tmp_path):
    source_path = tmp_path / "enhanced_cps_2024.h5"
    with h5py.File(source_path, "w") as h5_file:
        _write_period_group(h5_file, "person_id", {2024: np.array([11, 12, 13])})
        _write_period_group(
            h5_file, "person_is_puf_clone", {2024: np.array([0, 1, 0], dtype=np.int8)}
        )
        _write_period_group(h5_file, "tax_unit_id", {2024: np.array([21, 22])})
        _write_period_group(
            h5_file, "tax_unit_is_puf_clone", {2024: np.array([1, 0], dtype=np.int8)}
        )
        _write_period_group(h5_file, "spm_unit_id", {2024: np.array([31, 32])})
        _write_period_group(
            h5_file, "spm_unit_is_puf_clone", {2024: np.array([0, 1], dtype=np.int8)}
        )
        _write_period_group(h5_file, "family_id", {2024: np.array([41, 42])})
        _write_period_group(
            h5_file, "family_is_puf_clone", {2024: np.array([1, 1], dtype=np.int8)}
        )
        _write_period_group(h5_file, "household_id", {2024: np.array([51, 52])})
        _write_period_group(
            h5_file, "household_is_puf_clone", {2024: np.array([0, 1], dtype=np.int8)}
        )

    sampled_data = {
        "person_id": {2024: np.array([12, 11], dtype=np.int32)},
        "tax_unit_id": {2024: np.array([22, 21], dtype=np.int32)},
        "spm_unit_id": {2024: np.array([32, 31], dtype=np.int32)},
        "family_id": {2024: np.array([41, 42], dtype=np.int32)},
        "household_id": {2024: np.array([52, 51], dtype=np.int32)},
    }

    _attach_clone_origin_flags(sampled_data, source_path)

    assert sampled_data["person_is_puf_clone"][2024].tolist() == [1, 0]
    assert sampled_data["tax_unit_is_puf_clone"][2024].tolist() == [0, 1]
    assert sampled_data["spm_unit_is_puf_clone"][2024].tolist() == [1, 0]
    assert sampled_data["family_is_puf_clone"][2024].tolist() == [1, 1]
    assert sampled_data["household_is_puf_clone"][2024].tolist() == [1, 0]


def test_attach_clone_origin_flags_rejects_missing_sampled_id(tmp_path):
    source_path = tmp_path / "enhanced_cps_2024.h5"
    with h5py.File(source_path, "w") as h5_file:
        for entity_name, id_base in [
            ("person", 10),
            ("tax_unit", 20),
            ("spm_unit", 30),
            ("family", 40),
            ("household", 50),
        ]:
            _write_period_group(
                h5_file,
                f"{entity_name}_id",
                {2024: np.array([id_base], dtype=np.int32)},
            )
            _write_period_group(
                h5_file,
                f"{entity_name}_is_puf_clone",
                {2024: np.array([0], dtype=np.int8)},
            )

    sampled_data = {
        "person_id": {2024: np.array([999], dtype=np.int32)},
        "tax_unit_id": {2024: np.array([20], dtype=np.int32)},
        "spm_unit_id": {2024: np.array([30], dtype=np.int32)},
        "family_id": {2024: np.array([40], dtype=np.int32)},
        "household_id": {2024: np.array([50], dtype=np.int32)},
    }

    with pytest.raises(KeyError, match="person_is_puf_clone"):
        _attach_clone_origin_flags(sampled_data, source_path)


def test_attach_clone_origin_flags_accepts_period_keys(tmp_path):
    source_path = tmp_path / "enhanced_cps_2024.h5"
    with h5py.File(source_path, "w") as h5_file:
        _write_period_group(h5_file, "person_id", {2024: np.array([11, 12])})
        _write_period_group(
            h5_file,
            "person_is_puf_clone",
            {2024: np.array([0, 1], dtype=np.int8)},
        )
        for entity_name, id_base in [
            ("tax_unit", 20),
            ("spm_unit", 30),
            ("family", 40),
            ("household", 50),
        ]:
            _write_period_group(
                h5_file,
                f"{entity_name}_id",
                {2024: np.array([id_base], dtype=np.int32)},
            )
            _write_period_group(
                h5_file,
                f"{entity_name}_is_puf_clone",
                {2024: np.array([0], dtype=np.int8)},
            )

    period = pd.Period("2024", freq="Y")
    sampled_data = {
        "person_id": {period: np.array([12, 11], dtype=np.int32)},
        "tax_unit_id": {period: np.array([20], dtype=np.int32)},
        "spm_unit_id": {period: np.array([30], dtype=np.int32)},
        "family_id": {period: np.array([40], dtype=np.int32)},
        "household_id": {period: np.array([50], dtype=np.int32)},
    }

    _attach_clone_origin_flags(sampled_data, source_path)

    assert sampled_data["person_is_puf_clone"][period].tolist() == [1, 0]
