import h5py
import numpy as np
import pytest

from policyengine_us_data.datasets.puf import puf as puf_module
from policyengine_us_data.datasets.puf.puf import (
    PUF,
    QBI_SIMULATION_VERSION,
    QBI_SIMULATION_VERSION_ATTR,
    _person_financial_value_from_puf_row,
)


def _mark_current_qbi_simulation(file_handle):
    file_handle.attrs[QBI_SIMULATION_VERSION_ATTR] = QBI_SIMULATION_VERSION


def _write_capital_gains_basis_source_file(path):
    with h5py.File(path, "w") as file_handle:
        file_handle.create_dataset("person_id", data=np.array([1, 2, 3, 4]))
        file_handle.create_dataset("person_tax_unit_id", data=np.array([1, 1, 2, 2]))
        file_handle.create_dataset("person_household_id", data=np.array([1, 1, 2, 2]))
        file_handle.create_dataset("household_id", data=np.array([1, 2]))
        file_handle.create_dataset("household_weight", data=np.array([100.0, 200.0]))
        file_handle.create_dataset(
            "long_term_capital_gains",
            data=np.array([100.0, -40.0, 0.0, 200.0]),
        )


@pytest.mark.skip(reason="This test requires private data.")
@pytest.mark.parametrize("year", [2015])
def test_irs_puf_generates(year: int):
    from policyengine_us_data.datasets.puf.irs_puf import IRS_PUF_2015

    dataset_by_year = {
        2015: IRS_PUF_2015,
    }

    dataset_by_year[year](require=True)


def test_puf_person_split_keeps_capital_gains_holding_period_collapsed():
    row = {
        "long_term_capital_gains": 1_000.0,
        "long_term_capital_gains_years_held": 12.5,
    }

    assert (
        _person_financial_value_from_puf_row(
            "long_term_capital_gains_years_held",
            row,
            0.25,
        )
        == 12.5
    )
    assert (
        _person_financial_value_from_puf_row(
            "long_term_capital_gains_years_held",
            row,
            0.0,
        )
        == 0
    )


def test_puf_load_dataset_backfills_capital_gains_basis_inputs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        puf_module,
        "has_policyengine_us_variables",
        lambda *variables: True,
    )

    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    _write_capital_gains_basis_source_file(DummyPUF.file_path)

    arrays = DummyPUF().load_dataset()

    basis = arrays["long_term_capital_gains_basis"]
    years = arrays["long_term_capital_gains_years_held"]
    gains = arrays["long_term_capital_gains"]

    assert np.all(basis[gains != 0] > 0)
    assert np.all(years[gains != 0] > 0)
    assert np.all(basis[gains == 0] == 0)
    assert np.all(years[gains == 0] == 0)

    with h5py.File(DummyPUF.file_path, "r") as file_handle:
        assert "long_term_capital_gains_basis" in file_handle
        assert "long_term_capital_gains_years_held" in file_handle


def test_puf_load_key_backfills_read_only_capital_gains_basis_inputs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        puf_module,
        "has_policyengine_us_variables",
        lambda *variables: True,
    )

    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    _write_capital_gains_basis_source_file(DummyPUF.file_path)
    DummyPUF.file_path.chmod(0o444)

    dataset = DummyPUF()
    try:
        basis = dataset.load("long_term_capital_gains_basis")
        years = dataset.load("long_term_capital_gains_years_held")
        reader = dataset.load()
        np.testing.assert_array_equal(
            reader["long_term_capital_gains_basis"],
            basis,
        )
        reader.close()
    finally:
        DummyPUF.file_path.chmod(0o644)

    assert np.all(basis[[0, 1, 3]] > 0)
    assert basis[2] == 0
    assert np.all(years[[0, 1, 3]] > 0)
    assert years[2] == 0


def test_puf_load_dataset_backfills_sstb_split_inputs(tmp_path):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset(
            "self_employment_income", data=np.array([100.0, 200.0])
        )
        file_handle.create_dataset(
            "w2_wages_from_qualified_business", data=np.array([10.0, 20.0])
        )
        file_handle.create_dataset(
            "unadjusted_basis_qualified_property", data=np.array([5.0, 6.0])
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([1, 0]))
        _mark_current_qbi_simulation(file_handle)

    dataset = DummyPUF()
    arrays = dataset.load_dataset()

    np.testing.assert_array_equal(
        arrays["self_employment_income"], np.array([0.0, 200.0])
    )
    np.testing.assert_array_equal(
        arrays["sstb_self_employment_income"], np.array([100.0, 0.0])
    )
    np.testing.assert_array_equal(
        arrays["sstb_w2_wages_from_qualified_business"], np.array([10.0, 0.0])
    )
    np.testing.assert_array_equal(
        arrays["sstb_unadjusted_basis_qualified_property"], np.array([5.0, 0.0])
    )


def test_puf_load_key_backfills_sstb_split_inputs(tmp_path):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset(
            "self_employment_income", data=np.array([100.0, 200.0])
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([1, 0]))
        _mark_current_qbi_simulation(file_handle)

    dataset = DummyPUF()

    np.testing.assert_array_equal(
        dataset.load("self_employment_income"), np.array([0.0, 200.0])
    )
    np.testing.assert_array_equal(
        dataset.load("sstb_self_employment_income"), np.array([100.0, 0.0])
    )


def test_puf_load_key_repairs_partially_migrated_sstb_split_inputs(tmp_path):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset(
            "self_employment_income", data=np.array([100.0, 200.0])
        )
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([100.0, 0.0])
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([1, 0]))
        _mark_current_qbi_simulation(file_handle)

    dataset = DummyPUF()

    np.testing.assert_array_equal(
        dataset.load("self_employment_income"), np.array([0.0, 200.0])
    )
    np.testing.assert_array_equal(
        dataset.load("sstb_self_employment_income"), np.array([100.0, 0.0])
    )


def test_puf_load_read_only_backfilled_file_does_not_reopen_for_writes(tmp_path):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset(
            "self_employment_income", data=np.array([0.0, 200.0])
        )
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([100.0, 0.0])
        )
        file_handle.create_dataset(
            "w2_wages_from_qualified_business", data=np.array([10.0, 20.0])
        )
        file_handle.create_dataset(
            "sstb_w2_wages_from_qualified_business", data=np.array([10.0, 0.0])
        )
        file_handle.create_dataset(
            "unadjusted_basis_qualified_property", data=np.array([5.0, 6.0])
        )
        file_handle.create_dataset(
            "sstb_unadjusted_basis_qualified_property",
            data=np.array([5.0, 0.0]),
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([1, 0]))
        _mark_current_qbi_simulation(file_handle)

    DummyPUF.file_path.chmod(0o444)
    dataset = DummyPUF()

    try:
        np.testing.assert_array_equal(
            dataset.load("sstb_self_employment_income"), np.array([100.0, 0.0])
        )
        arrays = dataset.load_dataset()
    finally:
        DummyPUF.file_path.chmod(0o644)

    np.testing.assert_array_equal(
        arrays["sstb_self_employment_income"], np.array([100.0, 0.0])
    )


def test_puf_load_read_only_partially_migrated_file_uses_overrides(tmp_path):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset(
            "self_employment_income", data=np.array([100.0, 200.0])
        )
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([100.0, 0.0])
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([1, 0]))
        _mark_current_qbi_simulation(file_handle)

    DummyPUF.file_path.chmod(0o444)
    dataset = DummyPUF()

    try:
        np.testing.assert_array_equal(
            dataset.load("self_employment_income"), np.array([0.0, 200.0])
        )
        np.testing.assert_array_equal(
            dataset.load("sstb_self_employment_income"), np.array([100.0, 0.0])
        )
        reader = dataset.load()
        np.testing.assert_array_equal(
            reader["self_employment_income"], np.array([0.0, 200.0])
        )
        np.testing.assert_array_equal(
            reader.get("self_employment_income"), np.array([0.0, 200.0])
        )
        np.testing.assert_array_equal(
            dict(reader.items())["self_employment_income"],
            np.array([0.0, 200.0]),
        )
        reader.close()
        arrays = dataset.load_dataset()
    finally:
        DummyPUF.file_path.chmod(0o644)

    np.testing.assert_array_equal(
        arrays["self_employment_income"], np.array([0.0, 200.0])
    )
    np.testing.assert_array_equal(
        arrays["sstb_self_employment_income"], np.array([100.0, 0.0])
    )
