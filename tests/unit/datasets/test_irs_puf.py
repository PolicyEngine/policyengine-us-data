import h5py
import numpy as np
import pytest

from policyengine_us_data.datasets.puf.puf import PUF


@pytest.mark.skip(reason="This test requires private data.")
@pytest.mark.parametrize("year", [2015])
def test_irs_puf_generates(year: int):
    from policyengine_us_data.datasets.puf.irs_puf import IRS_PUF_2015

    dataset_by_year = {
        2015: IRS_PUF_2015,
    }

    dataset_by_year[year](require=True)


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
