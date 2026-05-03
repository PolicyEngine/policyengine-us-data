import copy

import h5py
import numpy as np
import pandas as pd

from policyengine_us_data.datasets.puf import puf as puf_module
from policyengine_us_data.datasets.puf.puf import PUF


def _qbi_frame(n=1):
    data = {source: np.zeros(n, dtype=float) for source in puf_module.QBI_SOURCE_NAMES}
    data.update(
        {
            "E00900": np.zeros(n, dtype=float),
            "E26270": np.zeros(n, dtype=float),
            "E26390": np.zeros(n, dtype=float),
            "E26400": np.zeros(n, dtype=float),
        }
    )
    return pd.DataFrame(data)


def _set_qbi_params(monkeypatch, mutate):
    params = copy.deepcopy(puf_module.QBI_PARAMS)
    mutate(params)
    monkeypatch.setattr(puf_module, "QBI_PARAMS", params)
    return params


def test_add_qbi_qualification_flags_to_puf_persists_source_flags(monkeypatch):
    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 0.0
        params["qbi_qualification_probabilities"]["self_employment_income"] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    puf = puf_module.add_qbi_qualification_flags_to_puf(_qbi_frame(), seed=0)

    assert bool(puf["self_employment_income_would_be_qualified"].iloc[0])
    for source in puf_module.QBI_SOURCE_NAMES:
        flag = puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE[source]
        assert flag in puf
    assert not bool(puf["rental_income_would_be_qualified"].iloc[0])


def test_puf_financial_subset_exports_qbi_qualification_flags():
    expected = {
        *puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE.values(),
        puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG,
    }

    assert expected <= set(puf_module.FINANCIAL_SUBSET)


def test_qualified_qbi_components_use_persisted_flags():
    puf = _qbi_frame()
    puf.loc[0, "self_employment_income"] = 100.0
    puf.loc[0, "rental_income"] = 200.0
    for source in puf_module.QBI_SOURCE_NAMES:
        puf[puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE[source]] = False
    puf["rental_income_would_be_qualified"] = True

    components = puf_module.qualified_qbi_components(puf)

    assert components.loc[0, "self_employment_income"] == 0.0
    assert components.loc[0, "rental_income"] == 200.0


def test_simulate_business_is_sstb_ignores_zero_and_unmapped_sources(monkeypatch):
    def mutate(params):
        for source in params["sstb_prob_map_by_name"]:
            params["sstb_prob_map_by_name"][source] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    puf = _qbi_frame(n=4)
    puf.loc[0, "rental_income"] = 10_000.0
    puf.loc[1, "E00900"] = 10_000.0
    puf.loc[2, "E00900"] = -10_000.0
    puf.loc[3, "E26270"] = 10_000.0

    is_sstb = puf_module.simulate_business_is_sstb(puf, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(is_sstb, np.array([False, True, False, True]))


def test_non_rental_capital_intensive_qbi_can_receive_ubia(monkeypatch):
    def mutate(params):
        for source in params["ubia_simulation"]["capital_intensity_probabilities"]:
            params["ubia_simulation"]["capital_intensity_probabilities"][source] = 0.0
        params["ubia_simulation"]["capital_intensity_probabilities"][
            "self_employment_income"
        ] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    puf = _qbi_frame()
    puf.loc[0, "self_employment_income"] = 10_000.0
    for source in puf_module.QBI_SOURCE_NAMES:
        puf[puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE[source]] = False
    puf["self_employment_income_would_be_qualified"] = True

    _, ubia = puf_module.simulate_w2_and_ubia_from_puf(puf, seed=0, diagnostics=False)

    assert ubia[0] > 0


def test_puf_load_dataset_backfills_qbi_simulation_inputs(tmp_path, monkeypatch):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 1.0
        for source in params["ubia_simulation"]["capital_intensity_probabilities"]:
            params["ubia_simulation"]["capital_intensity_probabilities"][source] = 1.0
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 0.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1, 2]))
        file_handle.create_dataset(
            "self_employment_income", data=np.array([10_000.0, 0.0])
        )
        file_handle.create_dataset(
            "partnership_s_corp_income", data=np.array([0.0, 20_000.0])
        )
        for source in set(puf_module.QBI_SOURCE_NAMES) - {
            "self_employment_income",
            "partnership_s_corp_income",
        }:
            file_handle.create_dataset(source, data=np.zeros(2))

    arrays = DummyPUF().load_dataset()

    for flag in puf_module.QBI_SIMULATION_REQUIRED_VARIABLES:
        assert flag in arrays
    assert "w2_wages_from_qualified_business" in arrays
    assert "unadjusted_basis_qualified_property" in arrays
    np.testing.assert_array_equal(arrays["business_is_sstb"], np.array([False, False]))
    assert np.all(arrays["unadjusted_basis_qualified_property"] > 0)
