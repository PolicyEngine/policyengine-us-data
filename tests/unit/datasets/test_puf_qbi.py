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


def test_draw_source_weighted_beta_uses_positive_qbi_weights():
    qbi_components = pd.DataFrame(
        {
            "self_employment_income": [100.0],
            "rental_income": [300.0],
        }
    )
    params = {
        "self_employment_income": {
            "beta_a": 1.0,
            "beta_b": 1.0,
            "scale": 0.0,
            "shift": 0.10,
        },
        "rental_income": {
            "beta_a": 1.0,
            "beta_b": 1.0,
            "scale": 0.0,
            "shift": 0.50,
        },
    }

    draw = puf_module.draw_source_weighted_beta(
        qbi_components, params, np.random.default_rng(0)
    )

    np.testing.assert_allclose(draw, np.array([0.40]))


def test_calibrate_logit_intercept_matches_positive_receipt_target():
    revenues = np.array([0.0, 10_000.0, 100_000.0, 1_000_000.0])
    target_share = 0.25

    intercept = puf_module.calibrate_logit_intercept(
        revenues, slope=1e-6, target_share=target_share
    )

    probabilities = puf_module.logistic(intercept + 1e-6 * revenues[revenues > 0])
    np.testing.assert_allclose(probabilities.mean(), target_share, atol=1e-12)


def test_calibrate_logit_intercept_handles_large_receipts():
    revenues = np.array([100_000_000.0, 200_000_000.0])
    slope = 1.2e-6
    target_share = 0.18

    intercept = puf_module.calibrate_logit_intercept(
        revenues, slope=slope, target_share=target_share
    )

    probabilities = puf_module.logistic(intercept + slope * revenues)
    np.testing.assert_allclose(probabilities.mean(), target_share, atol=1e-12)


def test_simulate_business_is_sstb_ignores_zero_and_unmapped_sources(monkeypatch):
    def mutate(params):
        for source in params["sstb_prob_map_by_name"]:
            params["sstb_prob_map_by_name"][source] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    assert "E26400" not in puf_module.QBI_PARAMS["sstb_prob_map_by_name"]
    puf = _qbi_frame(n=4)
    puf.loc[0, "rental_income"] = 10_000.0
    puf.loc[1, "E00900"] = 10_000.0
    puf.loc[2, "E00900"] = -10_000.0
    puf.loc[3, "E26270"] = 10_000.0

    is_sstb = puf_module.simulate_business_is_sstb(puf, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(is_sstb, np.array([False, True, False, True]))


def test_simulate_business_is_sstb_source_map_ignores_estate_loss_column(
    monkeypatch,
):
    def mutate(params):
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    puf = _qbi_frame(n=2)
    puf.loc[0, "rental_income"] = 10_000.0
    puf.loc[0, "E26400"] = 10_000.0
    puf.loc[1, "estate_income"] = 10_000.0

    is_sstb = puf_module.simulate_business_is_sstb(
        puf,
        rng=np.random.default_rng(0),
        probability_map=puf_module.QBI_PARAMS["sstb_prob_map_by_source_name"],
    )

    np.testing.assert_array_equal(is_sstb, np.array([False, True]))


def test_simulate_business_is_sstb_ignores_unqualified_sources(monkeypatch):
    def mutate(params):
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    puf = _qbi_frame(n=2)
    puf.loc[0, "self_employment_income"] = 10_000.0
    puf.loc[0, "rental_income"] = 20_000.0
    puf.loc[1, "self_employment_income"] = 10_000.0
    puf.loc[1, "rental_income"] = 20_000.0
    for source in puf_module.QBI_SOURCE_NAMES:
        puf[puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE[source]] = False
    puf.loc[:, "rental_income_would_be_qualified"] = True
    puf.loc[1, "self_employment_income_would_be_qualified"] = True

    is_sstb = puf_module.simulate_business_is_sstb(
        puf,
        rng=np.random.default_rng(0),
        probability_map=puf_module.QBI_PARAMS["sstb_prob_map_by_source_name"],
    )

    np.testing.assert_array_equal(is_sstb, np.array([False, True]))


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


def test_investment_qbi_is_scaled_to_observed_exposures(monkeypatch):
    def mutate(params):
        params["reit_ptp_income_distribution"] = {
            "non_qualified_dividend_income": {
                "probability_of_receiving": 1.0,
                "beta_a": 1.0,
                "beta_b": 1.0,
                "scale": 0.0,
                "shift": 0.20,
            },
            "partnership_s_corp_income": {
                "probability_of_receiving": 1.0,
                "beta_a": 1.0,
                "beta_b": 1.0,
                "scale": 0.0,
                "shift": 0.30,
            },
        }
        params["bdc_income_distribution"] = {
            "non_qualified_dividend_income": {
                "probability_of_receiving": 1.0,
                "beta_a": 1.0,
                "beta_b": 1.0,
                "scale": 0.0,
                "shift": 0.05,
            }
        }

    _set_qbi_params(monkeypatch, mutate)
    puf = pd.DataFrame(
        {
            "qualified_dividend_income": [900.0, 0.0, 0.0],
            "non_qualified_dividend_income": [100.0, 0.0, 0.0],
            "partnership_income": [0.0, 150.0, 0.0],
            "s_corp_income": [0.0, 50.0, 0.0],
        }
    )

    investment_qbi = puf_module.simulate_investment_qbi_income_from_puf(
        puf, rng=np.random.default_rng(0)
    )

    np.testing.assert_allclose(
        investment_qbi["qualified_reit_and_ptp_income"],
        np.array([20.0, 60.0, 0.0]),
    )
    np.testing.assert_allclose(
        investment_qbi["qualified_bdc_income"],
        np.array([5.0, 0.0, 0.0]),
    )


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
        params["reit_ptp_income_distribution"] = {
            "partnership_s_corp_income": {
                "probability_of_receiving": 1.0,
                "beta_a": 1.0,
                "beta_b": 1.0,
                "scale": 0.0,
                "shift": 0.25,
            }
        }
        params["bdc_income_distribution"] = {}

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1, 2]))
        file_handle.create_dataset(
            "self_employment_income", data=np.array([10_000.0, 0.0])
        )
        file_handle.create_dataset("partnership_income", data=np.array([0.0, 12_000.0]))
        file_handle.create_dataset("s_corp_income", data=np.array([0.0, 8_000.0]))
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
    assert "qualified_reit_and_ptp_income" in arrays
    assert "qualified_bdc_income" in arrays
    np.testing.assert_array_equal(arrays["business_is_sstb"], np.array([False, False]))
    assert np.all(arrays["unadjusted_basis_qualified_property"] > 0)
    np.testing.assert_allclose(
        arrays["qualified_reit_and_ptp_income"], np.array([0.0, 5_000.0])
    )


def test_puf_load_dataset_repairs_qbi_with_person_level_length(tmp_path, monkeypatch):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 1.0
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 0.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("person_id", data=np.array([101, 102]))
        file_handle.create_dataset(
            "self_employment_income", data=np.array([10_000.0, 0.0])
        )
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(2))

    arrays = DummyPUF().load_dataset()

    assert len(arrays["self_employment_income_would_be_qualified"]) == 2
    assert len(arrays["business_is_sstb"]) == 2
    np.testing.assert_array_equal(
        arrays["self_employment_income_would_be_qualified"],
        np.array([True, True]),
    )


def test_puf_save_current_qbi_dataset_marks_version(tmp_path):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

        def save_dataset(self, arrays):
            with h5py.File(self.file_path, "w") as file_handle:
                for key, values in arrays.items():
                    file_handle.create_dataset(key, data=values)

    DummyPUF()._save_current_qbi_dataset(
        {
            "person_id": np.array([101]),
            "self_employment_income": np.array([10_000.0]),
        }
    )

    with h5py.File(DummyPUF.file_path, "r") as file_handle:
        assert (
            file_handle.attrs[puf_module.QBI_SIMULATION_VERSION_ATTR]
            == puf_module.QBI_SIMULATION_VERSION
        )


def test_puf_load_dataset_repairs_partially_migrated_qbi_outputs(tmp_path, monkeypatch):
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
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("self_employment_income", data=np.array([10_000.0]))
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(1))
        for flag in puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE.values():
            file_handle.create_dataset(flag, data=np.array([True]))
        file_handle.create_dataset(
            puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG,
            data=np.array([False]),
        )
        file_handle.create_dataset(
            "qualified_reit_and_ptp_income", data=np.array([0.0])
        )
        file_handle.create_dataset("qualified_bdc_income", data=np.array([0.0]))

    arrays = DummyPUF().load_dataset()

    assert "w2_wages_from_qualified_business" in arrays
    assert "unadjusted_basis_qualified_property" in arrays
    assert "business_is_sstb" in arrays
    assert "sstb_w2_wages_from_qualified_business" in arrays
    assert "sstb_unadjusted_basis_qualified_property" in arrays


def test_puf_load_dataset_preserves_existing_qbi_qualification_flags(tmp_path):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    stored_flags = {}
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1, 2]))
        file_handle.create_dataset(
            "self_employment_income", data=np.array([0.0, 20_000.0])
        )
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([10_000.0, 0.0])
        )
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(2))
        for index, flag in enumerate(
            puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE.values()
        ):
            values = np.array([index % 2 == 0, index % 2 == 1])
            stored_flags[flag] = values
            file_handle.create_dataset(flag, data=values)
        stored_flags[puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG] = np.array(
            [True, False]
        )
        file_handle.create_dataset(
            puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG,
            data=stored_flags[puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG],
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([True, False]))
        file_handle.create_dataset(
            "w2_wages_from_qualified_business", data=np.array([10.0, 20.0])
        )
        file_handle.create_dataset(
            "unadjusted_basis_qualified_property", data=np.array([5.0, 6.0])
        )
        file_handle.create_dataset(
            "sstb_w2_wages_from_qualified_business", data=np.array([10.0, 0.0])
        )
        file_handle.create_dataset(
            "sstb_unadjusted_basis_qualified_property", data=np.array([5.0, 0.0])
        )
        file_handle.create_dataset(
            "qualified_reit_and_ptp_income", data=np.array([123.0, 456.0])
        )
        file_handle.attrs[puf_module.QBI_SIMULATION_VERSION_ATTR] = (
            puf_module.QBI_SIMULATION_VERSION
        )

    arrays = DummyPUF().load_dataset()

    assert "qualified_bdc_income" in arrays
    np.testing.assert_array_equal(
        arrays["qualified_reit_and_ptp_income"], np.array([123.0, 456.0])
    )
    for flag, values in stored_flags.items():
        np.testing.assert_array_equal(arrays[flag], values)


def test_puf_load_dataset_backfills_missing_sstb_self_employment_flag(
    tmp_path, monkeypatch
):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 0.0
        params["qbi_qualification_probabilities"]["self_employment_income"] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("self_employment_income", data=np.array([0.0]))
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([10_000.0])
        )
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(1))
        for flag in puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE.values():
            file_handle.create_dataset(flag, data=np.array([False]))
        file_handle.create_dataset("business_is_sstb", data=np.array([True]))
        file_handle.create_dataset(
            "w2_wages_from_qualified_business", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "unadjusted_basis_qualified_property", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "sstb_w2_wages_from_qualified_business", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "sstb_unadjusted_basis_qualified_property", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "qualified_reit_and_ptp_income", data=np.array([0.0])
        )
        file_handle.create_dataset("qualified_bdc_income", data=np.array([0.0]))
        file_handle.attrs[puf_module.QBI_SIMULATION_VERSION_ATTR] = (
            puf_module.QBI_SIMULATION_VERSION
        )

    arrays = DummyPUF().load_dataset()

    np.testing.assert_array_equal(
        arrays["self_employment_income_would_be_qualified"], np.array([False])
    )
    np.testing.assert_array_equal(
        arrays[puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG],
        np.array([True]),
    )


def test_puf_load_dataset_moves_self_employment_flags_with_current_sstb_split(
    tmp_path, monkeypatch
):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 0.0
        params["qbi_qualification_probabilities"]["self_employment_income"] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1, 2]))
        file_handle.create_dataset(
            "self_employment_income", data=np.array([10_000.0, 20_000.0])
        )
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([0.0, 0.0])
        )
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(2))
        for flag in puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE.values():
            file_handle.create_dataset(
                flag,
                data=np.array(
                    [flag == "self_employment_income_would_be_qualified", False]
                ),
            )
        file_handle.create_dataset(
            puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG,
            data=np.array([False, False]),
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([True, False]))
        file_handle.create_dataset(
            "w2_wages_from_qualified_business", data=np.array([0.0, 0.0])
        )
        file_handle.create_dataset(
            "unadjusted_basis_qualified_property", data=np.array([0.0, 0.0])
        )
        file_handle.create_dataset(
            "sstb_w2_wages_from_qualified_business", data=np.array([0.0, 0.0])
        )
        file_handle.create_dataset(
            "sstb_unadjusted_basis_qualified_property", data=np.array([0.0, 0.0])
        )
        file_handle.create_dataset(
            "qualified_reit_and_ptp_income", data=np.array([0.0, 0.0])
        )
        file_handle.create_dataset("qualified_bdc_income", data=np.array([0.0, 0.0]))
        file_handle.attrs[puf_module.QBI_SIMULATION_VERSION_ATTR] = (
            puf_module.QBI_SIMULATION_VERSION
        )

    arrays = DummyPUF().load_dataset()

    np.testing.assert_allclose(
        arrays["self_employment_income"], np.array([0.0, 20_000.0])
    )
    np.testing.assert_allclose(
        arrays["sstb_self_employment_income"], np.array([10_000.0, 0.0])
    )
    np.testing.assert_array_equal(
        arrays["self_employment_income_would_be_qualified"],
        np.array([False, False]),
    )
    np.testing.assert_array_equal(
        arrays[puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG],
        np.array([True, False]),
    )


def test_puf_load_dataset_recomputes_unversioned_qbi_outputs(tmp_path, monkeypatch):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 0.0
        params["qbi_qualification_probabilities"]["rental_income"] = 1.0
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 1.0
        for source in params["ubia_simulation"]["capital_intensity_probabilities"]:
            params["ubia_simulation"]["capital_intensity_probabilities"][source] = 0.0
        params["ubia_simulation"]["capital_intensity_probabilities"][
            "rental_income"
        ] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("self_employment_income", data=np.array([10_000.0]))
        file_handle.create_dataset("rental_income", data=np.array([20_000.0]))
        for source in set(puf_module.QBI_SOURCE_NAMES) - {
            "self_employment_income",
            "rental_income",
        }:
            file_handle.create_dataset(source, data=np.zeros(1))
        file_handle.create_dataset("business_is_sstb", data=np.array([True]))
        file_handle.create_dataset(
            "w2_wages_from_qualified_business", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "unadjusted_basis_qualified_property", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "qualified_reit_and_ptp_income", data=np.array([0.0])
        )

    arrays = DummyPUF().load_dataset()

    np.testing.assert_array_equal(arrays["business_is_sstb"], np.array([False]))
    assert arrays["unadjusted_basis_qualified_property"][0] > 0
    with h5py.File(DummyPUF.file_path, "r") as file_handle:
        assert (
            file_handle.attrs[puf_module.QBI_SIMULATION_VERSION_ATTR]
            == puf_module.QBI_SIMULATION_VERSION
        )


def test_puf_load_dataset_refreshes_unversioned_stale_self_employment_flags(
    tmp_path, monkeypatch
):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 0.0
        params["qbi_qualification_probabilities"]["self_employment_income"] = 1.0
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 0.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("self_employment_income", data=np.array([10_000.0]))
        file_handle.create_dataset("sstb_self_employment_income", data=np.array([0.0]))
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(1))
        for flag in puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE.values():
            file_handle.create_dataset(flag, data=np.array([False]))
        file_handle.create_dataset(
            puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG,
            data=np.array([False]),
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([False]))

    arrays = DummyPUF().load_dataset()

    np.testing.assert_array_equal(arrays["business_is_sstb"], np.array([False]))
    np.testing.assert_array_equal(
        arrays["self_employment_income_would_be_qualified"], np.array([True])
    )
    np.testing.assert_array_equal(
        arrays[puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG],
        np.array([False]),
    )


def test_puf_load_dataset_refreshes_unversioned_self_employment_qbi_flags(
    tmp_path, monkeypatch
):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 0.0
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 0.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("self_employment_income", data=np.array([0.0]))
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([10_000.0])
        )
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(1))
        for flag in puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE.values():
            file_handle.create_dataset(flag, data=np.array([False]))
        file_handle.create_dataset(
            puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG,
            data=np.array([True]),
        )
        file_handle.create_dataset("business_is_sstb", data=np.array([True]))
        file_handle.create_dataset(
            "w2_wages_from_qualified_business", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "unadjusted_basis_qualified_property", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "sstb_w2_wages_from_qualified_business", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "sstb_unadjusted_basis_qualified_property", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "qualified_reit_and_ptp_income", data=np.array([0.0])
        )
        file_handle.create_dataset("qualified_bdc_income", data=np.array([0.0]))

    arrays = DummyPUF().load_dataset()

    np.testing.assert_array_equal(arrays["business_is_sstb"], np.array([False]))
    np.testing.assert_allclose(arrays["self_employment_income"], np.array([10_000.0]))
    np.testing.assert_allclose(arrays["sstb_self_employment_income"], np.array([0.0]))
    np.testing.assert_array_equal(
        arrays["self_employment_income_would_be_qualified"], np.array([True])
    )
    np.testing.assert_array_equal(
        arrays[puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG],
        np.array([False]),
    )
    with h5py.File(DummyPUF.file_path, "r") as file_handle:
        assert (
            file_handle.attrs[puf_module.QBI_SIMULATION_VERSION_ATTR]
            == puf_module.QBI_SIMULATION_VERSION
        )


def test_puf_load_dataset_preserves_unversioned_sstb_self_employment_income(
    tmp_path, monkeypatch
):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 0.0
        params["qbi_qualification_probabilities"]["self_employment_income"] = 1.0
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 0.0
        params["sstb_prob_map_by_source_name"]["self_employment_income"] = 1.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("self_employment_income", data=np.array([0.0]))
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([10_000.0])
        )
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(1))
        file_handle.create_dataset("business_is_sstb", data=np.array([False]))

    arrays = DummyPUF().load_dataset()

    np.testing.assert_array_equal(arrays["business_is_sstb"], np.array([True]))
    np.testing.assert_allclose(arrays["self_employment_income"], np.array([0.0]))
    np.testing.assert_allclose(
        arrays["sstb_self_employment_income"], np.array([10_000.0])
    )


def test_puf_load_dataset_refreshes_self_employment_qbi_flags_when_sstb_missing(
    tmp_path, monkeypatch
):
    class DummyPUF(PUF):
        label = "Dummy PUF"
        name = "dummy_puf"
        time_period = 2024
        file_path = tmp_path / "dummy_puf.h5"

    def mutate(params):
        for source in params["qbi_qualification_probabilities"]:
            params["qbi_qualification_probabilities"][source] = 0.0
        for source in params["sstb_prob_map_by_source_name"]:
            params["sstb_prob_map_by_source_name"][source] = 0.0

    _set_qbi_params(monkeypatch, mutate)
    with h5py.File(DummyPUF.file_path, "w") as file_handle:
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("self_employment_income", data=np.array([0.0]))
        file_handle.create_dataset(
            "sstb_self_employment_income", data=np.array([10_000.0])
        )
        for source in set(puf_module.QBI_SOURCE_NAMES) - {"self_employment_income"}:
            file_handle.create_dataset(source, data=np.zeros(1))
        for flag in puf_module.QBI_QUALIFICATION_FLAG_BY_SOURCE.values():
            file_handle.create_dataset(flag, data=np.array([False]))
        file_handle.create_dataset(
            puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG,
            data=np.array([True]),
        )
        file_handle.create_dataset(
            "w2_wages_from_qualified_business", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "unadjusted_basis_qualified_property", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "sstb_w2_wages_from_qualified_business", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "sstb_unadjusted_basis_qualified_property", data=np.array([0.0])
        )
        file_handle.create_dataset(
            "qualified_reit_and_ptp_income", data=np.array([0.0])
        )
        file_handle.create_dataset("qualified_bdc_income", data=np.array([0.0]))
        file_handle.attrs[puf_module.QBI_SIMULATION_VERSION_ATTR] = (
            puf_module.QBI_SIMULATION_VERSION
        )

    arrays = DummyPUF().load_dataset()

    np.testing.assert_array_equal(arrays["business_is_sstb"], np.array([False]))
    np.testing.assert_allclose(arrays["self_employment_income"], np.array([10_000.0]))
    np.testing.assert_allclose(arrays["sstb_self_employment_income"], np.array([0.0]))
    np.testing.assert_array_equal(
        arrays["self_employment_income_would_be_qualified"], np.array([True])
    )
    np.testing.assert_array_equal(
        arrays[puf_module.SSTB_SELF_EMPLOYMENT_QUALIFICATION_FLAG],
        np.array([False]),
    )


def test_puf_load_dataset_repairs_missing_qbi_source_with_full_outputs(
    tmp_path, monkeypatch
):
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
        file_handle.create_dataset("household_id", data=np.array([1]))
        file_handle.create_dataset("self_employment_income", data=np.array([10_000.0]))
        for source in set(puf_module.QBI_SOURCE_NAMES) - {
            "self_employment_income",
            "estate_income",
        }:
            file_handle.create_dataset(source, data=np.zeros(1))

    arrays = DummyPUF().load_dataset()

    for variable in puf_module.QBI_SIMULATION_REQUIRED_VARIABLES:
        assert variable in arrays
    assert "estate_income" not in arrays
    assert arrays["unadjusted_basis_qualified_property"][0] > 0
