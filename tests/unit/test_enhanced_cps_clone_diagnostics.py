from pathlib import Path

import numpy as np
import pytest

from policyengine_us_data.datasets.cps.enhanced_cps import (
    build_clone_diagnostics_for_simulation,
    build_clone_diagnostics_payload,
    compute_clone_diagnostics_summary,
    clone_diagnostics_path,
    initialize_weight_priors,
    PUF_CLONE_PRIOR_TOTAL_SHARE,
    refresh_clone_diagnostics_report,
    save_clone_diagnostics_report,
    validate_clone_diagnostics,
)


def test_initialize_weight_priors_gives_zero_weight_records_support_mass():
    weights = np.array([1_500.0, 0.0, 625.0, 0.0], dtype=np.float64)

    priors = initialize_weight_priors(weights, seed=123)

    assert np.all(priors > 0)
    assert priors.sum() == pytest.approx(weights.sum())
    assert priors[[1, 3]].sum() == pytest.approx(
        weights.sum() * PUF_CLONE_PRIOR_TOTAL_SHARE
    )
    assert priors[[0, 2]].sum() == pytest.approx(
        weights.sum() * (1 - PUF_CLONE_PRIOR_TOTAL_SHARE)
    )
    assert priors[1] == pytest.approx(priors[3])
    assert priors[0] / priors[2] == pytest.approx(weights[0] / weights[2])


def test_initialize_weight_priors_preserves_positive_weights_exactly():
    weights = np.array([1_500.0, 625.0, 42.0], dtype=np.float64)

    priors = initialize_weight_priors(weights, seed=123)

    np.testing.assert_array_equal(priors, weights)


def test_initialize_weight_priors_is_reproducible():
    weights = np.array([400.0, 0.0, 100.0], dtype=np.float64)

    priors_a = initialize_weight_priors(weights, seed=77)
    priors_b = initialize_weight_priors(weights, seed=77)

    np.testing.assert_allclose(priors_a, priors_b)


def test_initialize_weight_priors_honors_configured_zero_weight_share():
    weights = np.array([80.0, 20.0, 0.0, 0.0])

    priors = initialize_weight_priors(weights, zero_weight_total_share=0.5)

    np.testing.assert_allclose(priors.sum(), 100.0)
    np.testing.assert_allclose(priors, np.array([40.0, 10.0, 25.0, 25.0]))


def test_compute_clone_diagnostics_summary():
    diagnostics = compute_clone_diagnostics_summary(
        household_is_puf_clone=[False, True],
        household_weight=[9.0, 1.0],
        person_is_puf_clone=[False, True, True],
        person_weight=[4.0, 3.0, 3.0],
        person_in_poverty=[False, True, True],
        spm_unit_is_puf_clone=[False, True, True],
        spm_unit_weight=[2.0, 3.0, 5.0],
        spm_unit_capped_work_childcare_expenses=[0.0, 6000.0, 7000.0],
        spm_unit_pre_subsidy_childcare_expenses=[0.0, 5000.0, 8000.0],
        spm_unit_taxes=[100.0, 9000.0, 200.0],
        spm_unit_market_income=[1000.0, 8000.0, 1000.0],
    )

    assert diagnostics["clone_household_weight_share_pct"] == pytest.approx(10.0)
    assert diagnostics["clone_poor_person_weight_share_pct"] == pytest.approx(60.0)
    assert diagnostics[
        "clone_childcare_exceeds_pre_subsidy_share_pct"
    ] == pytest.approx(37.5)
    assert diagnostics["clone_childcare_above_5000_share_pct"] == pytest.approx(100.0)
    assert diagnostics["clone_taxes_exceed_market_income_share_pct"] == pytest.approx(
        37.5
    )


def test_validate_clone_diagnostics_accepts_support_clone_share():
    validate_clone_diagnostics(
        {
            "clone_household_weight_share_pct": 10.0,
            "clone_taxes_exceed_market_income_share_pct": 5.0,
        }
    )


def test_validate_clone_diagnostics_rejects_clone_starvation():
    with pytest.raises(ValueError, match="floor"):
        validate_clone_diagnostics(
            {
                "clone_household_weight_share_pct": 2.0,
                "clone_taxes_exceed_market_income_share_pct": 5.0,
            }
        )


def test_validate_clone_diagnostics_accepts_high_share_no_cap():
    # No upper cap on clone weight share (the household-count loss target governs
    # it); a high share with healthy tax quality must pass.
    validate_clone_diagnostics(
        {
            "clone_household_weight_share_pct": 81.3,
            "clone_taxes_exceed_market_income_share_pct": 5.0,
        }
    )


def test_validate_clone_diagnostics_rejects_clone_tax_pathology():
    with pytest.raises(
        ValueError,
        match="PUF clone taxes-exceed-market-income share",
    ):
        validate_clone_diagnostics(
            {
                "clone_household_weight_share_pct": 10.0,
                "clone_taxes_exceed_market_income_share_pct": 66.6,
            }
        )


def test_build_clone_diagnostics_for_simulation_maps_household_weights(
    monkeypatch,
):
    class FakeResult:
        def __init__(self, values):
            self.values = np.asarray(values)

    class FakeSim:
        def calculate(self, variable, period=None, map_to=None):
            lookup = {
                ("household_weight", None): [9.0, 1.0],
                ("household_weight", "person"): [9.0, 1.0, 1.0],
                ("household_weight", "spm_unit"): [9.0, 1.0],
                # Trap values: diagnostics should not read these stale inputs.
                ("person_weight", None): [9.0, 0.0, 0.0],
                ("spm_unit_weight", None): [9.0, 0.0],
                ("person_in_poverty", None): [False, True, True],
                ("spm_unit_capped_work_childcare_expenses", None): [0.0, 6000.0],
                ("spm_unit_pre_subsidy_childcare_expenses", None): [0.0, 5000.0],
                ("spm_unit_taxes", None): [100.0, 9000.0],
                ("spm_unit_market_income", None): [1000.0, 8000.0],
            }
            return FakeResult(lookup[(variable, map_to)])

    saved_arrays = {
        "household_is_puf_clone": np.array([False, True]),
        "person_is_puf_clone": np.array([False, True, True]),
        "spm_unit_is_puf_clone": np.array([False, True]),
    }

    monkeypatch.setattr(
        "policyengine_us_data.datasets.cps.enhanced_cps._load_saved_period_array",
        lambda dataset_path, variable_name, period: saved_arrays[variable_name],
    )

    diagnostics = build_clone_diagnostics_for_simulation(
        FakeSim(),
        dataset_path=Path("enhanced_cps_2024.h5"),
        period=2024,
    )

    assert diagnostics["clone_household_weight_share_pct"] == pytest.approx(10.0)
    assert diagnostics["clone_person_weight_share_pct"] == pytest.approx(200.0 / 11.0)
    assert diagnostics["clone_poor_person_weight_share_pct"] == pytest.approx(
        200.0 / 11.0
    )
    assert diagnostics[
        "clone_childcare_exceeds_pre_subsidy_share_pct"
    ] == pytest.approx(100.0)
    assert diagnostics["clone_childcare_above_5000_share_pct"] == pytest.approx(100.0)
    assert diagnostics["clone_taxes_exceed_market_income_share_pct"] == pytest.approx(
        100.0
    )


def test_build_clone_diagnostics_payload_single_period():
    payload = build_clone_diagnostics_payload(
        {2024: {"clone_person_weight_share_pct": 12.5}}
    )

    assert payload == {
        "period": 2024,
        "clone_person_weight_share_pct": 12.5,
    }


def test_build_clone_diagnostics_payload_multiple_periods():
    payload = build_clone_diagnostics_payload(
        {
            2026: {"clone_person_weight_share_pct": 20.0},
            2024: {"clone_person_weight_share_pct": 10.0},
        }
    )

    assert payload == {
        "periods": {
            "2024": {"clone_person_weight_share_pct": 10.0},
            "2026": {"clone_person_weight_share_pct": 20.0},
        }
    }


def test_refresh_clone_diagnostics_report_removes_stale_sidecar_on_failure(tmp_path):
    file_path = tmp_path / "enhanced_cps_2024.h5"
    file_path.write_text("placeholder")
    stale_path = clone_diagnostics_path(file_path)
    stale_path.write_text("stale")

    def _raise():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        refresh_clone_diagnostics_report(file_path, _raise)

    assert stale_path == Path(file_path).with_suffix(".clone_diagnostics.json")
    assert not stale_path.exists()


def test_save_clone_diagnostics_report_removes_stale_sidecar_on_failure(
    tmp_path, monkeypatch
):
    class DummyDataset:
        file_path = tmp_path / "enhanced_cps_2024.h5"

    DummyDataset.file_path.write_text("placeholder")
    stale_path = clone_diagnostics_path(DummyDataset.file_path)
    stale_path.write_text("stale")

    monkeypatch.setattr(
        "policyengine_us_data.datasets.cps.enhanced_cps.build_clone_diagnostics_for_saved_dataset",
        lambda dataset_cls, period: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        save_clone_diagnostics_report(
            DummyDataset,
            start_year=2024,
            end_year=2024,
        )

    assert not stale_path.exists()


def test_save_clone_diagnostics_report_writes_fresh_payload(tmp_path, monkeypatch):
    class DummyDataset:
        file_path = tmp_path / "enhanced_cps_2024.h5"

    DummyDataset.file_path.write_text("placeholder")

    monkeypatch.setattr(
        "policyengine_us_data.datasets.cps.enhanced_cps.build_clone_diagnostics_for_saved_dataset",
        lambda dataset_cls, period: {
            "clone_person_weight_share_pct": float(period),
            "clone_household_weight_share_pct": 10.0,
            "clone_taxes_exceed_market_income_share_pct": 5.0,
        },
    )

    output_path, payload = save_clone_diagnostics_report(
        DummyDataset,
        start_year=2024,
        end_year=2025,
    )

    assert output_path == clone_diagnostics_path(DummyDataset.file_path)
    assert payload == {
        "periods": {
            "2024": {
                "clone_person_weight_share_pct": 2024.0,
                "clone_household_weight_share_pct": 10.0,
                "clone_taxes_exceed_market_income_share_pct": 5.0,
            },
            "2025": {
                "clone_person_weight_share_pct": 2025.0,
                "clone_household_weight_share_pct": 10.0,
                "clone_taxes_exceed_market_income_share_pct": 5.0,
            },
        }
    }
    assert output_path.exists()


def test_save_clone_diagnostics_report_rejects_bad_clone_payload(tmp_path, monkeypatch):
    class DummyDataset:
        file_path = tmp_path / "enhanced_cps_2024.h5"

    DummyDataset.file_path.write_text("placeholder")

    monkeypatch.setattr(
        "policyengine_us_data.datasets.cps.enhanced_cps.build_clone_diagnostics_for_saved_dataset",
        lambda dataset_cls, period: {
            "clone_person_weight_share_pct": 1.0,
            "clone_household_weight_share_pct": 2.0,
            "clone_taxes_exceed_market_income_share_pct": 5.0,
        },
    )

    with pytest.raises(ValueError, match="PUF clone household weight share"):
        save_clone_diagnostics_report(
            DummyDataset,
            start_year=2024,
            end_year=2024,
        )
