from pathlib import Path

import pytest

from policyengine_us_data.datasets.cps.enhanced_cps import (
    build_clone_diagnostics_payload,
    compute_clone_diagnostics_summary,
    clone_diagnostics_path,
    refresh_clone_diagnostics_report,
    save_clone_diagnostics_report,
)


def test_compute_clone_diagnostics_summary():
    diagnostics = compute_clone_diagnostics_summary(
        household_is_puf_clone=[False, True],
        household_weight=[9.0, 1.0],
        person_is_puf_clone=[False, True, True],
        person_weight=[4.0, 3.0, 3.0],
        person_in_poverty=[False, True, True],
        person_reported_in_poverty=[False, False, True],
        spm_unit_is_puf_clone=[False, True, True],
        spm_unit_weight=[2.0, 3.0, 5.0],
        spm_unit_capped_work_childcare_expenses=[0.0, 6000.0, 7000.0],
        spm_unit_pre_subsidy_childcare_expenses=[0.0, 5000.0, 8000.0],
        spm_unit_taxes=[100.0, 9000.0, 200.0],
        spm_unit_market_income=[1000.0, 8000.0, 1000.0],
    )

    assert diagnostics["clone_household_weight_share_pct"] == pytest.approx(10.0)
    assert diagnostics[
        "clone_poor_modeled_only_person_weight_share_pct"
    ] == pytest.approx(30.0)
    assert diagnostics[
        "poor_modeled_only_within_clone_person_weight_share_pct"
    ] == pytest.approx(50.0)
    assert diagnostics[
        "clone_childcare_exceeds_pre_subsidy_share_pct"
    ] == pytest.approx(37.5)
    assert diagnostics["clone_childcare_above_5000_share_pct"] == pytest.approx(100.0)
    assert diagnostics["clone_taxes_exceed_market_income_share_pct"] == pytest.approx(
        37.5
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
        lambda dataset_cls, period: {"clone_person_weight_share_pct": float(period)},
    )

    output_path, payload = save_clone_diagnostics_report(
        DummyDataset,
        start_year=2024,
        end_year=2025,
    )

    assert output_path == clone_diagnostics_path(DummyDataset.file_path)
    assert payload == {
        "periods": {
            "2024": {"clone_person_weight_share_pct": 2024.0},
            "2025": {"clone_person_weight_share_pct": 2025.0},
        }
    }
    assert output_path.exists()
