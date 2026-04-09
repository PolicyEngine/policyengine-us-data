import os

from policyengine_us_data.calibration.validate_national_h5 import (
    get_ctc_diagnostic_outputs,
    get_reference_values,
    resolve_dataset_path,
)


def test_reference_values_use_irs_ctc_component_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.validate_national_h5.get_national_geography_soi_target",
        lambda variable, reference_year: {
            "refundable_ctc": {
                "amount": 33_000_000_000.0,
                "source_year": 2022,
            },
            "non_refundable_ctc": {
                "amount": 81_600_000_000.0,
                "source_year": 2022,
            },
        }[variable],
    )

    references = get_reference_values()

    assert references["refundable_ctc"] == (
        33_000_000_000.0,
        "IRS SOI 2022 $33.0B",
    )
    assert references["non_refundable_ctc"] == (
        81_600_000_000.0,
        "IRS SOI 2022 $81.6B",
    )


def test_ctc_diagnostic_outputs_format_both_sections(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.validate_national_h5.create_ctc_diagnostic_tables",
        lambda sim: {
            "by_agi_band": "agi_table",
            "by_filing_status": "filing_status_table",
        },
    )
    monkeypatch.setattr(
        "policyengine_us_data.calibration.validate_national_h5.format_ctc_diagnostic_table",
        lambda table: f"formatted:{table}",
    )

    outputs = get_ctc_diagnostic_outputs(sim=object())

    assert outputs == {
        "CTC DIAGNOSTICS BY AGI BAND": "formatted:agi_table",
        "CTC DIAGNOSTICS BY FILING STATUS": "formatted:filing_status_table",
    }


def test_resolve_dataset_path_passes_through_local_paths():
    assert resolve_dataset_path("/tmp/national.h5") == "/tmp/national.h5"


def test_resolve_dataset_path_downloads_hf_paths(monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/downloaded.h5"

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        fake_download,
    )

    result = resolve_dataset_path(
        "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
    )

    assert result == "/tmp/downloaded.h5"
    assert calls == [
        {
            "repo_id": "policyengine/policyengine-us-data",
            "filename": "enhanced_cps_2024.h5",
            "repo_type": "model",
            "token": os.environ.get("HUGGING_FACE_TOKEN"),
        }
    ]
