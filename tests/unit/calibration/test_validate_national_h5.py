import os

import pandas as pd

from policyengine_us_data.calibration.validate_national_h5 import (
    build_canonical_ctc_reform_summary,
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


def test_ctc_diagnostic_outputs_format_all_sections(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.validate_national_h5.create_ctc_diagnostic_tables",
        lambda sim: {
            "by_agi_band": "agi_table",
            "by_filing_status": "filing_status_table",
            "by_agi_band_and_filing_status": "agi_filing_table",
            "by_child_count": "child_count_table",
            "by_child_age": "child_age_table",
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
        "CTC DIAGNOSTICS BY AGI BAND AND FILING STATUS": "formatted:agi_filing_table",
        "CTC DIAGNOSTICS BY QUALIFYING-CHILD COUNT": "formatted:child_count_table",
        "CTC DIAGNOSTICS BY QUALIFYING-CHILD AGE": "formatted:child_age_table",
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


class _FakeArrayResult:
    def __init__(self, values):
        self._values = values

    @property
    def values(self):
        return self._values

    def sum(self):
        return self._values.sum()


class _FakeSummarySim:
    def __init__(self, values_by_variable):
        self.values_by_variable = values_by_variable

    def calculate(self, variable, period=None, map_to=None):
        assert map_to is None
        return _FakeArrayResult(self.values_by_variable[variable])


def test_build_canonical_ctc_reform_summary_reports_level_and_delta():
    baseline = _FakeSummarySim(
        {
            "ctc_value": pd.Series([100.0, 50.0]).to_numpy(),
            "ctc": pd.Series([90.0]).to_numpy(),
            "refundable_ctc": pd.Series([40.0]).to_numpy(),
            "non_refundable_ctc": pd.Series([50.0]).to_numpy(),
            "eitc": pd.Series([20.0]).to_numpy(),
            "household_net_income": pd.Series([500.0, 200.0]).to_numpy(),
        }
    )
    reformed = _FakeSummarySim(
        {
            "ctc_value": pd.Series([130.0, 70.0]).to_numpy(),
            "ctc": pd.Series([120.0]).to_numpy(),
            "refundable_ctc": pd.Series([70.0]).to_numpy(),
            "non_refundable_ctc": pd.Series([50.0]).to_numpy(),
            "eitc": pd.Series([35.0]).to_numpy(),
            "household_net_income": pd.Series([540.0, 215.0]).to_numpy(),
        }
    )

    summary = build_canonical_ctc_reform_summary(
        baseline,
        reformed,
        period=2025,
    ).set_index("variable")

    assert summary.loc["ctc_value", "baseline"] == 150.0
    assert summary.loc["ctc_value", "reformed"] == 200.0
    assert summary.loc["ctc_value", "delta"] == 50.0
    assert summary.loc["refundable_ctc", "delta"] == 30.0
    assert summary.loc["non_refundable_ctc", "delta"] == 0.0
    assert summary.loc["household_net_income", "delta"] == 55.0
