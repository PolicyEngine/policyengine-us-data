from policyengine_us_data.calibration.check_staging_sums import (
    get_reference_summary,
)


def test_reference_summary_uses_irs_ctc_component_targets(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.check_staging_sums.get_national_geography_soi_target",
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

    summary = get_reference_summary()

    assert "refundable CTC ~$33.0B" in summary
    assert "non-refundable CTC ~$81.6B" in summary
    assert "IRS SOI 2022" in summary
