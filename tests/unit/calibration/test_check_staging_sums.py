from policyengine_us_data.calibration.check_staging_sums import (
    get_reference_summary,
)


def test_reference_summary_uses_irs_refundable_ctc_target(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.check_staging_sums.get_national_geography_soi_target",
        lambda variable, reference_year: {
            "amount": 33_000_000_000.0,
            "source_year": 2022,
        },
    )

    summary = get_reference_summary()

    assert "refundable CTC ~$33.0B" in summary
    assert "IRS SOI 2022" in summary
