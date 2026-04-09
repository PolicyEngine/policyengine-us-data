from policyengine_us_data.calibration.validate_national_h5 import (
    get_reference_values,
)


def test_reference_values_use_irs_refundable_ctc_target(monkeypatch):
    monkeypatch.setattr(
        "policyengine_us_data.calibration.validate_national_h5.get_national_geography_soi_target",
        lambda variable, reference_year: {
            "amount": 33_000_000_000.0,
            "source_year": 2022,
        },
    )

    references = get_reference_values()

    assert references["refundable_ctc"] == (
        33_000_000_000.0,
        "IRS SOI 2022 $33.0B",
    )
