from policyengine_us_data.calibration.validate_national_h5 import (
    get_reference_values,
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
