import numpy as np
import pandas as pd

from policyengine_us_data.calibration.ctc_diagnostics import (
    _assign_agi_bands,
    _normalize_filing_status,
    build_ctc_diagnostic_tables,
)


def test_assign_agi_bands_uses_irs_boundaries():
    bands = _assign_agi_bands(
        np.array(
            [
                -1.0,
                0.0,
                1.0,
                9_999.99,
                10_000.0,
                24_999.99,
                25_000.0,
                49_999.99,
                50_000.0,
                74_999.99,
                75_000.0,
                99_999.99,
                100_000.0,
                199_999.99,
                200_000.0,
                499_999.99,
                500_000.0,
            ]
        )
    )

    assert list(bands) == [
        "<$1",
        "<$1",
        "$1-$10k",
        "$1-$10k",
        "$10k-$25k",
        "$10k-$25k",
        "$25k-$50k",
        "$25k-$50k",
        "$50k-$75k",
        "$50k-$75k",
        "$75k-$100k",
        "$75k-$100k",
        "$100k-$200k",
        "$100k-$200k",
        "$200k-$500k",
        "$200k-$500k",
        "$500k+",
    ]

    assert list(bands.categories) == [
        "<$1",
        "$1-$10k",
        "$10k-$25k",
        "$25k-$50k",
        "$50k-$75k",
        "$75k-$100k",
        "$100k-$200k",
        "$200k-$500k",
        "$500k+",
    ]
    assert bands.ordered is True


def test_normalize_filing_status_collapses_joint_labels():
    statuses = _normalize_filing_status(
        pd.Series(
            [
                "SINGLE",
                "HEAD_OF_HOUSEHOLD",
                "JOINT",
                "SURVIVING_SPOUSE",
                "SEPARATE",
                "UNKNOWN",
            ]
        )
    )

    assert list(statuses) == [
        "Single",
        "Head of household",
        "Joint / surviving spouse",
        "Joint / surviving spouse",
        "Separate",
        "Other",
    ]
    assert list(statuses.categories) == [
        "Single",
        "Head of household",
        "Joint / surviving spouse",
        "Separate",
        "Other",
    ]
    assert statuses.ordered is True


def test_build_ctc_diagnostic_tables_aggregates_weights_by_group():
    frame = pd.DataFrame(
        {
            "adjusted_gross_income": [
                0.0,
                25_000.0,
                25_000.0,
                500_000.0,
            ],
            "filing_status": [
                "SINGLE",
                "JOINT",
                "SURVIVING_SPOUSE",
                "HEAD_OF_HOUSEHOLD",
            ],
            "tax_unit_weight": [
                2.0,
                1.5,
                0.5,
                3.0,
            ],
            "ctc_qualifying_children": [
                1.0,
                2.0,
                4.0,
                0.0,
            ],
            "ctc": [
                100.0,
                200.0,
                0.0,
                0.0,
            ],
            "refundable_ctc": [
                25.0,
                150.0,
                0.0,
                0.0,
            ],
            "non_refundable_ctc": [
                75.0,
                50.0,
                0.0,
                0.0,
            ],
        }
    )

    tables = build_ctc_diagnostic_tables(frame)

    by_agi_band = tables["by_agi_band"].set_index("group")
    assert by_agi_band.loc["<$1", "tax_unit_count"] == 2.0
    assert by_agi_band.loc["<$1", "ctc_qualifying_children"] == 2.0
    assert by_agi_band.loc["<$1", "ctc_recipient_count"] == 2.0
    assert by_agi_band.loc["<$1", "refundable_ctc_recipient_count"] == 2.0
    assert by_agi_band.loc["<$1", "non_refundable_ctc_recipient_count"] == 2.0
    assert by_agi_band.loc["<$1", "ctc"] == 200.0
    assert by_agi_band.loc["<$1", "refundable_ctc"] == 50.0
    assert by_agi_band.loc["<$1", "non_refundable_ctc"] == 150.0

    assert by_agi_band.loc["$25k-$50k", "tax_unit_count"] == 2.0
    assert by_agi_band.loc["$25k-$50k", "ctc_qualifying_children"] == 5.0
    assert by_agi_band.loc["$25k-$50k", "ctc_recipient_count"] == 1.5
    assert by_agi_band.loc["$25k-$50k", "refundable_ctc_recipient_count"] == 1.5
    assert by_agi_band.loc["$25k-$50k", "non_refundable_ctc_recipient_count"] == 1.5
    assert by_agi_band.loc["$25k-$50k", "ctc"] == 300.0
    assert by_agi_band.loc["$25k-$50k", "refundable_ctc"] == 225.0
    assert by_agi_band.loc["$25k-$50k", "non_refundable_ctc"] == 75.0

    by_filing_status = tables["by_filing_status"].set_index("group")
    assert by_filing_status.loc["Single", "tax_unit_count"] == 2.0
    assert by_filing_status.loc["Single", "ctc_qualifying_children"] == 2.0
    assert by_filing_status.loc["Single", "ctc_recipient_count"] == 2.0
    assert by_filing_status.loc["Single", "ctc"] == 200.0

    assert by_filing_status.loc["Joint / surviving spouse", "tax_unit_count"] == 2.0
    assert by_filing_status.loc["Joint / surviving spouse", "ctc_qualifying_children"] == 5.0
    assert by_filing_status.loc["Joint / surviving spouse", "ctc_recipient_count"] == 1.5
    assert by_filing_status.loc["Joint / surviving spouse", "refundable_ctc"] == 225.0
    assert by_filing_status.loc["Joint / surviving spouse", "non_refundable_ctc"] == 75.0
