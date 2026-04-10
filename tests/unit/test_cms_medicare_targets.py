import pytest

from policyengine_us_data.utils.cms_medicare import (
    get_beneficiary_paid_medicare_part_b_premiums_notes,
    get_beneficiary_paid_medicare_part_b_premiums_source,
    get_beneficiary_paid_medicare_part_b_premiums_target,
)


def test_beneficiary_paid_medicare_part_b_target_2024_is_sourced():
    assert get_beneficiary_paid_medicare_part_b_premiums_target(2024) == pytest.approx(
        112e9
    )


def test_beneficiary_paid_medicare_part_b_source_mentions_primary_sources():
    source = get_beneficiary_paid_medicare_part_b_premiums_source(2024)
    assert "2025 Medicare Trustees Report" in source
    assert "State Buy-In FAQ" in source


def test_beneficiary_paid_medicare_part_b_notes_describe_out_of_pocket_semantics():
    notes = get_beneficiary_paid_medicare_part_b_premiums_notes(2024)
    assert "out-of-pocket" in notes
    assert "gross trust-fund premium income" in notes
