import pytest

from policyengine_us_data.utils.cms_medicare import (
    get_beneficiary_paid_medicare_part_b_premiums_notes,
    get_beneficiary_paid_medicare_part_b_premiums_source,
    get_beneficiary_paid_medicare_part_b_premiums_target,
    get_medicare_part_b_enrollment_notes,
    get_medicare_part_b_enrollment_source,
    get_medicare_part_b_enrollment_target,
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


def test_medicare_part_b_enrollment_target_2024_is_sourced():
    assert get_medicare_part_b_enrollment_target(2024) == pytest.approx(62_084_000)


def test_medicare_part_b_enrollment_source_mentions_trustees_table():
    source = get_medicare_part_b_enrollment_source(2024)
    assert "2024 Medicare Trustees Report" in source
    assert "Table V.B3" in source


def test_medicare_part_b_enrollment_notes_describe_complementary_semantics():
    notes = get_medicare_part_b_enrollment_notes(2024)
    assert "separate calibration anchor" in notes
    assert "MSP/state buy-in" in notes
