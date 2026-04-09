"""
Tests for reference-person partner extraction from CPS ASEC.

The public CPS ASEC relationship-to-reference-person variable PERRP identifies
unmarried partners of the household head/reference person. We carry that
through so the SPM childcare cap can distinguish the reference person's partner
from unrelated adults in the same SPM unit.
"""

from pathlib import Path


class TestReferencePartner:
    """Test suite for CPS relationship-to-reference-person extraction."""

    def test_census_cps_includes_perrp(self):
        census_cps_path = Path(__file__).parent.parent.parent / (
            "policyengine_us_data/datasets/cps/census_cps.py"
        )
        content = census_cps_path.read_text()

        assert '"PERRP"' in content, "PERRP should be in PERSON_COLUMNS"

    def test_cps_maps_unmarried_partner_from_perrp(self):
        cps_path = Path(__file__).parent.parent.parent / (
            "policyengine_us_data/datasets/cps/cps.py"
        )
        content = cps_path.read_text()

        assert 'cps["is_unmarried_partner_of_household_head"]' in content
        for code in ("43", "44", "46", "47"):
            assert code in content
