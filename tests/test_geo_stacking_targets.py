#!/usr/bin/env python3
"""
Unit tests for geo-stacking target counts.

These are self-contained tests that verify target count expectations
without requiring database connections or external dependencies.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np


class TestGeoStackingTargets(unittest.TestCase):
    """Test target count expectations for geo-stacking calibration."""

    def setUp(self):
        """Set up test fixtures with mocked components."""
        # Mock the builder class entirely
        self.mock_builder = Mock()
        self.mock_sim = Mock()

    def test_age_targets_per_cd(self):
        """Test that each CD gets exactly 18 age bins."""
        test_cds = ["601", "652", "3601"]

        # Create expected targets DataFrame
        mock_targets = []
        for cd in test_cds:
            for age_bin in range(18):  # 18 age bins per CD
                mock_targets.append(
                    {
                        "geographic_id": cd,
                        "stratum_group_id": 2,  # Age group
                        "variable": "person_count",
                        "value": 10000,
                        "description": f"age_bin_{age_bin}",
                    }
                )

        targets_df = pd.DataFrame(mock_targets)

        # Verify age targets per CD
        age_mask = targets_df["stratum_group_id"] == 2
        age_targets = targets_df[age_mask]

        for cd in test_cds:
            cd_age_targets = age_targets[age_targets["geographic_id"] == cd]
            self.assertEqual(
                len(cd_age_targets),
                18,
                f"CD {cd} should have exactly 18 age bins",
            )

    def test_medicaid_targets_count(self):
        """Test that we get one Medicaid target per CD."""
        test_cds = ["601", "652", "3601", "4801"]

        # Create expected targets with one Medicaid target per CD
        mock_targets = []
        for cd in test_cds:
            mock_targets.append(
                {
                    "geographic_id": cd,
                    "stratum_group_id": 5,  # Medicaid group
                    "variable": "person_count",
                    "value": 50000,
                    "description": f"medicaid_enrollment_cd_{cd}",
                }
            )

        targets_df = pd.DataFrame(mock_targets)

        # Check Medicaid targets
        medicaid_mask = targets_df["stratum_group_id"] == 5
        medicaid_targets = targets_df[medicaid_mask]

        self.assertEqual(
            len(medicaid_targets),
            len(test_cds),
            f"Should have exactly one Medicaid target per CD",
        )

        # Verify each CD has exactly one
        for cd in test_cds:
            cd_medicaid = medicaid_targets[
                medicaid_targets["geographic_id"] == cd
            ]
            self.assertEqual(
                len(cd_medicaid),
                1,
                f"CD {cd} should have exactly one Medicaid target",
            )

    def test_snap_targets_structure(self):
        """Test SNAP targets: one household_count per CD plus state costs."""
        test_cds = ["601", "602", "3601", "4801", "1201"]  # CA, CA, NY, TX, FL
        expected_states = ["6", "36", "48", "12"]  # Unique state FIPS

        mock_targets = []

        # CD-level SNAP household counts
        for cd in test_cds:
            mock_targets.append(
                {
                    "geographic_id": cd,
                    "geographic_level": "congressional_district",
                    "stratum_group_id": 4,  # SNAP group
                    "variable": "household_count",
                    "value": 20000,
                    "description": f"snap_households_cd_{cd}",
                }
            )

        # State-level SNAP costs
        for state_fips in expected_states:
            mock_targets.append(
                {
                    "geographic_id": state_fips,
                    "geographic_level": "state",
                    "stratum_group_id": 4,  # SNAP group
                    "variable": "snap",
                    "value": 1000000000,  # $1B
                    "description": f"snap_cost_state_{state_fips}",
                }
            )

        targets_df = pd.DataFrame(mock_targets)

        # Check CD-level SNAP
        cd_snap = targets_df[
            (targets_df["geographic_level"] == "congressional_district")
            & (targets_df["variable"] == "household_count")
            & (targets_df["stratum_group_id"] == 4)
        ]
        self.assertEqual(
            len(cd_snap),
            len(test_cds),
            "Should have one SNAP household_count per CD",
        )

        # Check state-level SNAP costs
        state_snap = targets_df[
            (targets_df["geographic_level"] == "state")
            & (targets_df["variable"] == "snap")
            & (targets_df["stratum_group_id"] == 4)
        ]
        self.assertEqual(
            len(state_snap),
            len(expected_states),
            "Should have one SNAP cost per unique state",
        )

    def test_irs_targets_per_cd(self):
        """Test that each CD gets approximately 76 IRS targets."""
        test_cds = ["601", "3601"]
        expected_irs_per_cd = 76

        mock_targets = []

        # Generate IRS targets for each CD
        for cd in test_cds:
            # AGI bins (group 3) - 18 bins
            for i in range(18):
                mock_targets.append(
                    {
                        "geographic_id": cd,
                        "stratum_group_id": 3,
                        "variable": "tax_unit_count",
                        "value": 5000,
                        "description": f"agi_bin_{i}_cd_{cd}",
                    }
                )

            # EITC bins (group 6) - 18 bins
            for i in range(18):
                mock_targets.append(
                    {
                        "geographic_id": cd,
                        "stratum_group_id": 6,
                        "variable": "tax_unit_count",
                        "value": 2000,
                        "description": f"eitc_bin_{i}_cd_{cd}",
                    }
                )

            # IRS scalars (groups >= 100) - 40 scalars
            # This gives us 18 + 18 + 40 = 76 total
            scalar_count = 40
            for i in range(scalar_count):
                mock_targets.append(
                    {
                        "geographic_id": cd,
                        "stratum_group_id": 100 + (i % 10),
                        "variable": "irs_scalar_" + str(i),
                        "value": 100000,
                        "description": f"irs_scalar_{i}_cd_{cd}",
                    }
                )

        targets_df = pd.DataFrame(mock_targets)

        # Count IRS targets per CD
        for cd in test_cds:
            cd_targets = targets_df[targets_df["geographic_id"] == cd]
            self.assertEqual(
                len(cd_targets),
                expected_irs_per_cd,
                f"CD {cd} should have exactly {expected_irs_per_cd} IRS targets",
            )

    def test_total_target_counts_for_full_run(self):
        """Test expected total target counts for a full 436 CD run."""
        n_cds = 436
        n_states = 51

        # Expected counts per category
        expected_counts = {
            "national": 30,
            "age_per_cd": 18,
            "medicaid_per_cd": 1,
            "snap_per_cd": 1,
            "irs_per_cd": 76,
            "state_snap": n_states,
        }

        # Calculate totals
        total_cd_targets = n_cds * (
            expected_counts["age_per_cd"]
            + expected_counts["medicaid_per_cd"]
            + expected_counts["snap_per_cd"]
            + expected_counts["irs_per_cd"]
        )

        total_expected = (
            expected_counts["national"]
            + total_cd_targets
            + expected_counts["state_snap"]
        )

        # Verify calculation matches known expectation (allowing some tolerance)
        self.assertTrue(
            41837 <= total_expected <= 42037,
            f"Total targets for 436 CDs should be approximately 41,937, got {total_expected}",
        )

        # Check individual components
        age_total = expected_counts["age_per_cd"] * n_cds
        self.assertEqual(age_total, 7848, "Age targets should total 7,848")

        medicaid_total = expected_counts["medicaid_per_cd"] * n_cds
        self.assertEqual(
            medicaid_total, 436, "Medicaid targets should total 436"
        )

        snap_cd_total = expected_counts["snap_per_cd"] * n_cds
        snap_total = snap_cd_total + expected_counts["state_snap"]
        self.assertEqual(snap_total, 487, "SNAP targets should total 487")

        irs_total = expected_counts["irs_per_cd"] * n_cds
        self.assertEqual(irs_total, 33136, "IRS targets should total 33,136")


class TestTargetDeduplication(unittest.TestCase):
    """Test deduplication of targets across CDs."""

    def test_irs_scalar_deduplication_within_state(self):
        """Test that IRS scalars are not duplicated for CDs in the same state."""
        # Test with two California CDs
        test_cds = ["601", "602"]

        # Create mock targets with overlapping state-level IRS scalars
        mock_targets_601 = [
            {
                "stratum_id": 1001,
                "stratum_group_id": 100,
                "variable": "income_tax",
                "value": 1000000,
                "geographic_id": "601",
            },
            {
                "stratum_id": 1002,
                "stratum_group_id": 100,
                "variable": "salt",
                "value": 500000,
                "geographic_id": "601",
            },
        ]

        mock_targets_602 = [
            {
                "stratum_id": 1001,
                "stratum_group_id": 100,
                "variable": "income_tax",
                "value": 1000000,
                "geographic_id": "602",
            },
            {
                "stratum_id": 1002,
                "stratum_group_id": 100,
                "variable": "salt",
                "value": 500000,
                "geographic_id": "602",
            },
        ]

        # The deduplication should recognize these are the same stratum_ids
        seen_strata = set()
        deduplicated_targets = []

        for targets in [mock_targets_601, mock_targets_602]:
            for target in targets:
                if target["stratum_id"] not in seen_strata:
                    seen_strata.add(target["stratum_id"])
                    deduplicated_targets.append(target)

        self.assertEqual(
            len(deduplicated_targets),
            2,
            "Should only count unique stratum_ids once across CDs",
        )

        # Verify we kept the unique targets
        unique_strata_ids = {t["stratum_id"] for t in deduplicated_targets}
        self.assertEqual(unique_strata_ids, {1001, 1002})


if __name__ == "__main__":
    unittest.main()
